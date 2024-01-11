"""Compute the E_min of a target molecule"""
from functools import partial, update_wrapper
from concurrent.futures import Future
from argparse import ArgumentParser
from threading import Semaphore
from typing import Iterable
from pathlib import Path
import logging
import gzip
import sys

import parsl
from parsl import Config, HighThroughputExecutor, python_app, ThreadPoolExecutor
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem, RDLogger

from emin.app import load_database, write_result
from emin.generate import generate_molecules_with_surge, get_random_selection_with_surge
from emin.parsl import run_molecule, load_config
from emin.source import get_molecules_from_pubchem

RDLogger.DisableLog('rdApp.*')


def get_key(smiles: str) -> str:
    """Get InChI key from a SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'SMILES failed to parse: {smiles}')
    return Chem.MolToInchiKey(mol)


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--level', default='xtb', help='Accuracy level at which to compute energies')
    parser.add_argument('--no-relax', action='store_true', help='Skip relaxing the molecular structure')
    parser.add_argument('--skip-store', action='store_true', help='Skip storing the full QCEngine record')
    parser.add_argument('--num-parallel', default=10000, type=int,
                        help='Maximum number of chemistry computations to run at the same time')
    parser.add_argument('--compute-config',
                        help='Path to the file defining the Parsl configuration. Configuration should be in variable named ``config``')
    parser.add_argument('--surge-amount', type=float,
                        help='Maximum number or fraction of molecules to generate from Surge. Set to 0 or less to disable surge. Default is to run all')
    parser.add_argument('molecule', help='SMILES or InChI of the target molecule')
    args = parser.parse_args()

    # Parse the molecule
    if args.molecule.startswith('InChI='):
        mol = Chem.MolFromInchi(args.molecule)
    else:
        mol = Chem.MolFromSmiles(args.molecule)
    if mol is None:
        raise ValueError(f'Molecule failed to parse: "{args.molecule}"')

    # Get the composition of the molecule, which will define our output directory
    formula = rdMolDescriptors.CalcMolFormula(mol)
    out_dir = Path('runs') / formula
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up the logging
    handlers = [logging.FileHandler(out_dir / 'runtime.log'), logging.StreamHandler(sys.stdout)]

    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    our_key = Chem.MolToInchiKey(mol)
    logger.info(f'Starting E_min run for {args.molecule} (InChI Key: {our_key}) in {out_dir}')
    logger.info(f'Running accuracy level: {args.level}. Relaxation: {not args.no_relax}')

    # Make the Parsl configuration
    if args.compute_config is None:
        logger.info('Using default Parsl configuration of a single worker on the local machine')
        config = Config(
            executors=[HighThroughputExecutor(max_workers=1, address='127.0.0.1')]
        )
    else:
        logger.info(f'Loading Parsl configuration from {args.compute_config}')
        config = load_config(args.compute_config)

    compute_execs = [e.label for e in config.executors]
    config.executors = list(config.executors) + [ThreadPoolExecutor(max_threads=1, label='writer')]  # Add a process that only writes
    config.run_dir = str(out_dir / 'runinfo')

    dfk = parsl.load(config)

    pinned_fun = partial(run_molecule, level=args.level, relax=not args.no_relax)
    update_wrapper(pinned_fun, run_molecule)
    run_app = python_app(pinned_fun, executors=compute_execs)
    logger.info('Started Parsl and created the app to be run')

    # Load any previous computations
    energy_file, known_energies = load_database(out_dir, args.level, not args.no_relax)
    logger.info(f'Loaded {len(known_energies)} energies from previous runs')

    # Open the output files
    result_file = out_dir / 'results.json.gz'
    with gzip.open(result_file, 'at') as fr, energy_file.open('a') as fe:
        # Make utility functions
        write_fn = partial(write_result, relax=not args.no_relax, level=args.level, energy_database_fp=fe, record_fp=fr)
        update_wrapper(write_fn, write_result)
        write_app = python_app(write_fn, executors=['writer'])

        def _run_if_needed(my_smiles: str, my_save_results: bool = True) -> tuple[bool, str, float | Future]:
            """Get the energy either by looking up result or running a new computation

            Returns:
                - Whether the energy is done now
                - The InChI Key for the molecule
                - Either the energy or a future with the label "key" associated with it
            """
            my_key = get_key(my_smiles)
            if my_key not in known_energies:
                future = run_app(my_smiles, return_full_record=my_save_results)
                return False, my_key, future
            else:
                return True, my_key, known_energies[my_key]

        # Start by running our molecule
        our_smiles = Chem.MolToSmiles(mol)
        is_done, our_key, our_energy = _run_if_needed(our_smiles)
        if not is_done:
            our_write = write_app(our_key, our_smiles, our_energy, known_energies, save_result=True)
            our_energy, runtime, xyz, result = our_energy.result()
            our_write.result()  # Make sure we write
        logger.info(f'Target molecule has an energy of {our_energy:.3f} Ha')

        # Gather molecules from PubChem
        pubchem = get_molecules_from_pubchem(formula)
        logger.info(f'Pulled {len(pubchem)} molecules for {formula} from PubChem')

        def _submit_all(my_mol_list: Iterable[str], my_warnings: bool = False, my_save_results: bool = False) -> int:
            """Run all molecules

            Returns:
                Number submitted
            """
            count = 0  # Number of new computations

            # Submit all the molecules
            submit_controller = Semaphore(max(args.num_parallel, 2))  # Control the maximum number of submissions
            for my_smiles in my_mol_list:
                try:
                    submit_controller.acquire()  # Block until resources are freed by the callback
                    my_is_done, my_key, my_result = _run_if_needed(my_smiles, my_save_results)
                except ValueError:
                    if my_warnings:
                        logger.warning(f'Failed to parse SMILES: {my_smiles}')
                    continue

                # Add the write app, if needed
                if not my_is_done:
                    count += 1
                    my_result.add_done_callback(lambda x: submit_controller.release())
                    write_app(my_key, my_smiles, my_result, known_energies, save_result=my_save_results)
                else:
                    submit_controller.release()  # We didn't create a future

            # Block until all finish
            dfk.wait_for_current_tasks()
            return count

        # Run them all
        before_count = len(known_energies)
        submit_count = _submit_all(pubchem, my_warnings=True, my_save_results=True)
        success_count = len(known_energies) - before_count

        logger.info(f'Successfully ran {success_count} molecules from PubChem of {submit_count} submitted')
        logger.info(f'Emin of molecule compared to PubChem: {(our_energy - min(known_energies.values())) * 1000:.1f} mHa')

        # Test molecules from surge
        if args.surge_amount is not None and args.surge_amount <= 0:
            logger.info('Skipping comparison against Surge-generated molecules')
        else:
            if args.surge_amount is None:
                logger.info('Comparing against all molecules generated by Surge')
                mol_list = generate_molecules_with_surge(formula)
            else:
                logger.info(f'Selecting a random subset from Surge molecules. Amount to select: {args.surge_amount}')
                mol_list, total = get_random_selection_with_surge(formula, to_select=args.surge_amount)
                logger.info(f'Selected {len(mol_list)} molecules out of {total} created by Surge. ({len(mol_list) / total * 100:.3g}%)')

            # Run them all
            before_count = len(known_energies)
            submit_count = _submit_all(mol_list, my_warnings=False, my_save_results=not args.skip_store)
            success_count = len(known_energies) - before_count
            logger.info(f'Completed {success_count} molecules from Surge of {submit_count} submitted')

        logger.info(f'Final E_min compared against {len(known_energies)} molecules: {(our_energy - min(known_energies.values())) * 1000: .1f} mHa')
