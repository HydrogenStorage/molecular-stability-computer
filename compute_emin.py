"""Compute the E_min of a target molecule"""
from concurrent.futures import Future, as_completed
from functools import partial, update_wrapper
from argparse import ArgumentParser
from pathlib import Path
import logging
import gzip
import sys

import parsl
from parsl import Config, HighThroughputExecutor, python_app
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem, RDLogger

from emin.generate import generate_molecules_with_surge, get_random_selection_with_surge
from emin.parsl import run_molecule
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

    # Start Parsl
    config = Config(
        executors=[HighThroughputExecutor(max_workers=1, address='127.0.0.1')]
    )
    parsl.load(config)
    pinned_fun = partial(run_molecule, level=args.level, relax=not args.no_relax)
    update_wrapper(pinned_fun, run_molecule)
    run_app = python_app(pinned_fun)
    logger.info('Started Parsl and created the app to be run')

    # Load any previous computations
    energy_file: Path = out_dir / 'energies.csv'
    known_energies = {}
    if not energy_file.exists():
        energy_file.write_text('inchi_key,smiles,level,relax,energy\n')
    with energy_file.open() as fp:
        fp.readline()  # Skip the header
        for line in fp:
            key, _, level, relax, our_energy = line.strip().split(",")
            if level == args.level and bool(relax) != args.no_relax:
                known_energies[key] = float(our_energy)
    logger.info(f'Loaded {len(known_energies)} energies from previous runs')

    # Open the output files
    result_file = out_dir / 'results.json.gz'
    with gzip.open(result_file, 'at') as fr, energy_file.open('a') as fe:
        # Make utility functions
        def _store_result(new_key, new_smiles, new_energy, result):
            known_energies[new_key] = new_energy
            print(result.json(), file=fr)
            print(f'{new_key},{new_smiles},{args.level},{not args.no_relax},{new_energy}', file=fe)

        def _run_if_needed(my_smiles: str) -> tuple[bool, float | Future]:
            """Get the energy either by looking up result or running a new computation

            Returns:
                - Whether the energy is done now
                - Either the energy or a future with the label "key" associated with it
            """
            my_key = get_key(my_smiles)
            if my_key not in known_energies:
                future = run_app(my_smiles)
                future.key = my_key
                future.smiles = my_smiles
                return False, future
            else:
                return True, known_energies[my_key]

        # Start by running our molecule
        our_smiles = Chem.MolToSmiles(mol)
        is_done, our_energy = _run_if_needed(our_smiles)
        if not is_done:
            our_energy, result = our_energy.result()
            _store_result(our_key, our_smiles, our_energy, result)
        logger.info(f'Target molecule has an energy of {our_energy:.3f} Ha')

        # Gather molecules from PubChem
        pubchem = get_molecules_from_pubchem(formula)
        logger.info(f'Pulled {len(pubchem)} molecules for {formula} from PubChem')

        # Submit them all
        futures = []
        failures = 0
        for smiles in pubchem:
            try:
                is_done, result = _run_if_needed(smiles)
                if not is_done:
                    futures.append(result)
            except ValueError:
                logger.warning(f'Failed to parse SMILES from PubChem: {smiles}')
                failures += 1

        # Wait for the results to finish
        logger.info(f'Waiting for {len(futures)} computations to finish')

        def _process_futures(my_futures: list[Future]) -> int:
            my_failures = 0
            for future in as_completed(my_futures):
                if future.exception() is not None:
                    logger.warning(f'Failure running {future.key}: {future.exception()}')
                    my_failures += 1
                else:
                    energy, result = future.result()
                    _store_result(future.key, future.smiles, energy, result)
            return my_failures

        failures = _process_futures(futures)

        logger.info(f'Successfully ran {len(pubchem) - failures}/{len(pubchem)} molecules from PubChem')
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
                logger.info(f'Selected {len(mol_list)} molecules out of {total} created by Surge')

            # Run each
            surge_count = 0
            current_min = min(known_energies.values())

            futures = []
            for smiles in mol_list:
                surge_count += 1
                is_done, result = _run_if_needed(smiles)
                if not is_done:
                    futures.append(result)
            logger.info(f'Generated {surge_count} molecules and submitted {len(futures)} to run')

            failures = _process_futures(futures)
            logger.info(f'Ran {len(futures)} molecules from Surge. {failures} failed.')

        logger.info(f'Final E_min compared against {len(known_energies)} molecules: {(our_energy - min(known_energies.values())) * 1000: .1f} mHa')
