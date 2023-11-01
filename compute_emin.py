"""Compute the E_min of a target molecule"""
from argparse import ArgumentParser
from pathlib import Path
import logging
import gzip
import sys

from qcelemental.models.procedures import QCInputSpecification, OptimizationResult
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem, RDLogger

from emin.generate import generate_molecules_with_surge, get_random_selection_with_surge
from emin.qcengine import relax_molecule, generate_xyz
from emin.source import get_molecules_from_pubchem

RDLogger.DisableLog('rdApp.*')


def get_key(smiles: str) -> str:
    """Get InChI key from a SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'SMILES failed to parse: {smiles}')
    return Chem.MolToInchiKey(mol)


def run_molecule(smiles: str) -> tuple[float, OptimizationResult]:
    """Compute the energy of a molecule

    Args:
        smiles: SMILES string of the molecule
    Returns:
        - Energy. ``None`` if the computation failed
        - Complete record of the optimization
    """

    # Make a xTB spec
    spec = QCInputSpecification(
        driver='gradient',
        model={'method': 'GFN2-xTB'},
        keywords={"accuracy": 0.05}
    )

    # Run the relaxation
    xyz = generate_xyz(smiles)
    result = relax_molecule(xyz, 'xtb', spec)

    # If the result was successful, get the energy
    energy = None
    if result.success:
        energy = result.energies[-1]

    return energy, result


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
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

    # Load any previous computations
    energy_file: Path = out_dir / 'energies.csv'
    known_energies = {}
    if not energy_file.exists():
        energy_file.write_text('inchi_key,smiles,energy\n')
    with energy_file.open() as fp:
        fp.readline()  # Skip the header
        for line in fp:
            key, _, our_energy = line.strip().split(",")
            known_energies[key] = float(our_energy)
    logger.info(f'Loaded {len(known_energies)} energies from previous runs')

    # Open the output files
    result_file = out_dir / 'results.json.gz'
    with gzip.open(result_file, 'at') as fr, energy_file.open('a') as fe:
        # Make utility functions
        def _append_energy(new_key, new_smiles, new_energy):
            print(f'{new_key},{new_smiles},{new_energy}', file=fe)


        def _run_if_needed(smiles: str) -> float:
            """Get the energy either by looking up result or running a new computation"""
            key = get_key(smiles)
            if key not in known_energies:
                energy, result = run_molecule(Chem.MolToSmiles(mol))
                known_energies[key] = energy
                print(result.json(), file=fr)
                _append_energy(key, smiles, energy)
                return energy
            else:
                return known_energies[key]


        # Start by running our molecule
        our_energy = _run_if_needed(Chem.MolToSmiles(mol))
        logger.info(f'Target molecule has an energy of {our_energy:.3f} Ha')

        # Test molecules from PubChem
        pubchem = get_molecules_from_pubchem(formula)
        logger.info(f'Pulled {len(pubchem)} molecules for {formula} from PubChem')
        failures = 0
        for smiles in pubchem:
            try:
                energy = _run_if_needed(smiles)
            except ValueError:
                logger.warning(f'Failed to parse SMILES from PubChem: {smiles}')
                failures += 1
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
                logger.info(f'Selected {len(mol_list)} molecules out of {total}')

            # Run each
            surge_count = 0
            current_min = min(known_energies.values())

            for smiles in mol_list:
                surge_count += 1
                energy = _run_if_needed(smiles)
                if energy < current_min:
                    current_min = energy
                    logger.info(f'Updated E_min to ({(our_energy - current_min) * 1000:.1f}) mHa')
            logger.info(f'Ran {surge_count} molecules from Surge')

        logger.info(f'Final E_min compared against {len(known_energies)} molecules: {(our_energy - min(known_energies.values())) * 1000: .1f} mHa')
