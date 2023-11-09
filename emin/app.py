"""Functions relating to executing the full Emin workflow"""
from csv import DictReader
from typing import TextIO
from pathlib import Path
import json

from qcelemental.models import OptimizationResult, AtomicResult


def load_database(run_path: str | Path, level: str, relax: bool) -> tuple[Path, dict[str, float]]:
    """Load the energies of molecules for our known settings

    Creates the output file for the run if needed

    Args:
        run_path: Path to the run directory
        level: Quantum chemistry level
        relax: Whether to relax or not
    Returns:
        - Path to the output file
        - Mapping of InChI key to energy computed using the desired settings
    """

    energy_file = Path(run_path) / 'energies.csv'

    output = {}
    if not energy_file.exists():
        energy_file.write_text('inchi_key,smiles,level,relax,energy,runtime,xyz\n')
    with energy_file.open() as fp:
        reader = DictReader(fp)
        for row in reader:
            if row['level'] == level and row['relax'] == str(relax):
                output[row['inchi_key']] = float(row['energy'])

    return energy_file, output


def write_result(new_key: str, new_smiles: str,
                 compute_result: tuple[float, float, str | None, OptimizationResult | AtomicResult | None],
                 known_energies: dict[str, float],
                 energy_database_fp: TextIO,
                 record_fp: TextIO,
                 level: str,
                 relax: bool,
                 save_result: bool = False):
    """Write the result of a computation to disk and update the database files

    Args:
        new_key: InChI key of the molecule
        new_smiles: SMILES string of the molecule
        compute_result: Output from :meth:`~emin.parsl.run_molecule`
        known_energies: Database of energies of molecule given InChI Key
        energy_database_fp: Handle to the file which records energies
        record_fp: Handle to the file holding any result logs
        level: Level of the quantum chemistry computation
        relax: Whether the geometry was optimized
        save_result: Whether to save the result log to disk
    """
    # Resolve the future
    new_energy, new_runtime, new_xyz, new_result = compute_result

    # Always save the energy and such
    if new_result is None or new_result.success:
        known_energies[new_key] = new_energy
        print(f'{new_key},{new_smiles},{level},{relax},{new_energy},{new_runtime},{json.dumps(new_xyz)}', file=energy_database_fp)

    # Save the result only if the user wants
    if new_result is not None and save_result:
        print(new_result.json(), file=record_fp)
