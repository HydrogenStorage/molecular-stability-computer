"""Utilities associated with running functions via Parsl

Includes functions to be run remotely and other utilities
"""
from pathlib import Path
from time import perf_counter

from parsl import Config
from qcelemental.models import AtomicResult
from qcelemental.models.procedures import OptimizationResult

from emin.qcengine import generate_xyz, relax_molecule, get_qcengine_spec, compute_energy, evaluate_mmff94


def run_molecule(smiles: str, level: str, relax: bool = True, return_full_record: bool = True) \
        -> tuple[float, float, str | None, AtomicResult | OptimizationResult | None]:
    """Compute the energy of a molecule

    Args:
        smiles: SMILES string of the molecule
        level: Level of accuracy to run
        relax: Whether to relax the molecule
        return_full_record: Whether to return the full result record
    Returns:
        - Energy. ``inf`` if the computation failed
        - Runtime
        - XYZ of molecule
        - Complete record of the optimization
    """

    start_time = perf_counter()

    # Special case: MMFF94
    if level == 'mmff94':
        try:
            return evaluate_mmff94(smiles, relax), perf_counter() - start_time, None, None
        except ValueError:
            return float('inf'), perf_counter() - start_time, None, None

    # Make a QCengine spec
    code, spec = get_qcengine_spec(level)

    # Generate an initial structure
    try:
        xyz = generate_xyz(smiles)
    except ValueError:
        return float('inf'), perf_counter() - start_time, None, None

    if relax:
        # Relax, if requested
        result = relax_molecule(xyz, code, spec)

        # If the result was successful, get the energy
        energy = float('inf')
        if result.success:
            energy = result.energies[-1]

        return energy, perf_counter() - start_time, result.final_molecule.to_string('xyz'), result if return_full_record else None
    else:
        result = compute_energy(xyz, code, spec)
        energy = result.return_result if result.success else float('inf')
        return energy, perf_counter() - start_time, result.molecule.to_string('xyz'), result if return_full_record else None


def load_config(path: str | Path, var_name: str = 'config') -> Config:
    """Load a configuration from a file

    Args:
        path: Path to the configuration file
        var_name: Name of the configuration within that file
    Returns:
        What should be a Parsl configuration
    """

    spec_ns = {}
    path = Path(path)
    exec(path.read_text(), spec_ns)
    if var_name not in spec_ns:
        raise ValueError(f'Variable {var_name} not found in {path}')

    return spec_ns[var_name]
