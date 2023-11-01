"""Utilities associated with running functions via Parsl

Includes functions to be run remotely and other utilities
"""
from pathlib import Path

from parsl import Config
from qcelemental.models import AtomicResult
from qcelemental.models.procedures import OptimizationResult

from emin.qcengine import generate_xyz, relax_molecule, get_qcengine_spec, compute_energy, evaluate_mmff94


def run_molecule(smiles: str, level: str, relax: bool = True) -> tuple[float, AtomicResult | OptimizationResult | None]:
    """Compute the energy of a molecule

    Args:
        smiles: SMILES string of the molecule
        level: Level of accuracy to run
        relax: Whether to relax the molecule
    Returns:
        - Energy. ``None`` if the computation failed
        - Complete record of the optimization
    """

    # Special case: MMFF94
    if level == 'mmff94':
        return evaluate_mmff94(smiles, relax), None

    # Make a xTB spec
    code, spec = get_qcengine_spec(level)

    # Generate an initial structure
    xyz = generate_xyz(smiles)

    if relax:
        # Relax, if requested
        result = relax_molecule(xyz, code, spec)

        # If the result was successful, get the energy
        energy = None
        if result.success:
            energy = result.energies[-1]

        return energy, result
    else:
        result = compute_energy(xyz, code, spec)
        return result.return_result, result


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
