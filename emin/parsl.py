"""Utilities associated with running functions via Parsl

Includes functions to be run remotely and other utilities
"""
from qcelemental.models import AtomicResult
from qcelemental.models.procedures import OptimizationResult

from emin.qcengine import generate_xyz, relax_molecule, get_qcengine_spec, compute_energy


def run_molecule(smiles: str, level: str, relax: bool = True) -> tuple[float, AtomicResult | OptimizationResult]:
    """Compute the energy of a molecule

    Args:
        smiles: SMILES string of the molecule
        level: Level of accuracy to run
        relax: Whether to relax the molecule
    Returns:
        - Energy. ``None`` if the computation failed
        - Complete record of the optimization
    """

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
