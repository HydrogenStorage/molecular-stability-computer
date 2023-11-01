"""Utilities associated with running functions via Parsl

Includes functions to be run remotely and other utilities
"""
from qcelemental.models.procedures import QCInputSpecification, OptimizationResult

from emin.qcengine import generate_xyz, relax_molecule


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
