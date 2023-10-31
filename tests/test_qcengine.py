"""Test computing the atomizaton energy of a molecule"""
from qcelemental.models.procedures import QCInputSpecification

from emin.qcengine import generate_xyz, relax_molecule


def test_methane():
    xyz = generate_xyz('C')
    spec = QCInputSpecification(
        driver='gradient',
        model={'method': 'GFN2-xTB'},
        keywords={"accuracy": 0.05}
    )
    result = relax_molecule(xyz, 'xtb', spec)
    assert result.success
    assert result.energies[-1]
