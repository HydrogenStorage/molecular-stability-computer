"""Test computing the atomizaton energy of a molecule"""
from emin.qcengine import generate_xyz, relax_molecule, compute_energy, get_qcengine_spec


def test_methane():
    xyz = generate_xyz('C')
    code, spec = get_qcengine_spec('xtb')

    # Single point energy
    eng_result = compute_energy(xyz, code, spec)
    assert eng_result.success

    # Relaxation
    opt_result = relax_molecule(xyz, code, spec)
    assert opt_result.success
    assert opt_result.energies[-1] < eng_result.return_result
