from emin.parsl import run_molecule


def test_run_function():
    energy, result = run_molecule('C')
    assert result.energies[-1] == energy
