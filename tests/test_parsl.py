from emin.parsl import run_molecule


def test_run_function():
    # Run single point
    energy, result = run_molecule('C', 'xtb', relax=False)
    assert result.success

    # Run relaxed
    relaxed_energy, relaxed_result = run_molecule('C', 'xtb', relax=True)
    assert relaxed_result.energies[-1] == relaxed_energy
    assert relaxed_energy < energy
