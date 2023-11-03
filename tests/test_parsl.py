from pathlib import Path

from pytest import mark

from emin.parsl import run_molecule, load_config


@mark.parametrize('level', ['xtb', 'mmff94'])
def test_run_function(level):
    # Run single point
    energy, single_runtime, result = run_molecule('C', level, relax=False)
    assert level == 'mmff94' or result.success  # MMFF94 does not use QCEngine

    # Run relaxed
    relaxed_energy, relax_runtime, relaxed_result = run_molecule('C', level, relax=True)
    assert level == 'mmff94' or relaxed_result.energies[-1] == relaxed_energy
    assert relaxed_energy < energy


def test_load_config():
    config_path = Path(__file__).parent / 'files/test-spec.py'
    config = load_config(config_path)
    assert config.executors[0].label == 'test'
