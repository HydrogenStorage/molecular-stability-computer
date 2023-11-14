import math
from pathlib import Path

from pytest import mark

from emin.parsl import run_molecule, load_config


@mark.parametrize('level', ['xtb', 'mmff94', 'hf_def2-svpd'])
def test_run_function(level):
    # Run single point
    energy, single_runtime, xyz, result = run_molecule('C', level, relax=False)
    assert level == 'mmff94' or result.success  # MMFF94 does not use QCEngine

    # Run relaxed
    relaxed_energy, relax_runtime, xyz, relaxed_result = run_molecule('C', level, relax=True)
    assert level == 'mmff94' or relaxed_result.energies[-1] == relaxed_energy
    assert relaxed_energy < energy

    # Skip sending back the result
    _, _, xyz, result = run_molecule('C', level, relax=True, return_full_record=False)
    assert level == 'mmff94' or xyz is not None
    assert result is None

    _, _, xyz, result = run_molecule('C', level, relax=False, return_full_record=False)
    assert level == 'mmff94' or xyz is not None
    assert result is None


def test_load_config():
    config_path = Path(__file__).parent / 'files/test-spec.py'
    config = load_config(config_path)
    assert config.executors[0].label == 'test'


@mark.parametrize('level', ['mmff94', 'xtb'])
def test_failure(level):
    energy, time, xyz, _ = run_molecule('c1ccc2c3cc(cc2c1)CC3', level)
    assert math.isinf(energy)
    assert not math.isinf(time)
    assert xyz is None
