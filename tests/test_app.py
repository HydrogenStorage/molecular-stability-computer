"""Test components of the application"""
from pathlib import Path

from qcelemental.models import AtomicResult, OptimizationResult

from emin.app import load_database, write_result

files_dir = Path(__file__).parent / 'files'


def test_database(tmpdir):
    out_dir = Path(tmpdir) / 'H2O'
    out_dir.mkdir(parents=True)
    energy_file, known_energies = load_database(out_dir, 'test', True)
    assert len(known_energies) == 0
    assert energy_file.exists()

    # Write to the database
    with energy_file.open('a') as fe, open(out_dir / 'records.json', 'w') as fr:
        # Write the result without a result
        write_result('MOL1', 'H2O', (1., 2., None, None), known_energies,
                     energy_database_fp=fe, record_fp=fr, relax=False, level='mmff94')
        assert len(known_energies) == 1

    # Make sure it read property
    energy_file, new_energies = load_database(out_dir, 'mmff94', False)
    assert len(new_energies) == 1

    # Add a version with a result file
    records_path = out_dir / 'records.json'
    with energy_file.open('a') as fe, open(records_path, 'w') as fr:
        record = AtomicResult.parse_file(files_dir / 'water-no-relax.json')
        write_result('MOL2', 'H2O', (1., 2., record.molecule.to_string('xyz'), record), known_energies,
                     energy_database_fp=fe, record_fp=fr, relax=False, level='mmff94', save_result=True)
        assert len(known_energies) == 2
    assert records_path.read_text().startswith('{"id": null, "schema_name": "qcschema_output"')

    # Add a version with a result file
    records_path = out_dir / 'records.json'
    with energy_file.open('a') as fe, open(records_path, 'w') as fr:
        record = OptimizationResult.parse_file(files_dir / 'water-relax.json')
        write_result('MOL2', 'H2O', (1., 2., record.final_molecule.to_string('xyz'), record), known_energies,
                     energy_database_fp=fe, record_fp=fr, relax=True, level='mmff94', save_result=True)
        assert len(known_energies) == 2
    assert records_path.read_text().startswith('{"id": null, "hash_index": null, "schema_name": "qcschema_optimization_output",')
