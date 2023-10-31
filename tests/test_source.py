from emin.source import get_molecules_from_pubchem


def test_no_isotopes_or_charge():
    waters = get_molecules_from_pubchem('H2O')
    assert len(waters) == 1  # There is but one true water
