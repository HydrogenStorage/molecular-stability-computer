from pytest import raises

from emin.generate import generate_molecules_with_surge


def test_small():
    waters = list(generate_molecules_with_surge('H2O'))
    assert len(waters) == 1

    propanes = list(generate_molecules_with_surge('C3H8'))
    assert len(propanes) == 1

    propenes = list(generate_molecules_with_surge('C3H6'))
    assert len(propenes) == 2


def test_failure():
    with raises(ValueError) as exc:
        next(iter(generate_molecules_with_surge('X46')))
    assert 'unknown element name' in str(exc.value)
