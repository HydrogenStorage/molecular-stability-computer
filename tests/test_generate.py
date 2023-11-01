from pytest import raises

from emin.generate import generate_molecules_with_surge, get_random_selection_with_surge


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


def test_fixed_amount():
    # Test only a certain number
    fixed_number, total = get_random_selection_with_surge('C3H6', 1)
    assert len(fixed_number) == 1
    assert total == len(list(generate_molecules_with_surge('C3H6')))

    # Test only a certain fraction
    fixed_frac, _ = get_random_selection_with_surge('C3H6', 0.5)
    assert len(fixed_frac) == 1

    # Test a bigger case
    larger_case, _ = get_random_selection_with_surge('C6H12', to_select=10)
    assert len(larger_case) == 10
