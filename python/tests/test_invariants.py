"""Spec 3 generic invariants (criterion 24).

An invariant is a typed function StateSet -> Scalar built from a board
``integral(...)`` expression. Nothing about mass / charge / momentum / energy is
hardcoded in the framework: the value is whatever the user writes.
"""
import pytest

physics = pytest.importorskip("adc.physics")
amath = pytest.importorskip("adc.math")


def test_invariant_is_a_generic_integral():
    from adc.math import integral
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    inv = m.invariant("total_momentum_x", expression=integral(mx))
    assert inv.name == "total_momentum_x"
    assert isinstance(inv.value, amath.Integral)
    assert "total_momentum_x" in m.invariants()


def test_invariants_are_not_name_special_cased():
    from adc.math import integral
    m = physics.Model("euler")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    # an arbitrary, physically meaningless invariant name is accepted verbatim:
    inv = m.invariant("banana", expression=integral(rho * rho))
    assert m.invariants()["banana"] is inv
    assert inv.value is not None


def test_invariant_records_the_states_it_ranges_over():
    from adc.math import integral
    m = physics.Model("plasma")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    inv = m.invariant("charge", expression=integral(rho), over=[U])
    assert inv.over == (U,)
