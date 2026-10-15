"""Compiler/identity tests; native exact arithmetic has separate ledger receivers."""
from fractions import Fraction

import pytest

import pops
from pops.codegen.program_emit_amr import _emit_flux_expression_budget
from pops.codegen.program_emit_kernels import _coeff_metadata_cpp, _coeff_metadata_terms


@pytest.mark.parametrize("bad", (0, 7, -1, True, 1.5, 1 << 63))
def test_budget_is_an_explicit_checked_integer(bad):
    with pytest.raises(ValueError, match="max_exact_coefficient_bytes"):
        pops.time.Program("budget", max_exact_coefficient_bytes=bad)


def test_budget_is_frozen_ir_and_native_metadata():
    first = pops.time.Program("budget", max_exact_coefficient_bytes=512)
    second = pops.time.Program("budget", max_exact_coefficient_bytes=1024)
    assert first._serialize()["max_exact_coefficient_bytes"] == 512
    assert first._ir_hash() != second._ir_hash()
    source = _emit_flux_expression_budget(first)
    assert "pops_program_flux_exact_coefficient_bytes() { return UINT64_C(512); }" in source
    first.max_exact_coefficient_bytes = 0
    with pytest.raises(ValueError, match="max_exact_coefficient_bytes"):
        first._serialize()


@pytest.mark.parametrize("value", (1e-6, 5e-324, Fraction(10**100, 3), -Fraction(1, 10**100)))
def test_coefficient_metadata_uses_exact_text_above_int64(value):
    exact = Fraction(value)
    assert _coeff_metadata_terms({1: value}) == ((1, exact.numerator, exact.denominator),)
    source = _coeff_metadata_cpp({1: value})
    assert hex(exact.numerator) in source and hex(exact.denominator) in source
    assert '"' in source


def test_small_exact_metadata_remains_compact_and_power_is_separately_checked():
    assert _coeff_metadata_cpp({1: Fraction(-1, 2)}) == "{{1, -1, 2}}"
    with pytest.raises(OverflowError, match="dt power"):
        _coeff_metadata_terms({1 << 31: Fraction(1, 2)})


def test_large_integer_source_does_not_depend_on_python_decimal_digit_limit():
    numerator = 1 << 20000
    rendered = _coeff_metadata_cpp({1: Fraction(numerator, 3)})
    assert hex(numerator) in rendered and '"0x3"' in rendered
