"""Lossless integral v2 encoding is checked before declaration publication."""
from fractions import Fraction
import hashlib
import json

import pops
import pytest

from pops._ir.quantity import PhysicalDimension
from pops.time._program.integrals import integral_units_bytes
from tests.python.unit.codegen.test_integral_candidate_capture import build_feedback


@pytest.mark.parametrize("powers", [
    (("\ud800\udc00", 1),),
    (("\ud800\udc00", 1), ("\ue000", 1)),
    (("\ud800\udc00", 1), ("\U00010000", 2)),
])
def test_ambiguous_surrogate_names_refuse_before_declaration_mutation(powers):
    units = PhysicalDimension(powers)
    # PhysicalDimension's historical general-purpose constructor remains intact.
    assert len(units.powers) == len(powers)
    program = pops.Program("roundtrip_atomic")
    old = program.integral_state("old", initial=.9)
    before = program._serialize()
    state_table, unit_table = program._integral_states, program._integral_units
    states, dimensions = dict(state_table), dict(unit_table)
    with pytest.raises(ValueError, match="^integral units require a lossless canonical PhysicalDimension JSON roundtrip$"):
        integral_units_bytes(units)
    with pytest.raises(ValueError, match="^integral units require a lossless canonical PhysicalDimension JSON roundtrip$"):
        program.integral_state("new", initial=.7, units=units)
    assert program._serialize() == before
    assert program._integral_states is state_table and program._integral_states == states
    assert program._integral_units is unit_table and program._integral_units == dimensions
    assert old.program is program
    # The same name can be published correctly after the failed attempt.
    new = program.integral_state("new", initial=.7, units=PhysicalDimension())
    assert new.name == "new" and program._integral_states["new"] == .7


@pytest.mark.parametrize("units", [
    PhysicalDimension(),
    PhysicalDimension((("énergie", Fraction(-7, 3)), ("時間", Fraction(2, 5)))),
    PhysicalDimension((("\ue000", 1), ("\U00010000", Fraction(3, 7)))),
    PhysicalDimension((("large", Fraction(2**60 + 1, 2**60)),)),
])
def test_true_unicode_and_large_rationals_keep_exact_bytes_and_identity(units):
    original = json.dumps(units.to_data(), sort_keys=True, separators=(",", ":"))
    assert integral_units_bytes(units) == original
    assert PhysicalDimension.from_data(json.loads(original)) == units
    program = pops.Program("roundtrip_valid")
    quantity = program.integral_state("q", initial=.7, units=units)
    assert program._integral_units["q"] is units
    assert "/" + hashlib.sha256(original.encode()).hexdigest() + "/q" in quantity.identity


def test_giant_fraction_keeps_bytes_without_changing_existing_cbor_integer_contract():
    units = PhysicalDimension((("huge", Fraction(-(10**2048 + 1), 10**1024)),))
    original = json.dumps(units.to_data(), sort_keys=True, separators=(",", ":"))
    assert integral_units_bytes(units) == original
    assert PhysicalDimension.from_data(json.loads(original)) == units
    program = pops.Program("roundtrip_giant")
    quantity = program.integral_state("q", initial=.7, units=units)
    assert program._integral_units["q"] is units
    # The general identity codec's existing signed-int64 bound is separate from
    # the unbounded unit JSON reader/roundtrip guard. This lot does not extend it.
    with pytest.raises(OverflowError, match="outside signed int64"):
        _ = quantity.identity


def test_none_keeps_untyped_legacy_declaration_and_explicit_capture_refusal():
    program = pops.Program("roundtrip_legacy")
    quantity = program.integral_state("q", initial=.7)
    assert program._integral_units == {}
    assert "integral_units_v2" not in program._serialize()
    assert quantity.identity.startswith("pops.integral.v1/")
    with pytest.raises(ValueError, match="explicitly declared physical units"):
        program.integral_value(quantity, at=program.clock, scope="candidate")


@pytest.mark.parametrize("base", ("\ud800", "\udc00"))
def test_unpaired_surrogates_roundtrip_but_existing_public_cbor_refuses(base):
    units = PhysicalDimension(((base, 1),))
    # This guard does not broaden its purpose into general Unicode restrictions.
    assert integral_units_bytes(units) == json.dumps(units.to_data(), sort_keys=True, separators=(",", ":"))
    case, _, _, _, _ = build_feedback(units=units)
    with pytest.raises(ValueError, match="not valid Unicode"):
        pops.validate(case)
