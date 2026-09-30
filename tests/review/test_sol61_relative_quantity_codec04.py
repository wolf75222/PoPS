"""Independent binary64 oracles for the native .inc fixture; no PoPS import.

The two triples pin authenticated native diagnostic occurrences. This suite
qualifies the fixture mathematics and dispatch source, not C++ codec execution.
"""
from fractions import Fraction
from pathlib import Path
import re
import struct

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROWS = (
    ("abc", "-0x1.d7e1ff2b2db0ap-11", "0x1.8004eaba28d6cp+1", "-0x1.da88051ea8000p-15",
     "-0x1.7ee75a8e3ebabp-11", "-0x1.7ee75a8e3ebaap-11", (.5, 1., 2.)),
    ("cab", "0x1.134759d5cf776p-9", "0x1.8009d3c5cf16ep+1", "0x1.da88051ea9000p-15",
     "0x1.fa10cfbb6fc95p-10", "0x1.fa10cfbb6fc96p-10", (.5, 2., 1.)),
)


def bits(value):
    return struct.pack("<d", value)


@pytest.mark.parametrize("row", ROWS, ids=("abc", "cab"))
@pytest.mark.parametrize("component", range(3))
def test_three_component_independent_literals_and_reynolds(row, component):
    _, p, u, s, f, o, factors = row
    factor = factors[component]
    physical, density, sweep = float.fromhex(p)*factor, float.fromhex(u)*factor, float.fromhex(s)
    fused = float(Fraction(physical) - Fraction(density)*Fraction(sweep))
    product = float(Fraction(density)*Fraction(sweep))
    ordinary = float(Fraction(physical) - Fraction(product))
    assert bits(fused) == bits(float.fromhex(f)*factor)
    assert bits(ordinary) == bits(float.fromhex(o)*factor)
    assert bits(fused) != bits(ordinary)
    nodes = (0., .125+sweep, .25)
    assert nodes[1]-.125 == sweep
    for cell, (left, right) in enumerate(((physical, fused), (fused, physical))):
        volume = nodes[cell+1]-nodes[cell]
        current = (density*.125 + left - right)/volume
        residual = (current*volume-density*.125)+(right-left)
        assert volume > 0 and abs(residual) <= 64*2**-52


def test_new_format_has_separate_named_arithmetic_and_historical_branch():
    geometry = (ROOT / "include/pops/runtime/program/moving_interval_geometry.hpp").read_text()
    codec = (ROOT / "include/pops/runtime/program/moving_interval_checkpoint.hpp").read_text()
    producer = (ROOT / "include/pops/runtime/program/program_context_moving_interval.inc").read_text()
    assert "kLegacyExpression = 0, kFused = 1" in geometry
    assert "std::fma(-density, sweep, physical_amount)" in geometry
    assert "return physical_amount - density * sweep;" in geometry
    assert "candidate->checkpoint_wire_version=4" in producer.replace(" ", "")
    assert "moving_relative_face_amount(flux(offset),density(offset),sw(face))" in producer.replace(" ", "")
    assert "historical ? std::uint64_t{0} : in.u64()" in codec
    assert 'require(convention<=1,"unknown relative face amount realization")' in codec
    assert "geometry.checkpoint_wire_version=historical ? 3 : 4" in codec
    assert "out.u64(geometry.checkpoint_wire_version)" in codec
    assert "out.u64(static_cast<std::uint64_t>(geometry.last_receipt->relative_amount_convention))" in codec


def test_legacy_write_order_is_identical_to_frozen_old_layout():
    codec = (ROOT / "include/pops/runtime/program/moving_interval_checkpoint.hpp").read_text()
    writer = codec.split("Writer out; out.raw(historical ? legacy_magic : magic);", 1)[1]
    writer = writer.split("return std::move(out).take();", 1)[0]
    # An explicit list of wire-emitting calls from historical POPSEX03. New
    # receipt tag is omitted on that branch; no canonicalizing rewrite occurs.
    writer = re.sub(r"if \(!historical\) out.u64\([^;]+;", "", writer)
    calls = re.findall(r"out\.(bytes|size|string|i32|real|u64)\(|write_(field|faces|point)\(", writer)
    assert tuple(a or b for a, b in calls) == (
        "bytes", "size", "string", "i32", "string", "string", "real", "u64", "string",
        "field", "field", "faces", "faces", "u64", "point", "string", "string", "real",
        "field", "field", "field", "faces", "faces", "faces",
    )
