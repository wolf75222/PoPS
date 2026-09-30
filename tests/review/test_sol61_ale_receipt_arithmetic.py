"""Independent exact binary64 arithmetic proof; no PoPS/native import or execution.

Triples were reconstructed from authenticated saved pre-step data and the declared
projection. The native diagnostic then confirmed the resulting ordinary/fused
quantity bits at the exact same accepted-exchange occurrences.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

import pytest

BASE = Path(__file__).resolve().parents[4]
DIAGNOSTIC = BASE / "outputs/installed-public-ale-exact-receipt-diagnostic-dim1-sdk3f12-20260930"
# name, component, occurrence, physical_amount, density, swept_volume, ordinary, fused
WITNESSES = (
    ("abc", 1, "cell:9/side:1", "-0x1.d7e1ff2b2db0ap-11", "0x1.8004eaba28d6cp+1",
     "-0x1.da88051ea8000p-15", "-0x1.7ee75a8e3ebaap-11", "-0x1.7ee75a8e3ebabp-11"),
    ("cab", 2, "cell:1/side:1", "0x1.134759d5cf776p-9", "0x1.8009d3c5cf16ep+1",
     "0x1.da88051ea9000p-15", "0x1.fa10cfbb6fc96p-10", "0x1.fa10cfbb6fc95p-10"),
)


def fused_oracle(physical, density, swept):
    return float(Fraction(physical) - Fraction(density) * Fraction(swept))


def ordinary_oracle(physical, density, swept):
    product = float(Fraction(density) * Fraction(swept))
    return float(Fraction(physical) - Fraction(product))


@pytest.mark.parametrize("row", WITNESSES, ids=("abc-component1-step2", "cab-component2-step3"))
def test_native_diagnostic_quantities_are_distinct_exact_rounding_conventions(row):
    _, _, _, physical, density, swept, ordinary, fused = row
    operands = tuple(map(float.fromhex, (physical, density, swept)))
    assert ordinary_oracle(*operands) == float.fromhex(ordinary)
    assert fused_oracle(*operands) == float.fromhex(fused)
    assert struct.pack("<d", ordinary_oracle(*operands)) != struct.pack("<d", fused_oracle(*operands))


def test_authenticated_native_diagnostic_matches_both_independent_witnesses():
    # This is read-only evidence reception, not a native test. The bytes are
    # pinned externally to the root's exact diagnostic campaign.
    identity = DIAGNOSTIC / "identity.json"
    log = DIAGNOSTIC / "pytest.log"
    assert hashlib.sha256(identity.read_bytes()).hexdigest() == "8a6ae0c675b7f8cfb6e1a46c8d2d83f81fb07d48e3a6106dd6255f979729795e"
    assert hashlib.sha256(log.read_bytes()).hexdigest() == "664db77d1564d387b718dac928946294f9d350f95796930bc93476db7fcd88e5"
    data = json.loads(identity.read_text())
    assert data["source_commit"] == "5f2076e3ff9ee8c1558022a3b0295bb39737d6f9"
    assert data["native_sha256"] == "1cb02ada9cc8f0be2ed9dbc511becf77af37a243224f3d10cb88f0ae41995733"
    assert "headers=3f1286c0514edba336044d55c5e00b7965917455740adc31d0e21c621e30757d" in data["abi_key"]
    text = log.read_text()
    for _, component, occurrence, _, _, _, ordinary, fused in WITNESSES:
        assert "/component:%d %s; numerical_flux=%s expected=%s" % (component, occurrence, ordinary, fused) in text
        assert "face_measure=0x1p+0 expected=0x1p+0; temporal_weight=0x1p+0 expected=0x1p+0" in text


@pytest.mark.parametrize("origin,length,ghosts,width", [(-7,5,0,1), (0,16,2,3), (11,9,1,3), (-9,32,2,7)])
def test_source_view_offset_and_independent_grown_storage_enumeration_coincide(origin,length,ghosts,width):
    # Fab's grown storage is component-major. Enumerating it independently is
    # an offset oracle even for nonzero/negative patch origins and ghost cells.
    grown = tuple(range(origin-ghosts, origin+length+ghosts))
    storage = tuple((component, cell) for component in range(width) for cell in grown)
    for component in reversed(range(width)):
        for cell in range(origin,origin+length):
            view_offset = (cell - (origin-ghosts)) + component*len(grown)
            assert storage[view_offset] == (component, cell)
            assert view_offset == storage.index((component,cell))
