#!/usr/bin/env python3
"""ADC-644 -- vacuum_floor on the COMPILED (CompositeModel) isothermal transport brick.

The private native engine.Model path already threads FluidState(vacuum_floor=) through
spec.vacuum_floor into pops::IsothermalFlux (ADC-77). The compiled-model path did NOT: the AOT struct
baked only cs2. This test pins the fix at the emit layer (source-only, no compile / no _pops runtime):

  * a default IsothermalFlux() (vacuum_floor 0) bakes ONLY cs2, so the generated struct source is
    BYTE-IDENTICAL to the pre-644 emission (the quasi-vacuum clamp inactive);
  * an active vacuum_floor is carried into the baked struct's ctor (vacuum_floor = pops::Real(...)),
    reaching the native member;
  * a negative vacuum_floor is rejected at the python boundary.

Skips (never fakes) if pops is not importable.
"""
import sys

import pytest

pops = pytest.importorskip("pops")
from pops.runtime._bricks_model import IsothermalFlux, _native_to_brick  # noqa: E402


def test_default_vacuum_floor_omitted_from_baked_struct():
    brick = _native_to_brick(IsothermalFlux(cs2=1.0), "hyperbolic")
    assert brick.fields == {"cs2": 1.0}  # omit-when-default: no vacuum_floor key
    src = brick.emit("Iso")
    assert "cs2 = pops::Real(1.0);" in src
    assert "vacuum_floor" not in src  # byte-identical to the pre-644 quasi-vacuum-inactive struct


def test_active_vacuum_floor_reaches_baked_struct():
    brick = _native_to_brick(IsothermalFlux(cs2=1.0, vacuum_floor=1e-3), "hyperbolic")
    assert brick.fields == {"cs2": 1.0, "vacuum_floor": 1e-3}
    src = brick.emit("Iso")
    assert "vacuum_floor = pops::Real(0.001);" in src


def test_negative_vacuum_floor_rejected():
    with pytest.raises(ValueError, match="vacuum_floor >= 0"):
        IsothermalFlux(vacuum_floor=-1.0)


def main():
    test_default_vacuum_floor_omitted_from_baked_struct()
    test_active_vacuum_floor_reaches_baked_struct()
    test_negative_vacuum_floor_rejected()
    print("OK  ADC-644 compiled isothermal vacuum_floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
