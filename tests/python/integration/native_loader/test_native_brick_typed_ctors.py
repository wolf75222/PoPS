"""Spec 5 sec.14.2.5: typed native-brick constructors (ADC-504).

The native bricks are named by typed constructors. This test pins their lowering contract:

* ``engine.FluidState.compressible(gamma)`` / ``engine.FluidState.isothermal(cs2, vacuum_floor)`` build
  the SAME inert state as ``engine.FluidState(kind=...)`` and produce a bit-identical ``ModelSpec``
  through ``engine.Model(...)``.
* the native ``Dirichlet()`` / ``Neumann()`` / ``Periodic()`` boundary bricks lower
  to the ``bc=`` tokens ``"dirichlet"`` / ``"neumann"`` / ``"periodic"`` consumed by
  ``set_poisson`` ; they are DISTINCT objects from the ``pops.fields.bcs`` field-value descriptors.

Pure Python (no ``_pops`` / numpy / compile / Kokkos): ``Model`` builds a ``ModelSpec`` value
object without touching the runtime, so the round-trip is host-testable. Skips (never fakes the
engine) only if ``pops`` itself does not import.
"""
import sys

import pytest

pytest.importorskip("pops")
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._engine_descriptors import Dirichlet, Neumann, Periodic


_SPEC_ATTRS = ("transport", "gamma", "cs2", "vacuum_floor", "source", "elliptic")


def _spec_tuple(state, transport):
    """The ModelSpec fields a state/transport pair drives, as a comparable tuple."""
    m = engine.Model(state=state, transport=transport, source=engine.NoSource(),
                   elliptic=engine.ChargeDensity())
    return tuple(getattr(m, a, None) for a in _SPEC_ATTRS)


def test_fluidstate_compressible_classmethod_matches_kind():
    typed = engine.FluidState.compressible(gamma=1.7)
    string = engine.FluidState(kind="compressible", gamma=1.7)
    assert typed.kind == "compressible" and typed.gamma == 1.7
    assert (typed.kind, typed.gamma) == (string.kind, string.gamma)
    # Same ModelSpec round-trip through the existing kind= consumer (engine.Model).
    assert _spec_tuple(typed, engine.CompressibleFlux()) == \
        _spec_tuple(string, engine.CompressibleFlux())


def test_fluidstate_isothermal_classmethod_matches_kind():
    typed = engine.FluidState.isothermal(cs2=0.7, vacuum_floor=1e-9)
    string = engine.FluidState(kind="isothermal", cs2=0.7, vacuum_floor=1e-9)
    assert typed.kind == "isothermal" and typed.cs2 == 0.7 and typed.vacuum_floor == 1e-9
    assert (typed.kind, typed.cs2, typed.vacuum_floor) == \
        (string.kind, string.cs2, string.vacuum_floor)
    assert _spec_tuple(typed, engine.IsothermalFlux()) == \
        _spec_tuple(string, engine.IsothermalFlux())


def test_fluidstate_isothermal_default_vacuum_floor_is_inactive():
    typed = engine.FluidState.isothermal()
    assert typed.vacuum_floor == 0.0  # default = inactive, bit-identical
    assert _spec_tuple(typed, engine.IsothermalFlux()) == \
        _spec_tuple(engine.FluidState(kind="isothermal"), engine.IsothermalFlux())


def test_native_boundary_bricks_lower_to_tokens():
    cases = ((Dirichlet, "dirichlet"), (Neumann, "neumann"), (Periodic, "periodic"))
    for ctor, token in cases:
        brick = ctor()
        assert brick.bc == token
        assert brick.lower() == token
        assert brick == ctor()  # value equality
        assert brick != Dirichlet() or token == "dirichlet"


def test_native_boundary_bricks_are_distinct_from_fields_bcs():
    import pops.fields.bcs as field_bcs
    # The native elliptic-boundary brick and the field-VALUE descriptor share a name but are
    # different objects (different concern: Poisson bc= token vs per-face field value).
    assert Dirichlet is not field_bcs.Dirichlet
    assert Periodic is not field_bcs.Periodic
    assert not hasattr(Dirichlet(), "value")  # native brick carries only a bc token


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
