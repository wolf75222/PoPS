"""Spec 5 sec.14.2.5: typed native-brick constructors (ADC-504).

The native bricks are NAMED by typed constructors instead of magic ``kind=`` / ``bc=`` strings,
ADDITIVELY (the string path keeps working). This test pins the EQUIVALENCE of the typed forms to
the existing string forms:

* ``pops.FluidState.compressible(gamma)`` / ``pops.FluidState.isothermal(cs2, vacuum_floor)`` build
  the SAME inert state as ``pops.FluidState(kind=...)`` and produce a bit-identical ``ModelSpec``
  through ``pops.Model(...)``.
* the native ``pops.Dirichlet()`` / ``pops.Neumann()`` / ``pops.Periodic()`` boundary bricks lower
  to the ``bc=`` tokens ``"dirichlet"`` / ``"neumann"`` / ``"periodic"`` consumed by
  ``set_poisson`` ; they are DISTINCT objects from the ``pops.fields.bcs`` field-value descriptors.

Pure Python (no ``_pops`` / numpy / compile / Kokkos): ``Model`` builds a ``ModelSpec`` value
object without touching the runtime, so the round-trip is host-testable. Skips (never fakes the
engine) only if ``pops`` itself does not import.
"""
import sys

import pytest

pytest.importorskip("pops")
import pops  # noqa: E402


_SPEC_ATTRS = ("transport", "gamma", "cs2", "vacuum_floor", "source", "elliptic")


def _spec_tuple(state, transport):
    """The ModelSpec fields a state/transport pair drives, as a comparable tuple."""
    m = pops.Model(state=state, transport=transport, source=pops.NoSource(),
                   elliptic=pops.ChargeDensity())
    return tuple(getattr(m, a, None) for a in _SPEC_ATTRS)


def test_fluidstate_compressible_classmethod_matches_kind():
    typed = pops.FluidState.compressible(gamma=1.7)
    string = pops.FluidState(kind="compressible", gamma=1.7)
    assert typed.kind == "compressible" and typed.gamma == 1.7
    assert (typed.kind, typed.gamma) == (string.kind, string.gamma)
    # Same ModelSpec round-trip through the existing kind= consumer (pops.Model).
    assert _spec_tuple(typed, pops.CompressibleFlux()) == \
        _spec_tuple(string, pops.CompressibleFlux())


def test_fluidstate_isothermal_classmethod_matches_kind():
    typed = pops.FluidState.isothermal(cs2=0.7, vacuum_floor=1e-9)
    string = pops.FluidState(kind="isothermal", cs2=0.7, vacuum_floor=1e-9)
    assert typed.kind == "isothermal" and typed.cs2 == 0.7 and typed.vacuum_floor == 1e-9
    assert (typed.kind, typed.cs2, typed.vacuum_floor) == \
        (string.kind, string.cs2, string.vacuum_floor)
    assert _spec_tuple(typed, pops.IsothermalFlux()) == \
        _spec_tuple(string, pops.IsothermalFlux())


def test_fluidstate_isothermal_default_vacuum_floor_is_inactive():
    typed = pops.FluidState.isothermal()
    assert typed.vacuum_floor == 0.0  # default = inactive, bit-identical
    assert _spec_tuple(typed, pops.IsothermalFlux()) == \
        _spec_tuple(pops.FluidState(kind="isothermal"), pops.IsothermalFlux())


def test_native_boundary_bricks_lower_to_tokens():
    cases = ((pops.Dirichlet, "dirichlet"), (pops.Neumann, "neumann"),
             (pops.Periodic, "periodic"))
    for ctor, token in cases:
        brick = ctor()
        assert brick.bc == token
        assert brick.lower() == token
        assert brick == ctor()  # value equality
        assert brick != pops.Dirichlet() or token == "dirichlet"


def test_native_boundary_bricks_are_distinct_from_fields_bcs():
    import pops.fields.bcs as field_bcs
    # The native elliptic-boundary brick and the field-VALUE descriptor share a name but are
    # different objects (different concern: Poisson bc= token vs per-face field value).
    assert pops.Dirichlet is not field_bcs.Dirichlet
    assert pops.Periodic is not field_bcs.Periodic
    assert not hasattr(pops.Dirichlet(), "value")  # native brick carries only a bc token


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
