"""Spec 5 sec.14.2.5: typed native-brick constructors (ADC-504).

The native bricks are NAMED by typed constructors instead of magic ``kind=`` / ``bc=`` strings.
This test pins the typed forms and the clean-break rejection of the old string selectors:

* ``bricks.FluidState.compressible(gamma)`` / ``bricks.FluidState.isothermal(cs2, vacuum_floor)``
  build the inert state and produce a valid ``ModelSpec`` through ``bricks.Model(...)``.
* ``bricks.ElectrostaticLorentzSchur(...)`` pins
  ``kind="electrostatic_lorentz"`` internally and is accepted as the ``source=`` of
  ``bricks.Split`` / ``bricks.Strang``. ``CondensedSchur`` is not a public constructor.
* the native ``bricks.Dirichlet()`` / ``bricks.Neumann()`` / ``bricks.Periodic()`` boundary bricks lower
  to the ``bc=`` tokens ``"dirichlet"`` / ``"neumann"`` / ``"periodic"`` consumed by
  ``set_poisson`` ; they are DISTINCT objects from the ``pops.fields.bcs`` field-value descriptors.

Pure Python (no ``_pops`` / numpy / compile / Kokkos): ``Model`` builds a ``ModelSpec`` value
object without touching the runtime, so the round-trip is host-testable. Skips (never fakes the
engine) only if ``pops`` itself does not import.
"""
import sys

import pytest

pytest.importorskip("pops")
import pops.runtime.bricks as bricks  # noqa: E402


_SPEC_ATTRS = ("transport", "gamma", "cs2", "vacuum_floor", "source", "elliptic")
_SCHUR_ATTRS = ("kind", "theta", "alpha", "density_spec", "momentum_x_spec", "momentum_y_spec",
                "energy_spec", "bz_aux_component", "potential", "krylov_tol", "krylov_max_iters")


def _spec_tuple(state, transport):
    """The ModelSpec fields a state/transport pair drives, as a comparable tuple."""
    m = bricks.Model(state=state, transport=transport, source=bricks.NoSource(),
                     elliptic=bricks.ChargeDensity())
    return tuple(getattr(m, a, None) for a in _SPEC_ATTRS)


def test_fluidstate_compressible_classmethod_builds_model_spec():
    typed = bricks.FluidState.compressible(gamma=1.7)
    assert typed.kind == "compressible" and typed.gamma == 1.7
    assert _spec_tuple(typed, bricks.CompressibleFlux())[0] == "compressible"
    with pytest.raises(TypeError):
        bricks.FluidState(kind="compressible", gamma=1.7)


def test_fluidstate_isothermal_classmethod_builds_model_spec():
    typed = bricks.FluidState.isothermal(cs2=0.7, vacuum_floor=1e-9)
    assert typed.kind == "isothermal" and typed.cs2 == 0.7 and typed.vacuum_floor == 1e-9
    spec = _spec_tuple(typed, bricks.IsothermalFlux())
    assert spec[0] == "isothermal" and spec[2] == 0.7 and spec[3] == 1e-9
    with pytest.raises(TypeError):
        bricks.FluidState(kind="isothermal", cs2=0.7)


def test_fluidstate_isothermal_default_vacuum_floor_is_inactive():
    typed = bricks.FluidState.isothermal()
    assert typed.vacuum_floor == 0.0  # default = inactive, bit-identical
    assert _spec_tuple(typed, bricks.IsothermalFlux()) == \
        _spec_tuple(bricks.FluidState.isothermal(), bricks.IsothermalFlux())


def test_electrostatic_lorentz_schur_pins_internal_kind():
    typed = bricks.ElectrostaticLorentzSchur(theta=0.5, alpha=2.0)
    assert typed.kind == "electrostatic_lorentz"
    assert not hasattr(bricks, "CondensedSchur")


def test_electrostatic_lorentz_schur_carries_descriptors():
    typed = bricks.ElectrostaticLorentzSchur(
        theta=1.0, alpha=3.0, energy=bricks.Role.Energy, krylov_tol=1e-8, krylov_max_iters=500)
    assert typed.theta == 1.0 and typed.alpha == 3.0
    assert typed.energy == bricks.Role.Energy
    assert typed.krylov_tol == 1e-8 and typed.krylov_max_iters == 500
    assert not hasattr(bricks, "CondensedSchur")


def test_electrostatic_lorentz_schur_accepted_by_split_and_strang():
    src = bricks.ElectrostaticLorentzSchur(theta=0.5, alpha=1.0)
    split = bricks.Split(hyperbolic=bricks.Explicit(), source=src)
    strang = bricks.Strang(hyperbolic=bricks.Explicit(), source=src)
    assert split.source is src and split.scheme == "lie"
    assert strang.source is src and strang.scheme == "strang"


def test_electrostatic_lorentz_schur_pins_kind():
    # kind is fixed by the typed ctor (not a constructor argument); passing it is a TypeError.
    with pytest.raises(TypeError):
        bricks.ElectrostaticLorentzSchur(kind="something_else")


def test_native_boundary_bricks_lower_to_tokens():
    cases = ((bricks.Dirichlet, "dirichlet"), (bricks.Neumann, "neumann"),
             (bricks.Periodic, "periodic"))
    for ctor, token in cases:
        brick = ctor()
        assert brick.bc == token
        assert brick.lower() == token
        assert brick == ctor()  # value equality
        assert brick != bricks.Dirichlet() or token == "dirichlet"


def test_native_boundary_bricks_are_distinct_from_fields_bcs():
    import pops.fields.bcs as field_bcs
    # The native elliptic-boundary brick and the field-VALUE descriptor share a name but are
    # different objects (different concern: Poisson bc= token vs per-face field value).
    assert bricks.Dirichlet is not field_bcs.Dirichlet
    assert bricks.Periodic is not field_bcs.Periodic
    assert not hasattr(bricks.Dirichlet(), "value")  # native brick carries only a bc token


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
