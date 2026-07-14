#!/usr/bin/env python3
"""ADC-533: the composite FiniteVolume home + pre-runtime Riemann-flux refusals.

Spec 5 criterion 7 homes the ``FiniteVolume(riemann=HLL(...), reconstruction=MUSCL(...))``
composite in :mod:`pops.numerics.spatial`; ``pops.FiniteVolume`` / ``pops.runtime._bricks_scheme``
re-export it so every existing import path keeps working. The model-aware refusals (HLL without
signed wave speeds, HLLC without the star-state hook, Roe without a declared dissipation, an
explicit Euler route on a non-Euler layout, a WENO5 stencil past a too-thin explicit halo) surface
through the descriptor ``available(context)`` / ``validate(context)`` protocol, DELEGATING to the
exact install-time predicates in ``pops.runtime.routes`` (single source), so they are testable
before any compile.

Pure Python: it imports the inert authoring packages and a metadata-only ``CompiledModel`` (never
built into a ``.so``, never run). Skips when the ``pops`` package cannot be imported.
"""

import pytest

pops = pytest.importorskip("pops")

from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.numerics.reconstruction import MUSCL, WENO5, validate_ghost_depth  # noqa: E402
from pops.numerics.riemann import (  # noqa: E402
    HLL, HLLC, Roe, Rusanov, EulerHLLC2D, available, validate)
from pops.numerics.riemann.waves import ExplicitPair  # noqa: E402


def _model(*, hllc=False, roe=False, wave_speeds=True, n_vars=3,
           prim_names=("rho", "u", "v")):
    """A metadata-only CompiledModel (never built into a .so) carrying the capability flags the
    install-time predicates read (has_hllc / has_roe / has_wave_speeds / n_vars / prim_names)."""
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production",
        cons_names=["rho", "mx", "my"], cons_roles=["density", "momentum_x", "momentum_y"],
        prim_names=list(prim_names), n_vars=n_vars, gamma=None, n_aux=3, params={},
        caps={}, abi_key="", model_hash="", cxx="c++", std="23",
        hllc=hllc, roe=roe, wave_speeds=wave_speeds)


# --- the homed composite is reachable from every historical path ------------------------------
def test_composite_home_and_reexports_agree():
    from pops.numerics.spatial import FiniteVolume as home_fv
    from pops.runtime._bricks_scheme import FiniteVolume as scheme_fv

    # pops.FiniteVolume, the runtime re-export, and the numerics home are the SAME surface.
    for factory in (pops.FiniteVolume, scheme_fv, home_fv):
        s = factory(riemann=HLL(), reconstruction=MUSCL())
        assert isinstance(s, pops.Spatial)
        assert (s.limiter, s.flux) == ("minmod", "hll")


def test_composite_inspectable():
    s = pops.FiniteVolume(riemann=HLLC(), reconstruction=WENO5())
    # The composite lowers to a Spatial whose typed routes are inspectable (ADC-584 manifest).
    routes = s.routes()
    assert routes["riemann"]["token"] == "hllc"
    assert routes["limiter"]["token"] == "weno5"
    assert "flux=hllc" in str(s)


def test_catalog_descriptor_still_string_based():
    # pops.numerics.spatial.FiniteVolume (the NAMESPACE attr) stays the brick-catalog descriptor,
    # which stores its scheme choice as STRING options (lowered later by _lower_spatial). Distinct
    # from the module-level composite above, which requires TYPED descriptors.
    cat = pops.numerics.spatial.FiniteVolume(riemann="hllc", reconstruction="weno5")
    assert cat.options["riemann"] == "hllc"
    assert cat.category == "spatial"


# --- NEGATIVE: string riemann is rejected pointing at the typed descriptor ---------------------
def test_string_riemann_rejected_points_at_typed():
    with pytest.raises(TypeError) as exc:
        pops.FiniteVolume(riemann="hll")
    msg = str(exc.value)
    assert "riemann='hll'" in msg
    assert "pops.numerics.riemann" in msg


# --- NEGATIVE: HLL refuses a model without signed wave speeds (via context) --------------------
def test_hll_refuses_model_without_wave_speeds():
    ctx = {"model": _model(wave_speeds=False)}
    status = available(HLL(), ctx)
    assert status.status == "no"
    assert "wave_speeds" in status.missing
    assert "wave speed" in status.reason.lower()
    with pytest.raises(ValueError):
        validate(HLL(), ctx)
    # With signed wave speeds it is available (no false positive).
    assert available(HLL(), {"model": _model(wave_speeds=True)}).status == "yes"


def test_hll_provider_mismatch_refused_via_context():
    # A model with an explicit-pair source but an HLL pinned to a jacobian provider is a mismatch;
    # the descriptor surface refuses it through the same routes predicate the install guard runs.
    from pops.numerics.riemann.waves import FromJacobian

    class _Authoring:
        # Duck-typed authoring model: provider_of reads _wave_speeds -> ExplicitPair.
        _wave_speeds = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        has_wave_speeds = True
        prim_defs = {}

    status = available(HLL(waves=FromJacobian()), {"model": _Authoring()})
    assert status.status == "no"


# --- NEGATIVE: HLLC refuses a model without pressure/contact/star-state ------------------------
def test_hllc_refuses_model_without_star_state():
    ctx = {"model": _model(hllc=False)}
    status = available(HLLC(), ctx)
    assert status.status == "no"
    assert "hllc_star_state" in status.missing
    with pytest.raises(ValueError):
        validate(HLLC(), ctx)
    # A model that declared the hook (m.enable_hllc()) is served.
    assert available(HLLC(), {"model": _model(hllc=True)}).status == "yes"


# --- NEGATIVE: Roe refuses a model without a declared dissipation ------------------------------
def test_roe_refuses_model_without_dissipation():
    ctx = {"model": _model(roe=False)}
    status = available(Roe(), ctx)
    assert status.status == "no"
    assert "roe_dissipation" in status.missing
    with pytest.raises(ValueError):
        validate(Roe(), ctx)
    assert available(Roe(), {"model": _model(roe=True)}).status == "yes"


# --- NEGATIVE: an explicit Euler route refuses a non-Euler (3-var) layout ----------------------
def test_euler_route_refuses_non_euler_layout():
    ctx = {"model": _model(n_vars=3, prim_names=("rho", "u", "v"))}
    status = available(EulerHLLC2D(), ctx)
    assert status.status == "no"
    with pytest.raises(ValueError):
        validate(EulerHLLC2D(), ctx)


# --- NEGATIVE: WENO5 requires ghost_depth >= 3 against an explicit halo ------------------------
def test_weno5_refuses_explicit_shallow_halo():
    # An EXPLICIT block halo of 2 is below the WENO5 3-cell requirement -> refuse. The default
    # (no explicit constraint) never rejects: the runtime grows the halo to the scheme.
    with pytest.raises(ValueError) as exc:
        validate_ghost_depth(WENO5(), available=2)
    assert "ghost_depth >= 3" in str(exc.value)
    assert validate_ghost_depth(WENO5(), available=3) is True
    assert validate_ghost_depth(WENO5(), available=None) is True
    # The composite's own validate mirrors it (an explicit shallow halo is refused).
    s = pops.FiniteVolume(reconstruction=WENO5())
    with pytest.raises(ValueError):
        s.validate(ghost_depth=2)
    assert s.validate(ghost_depth=None) is True


# --- POSITIVE: Rusanov has no model requirement, always available ------------------------------
def test_rusanov_always_available():
    assert available(Rusanov(), {"model": _model(wave_speeds=False)}).status == "yes"
    assert validate(Rusanov(), {"model": _model(wave_speeds=False)}) is True


# --- NO FALSE POSITIVE: a context with no model cannot refuse -----------------------------------
def test_no_model_context_does_not_refuse():
    for flux in (HLL(), HLLC(), Roe(), EulerHLLC2D()):
        assert available(flux, None).status == "yes"
        assert available(flux, {}).status == "yes"
        assert validate(flux, None) is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
