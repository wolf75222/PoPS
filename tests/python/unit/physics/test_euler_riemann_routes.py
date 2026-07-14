"""Explicit canonical Euler 2D Riemann routes (ADC-590): euler_hllc / euler_roe.

ADC-590 removed the implicit Euler fallback from the generic HLLC/Roe routes. The generic
riemann='hllc'/'roe' now REQUIRE the model's emitted capability (has_hllc / has_roe); the
canonical 4-variable Euler layout is served by the EXPLICIT euler_hllc / euler_roe routes
(descriptors EulerHLLC2D() / EulerRoe2D()), which pin pops::EulerHLLCFlux2D / EulerRoeFlux2D
and never act as a fallback.

Pure-authoring / refusal test (no .so compiled): the descriptors lower to the new routes, and
the capability guard (_validate_riemann_capability, the unified install gate exercised by
test_no_fallback_compliance_matrix.py) enforces the ADC-590 acceptance matrix on fake
CompiledModel objects.
"""
import pytest
from pops.runtime._system import System  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.numerics.riemann import EulerHLLC2D, EulerRoe2D, HLLC, Roe, riemann  # noqa: E402


def _compiled(*, n_vars, prim_names, hllc=False, roe=False):
    """A fake CompiledModel with the capability flags / layout under test (no .so needed)."""
    cons = ["rho", "rho_u", "rho_v", "E"][:n_vars] or ["q0", "q1", "q2"]
    roles = ["density", "momentum_x", "momentum_y", "energy"][:n_vars]
    return CompiledModel(
        so_path="/no/such/pops-euler-route.so", backend="production",
        cons_names=cons, cons_roles=roles, prim_names=list(prim_names), n_vars=n_vars,
        gamma=1.4, n_aux=3, params={}, caps={"cpu": True}, abi_key="SIG|c++|c++23",
        model_hash="mh", cxx="c++", std="c++23", hllc=hllc, roe=roe, wave_speeds=True,
        target="system")


def _validate(model, flux_desc):
    """Run the unified-install riemann capability gate on a fake compiled model."""
    System(n=8, L=1.0, periodic=True)._validate_riemann_capability(
        model, pops.FiniteVolume(riemann=flux_desc))


# --- 1. descriptor lowering ---------------------------------------------------------------------

def test_euler_descriptors_lower_to_explicit_routes():
    assert EulerHLLC2D().scheme == "euler_hllc"
    assert EulerRoe2D().scheme == "euler_roe"
    # also reachable on the riemann namespace
    assert riemann.EulerHLLC2D().scheme == "euler_hllc"
    assert riemann.EulerRoe2D().scheme == "euler_roe"


def test_euler_descriptors_lower_to_native_flux_entries():
    hs = pops.Spatial(flux=EulerHLLC2D())
    assert hs.flux.id == "riemann.euler_hllc"
    assert hs.flux.native_entry == "pops::EulerHLLCFlux2D"
    assert hs.flux == "euler_hllc"
    rs = pops.Spatial(flux=EulerRoe2D())
    assert rs.flux.id == "riemann.euler_roe"
    assert rs.flux.native_entry == "pops::EulerRoeFlux2D"


# --- 2. ADC-590 acceptance matrix (generic hllc/roe now require the capability) ------------------

def test_generic_hllc_refuses_p_only_euler_model():
    # A 4-var Euler with 'p' but NO emitted HLLC capability: the generic route now REFUSES it and
    # names the missing capability + both remedies (enable_hllc / EulerHLLC2D).
    m = _compiled(n_vars=4, prim_names=("rho", "u", "v", "p"), hllc=False)
    with pytest.raises(ValueError, match="hllc_star_state"):
        _validate(m, HLLC())


def test_generic_roe_refuses_p_only_euler_model():
    m = _compiled(n_vars=4, prim_names=("rho", "u", "v", "p"), roe=False)
    with pytest.raises(ValueError, match="roe_dissipation"):
        _validate(m, Roe())


def test_euler_hllc_accepts_canonical_euler_without_generic_capability():
    # The explicit euler_hllc route accepts the SAME 4-var+p model the generic route now refuses.
    m = _compiled(n_vars=4, prim_names=("rho", "u", "v", "p"), hllc=False)
    _validate(m, EulerHLLC2D())  # must not raise


def test_euler_roe_accepts_canonical_euler_without_generic_capability():
    m = _compiled(n_vars=4, prim_names=("rho", "u", "v", "p"), roe=False)
    _validate(m, EulerRoe2D())  # must not raise


def test_euler_hllc_refuses_non_euler_layout():
    # A 3-var moment model: euler_hllc requires n_vars == 4.
    m = _compiled(n_vars=3, prim_names=("rho", "u", "v"), hllc=False)
    with pytest.raises(ValueError, match="4-variable Euler"):
        _validate(m, EulerHLLC2D())


def test_euler_hllc_refuses_when_generic_capability_emitted():
    # A model that called m.enable_hllc() must use the GENERIC route (no ambiguity): euler_hllc
    # refuses, and the generic hllc route accepts it.
    m = _compiled(n_vars=4, prim_names=("rho", "u", "v", "p"), hllc=True)
    with pytest.raises(ValueError, match="generic"):
        _validate(m, EulerHLLC2D())
    _validate(m, HLLC())  # generic route accepts the capability-carrying model


def test_euler_roe_refuses_when_generic_capability_emitted():
    m = _compiled(n_vars=4, prim_names=("rho", "u", "v", "p"), roe=True)
    with pytest.raises(ValueError, match="generic"):
        _validate(m, EulerRoe2D())
    _validate(m, Roe())  # generic route accepts the capability-carrying model


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
