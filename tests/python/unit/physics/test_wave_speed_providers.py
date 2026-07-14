#!/usr/bin/env python3
"""Typed wave-speed providers (ADC-552): HLL(waves=...) + m.capabilities.wave_speeds.

Replaces the fragile ``module.capability("wave_speeds")`` string lookup with a typed
:class:`pops.numerics.riemann.waves.WaveSpeedProvider`. The provider participates in the HLL
descriptor (options / requirements), is exposed through the ``capabilities`` handle on an
authoring / compiled model, and is cross-checked by the shared install guard.

Pure authoring / descriptor level -- no .so compiled. The install guard is exercised with a REAL
authoring model carried on a REAL CompiledModel (the accepted pattern of test_euler_riemann_routes;
never a fake/mock pops object). Runs under pytest and as ``python tests/...``.
"""
import pytest
from pops.runtime._system import System  # ADC-545 advanced runtime seam

pytest.importorskip("pops")
from pops.params import ConstParam
import pops  # noqa: E402
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.numerics.reconstruction import FirstOrder  # noqa: E402
from pops.numerics.riemann import HLL, Rusanov  # noqa: E402
from pops.numerics.riemann.waves import (  # noqa: E402
    WaveSpeedProvider, ExplicitPair, FromJacobian, FromPressure, Einfeldt, Davis,
    MaxWaveSpeed, provider_of)
from pops.physics._facade import Model  # noqa: E402
from pops.numerics.riemann.waves import check_hll_waves  # noqa: E402
from pops.runtime.routes import check_wave_speed_provider  # noqa: E402


def _model_pair():
    m = Model("pair")
    q1, q2 = m.conservative_vars("q1", "q2")
    a = m.value(m.param(ConstParam("a", 1.5)))
    m.flux(x=[a * q2, a * q1], y=[a * q2, a * q1])
    m.wave_speeds(x=(-1.0 * a, 1.0 * a), y=(-1.0 * a, 1.0 * a))
    return m


def _model_jacobian():
    m = Model("jac")
    w1, w2 = m.conservative_vars("w1", "w2")
    m.flux(x=[w2, w1], y=[w2, w1])
    m.wave_speeds_from_jacobian()
    return m


def _model_pressure():
    m = Model("press")
    rho, mx, my = m.conservative_vars("rho", "m_x", "m_y",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", 1.0 * rho)
    m.flux(x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p])
    m.eigenvalues(x=[u - 1.0, u, u + 1.0], y=[v - 1.0, v, v + 1.0])
    m.primitive_vars(rho, u, v)
    return m


def _compiled(model=None, *, wave_speeds=True, n_vars=2):
    """A REAL CompiledModel (no .so) with the flags under test; optionally carrying an authoring
    model (the source-kind derivation reads it)."""
    cons = ["w1", "w2", "w3"][:n_vars]
    c = CompiledModel(
        so_path="/no/such/pops-ws-provider.so", backend="production",
        cons_names=cons, cons_roles=["custom"] * n_vars, prim_names=[], n_vars=n_vars, gamma=1.4,
        n_aux=3, params={}, caps={"cpu": True}, abi_key="SIG|c++|c++23", model_hash="mh",
        cxx="c++", std="c++23", wave_speeds=wave_speeds, target="system")
    if model is not None:
        c.model = model
    return c


# --- 1. provider factories: kinds / signed_pair / describe ---------------------------------------
def test_factory_kinds_and_signed_pair():
    assert ExplicitPair().kind == "explicit_pair" and ExplicitPair().signed_pair
    assert FromJacobian().kind == "jacobian" and FromJacobian().signed_pair
    assert FromPressure().kind == "pressure_derived" and FromPressure().signed_pair
    assert Einfeldt().kind == "einfeldt" and Einfeldt().signed_pair
    assert Davis().kind == "davis" and Davis().signed_pair
    mws = MaxWaveSpeed()
    assert mws.kind == "max_wave_speed" and not mws.signed_pair


def test_provider_capabilities_and_options():
    assert ExplicitPair().capabilities().to_dict() == {"signed_pair": True}
    assert MaxWaveSpeed().capabilities().to_dict() == {"signed_pair": False, "majorant": True}
    assert FromJacobian(eig="fd").options()["eig"] == "fd"
    assert FromJacobian(blocks=[[0], [1]]).options()["blocks"] == [[0], [1]]
    with pytest.raises(ValueError, match="eig 'numeric' | 'fd'"):
        FromJacobian(eig="bogus")
    with pytest.raises(ValueError, match="must be one of"):
        WaveSpeedProvider("nonsense")


def test_provider_describe_one_liner():
    assert "signed pair" in ExplicitPair().describe()
    assert "majorant" in MaxWaveSpeed().describe() and "Rusanov" in MaxWaveSpeed().describe()


# --- 2. HLL(waves=...) descriptor contract -------------------------------------------------------
def test_hll_with_provider_reflects_in_options_and_requirements():
    d = HLL(waves=ExplicitPair())
    assert d.options["waves"] == "explicit_pair"
    assert d.requirements["wave_speed_provider"] == "explicit_pair"
    assert "wave_speeds" in d.requirements["capabilities"]


def test_hll_unchanged_without_provider():
    d = HLL()
    assert "waves" not in d.options
    assert "wave_speed_provider" not in d.requirements
    assert d.scheme == "hll" and d.native_id == "pops::HLLFlux"


def test_hll_string_selector_rejected():
    with pytest.raises(TypeError, match="String algorithm selector rejected"):
        HLL(waves="einfeldt")


def test_hll_max_wave_speed_refused_precise():
    with pytest.raises(ValueError, match="signed wave-speed provider"):
        HLL(waves=MaxWaveSpeed())
    try:
        HLL(waves=MaxWaveSpeed())
    except ValueError as err:
        assert "Rusanov majorant" in str(err) and "use Rusanov()" in str(err)


def test_hll_wrong_type_refused():
    with pytest.raises(TypeError, match="typed WaveSpeedProvider"):
        HLL(waves=Rusanov())


def test_hll_accepts_capabilities_handle_object():
    m = _model_pair()
    d = HLL(waves=m.capabilities.wave_speeds)  # same object type as the factories
    assert d.options["waves"] == "explicit_pair"


# --- 3. capability handles: m.capabilities.wave_speeds -------------------------------------------
def test_authoring_handle_explicit_pair():
    assert _model_pair().capabilities.wave_speeds.kind == "explicit_pair"


def test_authoring_handle_jacobian():
    assert _model_jacobian().capabilities.wave_speeds.kind == "jacobian"


def test_authoring_handle_pressure_derived():
    assert _model_pressure().capabilities.wave_speeds.kind == "pressure_derived"


def test_authoring_handle_none_raises_precise():
    m = Model("none")
    r1, r2 = m.conservative_vars("r1", "r2")
    m.flux(x=[r2, r1], y=[r2, r1])
    assert provider_of(m) is None
    with pytest.raises(ValueError, match="declares no wave speeds"):
        _ = m.capabilities.wave_speeds


def test_compiled_handle_derives_from_carried_model():
    c = _compiled(_model_jacobian())
    assert c.capabilities.wave_speeds.kind == "jacobian"
    assert provider_of(c).kind == "jacobian"


def test_bare_compiled_generic_provider_and_none():
    assert provider_of(_compiled()).kind == "explicit_pair"   # has_wave_speeds True, generic
    assert provider_of(_compiled(wave_speeds=False)) is None


# --- 4. install-time guard: the shared check_hll_waves the add_equation guard calls ---------------
# The real add_equation guard derives the model's actual source (routes.py stays pops-import-free)
# and delegates to check_hll_waves(provider_kind, model, ctx). Exercised directly on a REAL
# authoring model carried by a REAL CompiledModel (no .so dlopen, no ABI match needed).
def test_guard_accepts_matching_provider():
    check_hll_waves("jacobian", _compiled(_model_jacobian()), "add_equation")  # must not raise


def test_guard_refuses_mismatched_provider():
    with pytest.raises(ValueError, match="actual wave-speed source is"):
        check_hll_waves("explicit_pair", _compiled(_model_jacobian()), "add_equation")


def test_guard_refuses_provider_when_model_emits_no_wave_speeds():
    with pytest.raises(ValueError, match="emit signed wave speeds"):
        check_hll_waves("explicit_pair", _compiled(wave_speeds=False), "add_equation")


def test_guard_accepts_estimate_provider_on_bare_handle():
    # einfeldt / davis are compatible with any signed source (documented) -- must not raise.
    check_hll_waves("einfeldt", _compiled(), "add_equation")
    check_hll_waves("davis", _compiled(), "add_equation")


def test_check_wave_speed_provider_actual_kind_arg():
    # The low-level shared checker takes the caller-derived actual kind (routes.py import-free).
    with pytest.raises(ValueError, match="actual wave-speed source is"):
        check_wave_speed_provider("explicit_pair", _compiled(_model_jacobian()), "add_equation",
                                  actual_provider="jacobian")


def test_spatial_records_waves_provider():
    assert pops.Spatial(flux=HLL(waves=ExplicitPair())).waves_provider == "explicit_pair"
    assert pops.Spatial(flux=HLL()).waves_provider is None


# --- 5. no silent HLL -> Rusanov swap ------------------------------------------------------------
def test_no_silent_flux_swap_on_missing_wave_speeds():
    # HLL on a model without wave speeds is REFUSED; the flux token is never swapped to rusanov.
    spatial = pops.FiniteVolume(limiter=FirstOrder(), riemann=HLL())
    assert spatial.flux == "hll"  # the descriptor stays HLL, no silent swap
    with pytest.raises(ValueError, match="wave_speeds"):
        System(n=8, L=1.0, periodic=True)._validate_riemann_capability(
            _compiled(wave_speeds=False), spatial)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-rs"]))
