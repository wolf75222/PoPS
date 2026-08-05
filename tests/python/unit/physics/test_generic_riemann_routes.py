"""One capability-driven route for each public HLLC/Roe provider (ADC-752)."""

import pytest

pops = pytest.importorskip("pops")
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.numerics.riemann import HLLC, Roe  # noqa: E402
from pops.numerics.riemann.providers import (  # noqa: E402
    ENTROPY_HARTEN,
    ENTROPY_NONE,
    ENTROPY_PROVIDER_OWNED,
    HLLC_FLUID_ROLES,
    ROE_DIRECT_ACTION,
    ROE_FLUID_ROLES,
    ROE_FLUX_JACOBIAN,
    Harten,
)
from pops.runtime._bricks_scheme import Spatial  # noqa: E402
from pops.runtime.routes import check_riemann_requirement_contract  # noqa: E402


def _compiled(*, n_vars, hllc=False, roe=False, roe_provider=ROE_FLUID_ROLES):
    """Metadata-only compiled model carrying exact provider capabilities."""
    return CompiledModel(
        so_path="/no/such/pops-riemann-provider.so",
        backend="production",
        cons_names=["q%d" % i for i in range(n_vars)],
        cons_roles=["other"] * n_vars,
        prim_names=[],
        n_vars=n_vars,
        gamma=None,
        n_aux=3,
        params={},
        caps={"cpu": True},
        abi_key="SIG|c++|c++23",
        model_hash="mh",
        cxx="c++",
        std="c++23",
        native_dimension=2,
        hllc=hllc,
        roe=roe,
        hllc_provider=HLLC_FLUID_ROLES if hllc else None,
        roe_provider=roe_provider if roe else None,
        roe_entropy_policy=(
            ENTROPY_PROVIDER_OWNED
            if roe and roe_provider == ROE_DIRECT_ACTION
            else ENTROPY_NONE
            if roe and roe_provider == ROE_FLUX_JACOBIAN
            else ENTROPY_HARTEN
            if roe
            else None
        ),
        roe_entropy_delta=(Harten().delta_token if roe and roe_provider == ROE_FLUID_ROLES else None),
        wave_speeds=True,
        wave_speed_provider="explicit_pair",
        target="system",
    )


def _validate(model, flux_desc):
    spatial = Spatial(flux=flux_desc)
    check_riemann_requirement_contract(
        spatial.riemann_capability_contract,
        model,
        "test",
        flux=spatial.flux,
    )


def test_public_descriptors_lower_to_single_generic_routes():
    hllc = Spatial(flux=HLLC()).flux
    roe = Spatial(flux=Roe()).flux
    assert (hllc.id, hllc.native_entry) == ("riemann.hllc", "pops::HLLCFlux")
    assert (roe.id, roe.native_entry) == ("riemann.roe", "pops::RoeFlux")


@pytest.mark.parametrize("n_vars", [1, 3, 4, 7])
def test_availability_depends_on_capability_not_component_count(n_vars):
    _validate(_compiled(n_vars=n_vars, hllc=True), HLLC())
    _validate(_compiled(n_vars=n_vars, roe=True), Roe())


@pytest.mark.parametrize(
    "provider",
    [ROE_FLUID_ROLES, ROE_DIRECT_ACTION, ROE_FLUX_JACOBIAN],
)
def test_all_exact_roe_providers_feed_the_same_native_route(provider):
    _validate(_compiled(n_vars=5, roe=True, roe_provider=provider), Roe())


def test_detached_model_inspection_keeps_provider_and_options() -> None:
    compiled = _compiled(n_vars=4, hllc=True, roe=True)
    assert compiled.hllc_provider == HLLC_FLUID_ROLES
    assert compiled.roe_provider == ROE_FLUID_ROLES
    assert compiled.roe_entropy_policy == ENTROPY_HARTEN
    assert compiled.roe_entropy_delta == Harten().delta_token
    rendered = repr(compiled)
    assert "hllc_provider='fluid_roles_v1'" in rendered
    assert "roe_provider='fluid_roles_v1'" in rendered
    assert "roe_entropy_policy='harten_v1'" in rendered


def test_compiled_provider_evidence_fails_closed_on_missing_unknown_or_mismatch():
    kwargs = dict(
        so_path="/no/such/model.so",
        backend="production",
        cons_names=["q"],
        cons_roles=["other"],
        prim_names=[],
        n_vars=1,
        gamma=None,
        n_aux=0,
        params={},
        caps={},
        abi_key="abi",
        model_hash="hash",
        cxx="c++",
        std="c++23",
        native_dimension=2,
    )
    with pytest.raises(ValueError, match="hllc flag disagrees"):
        CompiledModel(**kwargs, hllc=True)
    with pytest.raises(ValueError, match="unknown HLLC provider"):
        CompiledModel(**kwargs, hllc=True, hllc_provider="guessed")
    with pytest.raises(ValueError, match="requires exact harten_v1 or none"):
        CompiledModel(**kwargs, roe=True, roe_provider=ROE_FLUID_ROLES)
    with pytest.raises(ValueError, match="canonical scalar JSON"):
        CompiledModel(
            **kwargs,
            roe=True,
            roe_provider=ROE_FLUID_ROLES,
            roe_entropy_policy=ENTROPY_HARTEN,
            roe_entropy_delta="not-json",
        )


def test_truthy_legacy_flags_without_provider_evidence_are_not_capabilities():
    class Forged:
        has_hllc = True
        has_roe = True

    with pytest.raises(ValueError, match="hllc_star_state"):
        _validate(Forged(), HLLC())
    with pytest.raises(ValueError, match="roe_dissipation"):
        _validate(Forged(), Roe())


def test_hllc_missing_capability_fails_before_native_install():
    with pytest.raises(ValueError, match="hllc_star_state"):
        _validate(_compiled(n_vars=4), HLLC())


def test_roe_missing_capability_fails_before_native_install():
    with pytest.raises(ValueError, match="roe_dissipation"):
        _validate(_compiled(n_vars=4), Roe())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
