"""One capability-driven route for each public HLLC/Roe provider (ADC-752)."""

import pytest

pops = pytest.importorskip("pops")
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.numerics.riemann import HLLC, Roe  # noqa: E402
from pops.runtime._bricks_scheme import Spatial  # noqa: E402
from pops.runtime.routes import check_riemann_requirement_contract  # noqa: E402


def _compiled(*, n_vars, hllc=False, roe=False):
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
        hllc=hllc,
        roe=roe,
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


def test_hllc_missing_capability_fails_before_native_install():
    with pytest.raises(ValueError, match="hllc_star_state"):
        _validate(_compiled(n_vars=4), HLLC())


def test_roe_missing_capability_fails_before_native_install():
    with pytest.raises(ValueError, match="roe_dissipation"):
        _validate(_compiled(n_vars=4), Roe())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
