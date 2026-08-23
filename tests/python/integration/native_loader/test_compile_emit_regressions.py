"""Focused source-emission regressions for the production native loaders."""

from __future__ import annotations

import pytest

from pops.codegen import Production
from pops.codegen._artifact_identity import model_artifact_spec
from pops.codegen._compile_emit import (
    _BACKEND_CAPS,
    _normalize_sealed_system_routes,
    _canonical_affine_scalar_advection,
    compiled_capability_flags,
)
from pops.model import Handle, OwnerPath
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod, VanLeer
from pops.numerics.riemann import Rusanov, ScalarUpwind
from pops.numerics.spatial import FiniteVolume
from pops.numerics.variables import Conservative
from pops.params import RuntimeParam
from pops.physics._facade import Model
from pops.runtime._bricks_scheme import Spatial
from pops.runtime.routes import (
    LIMITER_MINMOD,
    RECON_CONSERVATIVE,
    RIEMANN_RUSANOV,
)


_AXES = ("x", "y", "z")


def _runtime_elliptic_amr_roles() -> tuple[dict[str, object], ...]:
    return (
        {
            "kind": "output",
            "field": "tests.runtime-elliptic.slot",
            "block": "runtime",
            "output_keys": (
                {
                    "owner_qid": "tests/runtime",
                    "space_kind": "field",
                    "space_name": "runtime_elliptic",
                    "component": "psi",
                },
            ),
            "gradient_sign": 1,
        },
        {
            "kind": "rhs",
            "field": "tests.runtime-elliptic.slot",
            "block": "runtime",
            "binding_ordinal": 0,
            "binding_identity": "tests.runtime-elliptic.binding.0",
            "provider_key": "psi",
            "coefficient": 1.0,
        },
    )


def _runtime_elliptic_model() -> Model:
    model = Model("runtime_elliptic")
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho=rho)
    model.conservative_from([rho])
    scale = model.value(model.param(RuntimeParam("scale", default=2.0)))
    model.flux(x=[rho], y=[rho])
    model.eigenvalues(x=[rho], y=[rho])
    model.elliptic_rhs(scale * rho)
    model.aux("psi")
    model.elliptic_field("psi", rhs=scale * rho, aux=["psi"])
    return model


def _ranked_scalar_model(dimension: int) -> Model:
    """One authored model whose emitted C++ rank is exactly ``dimension``."""
    model = Model("ranked_loader_%d" % dimension)
    (state,) = model.conservative_vars("state")
    axes = _AXES[:dimension]
    model.flux(**{axis: [(ordinal + 1) * state] for ordinal, axis in enumerate(axes)})
    model.eigenvalues(**{axis: [ordinal + 1 + 0 * state] for ordinal, axis in enumerate(axes)})
    model.primitive_vars(state)
    model.conservative_from([state])
    return model


def _assert_exact_native_loader(loader: str, *, target: str, dimension: int) -> None:
    assert "static constexpr int dimension = %d;" % dimension in loader
    assert ("static_assert(ProdModel::dimension == pops::kNativeDimension" in loader) == (
        target == "system"
    )
    assert "void* sys" in loader  # the stable C ABI is erased only at its boundary
    assert "POPS_LOADER_API int pops_native_system_package_abi_version()" in loader
    assert "return pops::runtime::system::kNativeSystemPackageAbiVersion;" in loader

    if target == "system":
        assert "using Installer = pops::runtime::system::PreparedNativeBlockInstaller<" in loader
        assert "static_cast<Installer*>(sys)" in loader
        assert "pops::runtime::system::PreparedNativeSystemPackage<" in loader
        assert "s->commit(std::move(package));" in loader
        assert "pops::add_compiled_model<pops::kNativeDimension>" not in loader
        assert "pops::PreparedSystemBlock<pops::kNativeDimension>" in loader
        canonical = "prepare_canonical_affine_scalar_advection_system_block" in loader
        assert ("prepare_exact_system_block(" in loader) != canonical
        assert ("pops::CompiledSystemBlockPreparation<" in loader) != canonical
        assert "pops::System*" not in loader
        assert "pops::AmrSystem*" not in loader
    else:
        assert "pops::PreparedNativeAmrPackage<pops::kNativeDimension>" in loader
        assert "package.block = pops::prepare_compiled_amr_system_block<" in loader
        assert "s->install_prepared_native_amr_package(std::move(package));" in loader
        assert "pops::add_compiled_model<pops::kNativeDimension>" not in loader
        assert "s->set_block_elliptic_field(" not in loader
        assert "s->register_elliptic_field(" not in loader
        assert "using NativeAmrSystem = pops::AmrSystem<pops::kNativeDimension>;" in loader
        assert "reinterpret_cast<NativeAmrSystem*>(sys)" in loader
        assert "pops::AmrSystem*" not in loader
        assert "pops::System*" not in loader

    # Rank selection belongs to the artifact and its C++ specialization.  The loader must not
    # rediscover it dynamically or keep one branch per physical rank.
    assert "switch (dimension)" not in loader
    assert "if (dimension ==" not in loader
    assert "if constexpr (ProdModel::dimension" not in loader


def _assert_bound_elliptic_closures(loader: str) -> None:
    bind = loader.index("auto model = pops::compiled_model::bind_runtime_params(")
    named_model = loader.index("auto named_elliptic_model_0 =")
    named_params = loader.index("pops::compiled_model::apply_runtime_params(", named_model)
    named_rhs = loader.index(
        "auto named_elliptic_rhs_0 = pops::make_poisson_rhs(named_elliptic_model_0);"
    )
    default_rhs = loader.index("auto fields_from_state_rhs = pops::make_poisson_rhs(model);")
    if "PreparedNativeSystemPackage" in loader:
        install = loader.index("package.block = pops::prepare_compiled_system_block<")
        attach = loader.index('package.elliptic_attachments.push_back({"fields_from_state", ')
        named_attach = loader.index('package.elliptic_attachments.push_back({"psi", ')
    else:
        install = loader.index("package.block = pops::prepare_compiled_amr_system_block<")
        attach = loader.index('attachment.field = "fields_from_state";')
        named_attach = loader.index('attachment.field = "tests.runtime-elliptic.slot";')
        assert 'attachment.binding_identity = "tests.runtime-elliptic.binding.0";' in loader
        assert 'attachment.block_identity = "runtime";' in loader
    assert bind < named_model < named_params < named_rhs < default_rhs < install < named_attach
    assert install < attach
    assert "make_poisson_rhs(pops_generated::RuntimeEllipticGenEll{})" not in loader
    if "PreparedNativeSystemPackage" in loader:
        assert 'package.elliptic_attachments.push_back({"psi", ' in loader

    # The composable default elliptic brick keeps its rhs(State) contract.  The loader fixes the
    # call site by capturing ProdModel; it must not inflate GenEll into a second model interface.
    ell_start = loader.index("struct RuntimeEllipticGenEll {")
    ell_end = loader.index("}  // namespace pops_generated", ell_start)
    elliptic_brick = loader[ell_start:ell_end]
    assert "rhs(const State& U)" in elliptic_brick
    assert "using State =" not in elliptic_brick
    assert "elliptic_rhs(" not in elliptic_brick

    named_start = loader.index("struct RuntimeEllipticGenEll_psi {")
    named_end = loader.index("}  // namespace pops_generated", named_start)
    named_brick = loader[named_start:named_end]
    assert "pops::RuntimeParams params" in named_brick
    assert "params.get(0)" in named_brick


def test_uniform_loader_builds_elliptic_closures_before_moving_bound_model() -> None:
    loader = _runtime_elliptic_model().__pops_native_loader_source__(
        name="RuntimeEllipticGen", target="system"
    )
    _assert_bound_elliptic_closures(loader)
    _assert_exact_native_loader(loader, target="system", dimension=2)


def test_amr_loader_builds_elliptic_closures_before_moving_bound_model() -> None:
    model = _runtime_elliptic_model()
    model._model_hash()
    loader = model._m.emit_cpp_native_loader(
        name="RuntimeEllipticGen",
        target="amr_system",
        native_field_roles=_runtime_elliptic_amr_roles(),
    )
    _assert_bound_elliptic_closures(loader)
    _assert_exact_native_loader(loader, target="amr_system", dimension=2)


def test_unbound_private_carrier_native_loader_is_fail_closed() -> None:
    model = _ranked_scalar_model(2)
    with pytest.raises(ValueError, match="complete exact ProviderPack carrier"):
        model._m.emit_cpp_native_loader(name="UnboundRankedLoader", target="system")


@pytest.mark.parametrize("dimension", (1, 2, 3))
@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_generated_loader_retains_the_exact_authored_rank(dimension: int, target: str) -> None:
    loader = _ranked_scalar_model(dimension).__pops_native_loader_source__(
        name="RankedLoader%d" % dimension,
        target=target,
    )

    _assert_exact_native_loader(loader, target=target, dimension=dimension)
    for other_dimension in {1, 2, 3} - {dimension}:
        assert "static constexpr int dimension = %d;" % other_dimension not in loader


def _system_package_preparer(loader: str) -> str:
    start = loader.index("inline pops::PreparedSystemBlock<pops::kNativeDimension>")
    end = loader.index("}  // namespace pops_generated", start)
    return loader[start:end]


def test_plan_owned_system_loader_seals_one_spatial_specialization_and_abi_tokens() -> None:
    spatial = Spatial(limiter=VanLeer(), flux=Rusanov(), recon=Conservative())
    loader = _ranked_scalar_model(2).__pops_native_loader_source__(
        name="SealedVanLeerRusanov",
        target="system",
        sealed_system_routes=spatial,
    )
    assert _normalize_sealed_system_routes(spatial) == ("vanleer", "rusanov", "conservative")
    assert _normalize_sealed_system_routes(
        ("vanleer", "rusanov", "conservative")
    ) == ("vanleer", "rusanov", "conservative")
    with pytest.raises(ValueError, match="non-catalogue token"):
        _normalize_sealed_system_routes(("vanleer", "invented", "conservative"))
    assert "pops::prepare_canonical_affine_scalar_advection_system_block<" in loader
    install = loader[loader.index("POPS_LOADER_API void pops_install_native("):]
    assert "pops::compiled_model::bind_runtime_params(" not in install
    assert "pops_generated::ProdModel{}" not in install
    assert "prepare_exact_system_block(" not in loader
    assert "prepare_generated_system_block_exact<" not in loader
    assert "velocity[0] = pops::Real(1);" in loader
    assert "velocity[1] = pops::Real(2);" in loader
    assert "pops_compiled_nparams() {\n  return 0;" in loader


@pytest.mark.parametrize(
    "mutate",
    (
        lambda model, state: model.source([state]),
        lambda model, state: model.flux(x=[state], y=[2 * state]),
    ),
    ids=("source", "non_affine_flux"),
)
def test_canonical_affine_scalar_recognizer_fails_closed_for_extra_physics(mutate) -> None:
    model = _ranked_scalar_model(2)
    state = model._m._flux["x"][0].b
    mutate(model, state)
    spatial = Spatial(limiter=VanLeer(), flux=Rusanov(), recon=Conservative())
    assert _canonical_affine_scalar_advection(
        model._m, target="system", sealed_routes=_normalize_sealed_system_routes(spatial)
    ) is None
    loader = model.__pops_native_loader_source__(
        name="NotCanonicalAffine", target="system", sealed_system_routes=spatial
    )
    assert "prepare_canonical_affine_scalar_advection_system_block" not in loader
    assert "prepare_exact_system_block(" in loader
    preparer = _system_package_preparer(loader)
    assert "pops::prepare_generated_system_block_exact<" in preparer
    assert "pops::nd::ReconstructionVariables::Conservative" in preparer
    assert "pops::VanLeer{}" in preparer
    assert "pops::RusanovFlux{}" in preparer
    assert '"vanleer", "rusanov", "conservative"' in preparer
    assert "prepare_generated_system_block(std::move(request))" not in preparer
    assert "select_reconstruction" not in preparer
    assert "select_riemann" not in preparer
    assert "switch (" not in preparer


def test_canonical_affine_scalar_recognizer_requires_exact_target_and_routes() -> None:
    model = _ranked_scalar_model(2)
    assert _canonical_affine_scalar_advection(
        model._m, target="amr_system", sealed_routes=("vanleer", "rusanov", "conservative")
    ) is None
    assert _canonical_affine_scalar_advection(
        model._m, target="system", sealed_routes=("minmod", "rusanov", "conservative")
    ) is None


def test_canonical_affine_scalar_recognizer_requires_matching_wave_speed() -> None:
    model = _ranked_scalar_model(2)
    state = model._m._flux["x"][0].b
    model.eigenvalues(x=[2 + 0 * state], y=[2 + 0 * state])
    assert _canonical_affine_scalar_advection(
        model._m,
        target="system",
        sealed_routes=("vanleer", "rusanov", "conservative"),
    ) is None


def test_direct_model_loader_remains_generic_when_no_plan_route_is_supplied() -> None:
    preparer = _system_package_preparer(
        _ranked_scalar_model(2).__pops_native_loader_source__(
            name="GenericDirectLoader", target="system"
        )
    )
    assert "pops::prepare_generated_system_block(std::move(request))" in preparer
    assert "pops::prepare_generated_system_block_exact<" not in preparer


def test_route_specialized_artifact_identity_changes_with_the_sealed_route(monkeypatch) -> None:
    from pops.codegen import cache, toolchain

    monkeypatch.setattr(cache, "_dsl_optflags", lambda: ())
    monkeypatch.setattr(cache, "_platform_cache_key", lambda: "test-platform")
    monkeypatch.setattr(cache, "_precision_cache_key", lambda: "f64")
    monkeypatch.setattr(cache, "_registry_cache_key", lambda: "test-registry")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "test-features")
    model = _ranked_scalar_model(2)
    model.__pops_native_loader_source__(name="IdentityRouteCarrier", target="system")
    common = {
        "backend": "production",
        "target": "system",
        "name": "IdentityRouteCarrier",
        "compiler": "c++",
        "standard": "c++23",
        "abi_key": "test-abi",
        "hoist_reciprocals": False,
    }
    _, minmod = model_artifact_spec(
        model._m,
        **common,
        sealed_system_routes=("minmod", "rusanov", "conservative"),
    )
    _, vanleer = model_artifact_spec(
        model._m,
        **common,
        sealed_system_routes=("vanleer", "rusanov", "conservative"),
    )
    assert minmod != vanleer
    assert _normalize_sealed_system_routes(
        Spatial(limiter=Minmod(), flux=Rusanov(), recon=Conservative())
    ) == (
        str(LIMITER_MINMOD),
        str(RIEMANN_RUSANOV),
        str(RECON_CONSERVATIVE),
    )


def test_sealed_routes_accept_the_real_finite_volume_scalar_upwind_lowering() -> None:
    owner = OwnerPath.model("sealed-system-route-test")
    state = Handle("U", kind="state", owner=owner)
    method = FiniteVolume(
        flux=Handle("F", kind="flux", owner=owner),
        variables=Conservative(state),
        reconstruction=MUSCL(VanLeer()),
        riemann=ScalarUpwind(velocity=Handle("a", kind="vector", owner=owner)),
    )
    assert _normalize_sealed_system_routes(method) == ("vanleer", "rusanov", "conservative")

    class ForgedSpatialProtocol:
        def runtime_spatial(self):
            return Spatial(limiter=VanLeer(), flux=Rusanov(), recon=Conservative())

    with pytest.raises(TypeError, match="exact Spatial or FiniteVolume"):
        _normalize_sealed_system_routes(ForgedSpatialProtocol())


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_explicit_program_stub_uses_ranked_runtime_facades(target: str) -> None:
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    from tests.python.support.explicit_program import _source

    src = _source(
        target=target,
        block_names=("gas",),
        projection_indices=(),
        coupled_sources=False,
        identity="deadbeef",
    )
    facade = (
        "pops::AmrSystem<pops::kNativeDimension>"
        if target == "amr_system"
        else "pops::System<pops::kNativeDimension>"
    )
    assert "%s*" % facade in src
    assert "pops::MultiFab<pops::kNativeDimension>&" in src
    assert "pops::System*" not in src
    assert "pops::AmrSystem*" not in src
    assert "pops::MultiFab&" not in src


def test_backend_capabilities_keep_feature_flags_and_route_tier() -> None:
    assert _BACKEND_CAPS["production"] == {
        "cpu": True,
        "mpi": True,
        "amr": True,
        "gpu": False,
        "tier": "production",
    }
    assert all(
        isinstance(_BACKEND_CAPS["production"][name], bool) for name in ("cpu", "mpi", "amr", "gpu")
    )
    assert Production().tier == "production"
    assert compiled_capability_flags("production") == {
        "cpu": True,
        "mpi": True,
        "amr": True,
        "gpu": False,
    }
