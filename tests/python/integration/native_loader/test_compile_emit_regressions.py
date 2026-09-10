"""Focused source-emission regressions for the production native loaders."""

from __future__ import annotations

import pytest

from pops.codegen import Production
from pops.codegen._compile_emit import _BACKEND_CAPS, compiled_capability_flags
from pops.params import RuntimeParam
from pops.physics._facade import Model


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
        assert "prepare_exact_system_block(" in loader
        assert "pops::CompiledSystemBlockPreparation<" in loader
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
    assert "static constexpr int dimension = 2;" in named_brick
    assert "pops::RuntimeParams params" in named_brick
    assert "params.get(0)" in named_brick


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_named_elliptic_rhs_declares_its_exact_consumer_dimension(dimension: int) -> None:
    from pops.codegen.module_codegen import emit_cpp_elliptic_field

    model = _ranked_scalar_model(dimension)
    model.aux("phi")
    model.elliptic_field("phi", rhs=model._m.cons_from[0], aux=["phi"])
    model._model_hash()
    brick = emit_cpp_elliptic_field(model._m, "phi", "ExactRankEllipticRhs")
    assert "static constexpr int dimension = %d;" % dimension in brick
    assert "static constexpr int n_vars = 1;" in brick
    assert "using State = pops::StateVec<1>;" in brick
    assert "elliptic_rhs(const State& U)" in brick
    assert "return state;" in brick


def _distinct_provider_roles_model() -> Model:
    model = _ranked_scalar_model(2)
    state = model._m.cons_from[0]
    grad_x = model.aux("grad_x")
    grad_y = model.aux("grad_y")
    forcing = model.aux("forcing")
    model.flux(x=[-grad_y * state], y=[grad_x * state])
    model.source([forcing * state])
    model._model_hash()
    return model


def test_generated_bricks_project_the_native_input_union_into_each_role() -> None:
    from pops.codegen._native_model_provider_plan import native_model_provider_plan
    from pops.codegen.module_codegen import _emit_bricks, emit_cpp_source

    model = _distinct_provider_roles_model()

    hyperbolic = model._m.emit_cpp_brick(name="ProviderCarrierHyperbolic")
    source = emit_cpp_source(model._m, name="ProviderCarrierSource")
    assert model._m._total_n_aux() == 3
    assert "static constexpr int n_flux_providers = 2;" in hyperbolic
    assert "static constexpr int n_providers = 2;" in hyperbolic
    assert "static constexpr int n_providers = 1;" in source
    for brick in (hyperbolic, source):
        assert "static constexpr int n_aux = 3;" in brick
    inputs = native_model_provider_plan(model._m)
    by_component = {row["key"]["component"]: row["consumer_slot"] for row in inputs}
    assert set(by_component) == {"grad_x", "grad_y", "forcing"}
    _, native, _ = _emit_bricks(model._m, name="DistinctRoles")
    assert native.count("static constexpr int n_providers = 3;") == 2
    for component, slot in by_component.items():
        assert "const pops::Real %s = pops::provider_value<%d>(a);" % (component, slot) in native
    loader = model.__pops_native_loader_source__(name="DistinctRoles", target="system")
    assert '/native_model"' in loader


def test_named_field_outputs_do_not_become_native_model_inputs() -> None:
    from pops.codegen._native_model_provider_plan import native_model_provider_plan
    from pops.codegen.module_codegen import _emit_bricks

    model = _runtime_elliptic_model()
    model._model_hash()
    assert model._m._total_n_aux() == 1
    assert native_model_provider_plan(model._m) == ()
    _, native, _ = _emit_bricks(model._m, name="OutputOnlyField")
    assert "static constexpr int n_providers = 0;" in native
    assert "static constexpr int n_aux = 1;" in native


def test_field_free_flux_retains_source_only_native_inputs() -> None:
    from pops.codegen._native_model_provider_plan import native_model_provider_plan
    from pops.codegen.module_codegen import _emit_bricks

    model = _ranked_scalar_model(2)
    electric = model.aux("electric")
    model.source([electric * model._m.cons_from[0]])
    model._model_hash()
    assert model._m._component_flux_consumer_plan == ()
    assert len(native_model_provider_plan(model._m)) == 1
    _, native, _ = _emit_bricks(model._m, name="ElectricSource")
    assert "static constexpr int n_flux_providers = 0;" in native
    assert native.count("static constexpr int n_providers = 1;") == 2


def test_native_flux_and_source_read_their_distinct_qualified_inputs() -> None:
    from pops.codegen._native_model_provider_plan import native_model_provider_plan
    from pops.codegen.module_codegen import _emit_bricks
    from tests.python.unit.codegen.test_dsl_brick import _compile_and_run

    model = _distinct_provider_roles_model()
    _, bricks, composite = _emit_bricks(model._m, name="DistinctRolesOracle")
    slots = {
        row["key"]["component"]: row["consumer_slot"]
        for row in native_model_provider_plan(model._m)
    }
    source = r'''
#include <pops/physics/bricks/bricks.hpp>
%s
using Model = %s;
static_assert(Model::n_providers == 3);
static_assert(pops::qualified_flux_provider_requirements_valid<Model>());
struct Storage {
  pops::ProviderValues<Model::n_providers> values{};
  pops::Real operator()(const pops::Index<2>&, int slot) const { return values[slot]; }
};
int main() {
  Model model;
  Model::State state{};
  state[0] = 2;
  Storage storage;
  storage.values[%d] = 3;
  storage.values[%d] = 5;
  storage.values[%d] = 7;
  const auto flux_inputs = pops::bind_flux_providers_at<Model>(storage, pops::Index<2>{});
  const auto x = model.template flux<0>(state, flux_inputs);
  const auto y = model.template flux<1>(state, flux_inputs);
  const auto local = model.source(state, storage.values);
  return x[0] == -10 && y[0] == 6 && local[0] == 14 ? 0 : 1;
}
''' % (bricks, composite, slots["grad_x"], slots["grad_y"], slots["forcing"])
    _compile_and_run(source, "distinct_provider_roles")


def test_native_input_layout_version_invalidates_only_the_model_artifact(monkeypatch) -> None:
    from pops.codegen import _native_model_provider_plan as provider_layout
    from pops.codegen._artifact_identity import model_artifact_spec

    model = _distinct_provider_roles_model()
    options = dict(
        backend="production", target="system", name="InputLayoutIdentity",
        compiler="c++", standard="c++20", abi_key="test-native-abi", hoist_reciprocals=False,
    )
    semantic, artifact = model_artifact_spec(model._m, **options)
    monkeypatch.setattr(provider_layout, "NATIVE_MODEL_PROVIDER_CONTRACT", 2)
    changed_semantic, changed_artifact = model_artifact_spec(model._m, **options)
    assert semantic == changed_semantic
    assert artifact != changed_artifact


def test_generated_provider_free_brick_declares_zero_native_carrier() -> None:
    model = _ranked_scalar_model(2)
    model._model_hash()
    brick = model._m.emit_cpp_brick(name="ProviderFreeHyperbolic")
    assert "static constexpr int n_providers = 0;" in brick
    assert "static constexpr int n_aux" not in brick


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
