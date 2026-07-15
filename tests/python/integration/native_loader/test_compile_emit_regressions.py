"""Focused source-emission regressions for the production native loaders."""

from pops.codegen import Production
from pops.codegen._compile_emit import _BACKEND_CAPS, compiled_capability_flags
from pops.params import RuntimeParam
from pops.physics._facade import Model


def _runtime_elliptic_model() -> Model:
    model = Model("runtime_elliptic")
    (rho,) = model.conservative_vars("rho")
    scale = model.value(model.param(RuntimeParam("scale", default=2.0)))
    model.flux(x=[rho], y=[rho])
    model.eigenvalues(x=[rho], y=[rho])
    model.elliptic_rhs(scale * rho)
    model.aux_field("psi")
    model.elliptic_field("psi", rhs=scale * rho, aux=["psi"])
    return model


def _assert_bound_elliptic_closures(loader: str) -> None:
    bind = loader.index("auto model = pops::compiled_model::bind_runtime_params(")
    named_model = loader.index("auto named_elliptic_model_0 =")
    named_params = loader.index("pops::compiled_model::apply_runtime_params(", named_model)
    named_rhs = loader.index(
        "auto named_elliptic_rhs_0 = pops::make_poisson_rhs(named_elliptic_model_0);"
    )
    default_rhs = loader.index(
        "auto fields_from_state_rhs = pops::make_poisson_rhs(model);"
    )
    install = loader.index("pops::add_compiled_model<")
    attach = loader.index(
        's->set_block_elliptic_field(name, "fields_from_state", '
        "std::move(fields_from_state_rhs));"
    )

    assert bind < named_model < named_params < named_rhs < default_rhs < install < attach
    assert "make_poisson_rhs(pops_generated::RuntimeEllipticGenEll{})" not in loader
    assert 'set_block_elliptic_field(name, "psi", std::move(named_elliptic_rhs_0))' in loader

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
    loader = _runtime_elliptic_model()._m.emit_cpp_native_loader(
        name="RuntimeEllipticGen", target="system"
    )
    _assert_bound_elliptic_closures(loader)
    assert "pops::System*" in loader


def test_amr_loader_builds_elliptic_closures_before_moving_bound_model() -> None:
    loader = _runtime_elliptic_model()._m.emit_cpp_native_loader(
        name="RuntimeEllipticGen", target="amr_system"
    )
    _assert_bound_elliptic_closures(loader)
    assert "pops::AmrSystem*" in loader


def test_backend_capabilities_keep_feature_flags_and_route_tier() -> None:
    assert _BACKEND_CAPS["production"] == {
        "cpu": True,
        "mpi": True,
        "amr": True,
        "gpu": False,
        "tier": "production",
    }
    assert all(
        isinstance(_BACKEND_CAPS["production"][name], bool)
        for name in ("cpu", "mpi", "amr", "gpu")
    )
    assert Production().tier == "production"
    assert compiled_capability_flags("production") == {
        "cpu": True,
        "mpi": True,
        "amr": True,
        "gpu": False,
    }
