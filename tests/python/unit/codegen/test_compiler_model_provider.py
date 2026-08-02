"""The compiler model-provider boundary is explicit, extensible, and fail-closed."""

from __future__ import annotations

import pytest

from pops.codegen import CompilerLowerable, CompilerLowering
from pops.codegen._compiler_lowering import require_compiler_lowering
from pops.codegen._phases import _resolve_problem_model
from pops.codegen.module_lowering import lower_and_validate
from pops._ir.expr import Const
from pops.model import Module, Rate
from pops.model.provider_pack import MissingInputProvider
from pops.physics._facade import Model


def _facade_model(name: str = "provider") -> Model:
    model = Model(name)
    rho, mx, my = model.conservative_vars("rho", "mx", "my")
    grad_x = model.aux("grad_x")
    grad_y = model.aux("grad_y")
    model.flux(
        x=[mx, mx * mx / rho, mx * my / rho],
        y=[my, mx * my / rho, my * my / rho],
    )
    model.source_term("electric", [Const(0.0), -rho * grad_x, -rho * grad_y])
    model.elliptic_rhs(rho - 1.0)
    return model


def _field_dependent_flux_model(name: str, *, with_provider: bool = True) -> Model:
    model = Model(name)
    (rho,) = model.conservative_vars("rho")
    phi = model.aux("phi")
    grad_x = model.aux("grad_x")
    model.aux("grad_y")
    model.primitive_vars(rho=rho)
    model.conservative_from([rho])
    model.flux(x=[rho * grad_x], y=[rho * grad_x])
    model.eigenvalues(x=(Const(1.0),), y=(Const(1.0),))
    if with_provider:
        model.elliptic_rhs(rho + Const(0.0) * phi)
    return model


class _ThirdPartyProvider:
    """An external provider delegates only the documented compiler contract."""

    def __init__(self, delegate: Model) -> None:
        self._delegate = delegate
        self.name = delegate.name
        self.owner_path = delegate.owner_path

    def declaration_index(self) -> object:
        return self._delegate.declaration_index()

    def __pops_compiler_lowering__(self) -> CompilerLowering:
        return CompilerLowering(
            emit_model=self._delegate,
            source_module=self._delegate.module,
            facade=self,
        )


class _CheckEmitter:
    def check(self) -> None:
        return None

    def __pops_bind_component_provider_packs__(self, packs) -> None:
        self.provider_packs = packs

    def __pops_native_loader_source__(
            self, *, name=None, target="system", hoist_reciprocals=False):
        return "// test native loader\n"


def test_third_party_provider_enters_resolution_and_lowering_through_public_protocol():
    provider = _ThirdPartyProvider(_facade_model())

    assert isinstance(provider, CompilerLowerable)
    assert _resolve_problem_model(provider) is provider
    emit_model, source_module = lower_and_validate(provider)
    assert emit_model is provider._delegate
    assert source_module is provider._delegate.module


def test_module_is_a_real_compiler_provider_adapter():
    module = Module("module-provider")
    state = module.state_space("fluid", ("rho",))
    (rho,) = module.state_symbols(state)
    flux = module.operator(
        name="transport",
        signature=(state,) >> Rate(state),
        kind="grid_operator",
        expr={"x": (rho,), "y": (rho,)},
    )
    module.eigenvalues(x=(Const(1.0),), y=(Const(1.0),))
    module.rate_operator(
        "advance", state_space=module.state_handle(state), flux=True, fluxes=(flux,)
    )

    lowering = require_compiler_lowering(module)
    assert lowering.source_module is module
    assert lowering.facade is module
    assert _resolve_problem_model(module) is module


def test_frozen_module_remains_the_canonical_compiler_ir():
    model = _facade_model("frozen-provider")
    module = model.module
    model.freeze()

    assert type(module) is not Module
    lowering = require_compiler_lowering(model)
    assert isinstance(lowering.source_module, Module)
    assert lowering.source_module is module


def test_facade_and_formula_carrier_share_one_minimal_flux_provider_pack():
    model = _field_dependent_flux_model("facade-flux-pack")

    emitted, source_module = lower_and_validate(model, facade=model)

    assert emitted is model
    rows = model._component_flux_provider_metadata["entries"]
    assert rows == model._m._component_flux_provider_metadata["entries"]
    assert [row["key"]["component"] for row in rows] == ["grad_x"]
    assert rows[0]["provider"]["availability"] is True
    assert rows[0]["key"]["owner_qid"] == str(source_module.owner_path.canonical())

    source = model.__pops_native_loader_source__()
    assert rows[0]["key"]["owner_qid"] in source
    assert '"grad_x"' in source
    assert "true, 1" in source
    assert "static constexpr int n_flux_providers = 1;" in source
    assert "flux_provider_requirements" in source
    assert "flux(const State& U, const auto& a, int dir)" in source
    assert "a.template flux_provider<1>()" in source


def test_field_dependent_flux_without_provider_fails_before_native_source():
    model = _field_dependent_flux_model("missing-flux-provider", with_provider=False)

    with pytest.raises(MissingInputProvider, match="unset"):
        lower_and_validate(model, facade=model)


def test_same_field_spelling_under_distinct_model_owners_stays_distinct_in_emitted_pack():
    left = _field_dependent_flux_model("left-flux-owner")
    right = _field_dependent_flux_model("right-flux-owner")

    lower_and_validate(left, facade=left)
    lower_and_validate(right, facade=right)
    left_owner = left._component_flux_provider_metadata["entries"][0]["key"]["owner_qid"]
    right_owner = right._component_flux_provider_metadata["entries"][0]["key"]["owner_qid"]

    assert left_owner != right_owner
    assert left_owner in left.__pops_native_loader_source__()
    assert right_owner not in left.__pops_native_loader_source__()
    assert right_owner in right.__pops_native_loader_source__()


class _MissingProtocol:
    pass


class _WrongReturn:
    def __pops_compiler_lowering__(self) -> object:
        return object()


class _WrongEmitter:
    def __pops_compiler_lowering__(self) -> CompilerLowering:
        return CompilerLowering(
            emit_model=object(), source_module=Module("wrong-emitter"), facade=self
        )


class _WrongSourceModule:
    def __pops_compiler_lowering__(self) -> CompilerLowering:
        return CompilerLowering(emit_model=_CheckEmitter(), source_module=object(), facade=self)


class _MissingNativeSource:
    def check(self):
        return None


class _WrongNativeSourceProvider:
    def __pops_compiler_lowering__(self) -> CompilerLowering:
        return CompilerLowering(
            emit_model=_MissingNativeSource(),
            source_module=Module("missing-native-source"),
            facade=self,
        )


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (_MissingProtocol(), "does not implement the CompilerLowerable protocol"),
        (_WrongReturn(), "must return an exact CompilerLowering"),
        (_WrongEmitter(), "emit_model must implement check"),
        (_WrongNativeSourceProvider(), "must implement __pops_native_loader_source__"),
        (_WrongSourceModule(), "source_module must be a pops.model.Module"),
    ],
)
def test_incomplete_or_false_compiler_provider_is_rejected_before_compile(provider, message):
    with pytest.raises(TypeError, match=message):
        _resolve_problem_model(provider)
