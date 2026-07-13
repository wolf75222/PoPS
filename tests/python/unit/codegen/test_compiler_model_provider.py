"""The compiler model-provider boundary is explicit, extensible, and fail-closed."""

from __future__ import annotations

import pytest

from pops.codegen import CompilerLowerable, CompilerLowering
from pops.codegen._compiler_lowering import require_compiler_lowering
from pops.codegen._phases import _resolve_problem_model
from pops.codegen.module_lowering import lower_and_validate
from pops.ir.expr import Const
from pops.model import Module, Rate
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


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (_MissingProtocol(), "does not implement the CompilerLowerable protocol"),
        (_WrongReturn(), "must return an exact CompilerLowering"),
        (_WrongEmitter(), "emit_model must implement check"),
        (_WrongSourceModule(), "source_module must be an exact pops.model.Module"),
    ],
)
def test_incomplete_or_false_compiler_provider_is_rejected_before_compile(provider, message):
    with pytest.raises(TypeError, match=message):
        _resolve_problem_model(provider)
