"""Small explicit interface between resolved physics values and code generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pops.model import Module


@runtime_checkable
class _CompilerEmitter(Protocol):
    """Minimal executable half of a compiler lowering."""

    def check(self) -> object: ...


@dataclass(frozen=True, slots=True)
class CompilerLowering:
    """One total lowering: executable emitter plus its canonical IR authority."""

    emit_model: _CompilerEmitter
    source_module: Module
    facade: object


@runtime_checkable
class CompilerLowerable(Protocol):
    """Physics authoring values implement this one method to enter compilation."""

    def __pops_compiler_lowering__(self) -> CompilerLowering: ...


def require_compiler_lowering(value: Any) -> CompilerLowering:
    """Invoke the explicit protocol and reject incomplete/structural lookalikes."""
    if not isinstance(value, CompilerLowerable):
        raise TypeError(
            "%s does not implement the CompilerLowerable protocol"
            % type(value).__name__
        )
    lowering = value.__pops_compiler_lowering__()
    if type(lowering) is not CompilerLowering:
        raise TypeError("__pops_compiler_lowering__() must return an exact CompilerLowering")
    if not isinstance(lowering.emit_model, _CompilerEmitter):
        raise TypeError("CompilerLowering.emit_model must implement check()")
    if type(lowering.source_module) is not Module:
        raise TypeError("CompilerLowering.source_module must be an exact pops.model.Module")
    return lowering


__all__ = ["CompilerLowerable", "CompilerLowering", "require_compiler_lowering"]
