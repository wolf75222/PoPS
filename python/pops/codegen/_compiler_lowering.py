"""Small explicit interface between resolved physics values and code generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CompilerLowering:
    """One total lowering: executable emitter plus its canonical IR authority."""

    emit_model: Any
    source_module: Any
    facade: Any


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
    if lowering.emit_model is None or lowering.source_module is None:
        raise ValueError("CompilerLowering requires an emitter and a source Module")
    return lowering


__all__ = ["CompilerLowerable", "CompilerLowering", "require_compiler_lowering"]
