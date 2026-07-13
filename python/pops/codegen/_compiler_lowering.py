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
    # Physics authoring seals a Module by moving the same object to a framework-owned,
    # layout-compatible frozen subclass.  The canonical IR boundary is therefore nominal
    # (Module and its immutable framework subtype), not an exact-type check.  We deliberately do
    # not accept a structural lookalike here: extension happens through CompilerLowerable, whose
    # lowering must still nominate the one canonical operator-first Module authority.
    if not isinstance(lowering.source_module, Module):
        raise TypeError("CompilerLowering.source_module must be a pops.model.Module")
    return lowering


__all__ = ["CompilerLowerable", "CompilerLowering", "require_compiler_lowering"]
