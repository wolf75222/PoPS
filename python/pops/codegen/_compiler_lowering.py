"""Small explicit interface between resolved physics values and code generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pops.model import Module


@runtime_checkable
class _CompilerEmitter(Protocol):
    """Minimal executable half of a compiler lowering."""

    def check(self) -> object: ...
    def __pops_native_loader_source__(
        self, *, name: Any = None, target: str = "system",
        hoist_reciprocals: bool = False,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class CompilerLowering:
    """One total lowering: executable emitter plus its canonical IR authority."""

    emit_model: _CompilerEmitter
    source_module: Module
    facade: object

    def native_loader_source(
        self, *, name: Any = None, target: str = "system",
        hoist_reciprocals: bool = False,
    ) -> str:
        """Emit the native package through the emitter's explicit typed protocol."""
        source = self.emit_model.__pops_native_loader_source__(
            name=name, target=target, hoist_reciprocals=hoist_reciprocals)
        if not isinstance(source, str) or not source:
            raise TypeError("native loader source protocol must return non-empty text")
        return source


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
        if not callable(getattr(lowering.emit_model, "check", None)):
            raise TypeError("CompilerLowering.emit_model must implement check()")
        raise TypeError(
            "CompilerLowering.emit_model must implement "
            "__pops_native_loader_source__()"
        )
    # Physics authoring seals a Module by moving the same object to a framework-owned,
    # layout-compatible frozen subclass.  The canonical IR boundary is therefore nominal
    # (Module and its immutable framework subtype), not an exact-type check.  We deliberately do
    # not accept a structural lookalike here: extension happens through CompilerLowerable, whose
    # lowering must still nominate the one canonical operator-first Module authority.
    if not isinstance(lowering.source_module, Module):
        raise TypeError("CompilerLowering.source_module must be a pops.model.Module")
    return lowering


__all__ = ["CompilerLowerable", "CompilerLowering", "require_compiler_lowering"]
