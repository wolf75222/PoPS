"""Explicit compiler boundary for blackboard physics authoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._model_contract import _BoardModel
else:
    _BoardModel = object


class _BoardCompileMixin(_BoardModel):
    def __pops_compiler_lowering__(self) -> Any:
        """Pair the executable formula emitter with its operator-first authority."""
        from pops.codegen._compiler_lowering import CompilerLowering
        from pops.model.state_symbols import native_formula_view

        source_module = self.module
        emitter = self._dsl
        if self._multi_module is None:
            emitter = native_formula_view(
                self._dsl, source_module, quantity_handles=tuple(self._states.values()))

        return CompilerLowering(
            emit_model=emitter,
            source_module=source_module,
            facade=self,
        )


__all__ = ["_BoardCompileMixin"]
