"""Explicit compiler boundary for blackboard physics authoring."""
from __future__ import annotations

from typing import Any


class _BoardCompileMixin:
    def compiler_lowering(self) -> Any:
        """Pair the executable formula emitter with its operator-first authority."""
        from pops.codegen.compiler_lowering import CompilerLowering

        return CompilerLowering(
            emit_model=self._dsl,
            source_module=self.module,
            facade=self,
        )


__all__ = ["_BoardCompileMixin"]
