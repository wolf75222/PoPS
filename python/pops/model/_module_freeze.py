"""Minimal transactional freeze boundary for a raw operator-first Module."""
from __future__ import annotations

from typing import Any


class ModuleFreezable:
    """Seal Module authoring entry points when a Problem freezes its model graph.

    Only the Boolean flag is mutated by :meth:`freeze`, so the Problem freeze transaction can roll
    it back by restoring the Module's ordinary Python attributes if a later descriptor fails.
    """

    _frozen = False

    @property
    def frozen(self) -> bool:
        return bool(getattr(self, "_frozen", False))

    def freeze(self) -> Any:
        object.__setattr__(self, "_frozen", True)
        return self

    def _guard_mutable(self, operation: str) -> None:
        if self.frozen:
            raise RuntimeError(
                "pops.model.Module %r is frozen by Problem.freeze(); cannot %s. "
                "Author a fresh Module and recompile." % (self.name, operation)
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if self.frozen and name != "_frozen":
            self._guard_mutable("set %s" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.frozen:
            self._guard_mutable("delete %s" % name)
        object.__delattr__(self, name)


__all__ = ["ModuleFreezable"]
