"""Minimal transactional freeze boundary for a raw operator-first Module."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def deep_freeze_model_value(value: Any) -> Any:
    """Return a detached immutable container graph for model metadata.

    Leaves (typed spaces, handles and symbolic expressions) already carry their own
    immutable contracts.  Rebuilding every container is nevertheless essential: a
    caller may still hold the dict/list passed to an Operator or returned while the
    Module was being authored.  Reusing such a container under a mapping proxy would
    only make the outermost level read-only and would leave a stale-alias escape.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({
            deep_freeze_model_value(key): deep_freeze_model_value(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze_model_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze_model_value(item) for item in value)
    return value


class ModuleFreezable:
    """Seal Module authoring entry points when a Problem freezes its model graph.

    The freeze is transitive over the Module's own tables and its OperatorRegistry.  The private
    snapshot/restore hooks let the enclosing Problem transaction roll that complete cascade back if
    a later descriptor fails.
    """

    _frozen = False
    _registry: Any
    name: str

    @property
    def frozen(self) -> bool:
        return bool(getattr(self, "_frozen", False))

    def freeze(self) -> Any:
        if self.frozen:
            return self
        from pops.problem._freeze_transaction import freeze_atomically

        def commit() -> None:
            self._registry.freeze()
            replacements = {
                name: deep_freeze_model_value(value)
                for name, value in vars(self).items()
                if name != "_frozen" and isinstance(
                    value, (Mapping, list, tuple, set, frozenset))
            }
            for name, value in replacements.items():
                object.__setattr__(self, name, value)
            object.__setattr__(self, "_frozen", True)

        # The custom snapshot below includes the registry and every Operator record, so one
        # participant is enough for a direct Module.freeze() transaction.
        freeze_atomically((self,), commit)
        return self

    def _pops_freeze_snapshot(self, capability: Any) -> Any:
        """Capture the complete model cascade for the enclosing Problem transaction."""
        from pops.problem._freeze_transaction import _require_freeze_capability
        _require_freeze_capability(capability)
        registry = self._registry
        return {
            "module": dict(vars(self)),
            "registry": (registry, dict(vars(registry))),
            "operators": tuple((operator, dict(vars(operator))) for operator in registry),
        }

    def _pops_freeze_restore(self, capability: Any, state: Any) -> None:
        """Restore the exact authoring objects captured by :meth:`_pops_freeze_snapshot`."""
        from pops.problem._freeze_transaction import _require_freeze_capability
        _require_freeze_capability(capability)
        for operator, attributes in reversed(state["operators"]):
            vars(operator).clear()
            vars(operator).update(attributes)
        registry, attributes = state["registry"]
        vars(registry).clear()
        vars(registry).update(attributes)
        vars(self).clear()
        vars(self).update(state["module"])

    def _guard_mutable(self, operation: str) -> None:
        if self.frozen:
            raise RuntimeError(
                "pops.model.Module %r is frozen by Problem.freeze(); cannot %s. "
                "Author a fresh Module and recompile." % (self.name, operation)
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if self.frozen:
            self._guard_mutable("set %s" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.frozen:
            self._guard_mutable("delete %s" % name)
        object.__delattr__(self, name)


__all__ = ["ModuleFreezable", "deep_freeze_model_value"]
