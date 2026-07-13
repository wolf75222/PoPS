"""Owner-qualified declaration handles for a Case assembly."""
from __future__ import annotations

from typing import Any

from pops.model.handles import Handle
from pops.model.ownership import MissingOwnershipError, OwnerKind, OwnerPath


class BlockHandle(Handle):
    """Reference to one physics block declaration."""

    __slots__ = ("model_owner_path", "_instance_registry")

    @property
    def expression_readable(self) -> bool:
        """A block selects an owner scope; it is not a scientific value."""
        return False

    def __init__(
        self,
        name: Any,
        *,
        owner: Any,
        model_owner: Any,
        instance_registry: Any = None,
        schema_version: int = 1,
    ) -> None:
        super().__init__(
            name,
            kind="block",
            owner=owner,
            schema_version=schema_version,
        )
        object.__setattr__(self, "model_owner_path", OwnerPath.coerce(model_owner))
        object.__setattr__(self, "_instance_registry", instance_registry)

    def inspect(self) -> dict[str, Any]:
        result = super().inspect()
        result.update({
            "handle_type": "block",
            "model_owner_path": self.model_owner_path.presentation().to_data(),
        })
        return result

    def canonical_identity(self) -> dict[str, Any]:
        """Serialize the concrete subtype and exact model-definition provenance."""
        result = super().canonical_identity()
        model_owner = self.model_owner_path
        if model_owner.is_authoring:
            model_owner = model_owner.canonical()
        result.update({
            "handle_type": "block",
            "model_owner_path": model_owner.to_data(),
        })
        return result

    def _resolved(self, owner: Any = None) -> BlockHandle:
        """Detach a canonical block while preserving canonical model provenance."""
        result = super()._resolved(owner)
        object.__setattr__(result, "model_owner_path", self.model_owner_path.canonical())
        object.__setattr__(result, "_instance_registry", None)
        return result

    def _identity(self) -> tuple[Any, ...]:
        return super()._identity() + (self.model_owner_path,)

    @property
    def instance_owner_path(self) -> OwnerPath:
        """Owner of this block's instantiated model declarations."""
        block_owner = self.owner_path.child(OwnerKind.BLOCK, self.local_id)
        return block_owner.instance_of(self.model_owner_path)

    def accepts(self, declaration: Any) -> bool:
        if not isinstance(declaration, Handle):
            return False
        expected = (
            self.model_owner_path.canonical()
            if declaration.is_resolved
            else self.model_owner_path
        )
        return declaration.owner_path == expected

    def __getitem__(self, declaration: Any) -> Handle:
        """Qualify a model-local declaration into this exact block instance."""
        if not isinstance(declaration, Handle):
            raise TypeError(
                "block qualification expects a declared Handle, not %r"
                % type(declaration).__name__)
        if self._instance_registry is None:
            raise MissingOwnershipError(
                "block handle %s is detached from its authoritative case registry and cannot "
                "qualify declarations" % self.qualified_id)
        return self._instance_registry.qualify(
            declaration, block=self, allow_existing=False)


class StateHandle(Handle):
    """Reference to one state declared by a block."""

    __slots__ = ()

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(name, kind="state", owner=owner)


class FieldHandle(Handle):
    """Callable reference to one Case-owned field operator declaration."""

    __slots__ = ("_field_registry",)

    def __init__(self, name: Any, *, owner: Any, field_registry: Any = None) -> None:
        super().__init__(name, kind="field", owner=owner)
        object.__setattr__(self, "_field_registry", field_registry)

    def __call__(self, *states: Any, name: Any = None) -> Any:
        """Build this field solve from one or more exact temporal states."""
        program = next((getattr(state, "prog", None) for state in states
                        if getattr(state, "prog", None) is not None), None)
        if program is None:
            raise ValueError(
                "field operator %r must be called with one or more time-Program State values"
                % self.name)
        return program._solve_field_operator(self, states, name=name)


class OperatorHandle(Handle):
    """Reference to a Case-scoped coupling/local operator declaration."""

    __slots__ = ()

    @property
    def expression_readable(self) -> bool:
        """An operator is callable/control identity, never a pointwise value."""
        return False

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(name, kind="operator", owner=owner)


__all__ = ["BlockHandle", "StateHandle", "FieldHandle", "OperatorHandle"]
