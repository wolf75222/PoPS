"""Owner-qualified declaration handles for a Problem assembly."""
from __future__ import annotations

from typing import Any

from pops.model.handles import Handle


class BlockHandle(Handle):
    """Reference to one physics block declaration."""

    __slots__ = ()

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(name, kind="block", owner=owner)

    def state(self, name: Any) -> "StateHandle":
        return StateHandle(name, owner=self.owner_path.child("block", self.local_id))


class StateHandle(Handle):
    """Reference to one state declared by a block."""

    __slots__ = ()

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(name, kind="state", owner=owner)


class FieldHandle(Handle):
    """Reference to one field declaration."""

    __slots__ = ()

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(name, kind="field", owner=owner)


class OperatorHandle(Handle):
    """Reference to a Problem-scoped coupling/local operator declaration."""

    __slots__ = ()

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(name, kind="operator", owner=owner)


__all__ = ["BlockHandle", "StateHandle", "FieldHandle", "OperatorHandle"]
