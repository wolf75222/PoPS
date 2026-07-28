"""Typed hierarchy policies for authenticated checkpoint restart."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pops.identity import Identity, make_identity


class RestartHierarchy:
    """Immutable semantic contract for the hierarchy selected during restart."""

    __pops_ir_immutable__ = True
    mode: ClassVar[str]
    guarantee: ClassVar[str]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "guarantee": self.guarantee,
        }

    @property
    def identity(self) -> Identity:
        return make_identity("restart-hierarchy", self.to_data())


@dataclass(frozen=True, slots=True)
class RestoreRecordedHierarchy(RestartHierarchy):
    """Restore the checkpoint's exact hierarchy, ownership, and accepted state."""

    mode: ClassVar[str] = "restore_recorded_hierarchy"
    guarantee: ClassVar[str] = "accepted_state_with_recorded_hierarchy"


@dataclass(frozen=True, slots=True)
class RegridOnRestart(RestartHierarchy):
    """Rebuild the hierarchy from restored state under an explicitly weaker guarantee."""

    mode: ClassVar[str] = "regrid_on_restart"
    guarantee: ClassVar[str] = "accepted_state_after_regrid"


_RESTART_HIERARCHIES = (RestoreRecordedHierarchy, RegridOnRestart)


def require_restart_hierarchy(value: Any, *, where: str) -> RestartHierarchy:
    """Require one exact built-in hierarchy contract; structural lookalikes are refused."""
    if type(value) not in _RESTART_HIERARCHIES:
        raise TypeError(
            "%s must be RestoreRecordedHierarchy() or RegridOnRestart()" % where
        )
    return value


__all__ = ["RegridOnRestart", "RestartHierarchy", "RestoreRecordedHierarchy"]
