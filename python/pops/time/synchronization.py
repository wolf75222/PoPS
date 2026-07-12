"""Typed, extensible clock-domain synchronization relations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SynchronizationRelation(Protocol):
    """Small extension protocol for the semantics of one cross-clock transfer."""

    __pops_sync_relation__: bool

    def to_data(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SampleAndHold:
    """Read the latest source-clock value at the target clock coordinate."""

    __pops_sync_relation__ = True
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {"kind": "sample_and_hold", "schema_version": 1}


def relation_data(value: Any) -> dict[str, Any]:
    """Validate the extension protocol and return detached canonical relation data."""
    if not isinstance(value, SynchronizationRelation):
        raise TypeError(
            "synchronization relation must implement SynchronizationRelation "
            "(__pops_sync_relation__ and to_data())"
        )
    if value.__pops_sync_relation__ is not True:
        raise TypeError("synchronization relation marker must be exactly True")
    data = value.to_data()
    if not isinstance(data, dict) or not isinstance(data.get("kind"), str) or not data["kind"]:
        raise TypeError("synchronization relation to_data() must contain a non-empty kind")
    from pops.time.graph import CanonicalData

    return CanonicalData(data, where="SynchronizationRelation").to_data()


__all__ = ["SampleAndHold", "SynchronizationRelation", "relation_data"]
