"""Typed, extensible clock-domain synchronization relations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SynchronizationRelation(Protocol):
    """Small extension protocol for the semantics of one cross-clock transfer."""

    __pops_sync_relation__: bool

    def to_data(self) -> dict[str, Any]: ...

    def validate_transfer(self, source: Any, target: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class SampleAndHold:
    """Read the latest source-clock value at the target clock coordinate."""

    __pops_sync_relation__ = True
    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "sample_and_hold",
            "schema_version": 1,
            "provider": {"kind": "latest_accepted_sample", "schema_version": 1},
        }

    def validate_transfer(self, source: Any, target: Any) -> None:
        """The current/latest accepted source sample is the explicit provider."""
        del target
        from pops.time.values import ProgramValue
        if not isinstance(source, ProgramValue):
            raise TypeError("SampleAndHold source must resolve to a ProgramValue")


@dataclass(frozen=True, slots=True)
class InterpolateHistory:
    """Use one typed history's declared interpolation/dense-output capability."""

    history: Any
    __pops_sync_relation__ = True
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops.time.handles import HistoryHandle
        if not isinstance(self.history, HistoryHandle):
            raise TypeError("InterpolateHistory requires a HistoryHandle")
        if self.history.contract.interpolation.to_data()["kind"] == "none":
            raise ValueError(
                "InterpolateHistory requires keep_history(..., interpolation=...) capability")

    def validate_transfer(self, source: Any, target: Any) -> None:
        del target
        expected = self.history.value
        if source is not expected:
            raise ValueError(
                "InterpolateHistory provider must be the same HistoryHandle as the transferred value")

    def to_data(self) -> dict[str, Any]:
        contract = self.history.contract.to_data()
        return {
            "kind": "history_interpolation",
            "schema_version": 1,
            "provider": {"kind": "typed_history", "schema_version": 1,
                         "contract": contract},
            "interpolation": contract["interpolation"],
        }


def validate_relation_data(data: Any) -> dict[str, Any]:
    """Reject a cross-clock relation that names no explicit value provider."""
    if not isinstance(data, dict) or not isinstance(data.get("kind"), str) or not data["kind"]:
        raise TypeError("synchronization relation to_data() must contain a non-empty kind")
    provider = data.get("provider")
    if not isinstance(provider, dict) or not isinstance(provider.get("kind"), str) \
            or not provider["kind"]:
        raise ValueError(
            "cross-clock synchronization relation must declare an explicit provider")
    if data["kind"] == "history_interpolation":
        contract = provider.get("contract")
        expected = {
            "schema_version", "owner", "state", "space", "clock", "validity",
            "interpolation", "depth",
        }
        if provider["kind"] != "typed_history" or not isinstance(contract, dict) \
                or set(contract) != expected:
            raise ValueError(
                "history_interpolation requires a complete typed HistoryContract provider")
        capability = contract.get("interpolation")
        if not isinstance(capability, dict) or capability.get("kind") in (None, "none"):
            raise ValueError(
                "history_interpolation provider has no interpolation/dense-output capability")
    return data


def relation_data(value: Any, *, source: Any = None, target: Any = None) -> dict[str, Any]:
    """Validate the extension protocol and return detached canonical relation data."""
    if not isinstance(value, SynchronizationRelation):
        raise TypeError(
            "synchronization relation must implement SynchronizationRelation "
            "(__pops_sync_relation__, validate_transfer(), and to_data())"
        )
    if value.__pops_sync_relation__ is not True:
        raise TypeError("synchronization relation marker must be exactly True")
    data = value.to_data()
    if source is not None:
        value.validate_transfer(source, target)
    validate_relation_data(data)
    from pops.time.graph import CanonicalData

    return CanonicalData(data, where="SynchronizationRelation").to_data()


__all__ = [
    "InterpolateHistory", "SampleAndHold", "SynchronizationRelation",
    "relation_data", "validate_relation_data",
]
