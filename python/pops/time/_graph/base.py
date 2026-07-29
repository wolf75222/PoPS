"""Foundational immutable values shared by temporal graph records."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar, cast

from pops.identity.scalar import ScalarLiteral, scalar_data
from pops.model.ownership import OwnerPath
from pops.time.points import Clock, StagePoint, TimePoint


def nonempty(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def node_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("ProgramGraph node_id must be a non-negative Python int")
    return value


def strict_data(value: Any, *, where: str) -> Any:
    if isinstance(value, CanonicalData):
        return value.to_data()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, Decimal, Fraction, ScalarLiteral)):
        return {"scalar": scalar_data(value)}
    if isinstance(value, Enum):
        return {
            "enum": "%s.%s.%s"
            % (type(value).__module__, type(value).__qualname__, value.name)
        }
    if isinstance(value, OwnerPath):
        return {"owner_path": value.canonical().to_data()}
    if isinstance(value, (Clock, TimePoint, StagePoint)):
        return value.to_data()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return {
            key: strict_data(item, where="%s.%s" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            strict_data(item, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    canonical = getattr(value, "canonical_identity", None)
    if callable(canonical):
        return strict_data(canonical(), where=where)
    to_data = getattr(value, "to_data", None)
    if callable(to_data) and getattr(value, "__pops_ir_immutable__", False) is True:
        return strict_data(to_data(), where=where)
    raise TypeError(
        "%s contains mutable/opaque %s; provide canonical immutable data"
        % (where, type(value).__name__))


def _canonical_int(value: Any) -> int | None:
    """Read an integer from raw manifest data or graph-canonical scalar data."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping) and set(value) == {"scalar"}:
        scalar = value["scalar"]
        if isinstance(scalar, Mapping) and scalar.get("kind") == "integer":
            raw = scalar.get("value")
            if isinstance(raw, str):
                try:
                    return int(raw)
                except ValueError:
                    return None
    return None


def _validate_history_contract_data(contract: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "owner",
        "state",
        "space",
        "clock",
        "validity",
        "interpolation",
        "depth",
    }
    if not isinstance(contract, dict) or set(contract) != expected:
        raise ValueError(
            "history_interpolation requires a complete typed HistoryContract provider"
        )
    if _canonical_int(contract["schema_version"]) != 1:
        raise ValueError("history_interpolation provider has an unsupported contract schema")
    depth = _canonical_int(contract["depth"])
    if depth is None or depth < 1:
        raise ValueError("history_interpolation provider depth must be a positive integer")

    clock = contract["clock"]
    if not isinstance(clock, dict) or set(clock) != {"schema_version", "name", "owner"} \
            or _canonical_int(clock["schema_version"]) != 1 \
            or not isinstance(clock["name"], str) or not clock["name"]:
        raise ValueError("history_interpolation provider has an invalid logical clock")
    if contract["owner"] != clock["owner"]:
        raise ValueError(
            "history_interpolation provider owner does not own its logical clock"
        )

    state, space = contract["state"], contract["space"]
    if not isinstance(state, dict) or state.get("kind") != "state" \
            or not isinstance(state.get("block_ref"), dict):
        raise ValueError("history_interpolation provider state is not block-qualified")
    if not isinstance(space, dict) or space.get("kind") != "state":
        raise ValueError("history_interpolation provider space is not a StateSpace")

    validity = contract["validity"]
    if not isinstance(validity, dict) or set(validity) != {
            "schema_version", "oldest", "newest"} \
            or _canonical_int(validity["schema_version"]) != 1:
        raise ValueError("history_interpolation provider has an invalid validity interval")
    for endpoint in ("oldest", "newest"):
        point = validity[endpoint]
        if not isinstance(point, dict) or set(point) != {
                "schema_version", "clock", "step", "offset"} \
                or _canonical_int(point["schema_version"]) != 1 \
                or point["clock"] != clock:
            raise ValueError(
                "history_interpolation provider validity must use its logical clock"
            )

    capability = contract["interpolation"]
    if not isinstance(capability, dict) or capability.get("kind") in (None, "none"):
        raise ValueError(
            "history_interpolation provider has no interpolation/dense-output capability"
        )
    return contract


def validate_synchronization_relation_data(
    data: Any,
    *,
    source_history_contract: Any = None,
    source_clock: Any = None,
    source_clock_id: Any = None,
    history_rows: Any = None,
) -> dict[str, Any]:
    """Validate one relation and, when supplied, bind its provider to source context."""
    if not isinstance(data, dict) or not isinstance(data.get("kind"), str) or not data["kind"]:
        raise TypeError("synchronization relation to_data() must contain a non-empty kind")
    provider = data.get("provider")
    if not isinstance(provider, dict) or not isinstance(provider.get("kind"), str) \
            or not provider["kind"]:
        raise ValueError(
            "cross-clock synchronization relation must declare an explicit provider"
        )
    if data["kind"] == "history_interpolation":
        contract = provider.get("contract")
        if provider["kind"] != "typed_history":
            raise ValueError(
                "history_interpolation requires a complete typed HistoryContract provider"
            )
        contract = _validate_history_contract_data(contract)
        normalized_contract = strict_data(
            contract, where="history_interpolation provider contract")
        if strict_data(
                data.get("interpolation"),
                where="history_interpolation relation capability",
        ) != normalized_contract["interpolation"]:
            raise ValueError(
                "history_interpolation relation disagrees with its provider capability"
            )
        if source_clock is not None and normalized_contract["clock"] != strict_data(
                source_clock, where="history_interpolation source clock"):
            raise ValueError(
                "history_interpolation provider clock does not match the source clock"
            )
        if source_history_contract is not None and normalized_contract != strict_data(
                source_history_contract,
                where="history_interpolation source history contract",
        ):
            raise ValueError(
                "history_interpolation provider does not match its source history node"
            )
        if history_rows is not None:
            required = (
                "state", "space", "depth", "validity", "interpolation",
            )
            matches = [
                row for row in history_rows
                if isinstance(row, Mapping)
                and all(
                    strict_data(
                        row.get(key),
                        where="retained history %s" % key,
                    ) == normalized_contract[key]
                    for key in required
                )
                and row.get("clock") == (
                    source_clock if source_clock_id is None else source_clock_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "history_interpolation provider does not match exactly one retained history"
                )
    return data


@dataclass(frozen=True, slots=True, init=False)
class CanonicalData:
    """Hashable canonical-data snapshot used for semantic node metadata."""

    _json: str

    def __init__(self, value: Any, *, where: str = "ProgramGraph metadata") -> None:
        data = strict_data(value, where=where)
        object.__setattr__(
            self, "_json", json.dumps(data, sort_keys=True, separators=(",", ":")))

    def to_data(self) -> Any:
        return json.loads(self._json)


@dataclass(frozen=True, slots=True)
class ValueRef:
    """Readable SSA edge. Commit nodes deliberately cannot produce one."""

    node_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", node_id(self.node_id))

    def to_data(self) -> dict[str, int]:
        return {"node_id": self.node_id}


def refs(values: Any, *, where: str) -> tuple[ValueRef, ...]:
    result = tuple(values)
    if any(type(value) is not ValueRef for value in result):
        raise TypeError("%s must contain exact ValueRef values" % where)
    return result


def point(value: Any) -> TimePoint | StagePoint:
    if type(value) not in (TimePoint, StagePoint):
        raise TypeError("ProgramGraph node point must be an exact TimePoint or StagePoint")
    return value


def point_clocks(value: TimePoint | StagePoint) -> frozenset[Clock]:
    if type(value) is TimePoint:
        return frozenset((cast(TimePoint, value).clock,))
    return frozenset(
        item.clock for item in cast(StagePoint, value).partitions.values()
    )


def node_data(node: Any, **specific: Any) -> dict[str, Any]:
    return {
        "kind": node.kind,
        "node_id": node.node_id,
        "clock": node.clock.to_data(),
        "point": node.point.to_data(),
        **specific,
    }


class Node:
    """Closed internal base protocol for exact graph node records."""

    kind: ClassVar[str]
    readable: ClassVar[bool] = True
    node_id: int

    def references(self) -> tuple[ValueRef, ...]:
        return ()


__all__ = ["CanonicalData", "ValueRef", "validate_synchronization_relation_data"]
