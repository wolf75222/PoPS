"""Canonical boundary identities and explicit periodic topology."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pops.model import OwnerPath


_SCHEMA_VERSION = 1


def _handle_base() -> type:
    # Kept behind a function so the mesh-layer AST remains free of a module-scope model edge.
    from pops.model import Handle
    return Handle


def _canonical_owner(value: Any, *, where: str) -> OwnerPath:
    from pops.model import OwnerKind, OwnerPath

    if isinstance(value, str):
        raise TypeError("%s requires a canonical OwnerPath, never a string" % where)
    owner = OwnerPath.coerce(value)
    if owner.is_authoring:
        raise TypeError("%s is post-resolution and refuses authoring ownership" % where)
    if owner.kind is not OwnerKind.CASE and not owner.contains(OwnerKind.BLOCK):
        raise TypeError(
            "%s must be owned by a canonical Case or block-instance path" % where)
    return owner


def _name(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("%s must be a non-empty string" % where)
    value = value.strip()
    if "::" in value:
        raise ValueError("%s must not contain the reserved '::' separator" % where)
    return value


class BoundarySide(Enum):
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True)
class BoundaryOrientation:
    axis: int
    side: BoundarySide

    def __post_init__(self) -> None:
        if isinstance(self.axis, bool) or not isinstance(self.axis, int) or self.axis < 0:
            raise ValueError("BoundaryOrientation.axis must be an integer >= 0")
        if not isinstance(self.side, BoundarySide):
            raise TypeError("BoundaryOrientation.side must be a BoundarySide")

    @property
    def outward_sign(self) -> int:
        return -1 if self.side is BoundarySide.LOWER else 1

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "axis": self.axis,
                "side": self.side.value, "outward_sign": self.outward_sign}


class BoundaryHandle(_handle_base()):
    """Immutable Handle identity including the oriented domain face."""

    __slots__ = ("orientation",)

    def __init__(self, local_id: Any, *, owner: Any,
                 orientation: BoundaryOrientation, schema_version: int = 1) -> None:
        if not isinstance(orientation, BoundaryOrientation):
            raise TypeError("BoundaryHandle.orientation must be a BoundaryOrientation")
        super().__init__(_name(local_id, where="BoundaryHandle.local_id"), kind="boundary",
                         owner=_canonical_owner(owner, where="BoundaryHandle.owner"),
                         schema_version=schema_version)
        object.__setattr__(self, "orientation", orientation)

    def _qualified_id(self, owner_path: OwnerPath) -> str:
        return "%s::orientation::axis-%d-%s" % (
            super()._qualified_id(owner_path), self.orientation.axis, self.orientation.side.value)

    def _identity(self) -> tuple[Any, ...]:
        return super()._identity() + (self.orientation,)

    def inspect(self) -> dict[str, Any]:
        result = super().inspect()
        result.update({"handle_type": "boundary",
                       "orientation": self.orientation.canonical_identity()})
        return result

    def canonical_identity(self) -> dict[str, Any]:
        result = super().canonical_identity()
        result.update({"handle_type": "boundary",
                       "orientation": self.orientation.canonical_identity()})
        return result

    @classmethod
    def from_canonical_identity(cls, data: Any) -> BoundaryHandle:
        from pops.model import OwnerPath

        required = {"kind", "local_id", "owner_path", "qualified_id", "schema_version",
                    "handle_type", "orientation"}
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("BoundaryHandle canonical identity has an unsupported shape")
        raw = data["orientation"]
        if not isinstance(raw, Mapping) or set(raw) != {
                "schema_version", "axis", "side", "outward_sign"}:
            raise TypeError("BoundaryHandle orientation identity has an unsupported shape")
        orientation = BoundaryOrientation(raw["axis"], BoundarySide(raw["side"]))
        if raw["outward_sign"] != orientation.outward_sign:
            raise ValueError("BoundaryHandle orientation sign is inconsistent")
        result = cls(data["local_id"], owner=OwnerPath.from_data(data["owner_path"]),
                     orientation=orientation, schema_version=data["schema_version"])
        if result.canonical_identity() != dict(data):
            raise ValueError("BoundaryHandle identity failed round-trip authentication")
        return result


@dataclass(frozen=True, slots=True)
class PeriodicOrientation:
    """Signed axis mapping from the source face coordinates to the target face."""

    permutation: tuple[int, ...]
    signs: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.permutation, tuple) or not self.permutation:
            raise TypeError("PeriodicOrientation.permutation must be a non-empty tuple")
        if set(self.permutation) != set(range(len(self.permutation))):
            raise ValueError("PeriodicOrientation.permutation must be an axis permutation")
        if not isinstance(self.signs, tuple) or len(self.signs) != len(self.permutation) \
                or any(value not in (-1, 1) for value in self.signs):
            raise ValueError("PeriodicOrientation.signs must contain one -1/+1 per axis")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION,
                "permutation": list(self.permutation), "signs": list(self.signs)}


@dataclass(frozen=True, slots=True)
class PeriodicIdentification:
    source: BoundaryHandle
    target: BoundaryHandle
    orientation: PeriodicOrientation

    def __post_init__(self) -> None:
        if not isinstance(self.source, BoundaryHandle) or not isinstance(self.target, BoundaryHandle):
            raise TypeError("periodic endpoints must be BoundaryHandle objects")
        if self.source == self.target:
            raise ValueError("periodic identification requires two distinct boundaries")
        if self.source.owner_path != self.target.owner_path:
            raise ValueError("periodic endpoints must share the same Case owner")
        if not isinstance(self.orientation, PeriodicOrientation):
            raise TypeError("periodic identification requires a PeriodicOrientation")
        dimension = len(self.orientation.permutation)
        if self.source.orientation.axis >= dimension or self.target.orientation.axis >= dimension:
            raise ValueError("periodic orientation does not cover endpoint axes")
        if self.orientation.permutation[self.source.orientation.axis] != \
                self.target.orientation.axis:
            raise ValueError("periodic axis mapping does not map source normal to target normal")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "identification_type": "periodic",
                "source": self.source.canonical_identity(),
                "target": self.target.canonical_identity(),
                "orientation": self.orientation.canonical_identity()}


@dataclass(frozen=True, slots=True)
class BoundaryTopology:
    """Exact classification of all Case boundaries as periodic pairs or physical faces."""

    owner: OwnerPath
    boundaries: tuple[BoundaryHandle, ...]
    periodic: tuple[PeriodicIdentification, ...]
    physical: tuple[BoundaryHandle, ...]

    def __post_init__(self) -> None:
        owner = _canonical_owner(self.owner, where="BoundaryTopology.owner")
        for key in ("boundaries", "periodic", "physical"):
            if not isinstance(getattr(self, key), tuple):
                raise TypeError("BoundaryTopology.%s must be a tuple" % key)
        if any(not isinstance(row, BoundaryHandle) for row in self.boundaries):
            raise TypeError("BoundaryTopology.boundaries must contain BoundaryHandle objects")
        if any(not isinstance(row, PeriodicIdentification) for row in self.periodic):
            raise TypeError("BoundaryTopology.periodic must contain PeriodicIdentification objects")
        if any(not isinstance(row, BoundaryHandle) for row in self.physical):
            raise TypeError("BoundaryTopology.physical must contain BoundaryHandle objects")
        ids = [boundary.qualified_id for boundary in self.boundaries]
        if len(ids) != len(set(ids)):
            raise ValueError("double boundary declaration in BoundaryTopology")
        if any(row.owner_path != owner for row in self.boundaries):
            raise ValueError("every boundary must belong to BoundaryTopology.owner")
        known = set(ids)
        periodic_ids = [endpoint.qualified_id for row in self.periodic
                        for endpoint in (row.source, row.target)]
        physical_ids = [row.qualified_id for row in self.physical]
        if any(value not in known for value in periodic_ids + physical_ids):
            raise ValueError("extra topology classification references an undeclared boundary")
        if len(periodic_ids) != len(set(periodic_ids)):
            raise ValueError("double periodic identification for one boundary")
        if len(physical_ids) != len(set(physical_ids)):
            raise ValueError("double physical boundary classification")
        if set(periodic_ids) & set(physical_ids):
            raise ValueError("boundary cannot be periodic+physical")
        missing = known - set(periodic_ids) - set(physical_ids)
        if missing:
            raise ValueError("missing boundary topology classification: %s" % sorted(missing))
        object.__setattr__(self, "boundaries", tuple(sorted(
            self.boundaries, key=lambda row: row.qualified_id)))
        object.__setattr__(self, "periodic", tuple(sorted(
            self.periodic, key=lambda row: (row.source.qualified_id, row.target.qualified_id))))
        object.__setattr__(self, "physical", tuple(sorted(
            self.physical, key=lambda row: row.qualified_id)))

    def is_periodic(self, boundary: BoundaryHandle) -> bool:
        return any(boundary in (row.source, row.target) for row in self.periodic)

    def contains(self, boundary: BoundaryHandle) -> bool:
        return boundary in self.boundaries

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "topology_type": "boundary",
                "owner": self.owner.to_data(),
                "boundaries": [row.canonical_identity() for row in self.boundaries],
                "periodic": [row.canonical_identity() for row in self.periodic],
                "physical": [row.canonical_identity() for row in self.physical]}

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "boundary_topology", "canonical_id": self.canonical_id,
                **self.canonical_identity()}


__all__ = [
    "BoundaryHandle", "BoundaryOrientation", "BoundarySide", "BoundaryTopology",
    "PeriodicIdentification", "PeriodicOrientation",
]
