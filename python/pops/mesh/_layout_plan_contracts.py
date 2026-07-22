"""Immutable value contracts shared by layout-plan authoring and consumers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pops._geometry_contracts import (
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_CELL_AREA,
    POLAR_ANNULUS_2D_COORDINATES,
    POLAR_ANNULUS_CELL_AREA,
)
from pops.model import Handle, OwnerKind, OwnerPath


SCHEMA_VERSION = 1
SUBJECT_KINDS = frozenset(("state", "field", "block"))

class LayoutRepresentation(Enum):
    """Versioned numerical meaning of a mapping port's stored values."""

    CELL_AVERAGE_V1 = "pops://representations/cell-average@1"

    def to_data(self) -> dict[str, Any]:
        return {"uri": self.value}


class LayoutMappingOperation(IntEnum):
    """Generated Transfer ABI-v1 operation value, never a free selector string."""

    CONSERVATIVE_CELL_AVERAGE_V1 = 1

    def to_data(self) -> dict[str, Any]:
        return {
            "interface": "pops://interfaces/transfer@1",
            "name": self.name,
            "abi_value": int(self),
        }


class LayoutSynchronization(Enum):
    """Qualified point at which a directional mapping is part of the step transaction."""

    BEFORE_STEP_V1 = "pops://synchronization/before-step@1"

    def to_data(self) -> dict[str, Any]:
        return {"uri": self.value}


def name(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("%s must be a non-empty string" % where)
    value = value.strip()
    if "::" in value:
        raise ValueError("%s must not contain the reserved '::' separator" % where)
    return value


def canonical_owner(value: Any, *, where: str) -> OwnerPath:
    if isinstance(value, str):
        raise TypeError("%s requires an OwnerPath authority, never a string" % where)
    result = OwnerPath.coerce(value)
    if result.is_authoring:
        raise TypeError(
            "%s is a post-resolution contract and requires a canonical OwnerPath; "
            "resolve the authoritative registry before building a LayoutPlan" % where)
    return result


def handle_identity(value: Any, *, where: str, kind: str | None = None) -> str:
    if not isinstance(value, Handle):
        raise TypeError("%s must be a canonical pops.model.Handle" % where)
    if not value.is_resolved:
        raise TypeError("%s handle %s is not canonically owner-qualified" %
                        (where, value.qualified_id))
    if kind is not None and value.kind != kind:
        raise TypeError("%s requires Handle.kind=%r, got %r" % (where, kind, value.kind))
    identity = value.canonical_identity()
    if identity.get("qualified_id") != value.qualified_id:
        raise ValueError("%s handle canonical identity does not authenticate qualified_id" % where)
    return value.qualified_id


def json_data(value: Any, *, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("%s mappings require non-empty string keys" % where)
            out[key] = json_data(item, where="%s.%s" % (where, key))
        return out
    if isinstance(value, (list, tuple)):
        return [json_data(item, where=where) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return json_data(to_dict(), where=where)
    raise TypeError("%s must contain strict JSON data, got %r" % (where, value))


def freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def subject_kind(kind: str) -> str:
    kind = name(kind, where="subject kind")
    if kind not in SUBJECT_KINDS:
        raise ValueError("subject kind must be one of block, field, state (got %r)" % kind)
    return kind


class LayoutHandle(Handle):
    """Immutable identity of one layout declaration within a resolved Case."""

    __slots__ = ()

    def __init__(self, local_id: Any, *, owner: Any, schema_version: int = 1) -> None:
        local_id = name(local_id, where="LayoutHandle.local_id")
        owner = canonical_owner(owner, where="LayoutHandle.owner")
        owner.child(OwnerKind.LAYOUT, local_id)
        super().__init__(local_id, kind="layout", owner=owner, schema_version=schema_version)

    @property
    def layout_owner_path(self) -> OwnerPath:
        return self.owner_path.child(OwnerKind.LAYOUT, self.local_id)

    @classmethod
    def from_canonical_identity(cls, data: Any) -> LayoutHandle:
        if not isinstance(data, Mapping):
            raise TypeError("LayoutHandle canonical identity must be a mapping")
        required = {"kind", "local_id", "owner_path", "qualified_id", "schema_version"}
        if set(data) != required or data.get("kind") != "layout":
            raise TypeError("LayoutHandle canonical identity has an unsupported shape")
        result = cls(data["local_id"], owner=OwnerPath.from_data(data["owner_path"]),
                     schema_version=data["schema_version"])
        if result.canonical_identity() != dict(data):
            raise ValueError("LayoutHandle canonical identity failed round-trip authentication")
        return result


@dataclass(frozen=True, slots=True)
class LayoutLevel:
    index: int
    refinement: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("LayoutLevel.index must be an integer >= 0")
        if isinstance(self.refinement, bool) or not isinstance(self.refinement, int) \
                or self.refinement < 1:
            raise ValueError("LayoutLevel.refinement must be an integer >= 1")

    def to_data(self) -> dict[str, int]:
        return {"index": self.index, "refinement": self.refinement}


def _geometry_uri(value: Any, *, where: str) -> str:
    value = name(value, where=where)
    if not value.startswith("pops://") or "@" not in value:
        raise ValueError("%s must be a versioned pops:// URI" % where)
    return value


def _geometry_axis_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("NormalizedGeometry.axis_names must contain at least one name")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "NormalizedGeometry.axis_names must contain at least one name") from exc
    if not result:
        raise TypeError("NormalizedGeometry.axis_names must contain at least one name")
    names = tuple(name(item, where="NormalizedGeometry.axis_names") for item in result)
    if len(set(names)) != len(names):
        raise ValueError("NormalizedGeometry.axis_names must be distinct")
    return names


def _geometry_points(value: Any, *, where: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("%s must contain at least one finite binary64 value" % where)
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise TypeError("%s must contain at least one finite binary64 value" % where) from exc
    if not raw:
        raise TypeError("%s must contain at least one finite binary64 value" % where)
    result = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) for item in result):
        raise ValueError("%s must contain only finite binary64 values" % where)
    return result


def _geometry_cells(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("NormalizedGeometry.cells must contain at least one positive integer")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "NormalizedGeometry.cells must contain at least one positive integer") from exc
    if not result or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in result):
        raise ValueError(
            "NormalizedGeometry.cells must contain only positive integers")
    return result


@dataclass(frozen=True, slots=True)
class NormalizedGeometry:
    """Detached exact rank-generic geometry projected by any layout implementation.

    Coordinate and cell-measure semantics use versioned URIs so extension descriptors can project
    new geometries without teaching layout normalization about their implementation classes.
    Scientific consumers still fail closed when they do not implement a projected measure URI.
    """

    coordinate_system: str
    cell_measure: str
    axis_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    cells: tuple[int, ...]
    frame_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_system", _geometry_uri(
            self.coordinate_system, where="NormalizedGeometry.coordinate_system"))
        object.__setattr__(self, "cell_measure", _geometry_uri(
            self.cell_measure, where="NormalizedGeometry.cell_measure"))
        object.__setattr__(self, "axis_names", _geometry_axis_names(self.axis_names))
        if self.frame_id is not None \
                and (not isinstance(self.frame_id, str) or not self.frame_id):
            raise TypeError("NormalizedGeometry.frame_id must be non-empty text or None")
        lower = _geometry_points(self.lower, where="NormalizedGeometry.lower")
        upper = _geometry_points(self.upper, where="NormalizedGeometry.upper")
        cells = _geometry_cells(self.cells)
        rank = len(self.axis_names)
        if len(lower) != rank or len(upper) != rank or len(cells) != rank:
            raise ValueError(
                "NormalizedGeometry axes, bounds and cells must have one common rank")
        if any(high <= low for low, high in zip(lower, upper, strict=True)):
            raise ValueError("NormalizedGeometry.upper must be strictly above lower on every axis")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "cells", cells)

    @property
    def dimension(self) -> int:
        return len(self.axis_names)

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(high - low for low, high in zip(self.lower, self.upper, strict=True))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "coordinate_system": self.coordinate_system,
            "cell_measure": self.cell_measure,
            "axis_names": list(self.axis_names),
            "lower": [item.hex() for item in self.lower],
            "upper": [item.hex() for item in self.upper],
            "cells": list(self.cells),
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_data(cls, data: Any) -> NormalizedGeometry:
        required = {
            "schema_version", "coordinate_system", "cell_measure", "axis_names",
            "lower", "upper", "cells", "frame_id",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("NormalizedGeometry data has an unsupported shape")
        if data["schema_version"] != 1:
            raise ValueError("NormalizedGeometry data uses an unsupported schema")
        for key in ("lower", "upper"):
            values = data[key]
            if not isinstance(values, list) or not values \
                    or any(not isinstance(item, str) for item in values):
                raise TypeError(
                    "NormalizedGeometry.%s data must contain float.hex values" % key)
        try:
            lower = tuple(float.fromhex(item) for item in data["lower"])
            upper = tuple(float.fromhex(item) for item in data["upper"])
        except ValueError:
            raise ValueError("NormalizedGeometry bounds contain an invalid float.hex value") from None
        result = cls(
            data["coordinate_system"], data["cell_measure"], tuple(data["axis_names"]),
            lower, upper, tuple(data["cells"]), data["frame_id"],
        )
        if result.to_data() != dict(data):
            raise ValueError("NormalizedGeometry data is not canonical")
        return result


@runtime_checkable
class NormalizedGeometryProvider(Protocol):
    """Open descriptor protocol used before snapshots reach runtime consumers."""

    def normalized_geometry(self) -> NormalizedGeometry: ...


@dataclass(frozen=True, slots=True)
class NormalizedLayout:
    """Algorithm-neutral level plan; Uniform is the one-level degenerate case."""

    handle: LayoutHandle
    descriptor_type: str
    descriptor_name: str
    adaptive: bool
    transition_ratios: tuple[int, ...]
    levels: tuple[LayoutLevel, ...]
    geometry: NormalizedGeometry
    options: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    requirements: Mapping[str, Any]
    descriptor_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.handle, LayoutHandle):
            raise TypeError("NormalizedLayout.handle must be a LayoutHandle")
        handle_identity(self.handle, where="NormalizedLayout.handle", kind="layout")
        if type(self.geometry) is not NormalizedGeometry:
            raise TypeError("NormalizedLayout.geometry must be an exact NormalizedGeometry")
        object.__setattr__(self, "geometry", NormalizedGeometry.from_data(
            self.geometry.to_data()))
        ratios = tuple(self.transition_ratios)
        if len(ratios) != max(0, len(self.levels) - 1) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in ratios):
            raise ValueError(
                "NormalizedLayout.transition_ratios must contain one integer >= 2 per transition"
            )
        refinement = 1
        for index, level in enumerate(self.levels):
            if level.index != index or level.refinement != refinement:
                raise ValueError(
                    "NormalizedLayout.levels must preserve exact cumulative transition refinement"
                )
            if index < len(ratios):
                refinement *= ratios[index]
        object.__setattr__(self, "transition_ratios", ratios)
        for key in ("options", "capabilities", "requirements", "descriptor_snapshot"):
            data = json_data(getattr(self, key), where="NormalizedLayout.%s" % key)
            if not isinstance(data, dict):
                raise TypeError("NormalizedLayout.%s must be a mapping" % key)
            object.__setattr__(self, key, freeze(data))

    def to_data(self) -> dict[str, Any]:
        return {
            "handle": self.handle.canonical_identity(),
            "descriptor_type": self.descriptor_type,
            "descriptor_name": self.descriptor_name,
            "adaptive": self.adaptive,
            "transition_ratios": list(self.transition_ratios),
            "levels": [level.to_data() for level in self.levels],
            "geometry": self.geometry.to_data(),
            "options": thaw(self.options),
            "capabilities": thaw(self.capabilities),
            "requirements": thaw(self.requirements),
            "descriptor_snapshot": thaw(self.descriptor_snapshot),
        }


@dataclass(frozen=True, slots=True)
class LayoutAssignment:
    subject: Handle
    layout: LayoutHandle

    def __post_init__(self) -> None:
        if not isinstance(self.subject, Handle):
            raise TypeError("LayoutAssignment.subject must be a canonical Handle")
        kind = subject_kind(self.subject.kind)
        handle_identity(self.subject, where="LayoutAssignment.subject", kind=kind)
        if not isinstance(self.layout, LayoutHandle):
            raise TypeError("LayoutAssignment.layout must be a LayoutHandle")

    @property
    def subject_kind(self) -> str:
        return self.subject.kind

    @property
    def subject_id(self) -> str:
        return self.subject.qualified_id

    def to_data(self) -> dict[str, Any]:
        return {"subject": self.subject.canonical_identity(),
                "layout": self.layout.canonical_identity()}


@dataclass(frozen=True, slots=True)
class LayoutMappingPort:
    """One owner-qualified field crossing a layout boundary.

    A layout name alone cannot identify the storage that a transfer reads or writes.  Ports make
    the scientific representation part of the immutable plan and keep runtime routing independent
    from display labels or block ordering.
    """

    subject: Handle
    representation: LayoutRepresentation = LayoutRepresentation.CELL_AVERAGE_V1

    def __post_init__(self) -> None:
        if not isinstance(self.subject, Handle) or self.subject.kind not in ("state", "field"):
            raise TypeError("LayoutMappingPort.subject must be a canonical state or field Handle")
        handle_identity(
            self.subject, where="LayoutMappingPort.subject", kind=self.subject.kind)
        if type(self.representation) is not LayoutRepresentation:
            raise TypeError(
                "LayoutMappingPort.representation must be an exact LayoutRepresentation")

    def to_data(self) -> dict[str, Any]:
        return {
            "subject": self.subject.canonical_identity(),
            "representation": self.representation.to_data(),
        }


@dataclass(frozen=True, slots=True)
class LayoutMappingRequirement:
    source_layout: LayoutHandle
    target_layout: LayoutHandle
    source_port: LayoutMappingPort
    target_port: LayoutMappingPort
    operation: LayoutMappingOperation
    synchronization: LayoutSynchronization
    reverse_of: str | None = None

    def __post_init__(self) -> None:
        for endpoint in (self.source_layout, self.target_layout):
            if not isinstance(endpoint, LayoutHandle):
                raise TypeError("layout mapping endpoints must be LayoutHandle objects")
            handle_identity(endpoint, where="layout mapping endpoint", kind="layout")
        if self.source_layout == self.target_layout:
            raise ValueError("a layout mapping must cross two distinct layouts")
        for port in (self.source_port, self.target_port):
            if type(port) is not LayoutMappingPort:
                raise TypeError("layout mapping ports must be exact LayoutMappingPort values")
        if type(self.operation) is not LayoutMappingOperation:
            raise TypeError(
                "layout mapping operation must be an exact LayoutMappingOperation")
        if type(self.synchronization) is not LayoutSynchronization:
            raise TypeError(
                "layout mapping synchronization must be an exact LayoutSynchronization")
        if self.reverse_of is not None and (
                not isinstance(self.reverse_of, str) or not self.reverse_of):
            raise TypeError("reverse mapping identity must be a non-empty string")

    @property
    def qualified_id(self) -> str:
        raw = canonical({
            "source_layout": self.source_layout.canonical_identity(),
            "target_layout": self.target_layout.canonical_identity(),
            "source_port": self.source_port.to_data(),
            "target_port": self.target_port.to_data(),
            "operation": self.operation.to_data(),
            "synchronization": self.synchronization.to_data(),
            "reverse_of": self.reverse_of,
        })
        return "pops.layout-mapping.v2::" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_data(self) -> dict[str, Any]:
        return {
            "qualified_id": self.qualified_id,
            "source_layout": self.source_layout.canonical_identity(),
            "target_layout": self.target_layout.canonical_identity(),
            "source_port": self.source_port.to_data(),
            "target_port": self.target_port.to_data(),
            "operation": self.operation.to_data(),
            "synchronization": self.synchronization.to_data(),
            "reverse_of": self.reverse_of,
        }


def reject_concurrent_overwrite_mappings(requirements: Any) -> None:
    """Require one writer per target storage/synchronization for overwrite operations."""
    writers: dict[tuple[str, str, str], str] = {}
    for requirement in requirements:
        if requirement.operation is not \
                LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1:
            continue
        key = (
            requirement.target_layout.qualified_id,
            requirement.target_port.subject.qualified_id,
            requirement.synchronization.value,
        )
        previous = writers.get(key)
        if previous is not None:
            raise ValueError(
                "concurrent overwrite mappings %s and %s target the same subject at %s; "
                "declare an explicit merge operation/provider"
                % (previous, requirement.qualified_id, requirement.synchronization.value))
        writers[key] = requirement.qualified_id


@runtime_checkable
class LayoutMappingProvider(Protocol):
    @property
    def qualified_id(self) -> str: ...

    def canonical_identity(self) -> Mapping[str, Any]: ...

    def supports_layout_mapping(self, requirement: LayoutMappingRequirement) -> bool: ...


@dataclass(frozen=True, slots=True)
class ResolvedLayoutMapping:
    requirement: LayoutMappingRequirement
    provider_id: str
    provider_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        data = json_data(self.provider_identity, where="mapping provider identity")
        if not isinstance(data, dict) or data.get("qualified_id") != self.provider_id:
            raise ValueError("provider identity does not authenticate provider_id")
        object.__setattr__(self, "provider_identity", freeze(data))

    def to_data(self) -> dict[str, Any]:
        return {"requirement": self.requirement.to_data(), "provider_id": self.provider_id,
                "provider_identity": thaw(self.provider_identity)}


def plan_payload(owner_path: OwnerPath, layouts: tuple[NormalizedLayout, ...],
                 assignments: tuple[LayoutAssignment, ...],
                 mappings: tuple[ResolvedLayoutMapping, ...]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "owner": owner_path.to_data(),
            "layouts": [row.to_data() for row in layouts],
            "assignments": [row.to_data() for row in assignments],
            "mappings": [row.to_data() for row in mappings]}


@dataclass(frozen=True, slots=True)
class LayoutPlan:
    """Detached immutable result consumed by later compile/install phases."""

    owner: OwnerPath
    layouts: tuple[NormalizedLayout, ...]
    assignments: tuple[LayoutAssignment, ...]
    mappings: tuple[ResolvedLayoutMapping, ...]
    canonical_id: str

    def __post_init__(self) -> None:
        owner_path = canonical_owner(self.owner, where="LayoutPlan.owner")
        for key in ("layouts", "assignments", "mappings"):
            if not isinstance(getattr(self, key), tuple):
                raise TypeError("LayoutPlan.%s must be a tuple" % key)
        layout_ids = [row.handle.qualified_id for row in self.layouts]
        if len(layout_ids) != len(set(layout_ids)):
            raise ValueError("LayoutPlan contains duplicate layouts")
        if any(row.handle.owner_path != owner_path for row in self.layouts):
            raise ValueError("every LayoutPlan layout must belong to the plan owner")
        known_layouts = set(layout_ids)
        assignment_keys = [(row.subject_kind, row.subject_id) for row in self.assignments]
        if len(assignment_keys) != len(set(assignment_keys)):
            raise ValueError("LayoutPlan contains duplicate assignments")
        if any(row.layout.qualified_id not in known_layouts for row in self.assignments):
            raise ValueError("LayoutPlan assignment references an undeclared layout")
        if any(row.requirement.source_layout.qualified_id not in known_layouts or
               row.requirement.target_layout.qualified_id not in known_layouts
               for row in self.mappings):
            raise ValueError("LayoutPlan mapping references an undeclared layout")
        assignments = {
            (row.subject_kind, row.subject_id): row.layout for row in self.assignments
        }
        for row in self.mappings:
            requirement = row.requirement
            source = requirement.source_port.subject
            target = requirement.target_port.subject
            if assignments.get((source.kind, source.qualified_id)) != requirement.source_layout:
                raise ValueError("LayoutPlan mapping source port is not assigned to source_layout")
            if assignments.get((target.kind, target.qualified_id)) != requirement.target_layout:
                raise ValueError("LayoutPlan mapping target port is not assigned to target_layout")
        reject_concurrent_overwrite_mappings(
            row.requirement for row in self.mappings)
        expected = hashlib.sha256(canonical(plan_payload(
            owner_path, self.layouts, self.assignments, self.mappings)).encode("utf-8")).hexdigest()
        if self.canonical_id != expected:
            raise ValueError("LayoutPlan canonical_id does not authenticate its complete payload")

    @property
    def qualified_id(self) -> str:
        return "pops.layout-plan.v1::%s" % self.canonical_id

    def inspect(self) -> dict[str, Any]:
        data = plan_payload(self.owner, self.layouts, self.assignments, self.mappings)
        data.update({"report_type": "layout_plan", "qualified_id": self.qualified_id,
                     "canonical_id": self.canonical_id})
        return data

    def canonical_identity(self) -> dict[str, Any]:
        return self.inspect()

    def layout_for(self, subject: Any) -> LayoutHandle:
        if not isinstance(subject, Handle):
            raise TypeError("layout_for subject must be a canonical pops.model.Handle")
        key = (subject_kind(subject.kind), handle_identity(subject, where="layout_for subject"))
        matches = [row.layout for row in self.assignments
                   if (row.subject_kind, row.subject_id) == key]
        if len(matches) != 1:
            raise KeyError("no exact %s layout assignment for %s" % key)
        return matches[0]

    def normalized(self, handle: Any) -> NormalizedLayout:
        """Return the one authenticated normalized row for a plan-owned layout handle."""
        if not isinstance(handle, LayoutHandle):
            raise TypeError("normalized layout lookup requires a LayoutHandle")
        matches = [row for row in self.layouts if row.handle == handle]
        if len(matches) != 1:
            raise KeyError("layout %s is not declared by this LayoutPlan" % handle.qualified_id)
        return matches[0]

    def validate_subjects(self, *, states: Any = (), fields: Any = (), blocks: Any = ()) -> None:
        """Prove every materialized subject has exactly one assignment and no extras."""
        expected = set()
        for kind, values in (("state", states), ("field", fields), ("block", blocks)):
            for value in values:
                key = (kind, handle_identity(value, where="expected %s" % kind, kind=kind))
                if key in expected:
                    raise ValueError("duplicate materialized layout subject %s" % (key,))
                expected.add(key)
        authored = {(row.subject_kind, row.subject_id) for row in self.assignments}
        missing, extra = sorted(expected - authored), sorted(authored - expected)
        if missing:
            raise ValueError("unassigned layout subjects: %s" % missing)
        if extra:
            raise ValueError("layout assignments are not exact; unexpected subjects: %s" % extra)

    def capability_evidence(self) -> dict[str, Any]:
        """Detached per-layout evidence; independent layouts never contaminate another assignment."""
        rows = {row.handle.qualified_id: row for row in self.layouts}
        return {
            "layouts": [
                {"layout": row.handle.canonical_identity(),
                 "capabilities": thaw(row.capabilities),
                 "requirements": thaw(row.requirements)}
                for row in self.layouts
            ],
            "assignments": [
                {**assignment.to_data(),
                 "capabilities": thaw(rows[assignment.layout.qualified_id].capabilities)}
                for assignment in self.assignments
            ],
            "mappings": [mapping.to_data() for mapping in self.mappings],
        }

    def resource_requirements(self) -> tuple[dict[str, Any], ...]:
        """Exact directional mapping resources consumed by lowering/runtime planning."""
        return tuple({"kind": "layout_mapping", **mapping.to_data()}
                     for mapping in self.mappings)


__all__ = [
    "CARTESIAN_2D_COORDINATES", "CARTESIAN_CELL_AREA",
    "LayoutAssignment", "LayoutHandle", "LayoutLevel", "LayoutMappingOperation",
    "LayoutMappingProvider", "LayoutMappingPort", "LayoutMappingRequirement",
    "LayoutRepresentation", "LayoutSynchronization", "LayoutPlan", "NormalizedLayout",
    "NormalizedGeometry", "NormalizedGeometryProvider", "POLAR_ANNULUS_2D_COORDINATES",
    "POLAR_ANNULUS_CELL_AREA", "ResolvedLayoutMapping",
]
