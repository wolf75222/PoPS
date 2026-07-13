"""Canonical, provider-driven AMR hierarchy authoring and resolution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops.identity import Identity, make_identity
from pops.model import Handle
from pops.time import AcceptedStep, EventHandle, Schedule


_SCHEMA_VERSION = 1


class HierarchyPhaseError(ValueError):
    """A runtime hierarchy effect was requested outside an accepted commit phase."""


def _handle(value: Any, *, where: str, kind: str) -> Handle:
    if not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if value.kind != kind:
        raise TypeError("%s requires Handle.kind=%r, got %r" % (where, kind, value.kind))
    value.canonical_identity()
    return value


def _positive_int(value: Any, *, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (where, minimum))
    return value


def _axes(value: Any, *, where: str, minimum: int) -> tuple[int, ...]:
    result = tuple(value)
    if not result or len(result) > 3:
        raise ValueError("%s must contain one, two, or three spatial axes" % where)
    for item in result:
        _positive_int(item, where=where, minimum=minimum)
    return result


def _freeze_data(value: Any, *, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return ("scalar", value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("%s contains a non-finite float" % where)
        return ("binary64", value.hex())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return (
            "mapping",
            tuple(
                (key, _freeze_data(value[key], where="%s.%s" % (where, key)))
                for key in sorted(value)
            ),
        )
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_freeze_data(item, where="%s[]" % where) for item in value))
    raise TypeError(
        "%s contains non-data value %s; callbacks/objects are forbidden"
        % (where, type(value).__name__)
    )


def _thaw_data(value: Any) -> Any:
    tag, payload = value
    if tag == "scalar":
        return payload
    if tag == "binary64":
        return {"binary64": payload}
    if tag == "mapping":
        return {key: _thaw_data(item) for key, item in payload}
    if tag == "sequence":
        return [_thaw_data(item) for item in payload]
    raise ValueError("invalid frozen hierarchy data tag")


@dataclass(frozen=True, slots=True, init=False)
class CanonicalOptions:
    """Deeply immutable data-only configuration for an extension provider."""

    _data: Any
    __pops_ir_immutable__ = True

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        values = {} if values is None else values
        if not isinstance(values, Mapping):
            raise TypeError("CanonicalOptions requires a mapping")
        object.__setattr__(self, "_data", _freeze_data(values, where="CanonicalOptions"))

    def to_data(self) -> dict[str, Any]:
        return _thaw_data(self._data)


@dataclass(frozen=True, slots=True)
class ClusteringPolicy:
    provider: Handle
    options: CanonicalOptions
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _handle(self.provider, where="ClusteringPolicy.provider", kind="amr_clustering_provider")
        if type(self.options) is not CanonicalOptions:
            raise TypeError("ClusteringPolicy.options must be CanonicalOptions")

    def canonical_identity(self) -> dict[str, Any]:
        return {"provider": self.provider.canonical_identity(), "options": self.options.to_data()}


@dataclass(frozen=True, slots=True)
class PatchGenerationPolicy:
    provider: Handle
    options: CanonicalOptions
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _handle(
            self.provider,
            where="PatchGenerationPolicy.provider",
            kind="amr_patch_generation_provider",
        )
        if type(self.options) is not CanonicalOptions:
            raise TypeError("PatchGenerationPolicy.options must be CanonicalOptions")

    def canonical_identity(self) -> dict[str, Any]:
        return {"provider": self.provider.canonical_identity(), "options": self.options.to_data()}


@dataclass(frozen=True, slots=True)
class LoadBalancePolicy:
    provider: Handle
    options: CanonicalOptions
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _handle(self.provider, where="LoadBalancePolicy.provider", kind="amr_load_balance_provider")
        if type(self.options) is not CanonicalOptions:
            raise TypeError("LoadBalancePolicy.options must be CanonicalOptions")

    def canonical_identity(self) -> dict[str, Any]:
        return {"provider": self.provider.canonical_identity(), "options": self.options.to_data()}


@dataclass(frozen=True, slots=True)
class NestingRequirementSource:
    """Canonical provider manifest contributing a derived nesting need."""

    provider: Handle
    minimum_buffer: tuple[int, ...]
    minimum_lookahead: int
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        allowed = {
            "amr_stencil_requirement",
            "amr_transfer_requirement",
            "amr_reflux_requirement",
            "amr_boundary_requirement",
        }
        if not isinstance(self.provider, Handle) or not self.provider.is_resolved:
            raise TypeError("NestingRequirementSource.provider must be a canonical Handle")
        if self.provider.kind not in allowed:
            raise TypeError(
                "NestingRequirementSource provider has unsupported kind %r" % self.provider.kind
            )
        self.provider.canonical_identity()
        object.__setattr__(
            self, "minimum_buffer", _axes(self.minimum_buffer, where="minimum_buffer", minimum=0)
        )
        object.__setattr__(
            self,
            "minimum_lookahead",
            _positive_int(self.minimum_lookahead, where="minimum_lookahead", minimum=0),
        )

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "provider": self.provider.canonical_identity(),
            "minimum_buffer": list(self.minimum_buffer),
            "minimum_lookahead": self.minimum_lookahead,
        }


@dataclass(frozen=True, slots=True)
class DerivedNestingRequirements:
    """Nesting derived only from stencil/transfer/reflux/boundary manifests."""

    stencil: NestingRequirementSource
    transfer: NestingRequirementSource
    reflux: NestingRequirementSource
    boundary: NestingRequirementSource
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        expected = (
            ("stencil", self.stencil, "amr_stencil_requirement"),
            ("transfer", self.transfer, "amr_transfer_requirement"),
            ("reflux", self.reflux, "amr_reflux_requirement"),
            ("boundary", self.boundary, "amr_boundary_requirement"),
        )
        for name, source, kind in expected:
            if type(source) is not NestingRequirementSource:
                raise TypeError("DerivedNestingRequirements.%s must be a requirement source" % name)
            if source.provider.kind != kind:
                raise TypeError(
                    "DerivedNestingRequirements.%s requires provider kind %r" % (name, kind)
                )
        dimensions = {len(source.minimum_buffer) for _, source, _ in expected}
        if len(dimensions) != 1:
            raise ValueError("all nesting requirement manifests must have the same dimension")

    @property
    def dimension(self) -> int:
        return len(self.stencil.minimum_buffer)

    @property
    def minimum_buffer(self) -> tuple[int, ...]:
        sources = (self.stencil, self.transfer, self.reflux, self.boundary)
        return tuple(
            max(source.minimum_buffer[axis] for source in sources) for axis in range(self.dimension)
        )

    @property
    def minimum_lookahead(self) -> int:
        return max(
            source.minimum_lookahead
            for source in (self.stencil, self.transfer, self.reflux, self.boundary)
        )

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "stencil": self.stencil.canonical_identity(),
            "transfer": self.transfer.canonical_identity(),
            "reflux": self.reflux.canonical_identity(),
            "boundary": self.boundary.canonical_identity(),
            "derived_minimum_buffer": list(self.minimum_buffer),
            "derived_minimum_lookahead": self.minimum_lookahead,
        }


@dataclass(frozen=True, slots=True)
class LevelTransition:
    """One explicit coarse-to-fine transition; levels are never inferred from a max knob."""

    coarse_level: int
    fine_level: int
    ratio: tuple[int, ...]
    buffer: tuple[int, ...]
    lookahead: int
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        coarse = _positive_int(self.coarse_level, where="coarse_level", minimum=0)
        fine = _positive_int(self.fine_level, where="fine_level", minimum=1)
        if fine != coarse + 1:
            raise ValueError("LevelTransition fine_level must equal coarse_level + 1")
        ratio = _axes(self.ratio, where="LevelTransition.ratio", minimum=2)
        buffer = _axes(self.buffer, where="LevelTransition.buffer", minimum=0)
        if len(ratio) != len(buffer):
            raise ValueError("LevelTransition ratio and buffer dimensions must match")
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(
            self, "lookahead", _positive_int(self.lookahead, where="lookahead", minimum=0)
        )

    @property
    def dimension(self) -> int:
        return len(self.ratio)

    @property
    def anisotropic(self) -> bool:
        return len(set(self.ratio)) > 1

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "coarse_level": self.coarse_level,
            "fine_level": self.fine_level,
            "ratio": list(self.ratio),
            "buffer": list(self.buffer),
            "lookahead": self.lookahead,
        }


@dataclass(frozen=True, slots=True)
class RegridSchedule:
    """A Program-owned regrid cadence bound to committed AcceptedStep time."""

    schedule: Schedule
    due_event: EventHandle
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.schedule) is not Schedule:
            raise TypeError("RegridSchedule.schedule must be an exact Schedule")
        if type(self.schedule.domain) is not AcceptedStep:
            raise ValueError("regrid schedules must use the AcceptedStep domain")
        if self.schedule.off is not None:
            raise ValueError("regrid schedules are event cadences and cannot define an off policy")
        if self.schedule.clock.owner is None:
            raise ValueError("regrid schedule clock must be owner-qualified")
        if type(self.due_event) is not EventHandle:
            raise TypeError("RegridSchedule.due_event must be an exact EventHandle")
        if self.due_event.owner != self.schedule.clock.owner:
            raise ValueError("regrid due event and schedule clock must share one Program owner")
        _freeze_data(self.schedule.to_data(), where="RegridSchedule.schedule")

    @property
    def identity(self) -> Identity:
        return make_identity("amr-regrid-schedule", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "schedule": self.schedule.to_data(),
            "due_event": self.due_event.to_data(),
        }


@dataclass(frozen=True, slots=True)
class FrozenHierarchy:
    """Typed static hierarchy: materialize once and never schedule runtime regrids."""

    __pops_ir_immutable__ = True

    @property
    def identity(self) -> Identity:
        return make_identity("amr-frozen-hierarchy", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "mode": "frozen"}


@dataclass(frozen=True, slots=True)
class HierarchyPlan:
    transitions: tuple[LevelTransition, ...]
    nesting: DerivedNestingRequirements
    clustering: ClusteringPolicy
    patch_generation: PatchGenerationPolicy
    load_balance: LoadBalancePolicy
    regrid: RegridSchedule | FrozenHierarchy
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if any(type(row) is not LevelTransition for row in self.transitions):
            raise TypeError("HierarchyPlan.transitions must contain LevelTransition values")
        for index, row in enumerate(self.transitions):
            if row.coarse_level != index or row.fine_level != index + 1:
                raise ValueError(
                    "HierarchyPlan transitions must form the contiguous chain 0->1->..."
                )
        if type(self.nesting) is not DerivedNestingRequirements:
            raise TypeError("HierarchyPlan.nesting must be DerivedNestingRequirements")
        expected_types = (
            (self.clustering, ClusteringPolicy, "clustering"),
            (self.patch_generation, PatchGenerationPolicy, "patch_generation"),
            (self.load_balance, LoadBalancePolicy, "load_balance"),
        )
        for value, expected, name in expected_types:
            if type(value) is not expected:
                raise TypeError("HierarchyPlan.%s must be %s" % (name, expected.__name__))
        if type(self.regrid) not in (RegridSchedule, FrozenHierarchy):
            raise TypeError("HierarchyPlan.regrid must be RegridSchedule or FrozenHierarchy")
        dimensions = {row.dimension for row in self.transitions}
        dimensions.add(self.nesting.dimension)
        if len(dimensions) != 1:
            raise ValueError("HierarchyPlan transitions and nesting must share one dimension")

    @property
    def dimension(self) -> int:
        return self.nesting.dimension

    @property
    def level_count(self) -> int:
        return len(self.transitions) + 1

    @property
    def identity(self) -> Identity:
        return make_identity("amr-hierarchy-plan", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "transitions": [row.canonical_identity() for row in self.transitions],
            "level_count": self.level_count,
            "nesting": self.nesting.canonical_identity(),
            "clustering": self.clustering.canonical_identity(),
            "patch_generation": self.patch_generation.canonical_identity(),
            "load_balance": self.load_balance.canonical_identity(),
            "regrid": self.regrid.canonical_identity(),
        }

    def inspect(self) -> dict[str, Any]:
        return {
            "report_type": "amr_hierarchy_plan",
            "identity": self.identity.token,
            **self.canonical_identity(),
        }


__all__ = [
    "CanonicalOptions",
    "ClusteringPolicy",
    "DerivedNestingRequirements",
    "FrozenHierarchy",
    "HierarchyPhaseError",
    "HierarchyPlan",
    "LevelTransition",
    "LoadBalancePolicy",
    "NestingRequirementSource",
    "PatchGenerationPolicy",
    "RegridSchedule",
]
