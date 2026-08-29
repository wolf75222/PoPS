"""Deterministic persistent-resource lowering for a temporal Program.

The native Program ABI receives a compact, immutable table.  This module is the
Python authority that builds that table from the authored SSA graph.  It is kept
independent from the native runtime on purpose: discovery, validation, sorting and
slot assignment all happen before C++ source is emitted.

An occurrence is identified by the complete key ``(value_id, occurrence_path,
owner, space, clock, level)``.  The human-readable occurrence path is retained in
the identity; a short path digest is only an acceleration/identity field and is
collision checked.  Generated C++ receives the assigned dense slot, never a node
id or a string lookup.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from pops._program_resource_plan_contract import ProgramResourcePlanCapacityAuthority


_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_RESOURCE_PLAN_SCHEMA = "program-resource-plan:v1"
_LEVEL_UNSPECIFIED = object()

# These are the structured regions owned by a ProgramValue.  Keep this list in
# lock-step with Program serialization/validation, but do not import the authoring
# module here: codegen must remain importable without the native extension.
_STRUCTURED_BLOCK_KEYS = (
    "cond_block",
    "body_block",
    "apply_block",
    "residual_block",
    "true_block",
    "false_block",
)

# Values in these families either bind a resident field or allocate a field-backed
# scratch.  A scheduled value is included even when it is an alias (for example a
# scheduled field solve), because its schedule/cache decision still needs a slot.
_RESOURCE_OPS = frozenset(
    {
        "state",
        "rhs",
        "source",
        "apply",
        "implicit_source",
        "solve_implicit_source",
        "linear_combine",
        "solve_local_linear",
        "solve_local_nonlinear",
        "solve_coupled_implicit",
        "coupled_rate",
        "coupled_rate_out",
        "local_transform",
        "history",
        "store_history",
        "while",
        "range",
        "subcycle",
        "branch",
        "synchronize",
        "solve_fields",
        "solve_fields_from_blocks",
        "cell_compare",
        "where",
        "scalar_field",
        "vector_field",
        "condensed_coeffs",
        "condensed_rhs",
        "condensed_reconstruct",
        "condensed_energy",
    }
)


def _canonical(value: Any) -> Any:
    """Return a strict JSON-ready projection used only for codegen identities."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return _canonical(hook())
    qualified = getattr(value, "qualified_id", None)
    if isinstance(qualified, str) and qualified:
        return qualified
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text_identity(value: Any, default: str) -> str:
    """Project an owner/space/clock/provider to a stable non-empty identity string."""

    if value is None:
        return default
    if isinstance(value, str):
        if value:
            return value
        raise ValueError("persistent resource identities cannot be empty")
    qualified = getattr(value, "qualified_id", None)
    if isinstance(qualified, str) and qualified:
        return qualified
    canonical = getattr(value, "canonical", None)
    if callable(canonical):
        result = canonical()
        if isinstance(result, str) and result:
            return result
        value = result
    if hasattr(value, "to_data"):
        return _canonical_json(value)
    result = str(value)
    if not result:
        raise ValueError("persistent resource identities cannot be empty")
    return result


def _exact_uint(value: Any, *, name: str, maximum: int = _UINT64_MAX, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an exact integer" % name)
    if value < minimum or value > maximum:
        raise ValueError("%s must be in [%d, %d]" % (name, minimum, maximum))
    return value


def _digest_bytes(payload: str) -> bytes:
    return hashlib.sha256(payload.encode("utf-8")).digest()


def occurrence_path_digest(path: str) -> int:
    """Return the compact path identity used by the native key.

    The full path remains in :meth:`ProgramPersistentValueKey.to_data`; callers
    must never use this integer as the sole identity.
    """

    if not isinstance(path, str) or not path:
        raise ValueError("occurrence path must be a non-empty string")
    return int.from_bytes(_digest_bytes(path)[:8], "big", signed=False)


def _occurrence_path_digest(path: str) -> int:
    """Compatibility seam used by collision-focused codegen tests.

    Keep the public helper as the implementation while routing the key through
    this private seam as well.  Tests and downstream extensions can therefore
    force a compact-hash collision without ever treating the digest as the
    authoritative occurrence identity.
    """

    return occurrence_path_digest(path)


def _identity_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProgramPersistentValueKey:
    """Complete static identity of one Program resource occurrence."""

    value_id: int
    occurrence_path: str
    owner: str
    space: str
    clock: str
    level: int | None = None

    def __post_init__(self) -> None:
        _exact_uint(self.value_id, name="ProgramPersistentValueKey.value_id")
        if not isinstance(self.occurrence_path, str) or not self.occurrence_path:
            raise ValueError("ProgramPersistentValueKey.occurrence_path must be non-empty")
        for name in ("owner", "space", "clock"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError("ProgramPersistentValueKey.%s must be non-empty" % name)
        if self.level is not None:
            _exact_uint(self.level, name="ProgramPersistentValueKey.level", maximum=_UINT32_MAX)

    @property
    def occurrence_path_id(self) -> int:
        return _occurrence_path_digest(self.occurrence_path)

    @property
    def canonical_identity(self) -> str:
        return _canonical_json(self.to_data(include_digest=False))

    def to_data(self, *, include_digest: bool = True) -> dict[str, Any]:
        data = {
            "value_id": self.value_id,
            "occurrence_path": self.occurrence_path,
            "owner": self.owner,
            "space": self.space,
            "clock": self.clock,
            "level": self.level,
        }
        if include_digest:
            data["occurrence_path_id"] = self.occurrence_path_id
        return data

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ProgramPersistentValueKey):
            return NotImplemented
        return (
            self.value_id,
            self.occurrence_path,
            self.owner,
            self.space,
            self.clock,
            -1 if self.level is None else self.level,
        ) < (
            other.value_id,
            other.occurrence_path,
            other.owner,
            other.space,
            other.clock,
            -1 if other.level is None else other.level,
        )


@dataclass(frozen=True, slots=True, init=False)
class ProgramResourcePlanEntry:
    """One fully described allocation/resource occurrence."""

    key: ProgramPersistentValueKey
    slot: int = 0
    lifetime: str = "transient"
    centering: str = "cell"
    ghosts: int = 0
    communication: str = "none"
    transfer_provider: str = "none"
    restart_provider: str = "none"
    components: int = 1
    # There is no truthful default for a resource footprint.  Lowering from a
    # Program may derive this only from explicit shape/item-size metadata;
    # callers constructing a sealed row must provide an exact positive value.
    bytes: int | None = None
    maximum_bytes: int | None = None
    off_policy: str = "none"
    component_names: tuple[str, ...] = ()
    communicates: bool = False
    restart_required: bool = False
    # Optional exact shape evidence.  Runtime geometry is not guessed here: an
    # empty shape means that the author supplied an exact byte count without a
    # compile-time extent; ``cells``/``itemsize`` are retained when the byte
    # count was derived from those exact metadata fields.
    shape: tuple[int, ...] = ()
    cells: int | None = None
    itemsize: int | None = None
    # A runtime-sized row is an ABI declaration, not a guessed allocation.  In
    # particular, ``bytes``, ``maximum_bytes``, ``cells`` and ``itemsize`` must
    # all remain absent until the host has observed the exact runtime layout.
    runtime_sized: bool = False
    resource_type: str = "exact"

    def __init__(
        self,
        key: ProgramPersistentValueKey,
        slot: int = 0,
        lifetime: str = "transient",
        centering: str = "cell",
        ghosts: int = 0,
        communication: str = "none",
        transfer_provider: str = "none",
        restart_provider: str = "none",
        components: int = 1,
        bytes: int | None = None,
        maximum_bytes: int | None = None,
        off_policy: str = "none",
        component_names: tuple[str, ...] = (),
        communicates: bool = False,
        restart_required: bool = False,
        shape: tuple[int, ...] = (),
        cells: int | None = None,
        itemsize: int | None = None,
        *,
        runtime_sized: bool = False,
        resource_type: str | None = None,
    ) -> None:
        for name, value in (
            ("key", key),
            ("slot", slot),
            ("lifetime", lifetime),
            ("centering", centering),
            ("ghosts", ghosts),
            ("communication", communication),
            ("transfer_provider", transfer_provider),
            ("restart_provider", restart_provider),
            ("components", components),
            ("bytes", bytes),
            ("maximum_bytes", maximum_bytes),
            ("off_policy", off_policy),
            ("component_names", component_names),
            ("communicates", communicates),
            ("restart_required", restart_required),
            ("shape", shape),
            ("cells", cells),
            ("itemsize", itemsize),
            ("runtime_sized", runtime_sized),
        ):
            object.__setattr__(self, name, value)
        if type(runtime_sized) is not bool:
            raise TypeError("ProgramResourcePlanEntry.runtime_sized must be an exact bool")
        supplied_type = resource_type
        if supplied_type is None:
            supplied_type = "runtime_sized" if runtime_sized else "exact"
        if supplied_type not in {"exact", "runtime_sized"}:
            raise ValueError("ProgramResourcePlanEntry.resource_type is unsupported")
        if runtime_sized != (supplied_type == "runtime_sized"):
            raise ValueError("runtime_sized and resource_type disagree")
        object.__setattr__(self, "resource_type", supplied_type)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.key) is not ProgramPersistentValueKey:
            raise TypeError("ProgramResourcePlanEntry.key must be an exact ProgramPersistentValueKey")
        _exact_uint(self.slot, name="ProgramResourcePlanEntry.slot", maximum=_UINT32_MAX)
        if self.lifetime not in {"transient", "persistent", "persistent_schedule"}:
            raise ValueError("unsupported persistent resource lifetime %r" % self.lifetime)
        for name in ("centering", "communication", "transfer_provider", "restart_provider", "off_policy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError("ProgramResourcePlanEntry.%s must be non-empty" % name)
        _exact_uint(self.ghosts, name="ProgramResourcePlanEntry.ghosts", maximum=_UINT32_MAX)
        _exact_uint(self.components, name="ProgramResourcePlanEntry.components", maximum=_UINT32_MAX, minimum=1)
        if self.runtime_sized:
            if self.resource_type != "runtime_sized":
                raise ValueError("runtime-sized resource has a non-symbolic type")
            if any(item is not None for item in (self.bytes, self.maximum_bytes, self.cells, self.itemsize)):
                raise ValueError(
                    "runtime-sized resource cannot claim exact cells/itemsize/bytes/maximum_bytes"
                )
            object.__setattr__(self, "maximum_bytes", None)
        else:
            if self.resource_type != "exact":
                raise ValueError("exact resource has a non-exact type")
            if self.bytes is None:
                raise ValueError("ProgramResourcePlanEntry.bytes is unknown")
            _exact_uint(self.bytes, name="ProgramResourcePlanEntry.bytes", minimum=1)
            maximum = self.bytes if self.maximum_bytes is None else self.maximum_bytes
            _exact_uint(maximum, name="ProgramResourcePlanEntry.maximum_bytes", minimum=1)
            if maximum < self.bytes:
                raise ValueError("ProgramResourcePlanEntry.maximum_bytes is below bytes")
            object.__setattr__(self, "maximum_bytes", maximum)
        if type(self.communicates) is not bool or type(self.restart_required) is not bool:
            raise TypeError("ProgramResourcePlanEntry communication flags must be exact bools")
        names = tuple(self.component_names)
        if any(type(name) is not str or not name for name in names):
            raise TypeError("ProgramResourcePlanEntry.component_names must contain non-empty strings")
        if names and len(names) != self.components:
            raise ValueError("ProgramResourcePlanEntry.component_names disagrees with components")
        object.__setattr__(self, "component_names", names)
        shape = tuple(self.shape)
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in shape
        ):
            raise TypeError("ProgramResourcePlanEntry.shape must contain positive exact integers")
        object.__setattr__(self, "shape", shape)
        if self.cells is not None:
            _exact_uint(self.cells, name="ProgramResourcePlanEntry.cells", minimum=1)
        if self.itemsize is not None:
            _exact_uint(self.itemsize, name="ProgramResourcePlanEntry.itemsize", minimum=1)

    def to_data(self, *, include_slot: bool = True) -> dict[str, Any]:
        data = {
            "key": self.key.to_data(),
            "lifetime": self.lifetime,
            "centering": self.centering,
            "ghosts": self.ghosts,
            "communication": self.communication,
            "transfer_provider": self.transfer_provider,
            "restart_provider": self.restart_provider,
            "components": self.components,
            "component_names": list(self.component_names),
            "bytes": self.bytes,
            "maximum_bytes": self.maximum_bytes,
            "off_policy": self.off_policy,
            "communicates": self.communicates,
            "restart_required": self.restart_required,
            "shape": list(self.shape),
            "cells": self.cells,
            "itemsize": self.itemsize,
            "runtime_sized": self.runtime_sized,
            "resource_type": self.resource_type,
        }
        if include_slot:
            data["slot"] = self.slot
        return data

    @property
    def bytes_exact(self) -> int | None:
        """The authenticated allocation size, or ``None`` for a runtime declaration."""

        return self.bytes

    @property
    def memory_ceiling(self) -> int | None:
        """The per-entry ceiling, or ``None`` until runtime materialization."""

        return self.maximum_bytes


@dataclass(frozen=True, slots=True)
class ProgramResourcePlanAbiRow:
    """Lossless compiler/ABI projection of one sealed resource entry.

    ``ProgramResourcePlanRecord`` is a native POD and therefore cannot carry a
    Python object.  This row deliberately retains every source field before a
    target-specific emitter serializes it.  Named attributes and
    :meth:`to_data` are the sole authoritative contract.
    """

    schema: str
    plan_digest: str
    slot: int
    key: ProgramPersistentValueKey
    lifetime: str
    centering: str
    off_policy: str
    communication: str
    transfer_provider: str
    restart_provider: str
    components: int
    component_names: tuple[str, ...]
    ghosts: int
    bytes: int | None
    maximum_bytes: int | None
    shape: tuple[int, ...]
    cells: int | None
    itemsize: int | None
    communicates: bool
    restart_required: bool
    identity: str
    runtime_sized: bool
    resource_type: str

    def __post_init__(self) -> None:
        if self.schema != _RESOURCE_PLAN_SCHEMA:
            raise ValueError("unsupported Program resource ABI schema %r" % self.schema)
        if not isinstance(self.plan_digest, str) or not self.plan_digest:
            raise ValueError("Program resource ABI row requires a plan digest")
        _exact_uint(self.slot, name="ProgramResourcePlanAbiRow.slot", maximum=_UINT32_MAX)
        if type(self.key) is not ProgramPersistentValueKey:
            raise TypeError("ProgramResourcePlanAbiRow.key must be an exact key")
        for name in (
            "lifetime",
            "centering",
            "off_policy",
            "communication",
            "transfer_provider",
            "restart_provider",
            "identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError("ProgramResourcePlanAbiRow.%s must be non-empty" % name)
        names = tuple(self.component_names)
        if any(type(name) is not str or not name for name in names):
            raise TypeError("ProgramResourcePlanAbiRow.component_names must contain strings")
        if names and len(names) != self.components:
            raise ValueError("ProgramResourcePlanAbiRow.component_names disagrees with components")
        object.__setattr__(self, "component_names", names)
        shape = tuple(self.shape)
        object.__setattr__(self, "shape", shape)
        _exact_uint(self.components, name="ProgramResourcePlanAbiRow.components", maximum=_UINT32_MAX, minimum=1)
        _exact_uint(self.ghosts, name="ProgramResourcePlanAbiRow.ghosts", maximum=_UINT32_MAX)
        if type(self.runtime_sized) is not bool:
            raise TypeError("ProgramResourcePlanAbiRow.runtime_sized must be an exact bool")
        if self.runtime_sized:
            if self.resource_type != "runtime_sized":
                raise ValueError("runtime-sized ABI row has a non-symbolic type")
            if any(item is not None for item in (self.bytes, self.maximum_bytes, self.cells, self.itemsize)):
                raise ValueError(
                    "runtime-sized ABI row cannot claim exact cells/itemsize/bytes/maximum_bytes"
                )
        else:
            if self.resource_type != "exact":
                raise ValueError("exact ABI row has a non-exact type")
            if self.bytes is None:
                raise ValueError("ProgramResourcePlanAbiRow.bytes is unknown")
            _exact_uint(self.bytes, name="ProgramResourcePlanAbiRow.bytes", minimum=1)
            if self.maximum_bytes is None:
                raise ValueError("ProgramResourcePlanAbiRow.maximum_bytes is unknown")
            _exact_uint(self.maximum_bytes, name="ProgramResourcePlanAbiRow.maximum_bytes", minimum=1)
            if self.maximum_bytes < self.bytes:
                raise ValueError("ProgramResourcePlanAbiRow.maximum_bytes is below bytes")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in self.shape
        ):
            raise TypeError("ProgramResourcePlanAbiRow.shape must contain positive exact integers")
        if self.cells is not None:
            _exact_uint(self.cells, name="ProgramResourcePlanAbiRow.cells", minimum=1)
        if self.itemsize is not None:
            _exact_uint(self.itemsize, name="ProgramResourcePlanAbiRow.itemsize", minimum=1)
        if type(self.communicates) is not bool or type(self.restart_required) is not bool:
            raise TypeError("ProgramResourcePlanAbiRow flags must be exact bools")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "slot": self.slot,
            "key": self.key.to_data(),
            "lifetime": self.lifetime,
            "centering": self.centering,
            "off_policy": self.off_policy,
            "communication": self.communication,
            "transfer_provider": self.transfer_provider,
            "restart_provider": self.restart_provider,
            "components": self.components,
            "component_names": list(self.component_names),
            "ghosts": self.ghosts,
            "bytes": self.bytes,
            "maximum_bytes": self.maximum_bytes,
            "shape": list(self.shape),
            "cells": self.cells,
            "itemsize": self.itemsize,
            "communicates": self.communicates,
            "restart_required": self.restart_required,
            "identity": self.identity,
            "runtime_sized": self.runtime_sized,
            "resource_type": self.resource_type,
        }

def _entry_from_data(data: Mapping[str, Any]) -> ProgramResourcePlanEntry:
    """Decode one manifest row while authenticating the compact path digest."""

    if not isinstance(data, Mapping):
        raise TypeError("Program resource plan entries must be mappings or exact entries")
    required_fields = {
        "key",
        "slot",
        "lifetime",
        "centering",
        "ghosts",
        "communication",
        "transfer_provider",
        "restart_provider",
        "components",
        "bytes",
        "maximum_bytes",
        "off_policy",
        "component_names",
        "communicates",
        "restart_required",
        "shape",
        "cells",
        "itemsize",
        "runtime_sized",
        "resource_type",
    }
    if set(data) != required_fields:
        missing = sorted(required_fields - set(data))
        extra = sorted(set(data) - required_fields)
        raise ValueError(
            "Program resource plan entry schema mismatch: missing=%r extra=%r"
            % (missing, extra)
        )
    raw_key = data.get("key")
    if isinstance(raw_key, ProgramPersistentValueKey):
        key = raw_key
    elif isinstance(raw_key, Mapping):
        required = {
            "value_id",
            "occurrence_path",
            "occurrence_path_id",
            "owner",
            "space",
            "clock",
            "level",
        }
        if set(raw_key) != required:
            raise ValueError("Program resource plan key schema is not exact")
        key = ProgramPersistentValueKey(
            value_id=raw_key["value_id"],
            occurrence_path=raw_key["occurrence_path"],
            owner=raw_key["owner"],
            space=raw_key["space"],
            clock=raw_key["clock"],
            level=raw_key["level"],
        )
        compact = raw_key["occurrence_path_id"]
        if compact != key.occurrence_path_id:
            raise ValueError("Program resource plan occurrence path digest is unauthenticated")
    else:
        raise TypeError("Program resource plan entry has no complete key")
    fields = dict(data)
    fields.pop("key", None)
    return ProgramResourcePlanEntry(key=key, **fields)


class ProgramResourcePlan(ProgramResourcePlanCapacityAuthority):
    """Sorted, collision-checked and dense-slotted resource plan."""

    schema_version = 1
    schema = _RESOURCE_PLAN_SCHEMA

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ProgramResourcePlan is immutable after lowering")
        object.__setattr__(self, name, value)

    def __init__(self, entries: Iterable[ProgramResourcePlanEntry], maximum_bytes: Any = None,
                 *, digest: str | None = None,
                 occurrence_values: Mapping[
                     int,
                     ProgramPersistentValueKey | Iterable[ProgramPersistentValueKey],
                 ] | None = None) -> None:
        raw_rows = tuple(entries)
        rows = tuple(
            _entry_from_data(row) if isinstance(row, Mapping) else row
            for row in raw_rows
        )
        if any(type(row) is not ProgramResourcePlanEntry for row in rows):
            raise TypeError("ProgramResourcePlan entries must contain exact ProgramResourcePlanEntry values")
        if len(rows) > _UINT32_MAX:
            raise OverflowError("Program resource plan has more than UINT32_MAX entries")
        # Discovery order is retained as a tie-breaker even though complete keys
        # should already be unique.  This makes sorting deterministic for extension
        # records that compare equal before duplicate rejection.
        indexed = list(enumerate(rows))
        indexed.sort(key=lambda item: (item[1].key, item[0]))
        ordered = [row for _index, row in indexed]
        seen_keys: dict[ProgramPersistentValueKey, int] = {}
        path_digests: dict[int, str] = {}
        identity_digests: dict[str, str] = {}
        assigned: list[ProgramResourcePlanEntry] = []
        total = 0
        for index, row in enumerate(ordered):
            key = row.key
            if key in seen_keys:
                raise ValueError(
                    "duplicate Program resource occurrence key %s (entries %d and %d)"
                    % (key.canonical_identity, seen_keys[key], index)
                )
            seen_keys[key] = index
            path_id = key.occurrence_path_id
            prior_path = path_digests.get(path_id)
            if prior_path is not None and prior_path != key.occurrence_path:
                raise ValueError(
                    "Program resource occurrence-path digest collision between %r and %r"
                    % (prior_path, key.occurrence_path)
                )
            path_digests[path_id] = key.occurrence_path
            identity = _identity_digest(key.to_data(include_digest=False))
            prior_identity = identity_digests.get(identity)
            if prior_identity is not None and prior_identity != key.canonical_identity:
                raise ValueError("Program resource identity digest collision")
            identity_digests[identity] = key.canonical_identity
            if row.maximum_bytes is not None:
                if row.maximum_bytes > _UINT64_MAX - total:
                    raise OverflowError("Program resource plan byte bound overflows uint64")
                total += row.maximum_bytes
            assigned.append(
                ProgramResourcePlanEntry(
                    key=key,
                    slot=index,
                    lifetime=row.lifetime,
                    centering=row.centering,
                    ghosts=row.ghosts,
                    communication=row.communication,
                    transfer_provider=row.transfer_provider,
                    restart_provider=row.restart_provider,
                    components=row.components,
                    bytes=row.bytes,
                    maximum_bytes=row.maximum_bytes,
                    off_policy=row.off_policy,
                    component_names=row.component_names,
                    communicates=row.communicates,
                    restart_required=row.restart_required,
                    shape=row.shape,
                    cells=row.cells,
                    itemsize=row.itemsize,
                    runtime_sized=row.runtime_sized,
                    resource_type=row.resource_type,
                )
            )
        symbolic = any(row.runtime_sized for row in assigned)
        if symbolic:
            # A symbolic plan is an install-time declaration only.  It is
            # deliberately unbounded here: the host must observe every exact
            # prototype/subslot and construct a new, fully materialized plan
            # before publication.
            if maximum_bytes is not None:
                raise ValueError(
                    "runtime-sized Program resource plan cannot claim a maximum byte bound"
                )
            bound = None
        elif maximum_bytes is None:
            bound = total
        else:
            if isinstance(maximum_bytes, str) or maximum_bytes is ...:
                raise ValueError("Program resource plan memory bound is unknown")
            bound = _exact_uint(maximum_bytes, name="ProgramResourcePlan.maximum_bytes", minimum=0)
            if total > bound:
                raise ValueError("Program resource plan exceeds its exact memory bound")
        if bound is not None and bound > _UINT64_MAX:
            raise OverflowError("Program resource plan memory bound overflows uint64")
        payload = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "maximum_bytes": bound,
            "entries": [row.to_data(include_slot=True) for row in assigned],
        }
        computed_digest = _identity_digest(payload)
        if digest is not None and digest != computed_digest:
            raise ValueError("Program resource plan digest does not authenticate its payload")
        self.entries = tuple(assigned)
        self.maximum_bytes = bound
        self.digest = computed_digest
        self.identity = computed_digest
        self._by_key = {row.key: row.slot for row in self.entries}
        self._by_value_id: dict[int, int] = {}
        for row in self.entries:
            prior = self._by_value_id.get(row.key.value_id)
            if prior is not None and prior != row.slot:
                # A value id may occur in different regions.  The caller must use
                # the full key for such a graph; generated ProgramValue ids are
                # globally unique, but this keeps forged plans explicit.
                self._by_value_id[row.key.value_id] = -1
            else:
                self._by_value_id[row.key.value_id] = row.slot
        # A forged/extension graph can legitimately reuse a value id in two
        # static regions because the complete key also carries its occurrence
        # path and owner.  Lowering normally sees globally unique ProgramValue
        # ids, but retaining object-to-key bindings keeps slot resolution exact
        # for such graphs without introducing a generated lookup.
        object_slots: dict[int, int | tuple[int, ...]] = {}
        if occurrence_values is not None:
            if not isinstance(occurrence_values, Mapping):
                raise TypeError("ProgramResourcePlan occurrence_values must be a mapping")
            for object_id, key in occurrence_values.items():
                if isinstance(object_id, bool) or not isinstance(object_id, int):
                    raise TypeError("ProgramResourcePlan occurrence object identities must be integers")
                if type(key) is ProgramPersistentValueKey:
                    keys = (key,)
                elif isinstance(key, (tuple, list)):
                    keys = tuple(key)
                    if not keys or any(type(item) is not ProgramPersistentValueKey for item in keys):
                        raise TypeError(
                            "ProgramResourcePlan occurrence bindings require exact keys"
                        )
                else:
                    raise TypeError("ProgramResourcePlan occurrence bindings require exact keys")
                slots = []
                for item in keys:
                    slot = self._by_key.get(item)
                    if slot is None:
                        raise ValueError(
                            "ProgramResourcePlan occurrence binding is absent from the plan"
                        )
                    slots.append(slot)
                object_slots[object_id] = slots[0] if len(slots) == 1 else tuple(slots)
        self._by_object_id = object_slots
        self._sealed = True

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> ProgramResourcePlan:
        """Rehydrate and authenticate a serialized plan before code generation."""

        if not isinstance(data, Mapping):
            raise TypeError("Program resource plan data must be a mapping")
        required = {"schema", "schema_version", "digest", "maximum_bytes", "entries"}
        if set(data) != required:
            raise ValueError("Program resource plan schema fields are not exact")
        schema = data["schema"]
        if schema != cls.schema:
            raise ValueError("unsupported Program resource plan schema %r" % schema)
        if data["schema_version"] != cls.schema_version:
            raise ValueError(
                "unsupported Program resource plan schema version %r"
                % data["schema_version"]
            )
        digest = data["digest"]
        if type(digest) is not str or len(digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in digest
        ):
            raise ValueError("Program resource plan digest must be exact lowercase SHA-256")
        rows = data["entries"]
        if not isinstance(rows, (list, tuple)):
            raise TypeError("Program resource plan data requires an entries sequence")
        return cls(
            rows,
            data["maximum_bytes"],
            digest=digest,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ProgramResourcePlanEntry]:
        return iter(self.entries)

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(row.slot for row in self.entries)

    def slot_for_key(self, key: ProgramPersistentValueKey) -> int:
        if type(key) is not ProgramPersistentValueKey:
            raise TypeError("ProgramResourcePlan.slot_for_key requires an exact key")
        try:
            slot = self._by_key[key]
        except KeyError:
            raise KeyError("Program resource key is absent from the sealed plan") from None
        return slot

    def slot_for_value(
        self,
        value: Any,
        *,
        occurrence_path: str | None = None,
        level: int | None | object = _LEVEL_UNSPECIFIED,
    ) -> int:
        """Return a compile-time slot without introducing a runtime lookup."""

        if isinstance(value, ProgramPersistentValueKey):
            return self.slot_for_key(value)
        object_slot = self._by_object_id.get(id(value))
        if object_slot is not None:
            slots = (object_slot,) if isinstance(object_slot, int) else object_slot
            if occurrence_path is not None and any(
                self.entries[slot].key.occurrence_path != occurrence_path for slot in slots
            ):
                raise KeyError("Program resource occurrence path does not match the sealed plan")
            if level is _LEVEL_UNSPECIFIED:
                attrs = getattr(value, "attrs", {})
                candidate = attrs.get("level") if isinstance(attrs, Mapping) else None
                if candidate is None and isinstance(attrs, Mapping):
                    schedule = attrs.get("schedule")
                    if schedule is not None:
                        lowered = schedule.native_schedule_ir(
                            where="resource occurrence level resolution"
                        )
                        candidate = getattr(lowered.domain, "level", None)
                if candidate is None and len(slots) != 1:
                    raise KeyError(
                        "Program resource occurrence requires its resolved level"
                    )
                level = candidate if candidate is not None else self.entries[slots[0]].key.level
            matches = [slot for slot in slots if self.entries[slot].key.level == level]
            if len(matches) != 1:
                raise KeyError("Program resource occurrence level is absent from the sealed plan")
            return matches[0]
        value_id = _exact_uint(getattr(value, "id", value), name="Program resource value id")
        slot = self._by_value_id.get(value_id)
        if slot is None or slot < 0:
            if occurrence_path is None:
                raise KeyError("Program resource value id %d needs its full occurrence path" % value_id)
            matches = [
                row.slot for row in self.entries
                if row.key.value_id == value_id and row.key.occurrence_path == occurrence_path
                and (level is _LEVEL_UNSPECIFIED or row.key.level == level)
            ]
            if len(matches) != 1:
                raise KeyError("Program resource occurrence is absent from the sealed plan")
            return matches[0]
        if occurrence_path is not None and self.entries[slot].key.occurrence_path != occurrence_path:
            raise KeyError("Program resource occurrence path does not match the sealed plan")
        if level is not _LEVEL_UNSPECIFIED and self.entries[slot].key.level != level:
            raise KeyError("Program resource occurrence level does not match the sealed plan")
        return slot

    def row_for_value(
        self,
        value: Any,
        *,
        occurrence_path: str | None = None,
        level: int | None | object = _LEVEL_UNSPECIFIED,
    ) -> ProgramResourcePlanEntry:
        return self.entries[
            self.slot_for_value(value, occurrence_path=occurrence_path, level=level)
        ]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "digest": self.digest,
            "maximum_bytes": self.maximum_bytes,
            "entries": [row.to_data() for row in self.entries],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_data(), sort_keys=True, separators=(",", ":") if indent is None else None,
                          indent=indent, allow_nan=False)

    def abi_rows(self) -> tuple[ProgramResourcePlanAbiRow, ...]:
        """Return lossless, versioned rows for native candidate emission.

        The named row carries the dense slot, complete key/path, typed policies,
        exact bytes and shape evidence, plus the plan digest/schema.
        """

        rows = []
        for entry in self.entries:
            identity = "program-resource:v1:%s:%s:components=%d:bytes=%s:maximum_bytes=%s" % (
                self.digest,
                entry.key.canonical_identity,
                entry.components,
                "unknown" if entry.bytes is None else entry.bytes,
                "unknown" if entry.maximum_bytes is None else entry.maximum_bytes,
            )
            rows.append(
                ProgramResourcePlanAbiRow(
                    schema=self.schema,
                    plan_digest=self.digest,
                    slot=entry.slot,
                    key=entry.key,
                    lifetime=entry.lifetime,
                    centering=entry.centering,
                    off_policy=entry.off_policy,
                    communication=entry.communication,
                    transfer_provider=entry.transfer_provider,
                    restart_provider=entry.restart_provider,
                    components=entry.components,
                    component_names=entry.component_names,
                    ghosts=entry.ghosts,
                    bytes=entry.bytes,
                    maximum_bytes=entry.maximum_bytes,
                    shape=entry.shape,
                    cells=entry.cells,
                    itemsize=entry.itemsize,
                    communicates=entry.communicates,
                    restart_required=entry.restart_required,
                    identity=identity,
                    runtime_sized=entry.runtime_sized,
                    resource_type=entry.resource_type,
                )
            )
        return tuple(rows)

    def abi_data(self) -> dict[str, Any]:
        """Return the complete authenticated payload carried by candidate tables."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "digest": self.digest,
            "maximum_bytes": self.maximum_bytes,
            "entries": [row.to_data() for row in self.abi_rows()],
        }


def _program_values(program: Any) -> tuple[Any, ...]:
    values = getattr(program, "_values", None)
    if values is None:
        values = getattr(program, "values", None)
        values = values() if callable(values) else values
    if values is None:
        raise TypeError("program resource lowering requires Program values")
    return tuple(values)


def iter_program_occurrences(program: Any) -> Iterator[tuple[Any, str]]:
    """Yield static ProgramValue occurrences in canonical pre-order."""

    seen_objects: set[int] = set()

    def walk(values: Iterable[Any], prefix: str) -> Iterator[tuple[Any, str]]:
        for index, value in enumerate(values):
            if not hasattr(value, "id"):
                raise TypeError("Program resource occurrence is not a value record")
            path = "%s/%d" % (prefix, index)
            marker = id(value)
            if marker in seen_objects:
                raise ValueError("Program value occurs more than once at %s" % path)
            seen_objects.add(marker)
            yield value, path
            attrs = getattr(value, "attrs", {})
            for key in _STRUCTURED_BLOCK_KEYS:
                nested = attrs.get(key) if isinstance(attrs, Mapping) else None
                if isinstance(nested, (list, tuple)):
                    yield from walk(nested, "%s/%s" % (path, key))

    yield from walk(_program_values(program), "root")


def _embedded_program_values(value: Any, *, seen: set[int] | None = None) -> Iterator[Any]:
    """Yield ProgramValue-like references embedded in attrs/control metadata.

    Schedule metadata is a typed object graph rather than a mapping: a ``Schedule`` owns a
    ``When`` trigger, whose ``condition`` is the authored Bool ProgramValue.  Walk those exact
    trigger/predicate edges as well as the historical mapping/terms containers so a scheduled
    node's predicate is retained by the same reachability closure that drives body emission.
    """

    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    if hasattr(value, "id") and hasattr(value, "op") and hasattr(value, "inputs"):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _embedded_program_values(item, seen=seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _embedded_program_values(item, seen=seen)
        return
    terms = getattr(value, "terms", None)
    if isinstance(terms, (list, tuple)):
        for item in terms:
            yield from _embedded_program_values(item, seen=seen)
    # Schedule/IR descriptors are immutable dataclass-like objects, not mappings.  These are the
    # only metadata edges that can carry an executable Program Bool condition; keep the list exact
    # so owner/field descriptors are not recursively treated as arbitrary object graphs.
    for attribute in ("trigger", "condition", "predicate", "due"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            yield from _embedded_program_values(candidate, seen=seen)


def _reachable_program_occurrences(program: Any) -> tuple[tuple[Any, str], ...]:
    """Return only resource occurrences reachable from executable Program roots.

    A matrix-free operator declaration is inert until a solve consumes it.  The
    old resource walk treated every top-level declaration as live, causing an
    unconsumed ``coupled_interface_jacobian`` apply block to acquire a slot and
    an install-time owner.  Real ``Program`` instances expose commit/control
    metadata, so use an explicit root set and reverse dataflow closure there;
    tiny external test doubles without that metadata retain the historical
    all-occurrences behavior.
    """

    occurrences = tuple(iter_program_occurrences(program))
    if not any(hasattr(program, name) for name in ("_commits", "_post_sync_commits", "_dt_bound")):
        return occurrences
    by_object = {id(value): value for value, _path in occurrences}
    roots: list[Any] = []

    def add_refs(candidate: Any) -> None:
        for value in _embedded_program_values(candidate):
            if id(value) in by_object:
                roots.append(value)

    for name in ("_commits", "_post_sync_commits", "_outputs", "outputs", "_dt_bound",
                 "_acceptance_guards"):
        if hasattr(program, name):
            add_refs(getattr(program, name))

    # Solves and executable control regions are roots even when their return
    # value is consumed through an indirect buffer/side-effect path.  A bare
    # matrix-free operator is deliberately absent: only a solve/control node
    # that references it can make its apply block reachable.
    control_ops = {
        "while", "branch", "range", "subcycle", "post_synchronization",
    }
    side_effect_ops = {
        "project", "fill_boundary", "store_history", "record_scalar",
        "record_balance_term", "record_balance", "synchronize",
    }
    for value, _path in occurrences:
        op = getattr(value, "op", None)
        if (isinstance(op, str) and op.startswith("solve")) \
                or op in control_ops or op in side_effect_ops:
            roots.append(value)
        attrs = getattr(value, "attrs", {})
        if isinstance(attrs, Mapping) and attrs.get("output") is True:
            roots.append(value)

    reachable: set[int] = set()
    stack = list(roots)
    while stack:
        value = stack.pop()
        marker = id(value)
        if marker in reachable or marker not in by_object:
            continue
        reachable.add(marker)
        for reference in getattr(value, "inputs", ()):
            if id(reference) in by_object:
                stack.append(reference)
        attrs = getattr(value, "attrs", {})
        for reference in _embedded_program_values(attrs):
            if id(reference) in by_object:
                stack.append(reference)
    return tuple((value, path) for value, path in occurrences if id(value) in reachable)


def _resource_descriptor(value: Any) -> dict[str, Any]:
    attrs = getattr(value, "attrs", {})
    if not isinstance(attrs, Mapping):
        attrs = {}
    descriptor = None
    for name in ("resource_plan", "persistent_resource", "resource"):
        candidate = attrs.get(name)
        if candidate is None:
            candidate = getattr(value, name, None)
        if candidate is not None:
            descriptor = candidate
            break
    if descriptor is None:
        return dict(attrs)
    if not isinstance(descriptor, Mapping):
        hook = getattr(descriptor, "to_data", None)
        descriptor = hook() if callable(hook) else descriptor
    if not isinstance(descriptor, Mapping):
        raise TypeError("resource metadata on node %r must be a mapping" % getattr(value, "name", "<?>"))
    merged = dict(attrs)
    merged.update(descriptor)
    return merged


def _metadata(mapping: Mapping[str, Any], names: tuple[str, ...], *, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _level_value(
    mapping: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    level_index: int,
    levels: tuple[int | None, ...],
    default: Any = _LEVEL_UNSPECIFIED,
) -> Any:
    """Select one exact per-level metadata value without inventing geometry.

    A producer may provide a scalar, a sequence aligned with ``levels``, or an explicit mapping
    keyed by the resolved level.  Ambiguous/missing sequences are rejected before an artifact is
    emitted; in particular this helper never turns an absent cell count or item size into a dummy
    ``1``/``8`` cache footprint.
    """

    raw = _metadata(mapping, names, default=default)
    if raw is default:
        return default
    if isinstance(raw, Mapping):
        level = levels[level_index]
        keys = (level, str(level)) if level is not None else (None, "none")
        for key in keys:
            if key in raw:
                return raw[key]
        if "default" in raw:
            return raw["default"]
        raise ValueError(
            "resource metadata %s has no value for resolved level %r"
            % ("/".join(names), level)
        )
    if isinstance(raw, (list, tuple)):
        if len(raw) == len(levels):
            return raw[level_index]
        level = levels[level_index]
        if level is not None and level >= 0 and level < len(raw):
            return raw[level]
        raise ValueError(
            "resource metadata %s is not aligned with its resolved levels"
            % "/".join(names)
        )
    return raw


def _level_attr(
    value: Any,
    names: tuple[str, ...],
    *,
    level_index: int,
    levels: tuple[int | None, ...],
) -> Any:
    """Read optional geometry evidence from a space object using the same exact selector."""

    if isinstance(value, Mapping):
        data = {name: value[name] for name in names if name in value}
    else:
        data = {
            name: getattr(value, name)
            for name in names
            if hasattr(value, name)
        }
    return _level_value(
        data,
        names,
        level_index=level_index,
        levels=levels,
        default=_LEVEL_UNSPECIFIED,
    )


def _resolved_levels(value: Any, metadata: Mapping[str, Any], *, target: str | None) -> tuple[int | None, ...]:
    """Resolve the static level expansion for one Program occurrence.

    ``resolved_levels``/``amr_levels`` are authoritative when present.  A scalar ``level`` or an
    AMR-level schedule describes one row.  The schedule level must be covered by an explicit
    expansion, otherwise the complete key would not identify the emitted resource.
    """

    del target  # The target is retained in the call contract for future topology-specific checks.
    schedule = metadata.get("schedule")
    scheduled_level = None
    if schedule is not None:
        lowered = schedule.native_schedule_ir(where="resource schedule")
        scheduled_level = getattr(lowered.domain, "level", None)
    raw = _metadata(
        metadata,
        ("resolved_levels", "amr_levels", "levels"),
        default=_LEVEL_UNSPECIFIED,
    )
    if raw is _LEVEL_UNSPECIFIED:
        raw = _metadata(metadata, ("level", "amr_level"), default=_LEVEL_UNSPECIFIED)
    if raw is _LEVEL_UNSPECIFIED:
        levels = (scheduled_level,)
    elif raw is None:
        levels = (None,)
    elif isinstance(raw, Mapping):
        nested = _metadata(raw, ("resolved_levels", "amr_levels", "levels"), default=_LEVEL_UNSPECIFIED)
        if nested is _LEVEL_UNSPECIFIED:
            raise TypeError("resource level metadata mapping must declare resolved levels")
        raw = nested
        if raw is None:
            levels = (None,)
        elif isinstance(raw, int) and not isinstance(raw, bool):
            levels = (raw,)
        else:
            levels = tuple(raw)
    elif isinstance(raw, int) and not isinstance(raw, bool):
        levels = (raw,)
    elif isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("resource resolved levels cannot be empty")
        levels = tuple(raw)
    else:
        raise TypeError("resource resolved levels must be an integer or a finite sequence")
    normalized: list[int | None] = []
    for level in levels:
        if level is None:
            normalized.append(None)
        else:
            normalized.append(
                _exact_uint(level, name="resource AMR level", maximum=_UINT32_MAX)
            )
    result = tuple(normalized)
    if len(result) != len(set(result)):
        raise ValueError("resource resolved levels contain duplicates")
    if scheduled_level is not None and scheduled_level not in result:
        raise ValueError(
            "resource resolved levels %r do not cover schedule level %r"
            % (result, scheduled_level)
        )
    return result


def _component_info(value: Any, metadata: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
    raw = _metadata(metadata, ("components", "component_count", "ncomp"))
    names: tuple[str, ...] = ()
    if isinstance(raw, (list, tuple)):
        names = tuple(raw)
        count = len(names)
    elif raw is not None:
        count = _exact_uint(raw, name="resource components", maximum=_UINT32_MAX, minimum=1)
    else:
        space = getattr(value, "space", None)
        components = getattr(space, "components", None)
        if components:
            names = tuple(components)
            count = len(names)
        else:
            context = getattr(value, "field_context", None)
            context_outputs = getattr(context, "outputs", None)
            if context_outputs:
                names = tuple(context_outputs)
                count = len(names)
            else:
                count = 0
                for candidate in getattr(value, "inputs", ()):
                    candidate_attrs = getattr(candidate, "attrs", {})
                    candidate_count = (
                        candidate_attrs.get("ncomp")
                        if isinstance(candidate_attrs, Mapping)
                        else None
                    )
                    if candidate_count is not None:
                        count = _exact_uint(
                            candidate_count,
                            name="resource components",
                            maximum=_UINT32_MAX,
                            minimum=1,
                        )
                        break
                    candidate_space = getattr(candidate, "space", None)
                    candidate_components = getattr(candidate_space, "components", None)
                    if candidate_components:
                        names = tuple(candidate_components)
                        count = len(names)
                        break
                if count == 0 and getattr(value, "vtype", None) in {
                    "scalar", "bool", "scalar_field"
                }:
                    count = 1
    if count < 1:
        raise ValueError(
            "Program resource node %r has an unknown component count; exact bytes are required"
            % getattr(value, "name", "<?>")
        )
    return count, names


def _int_metadata(mapping: Mapping[str, Any], names: tuple[str, ...], *, default: Any = None,
                  minimum: int = 0, maximum: int = _UINT64_MAX) -> int:
    raw = _metadata(mapping, names, default=default)
    if isinstance(raw, (tuple, list)):
        if not raw:
            return minimum
        raw = max(raw)
    return _exact_uint(raw, name="resource metadata %s" % "/".join(names), maximum=maximum, minimum=minimum)


def _schedule_metadata(value: Any, *, target: str | None) -> tuple[str, str, bool, bool]:
    schedule = _resource_descriptor(value).get("schedule")
    if schedule is None:
        return "transient", "none", False, False
    from pops.time._schedule.api import Schedule, ScheduleDueKind

    where = "schedule on node %r" % getattr(value, "name", "<?>")
    if not isinstance(schedule, Schedule):
        raise TypeError("%s must implement Schedule" % where)
    schedule.validate_site(clock=value.clock, point=getattr(value, "point", None), where=where)
    lowered = schedule.native_schedule_ir(where=where)
    if lowered.due.kind is ScheduleDueKind.AT_END:
        raise NotImplementedError(
            "schedule AtEnd on %s is not lowerable: compiled Program steps have no end signal; "
            "use an AtEnd ConsumerGraph hook" % where
        )
    always = lowered.due.kind is ScheduleDueKind.ALWAYS
    if not always and schedule.off is None:
        raise ValueError(
            "scheduled value %r has no explicit OffPolicy; persistent resource policy is unknown"
            % getattr(value, "name", "<?>")
        )
    if schedule.off is None:
        off = "none"
    else:
        # ``AccumulateDt`` is not safely lower-cased from its Python class name: the canonical
        # wire spelling is ``accumulate_dt``.  Use the typed policy's manifest tag, the same
        # authority consumed by Schedule.to_data(), and refuse an extension that has not declared
        # one rather than leaking a non-canonical class spelling into the plan digest.
        off = getattr(type(schedule.off), "manifest_tag", None)
        if not isinstance(off, str) or not off:
            raise TypeError(
                "%s must declare a non-empty canonical manifest_tag" % where
            )
    persistent = not always
    if target == "amr_system" and lowered.domain.timeline.value == "amr_level" and persistent:
        descriptor = _resource_descriptor(value)
        provider = _metadata(
            descriptor,
            (
                "transfer_provider",
                "spatial_transfer_provider",
                "regrid_provider",
                "spatial_transfer",
                "transfer",
            ),
        )
        if provider is None:
            raise ValueError(
                "AMR-level persistent resource %r has no transfer provider" % getattr(value, "name", "<?>")
            )
    return "persistent_schedule" if persistent else "transient", off, persistent, always


def _runtime_sized_metadata(metadata: Mapping[str, Any]) -> bool:
    """Return the explicit runtime-sizing declaration carried by resource metadata.

    The authoring and wire contracts use the same two canonical fields.  A
    missing exact footprint is also symbolic: classifying it here prevents a
    later path from manufacturing a one-cell, eight-byte row.
    """

    raw = _metadata(
        metadata,
        ("runtime_sized",),
        default=_LEVEL_UNSPECIFIED,
    )
    type_raw = _metadata(
        metadata,
        ("resource_type",),
        default=_LEVEL_UNSPECIFIED,
    )
    if raw is not _LEVEL_UNSPECIFIED and type(raw) is not bool:
        raise TypeError("resource runtime_sized must be an exact bool")
    if type_raw is not _LEVEL_UNSPECIFIED:
        if not isinstance(type_raw, str):
            raise TypeError("resource_type must be a non-empty string")
        if type_raw == "runtime_sized":
            typed = True
        elif type_raw == "exact":
            typed = False
        else:
            raise ValueError("resource_type is unsupported")
        if raw is not _LEVEL_UNSPECIFIED and raw != typed:
            raise ValueError("resource runtime_sized and resource_type disagree")
        return typed
    return False if raw is _LEVEL_UNSPECIFIED else raw


def _shape_metadata(
    metadata: Mapping[str, Any],
    space: Any,
    *,
    level_index: int,
    levels: tuple[int | None, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(logical_shape, allocated_shape)`` from resolved layout evidence.

    ``logical_shape`` identifies the value in the artifact.  It is sufficient
    to count storage only when there are no ghosts.  A ghosted field needs an
    explicit allocation shape because inferring a patch-local allocation from a
    global logical extent would silently under-budget a multi-box hierarchy.
    The allocation aliases are deliberately separate from ``shape`` for that
    reason.
    """

    def normalize(raw: Any, *, name: str) -> tuple[int, ...]:
        if raw is _LEVEL_UNSPECIFIED or raw is None:
            return ()
        if isinstance(raw, Mapping):
            raw = _metadata(raw, ("shape", "extent", "dimensions"), default=None)
        if isinstance(raw, int) and not isinstance(raw, bool):
            return (_exact_uint(raw, name=name, minimum=1),)
        if isinstance(raw, (tuple, list)):
            return tuple(_exact_uint(item, name=name, minimum=1) for item in raw)
        raise TypeError("resource layout %s must be a positive integer extent sequence" % name)

    # Shape tuples are dimension vectors, not a per-level scalar sequence.
    # Only the explicit ``*_by_level`` keys use the level selector; otherwise
    # a 2-D shape on a two-level hierarchy would be misread as two 1-D rows.
    logical = _level_value(
        metadata,
        ("shape_by_level", "logical_shape_by_level"),
        level_index=level_index,
        levels=levels,
        default=_LEVEL_UNSPECIFIED,
    )
    if logical is _LEVEL_UNSPECIFIED:
        logical = _metadata(
            metadata, ("shape", "logical_shape", "extent", "global_shape"),
            default=_LEVEL_UNSPECIFIED,
        )
    if logical is _LEVEL_UNSPECIFIED and space is not None:
        logical = next(
            (getattr(space, name) for name in ("shape", "logical_shape", "extent", "global_shape")
             if hasattr(space, name)),
            _LEVEL_UNSPECIFIED,
        )
    allocated = _level_value(
        metadata,
        ("allocation_shape_by_level", "allocated_shape_by_level", "storage_shape_by_level"),
        level_index=level_index,
        levels=levels,
        default=_LEVEL_UNSPECIFIED,
    )
    if allocated is _LEVEL_UNSPECIFIED:
        allocated = _metadata(
            metadata, ("allocation_shape", "allocated_shape", "storage_shape"),
            default=_LEVEL_UNSPECIFIED,
        )
    if allocated is _LEVEL_UNSPECIFIED and space is not None:
        allocated = next(
            (getattr(space, name) for name in ("allocation_shape", "allocated_shape", "storage_shape")
             if hasattr(space, name)),
            _LEVEL_UNSPECIFIED,
        )
    return normalize(logical, name="resource logical shape"), normalize(
        allocated, name="resource allocated shape"
    )


def _shape_cells(shape: tuple[int, ...], *, name: str) -> int:
    """Return the checked product of a complete, non-empty storage shape."""

    if not shape:
        raise ValueError("%s is absent" % name)
    cells = 1
    for extent in shape:
        if cells > _UINT64_MAX // extent:
            raise OverflowError("%s overflows uint64" % name)
        cells *= extent
    return cells


def _entry_for_occurrence(
    value: Any,
    path: str,
    *,
    target: str | None,
    resolved_level: int | None | object = _LEVEL_UNSPECIFIED,
    level_index: int = 0,
    resolved_levels: tuple[int | None, ...] | None = None,
) -> ProgramResourcePlanEntry | None:
    metadata = _resource_descriptor(value)
    op = getattr(value, "op", None)
    vtype = getattr(value, "vtype", None)
    schedule = metadata.get("schedule")
    if op not in _RESOURCE_OPS and schedule is None and vtype not in {
        "state", "rhs", "scalar_field", "fields"
    }:
        return None
    lifetime, off, scheduled, _always = _schedule_metadata(value, target=target)
    explicit_lifetime = metadata.get("lifetime")
    if explicit_lifetime is not None:
        if not isinstance(explicit_lifetime, str) or not explicit_lifetime:
            raise TypeError("resource lifetime must be a non-empty string")
        lifetime = explicit_lifetime
    explicit_off = metadata.get("off_policy")
    if explicit_off is not None:
        if not isinstance(explicit_off, str) or not explicit_off:
            raise TypeError("resource off_policy must be a non-empty string")
        off = explicit_off
    components, component_names = _component_info(value, metadata)
    centering = _metadata(metadata, ("centering", "layout"))
    if centering is None:
        space = getattr(value, "space", None)
        centering = getattr(space, "centering", None) or getattr(space, "layout", None) or "cell"
    if not isinstance(centering, str) or not centering:
        raise ValueError("resource node %r has an unknown centering" % getattr(value, "name", "<?>"))
    ghosts = _int_metadata(
        metadata,
        ("ghosts", "ghost_depth", "nghost", "required_ghosts"),
        default=0,
        maximum=_UINT32_MAX,
    )
    communication = _metadata(metadata, ("communication", "communication_provider"), default="none")
    if communication is None:
        raise ValueError("resource node %r has no communication policy" % getattr(value, "name", "<?>"))
    communication = _text_identity(communication, "none")
    transfer = _metadata(
        metadata,
        (
            "transfer_provider",
            "spatial_transfer_provider",
            "regrid_provider",
            "spatial_transfer",
            "transfer",
        ),
        default="none",
    )
    restart = _metadata(
        metadata,
        ("restart_provider", "checkpoint_provider", "restart"),
        default="none",
    )
    if transfer is None or restart is None:
        raise ValueError("resource node %r has an incomplete transfer/restart provider policy" % getattr(value, "name", "<?>"))
    transfer = _text_identity(transfer, "none")
    restart = _text_identity(restart, "none")
    if resolved_levels is None:
        resolved_levels = _resolved_levels(value, metadata, target=target)
    if not 0 <= level_index < len(resolved_levels):
        raise IndexError("resource resolved level index is outside its exact level expansion")
    if resolved_level is _LEVEL_UNSPECIFIED:
        resolved_level = resolved_levels[level_index]
    elif resolved_level != resolved_levels[level_index]:
        raise ValueError("resource resolved level disagrees with its level expansion")
    space_shape = getattr(value, "space", None)
    shape, allocated_shape = _shape_metadata(
        metadata,
        space_shape,
        level_index=level_index,
        levels=resolved_levels,
    )
    explicit_bytes = _level_value(
        metadata,
        ("bytes_by_level", "size_bytes_by_level", "bytes", "size_bytes", "byte_count", "allocation_bytes", "memory_bytes"),
        level_index=level_index,
        levels=resolved_levels,
        default=_LEVEL_UNSPECIFIED,
    )
    if explicit_bytes is _LEVEL_UNSPECIFIED:
        explicit_bytes = _level_attr(
            space_shape,
            ("bytes", "memory_bytes"),
            level_index=level_index,
            levels=resolved_levels,
        )
    runtime_sized = _runtime_sized_metadata(metadata)
    if explicit_bytes is None:
        # ``bytes=None`` is an explicit unresolved declaration.  It is kept
        # symbolic so the host can bind the exact prepared layout later.
        runtime_sized = True

    cells = _level_value(
        metadata,
        ("cells_by_level", "cell_count_by_level", "elements_by_level", "cells", "cell_count", "elements"),
        level_index=level_index,
        levels=resolved_levels,
        default=_LEVEL_UNSPECIFIED,
    )
    if cells is _LEVEL_UNSPECIFIED:
        cells = _level_attr(
            space_shape,
            ("cells", "cell_count", "elements"),
            level_index=level_index,
            levels=resolved_levels,
        )
    if cells is _LEVEL_UNSPECIFIED:
        # A storage shape is complete by construction.  A logical shape is
        # complete only without ghosts; otherwise it does not say how every
        # AMR patch allocated its halo cells.
        storage_shape = allocated_shape or (shape if ghosts == 0 else ())
        cells = _shape_cells(storage_shape, name="resource allocated shape") \
            if storage_shape else _LEVEL_UNSPECIFIED
    itemsize = _level_value(
        metadata,
        ("itemsize_by_level", "itemsize_bytes_by_level", "dtype_bytes_by_level", "real_bytes_by_level", "itemsize", "itemsize_bytes", "dtype_bytes", "real_bytes"),
        level_index=level_index,
        levels=resolved_levels,
        default=_LEVEL_UNSPECIFIED,
    )
    if itemsize is _LEVEL_UNSPECIFIED:
        itemsize = _level_attr(
            space_shape,
            ("itemsize", "itemsize_bytes", "dtype_bytes", "real_bytes"),
            level_index=level_index,
            levels=resolved_levels,
        )

    # The native bind-sealed plan accepts an exact row only when it carries a
    # complete post-ghost cell count, precision and canonical shape.  A source
    # byte hint without that witness is not an exact layout; preserve it as a
    # runtime-sized declaration so the host materializer, rather than codegen,
    # supplies the missing facts during prepare_*.
    has_complete_layout = (
        cells is not _LEVEL_UNSPECIFIED and cells is not None
        and itemsize is not _LEVEL_UNSPECIFIED and itemsize is not None
        and bool(shape or allocated_shape)
    )
    exact_layout_requested = (
        metadata.get("resource_type") == "exact"
        or metadata.get("runtime_sized") is False
    )
    if not runtime_sized and not has_complete_layout and exact_layout_requested:
        raise ValueError(
            "resource exact layout is incomplete; provide cells, itemsize and a complete "
            "shape/allocation_shape or declare runtime_sized"
        )
    if runtime_sized or not has_complete_layout:
        runtime_sized = True
        explicit_bytes = None
        derived_cells = None
        derived_itemsize = None
    else:
        derived_cells = _exact_uint(cells, name="resource cell count", minimum=1)
        derived_itemsize = _exact_uint(itemsize, name="resource item size", minimum=1)
        if components > _UINT64_MAX // derived_itemsize or \
                components * derived_itemsize > _UINT64_MAX // derived_cells:
            raise OverflowError("resource byte bound overflows uint64")
        derived_bytes = components * derived_itemsize * derived_cells
        if explicit_bytes is _LEVEL_UNSPECIFIED:
            explicit_bytes = derived_bytes
        else:
            explicit_bytes = _exact_uint(explicit_bytes, name="resource bytes", minimum=1)
            if explicit_bytes != derived_bytes:
                raise ValueError(
                    "resource byte count disagrees with resolved cells/itemsize/components"
                )
    bytes_count = None if runtime_sized else _exact_uint(explicit_bytes, name="resource bytes", minimum=1)
    if not shape:
        # The allocation shape is a canonical, exact spatial witness when a
        # producer intentionally exposes storage rather than logical extents.
        shape = allocated_shape
    if runtime_sized:
        maximum = None
    else:
        maximum = _level_value(
            metadata,
            ("maximum_bytes_by_level", "memory_cap_by_level", "memory_limit_by_level", "maximum_bytes", "memory_cap", "memory_limit"),
            level_index=level_index,
            levels=resolved_levels,
            default=_LEVEL_UNSPECIFIED,
        )
        if maximum is _LEVEL_UNSPECIFIED:
            maximum = _level_attr(
                space_shape,
                ("maximum_bytes", "memory_cap"),
                level_index=level_index,
                levels=resolved_levels,
            )
        if maximum is _LEVEL_UNSPECIFIED:
            maximum = bytes_count
        if maximum is None:
            raise ValueError("resource node %r has unknown maximum bytes" % getattr(value, "name", "<?>"))
        maximum = _exact_uint(maximum, name="resource maximum bytes", minimum=1)
    communicates = _metadata(metadata, ("communicates", "communication_required"), default=False)
    restart_required = _metadata(metadata, ("restart_required", "requires_restart"), default=False)
    if type(communicates) is not bool or type(restart_required) is not bool:
        raise TypeError("resource communication/restart flags must be exact bools")
    if restart_required and restart == "none":
        raise ValueError("resource node %r requires a restart provider" % getattr(value, "name", "<?>"))
    owner = _text_identity(
        _metadata(metadata, ("owner", "owner_id"), default=getattr(value, "block", None)),
        "global",
    )
    state_ref = getattr(value, "state_ref", None)
    if owner == "global" and state_ref is not None:
        owner = _text_identity(getattr(state_ref, "block_ref", None), "global")
    space = _metadata(metadata, ("space", "space_id"), default=getattr(value, "space", None))
    space_id = _text_identity(space, "none")
    clock = _text_identity(
        _metadata(metadata, ("clock", "clock_id"), default=getattr(value, "clock", None)),
        "clock.unknown",
    )
    level = resolved_level
    key = ProgramPersistentValueKey(
        _exact_uint(getattr(value, "id", None), name="Program resource value id"),
        path,
        owner,
        space_id,
        clock,
        level,
    )
    return ProgramResourcePlanEntry(
        key=key,
        lifetime=lifetime,
        centering=centering,
        ghosts=ghosts,
        communication=communication,
        transfer_provider=transfer,
        restart_provider=restart,
        components=components,
        bytes=bytes_count,
        maximum_bytes=maximum,
        off_policy=off,
        component_names=component_names,
        communicates=communicates,
        restart_required=restart_required,
        shape=shape,
        cells=derived_cells,
        itemsize=derived_itemsize,
        runtime_sized=runtime_sized,
        resource_type="runtime_sized" if runtime_sized else "exact",
    )


def lower_program_resource_plan(program: Any, *, target: str | None = None,
                                maximum_bytes: Any = None) -> ProgramResourcePlan:
    """Build the complete deterministic resource plan before C++ emission."""

    if type(program) is ProgramResourcePlan:
        if maximum_bytes is not None:
            raise ValueError("cannot override the memory bound of a sealed ProgramResourcePlan")
        return program
    declared = getattr(program, "resource_plan", None)
    if type(declared) is ProgramResourcePlan:
        if maximum_bytes is not None:
            raise ValueError("cannot override the memory bound of a declared ProgramResourcePlan")
        return declared
    if isinstance(declared, Mapping):
        if maximum_bytes is not None:
            raise ValueError("cannot override the memory bound of a declared Program resource plan")
        return ProgramResourcePlan.from_data(declared)

    rows = []
    occurrences = list(_reachable_program_occurrences(program))
    occurrence_values: dict[int, list[ProgramPersistentValueKey]] = {}
    for value, path in occurrences:
        metadata = _resource_descriptor(value)
        resolved_levels = _resolved_levels(value, metadata, target=target)
        for level_index, resolved_level in enumerate(resolved_levels):
            row = _entry_for_occurrence(
                value,
                path,
                target=target,
                resolved_level=resolved_level,
                level_index=level_index,
                resolved_levels=resolved_levels,
            )
            if row is not None:
                rows.append(row)
                occurrence_values.setdefault(id(value), []).append(row.key)
    cap = maximum_bytes
    if cap is None:
        for name in ("_persistent_memory_limit", "_resource_memory_limit", "maximum_resource_bytes"):
            if hasattr(program, name):
                cap = getattr(program, name)
                if cap is None:
                    raise ValueError("Program resource plan memory bound is unknown")
                break
    bindings: dict[int, ProgramPersistentValueKey | tuple[ProgramPersistentValueKey, ...]] = {}
    for object_id, keys in occurrence_values.items():
        if len(keys) == 1:
            bindings[object_id] = keys[0]
        else:
            bindings[object_id] = tuple(keys)
    return ProgramResourcePlan(rows, cap, occurrence_values=bindings)


_PLAN_CACHE: dict[tuple[int, str, str | None], tuple[Any, ProgramResourcePlan]] = {}


def get_program_resource_plan(program: Any, *, target: str | None = None) -> ProgramResourcePlan:
    """Return a read-only plan cache for one emission pass."""

    if type(program) is ProgramResourcePlan:
        return program
    declared = getattr(program, "resource_plan", None)
    if type(declared) is ProgramResourcePlan:
        return declared
    if isinstance(declared, Mapping):
        return ProgramResourcePlan.from_data(declared)

    try:
        fingerprint = program._ir_hash()
    except (AttributeError, TypeError, ValueError):
        fingerprint = repr(_program_values(program))
    cache_key = (id(program), fingerprint, target)
    cached = _PLAN_CACHE.get(cache_key)
    if cached is not None and cached[0] is program:
        return cached[1]
    result = lower_program_resource_plan(program, target=target)
    _PLAN_CACHE[cache_key] = (program, result)
    return result


def persistent_slot(
    program: Any,
    value: Any,
    *,
    target: str | None = None,
    occurrence_path: str | None = None,
    level: int | None | object = _LEVEL_UNSPECIFIED,
) -> int:
    """Resolve a resource slot at compile time; no generated lookup is involved."""

    if program is None:
        raise TypeError("Program resource slot resolution requires a sealed ProgramResourcePlan")
    return get_program_resource_plan(program, target=target).slot_for_value(
        value, occurrence_path=occurrence_path, level=level
    )


def persistent_slot_token(
    program: Any,
    value: Any,
    *,
    target: str | None = None,
    occurrence_path: str | None = None,
    level: int | None | object = _LEVEL_UNSPECIFIED,
) -> str:
    """Return the dense slot spelling for a sealed resource occurrence.

    Slot resolution is deliberately performed at the point where an emitter
    lowers the occurrence.  There is no post-lowering value-id-to-slot rewrite:
    an artifact either contains the exact slot returned here or is rejected by
    the codegen source validator.  Every caller supplies an authenticated plan;
    there is no numeric value-id fallback.
    """

    slot = persistent_slot(
        program,
        value,
        target=target,
        occurrence_path=occurrence_path,
        level=level,
    )
    return str(slot)


__all__ = [
    "ProgramPersistentValueKey",
    "ProgramResourcePlanAbiRow",
    "ProgramResourcePlan",
    "ProgramResourcePlanEntry",
    "get_program_resource_plan",
    "iter_program_occurrences",
    "_reachable_program_occurrences",
    "lower_program_resource_plan",
    "occurrence_path_digest",
    "persistent_slot",
    "persistent_slot_token",
]
