"""Strict, immutable documentary provenance for authored and transformed PoPS IR.

Provenance is deliberately data-only.  It never retains a frame, callable, builder, registry,
``OwnerPath`` or graph node, and therefore remains safe to carry across snapshots and manifests.
It documents an artifact but is not part of that artifact's semantic identity.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_TRANSFORMATIONS = frozenset({
    "direct", "factory_expand", "normalize", "cse", "fuse", "lower",
})
_PHASES = frozenset({"authoring", "transform", "lowering"})


def _require_int(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % where)
    if value < minimum:
        raise ValueError("%s must be >= %d" % (where, minimum))
    return value


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """One exact source span in protocol version 1."""

    file: str
    line: int
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    VERSION = 1
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.file, str) or not self.file:
            raise ValueError("SourceSpan.file must be a non-empty string")
        object.__setattr__(self, "file", os.path.abspath(self.file) if self.file != "<unknown>" else self.file)
        _require_int(self.line, "SourceSpan.line", minimum=0)
        _require_int(self.column, "SourceSpan.column", minimum=0)
        if self.end_line is not None:
            _require_int(self.end_line, "SourceSpan.end_line", minimum=self.line)
        if self.end_column is not None:
            if self.end_line is None:
                raise ValueError("SourceSpan.end_column requires end_line")
            _require_int(self.end_column, "SourceSpan.end_column", minimum=0)

    def to_data(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    @classmethod
    def from_data(cls, data: Any) -> SourceSpan:
        expected = {"version", "file", "line", "column", "end_line", "end_column"}
        if not isinstance(data, Mapping) or set(data) != expected:
            raise TypeError("SourceSpan data must contain exactly %s" % sorted(expected))
        if isinstance(data["version"], bool) or not isinstance(data["version"], int):
            raise TypeError("SourceSpan version must be an integer")
        if data["version"] != cls.VERSION:
            raise ValueError("unsupported SourceSpan version %r" % data["version"])
        result = cls(data["file"], data["line"], data["column"], data["end_line"], data["end_column"])
        if result.to_data() != dict(data):
            raise ValueError("SourceSpan data is not canonical")
        return result


def source_span(*, skip_package: bool = True, depth: int = 1) -> SourceSpan:
    """Capture the first useful caller as immutable scalar data."""
    package_root = os.path.dirname(__file__)
    frames = inspect.stack(context=0)[depth + 1:]
    try:
        for frame in frames:
            filename = os.path.abspath(frame.filename)
            if not skip_package or not filename.startswith(package_root + os.sep):
                return SourceSpan(filename, frame.lineno)
    finally:
        del frames
    return SourceSpan("<unknown>", 0)


def callable_span(value: Any) -> SourceSpan:
    """Return the declaration span of a Python callable without retaining it."""
    try:
        filename = inspect.getsourcefile(value) or inspect.getfile(value)
        _, line = inspect.getsourcelines(value)
    except (OSError, TypeError):
        return SourceSpan("<unknown>", 0)
    return SourceSpan(filename, line)


def _owner_data(owner: Any) -> tuple[str, dict[str, Any]]:
    from pops.model.ownership import OwnerPath
    path = OwnerPath.coerce(owner)
    # Documentary capture must not force a live model-definition fingerprint: that provider hashes
    # the complete Module, so invoking it while an Operator is being registered would authenticate
    # the source against a transient partial definition. Strip only the process-local authority and
    # retain the exact logical topology. Already-canonical owners remain exact, including their
    # definition fingerprint.
    if path.is_authoring:
        path = OwnerPath(path.nodes)
    data = path.to_data()
    return str(path), data


def _canonical_blob(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _freeze_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_data(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_data(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError("provenance data must contain only strict JSON values")


def _thaw_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_data(item) for item in value]
    return value


class ProvenanceRecord:
    """Content-addressed immutable provenance protocol version 1."""

    VERSION = 1
    __pops_ir_immutable__ = True
    __slots__ = (
        "id", "primary", "owner", "owner_data", "authoring_api", "origins", "parents",
        "phase", "transformation",
    )

    def __init__(
        self,
        *,
        primary: SourceSpan,
        owner: Any,
        authoring_api: str,
        origins: Sequence[SourceSpan] = (),
        parents: Sequence[str] = (),
        phase: str = "authoring",
        transformation: str = "direct",
    ) -> None:
        if not isinstance(primary, SourceSpan):
            raise TypeError("ProvenanceRecord.primary must be a SourceSpan")
        owner_string, owner_data = _owner_data(owner)
        self._initialize(
            primary=primary, owner=owner_string, owner_data=owner_data,
            authoring_api=authoring_api, origins=origins, parents=parents,
            phase=phase, transformation=transformation,
        )

    def _initialize(
        self, *, primary: SourceSpan, owner: str, owner_data: Mapping[str, Any],
        authoring_api: str, origins: Sequence[SourceSpan], parents: Sequence[str],
        phase: str, transformation: str,
    ) -> None:
        if not isinstance(authoring_api, str) or not authoring_api:
            raise ValueError("ProvenanceRecord.authoring_api must be a non-empty string")
        if phase not in _PHASES:
            raise ValueError("ProvenanceRecord.phase must be one of %s" % sorted(_PHASES))
        if transformation not in _TRANSFORMATIONS:
            raise ValueError("ProvenanceRecord.transformation must be one of %s" % sorted(_TRANSFORMATIONS))
        frozen_origins = tuple(origins) or (primary,)
        if any(not isinstance(item, SourceSpan) for item in frozen_origins):
            raise TypeError("ProvenanceRecord.origins must contain only SourceSpan values")
        frozen_parents = tuple(parents)
        if any(not isinstance(item, str) or not item for item in frozen_parents):
            raise TypeError("ProvenanceRecord.parents must contain only non-empty ids")
        if len(set(frozen_parents)) != len(frozen_parents):
            raise ValueError("ProvenanceRecord.parents must be ordered and unique")
        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "owner_data", _freeze_data(json.loads(json.dumps(owner_data))))
        object.__setattr__(self, "authoring_api", authoring_api)
        object.__setattr__(self, "origins", frozen_origins)
        object.__setattr__(self, "parents", frozen_parents)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "transformation", transformation)
        object.__setattr__(self, "id", "sha256:" + hashlib.sha256(_canonical_blob(self._content_data())).hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ProvenanceRecord is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ProvenanceRecord is immutable")

    def _content_data(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "primary": self.primary.to_data(),
            "owner": self.owner,
            "owner_data": _thaw_data(self.owner_data),
            "authoring_api": self.authoring_api,
            "origins": [item.to_data() for item in self.origins],
            "parents": list(self.parents),
            "phase": self.phase,
            "transformation": self.transformation,
        }

    def to_data(self) -> dict[str, Any]:
        return {"id": self.id, **self._content_data()}

    @classmethod
    def from_data(cls, data: Any) -> ProvenanceRecord:
        expected = {
            "version", "id", "primary", "owner", "owner_data", "authoring_api", "origins",
            "parents", "phase", "transformation",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise TypeError("ProvenanceRecord data must contain exactly %s" % sorted(expected))
        if isinstance(data["version"], bool) or not isinstance(data["version"], int):
            raise TypeError("ProvenanceRecord version must be an integer")
        if data["version"] != cls.VERSION:
            raise ValueError("unsupported ProvenanceRecord version %r" % data["version"])
        if not isinstance(data["owner"], str) or not data["owner"]:
            raise TypeError("ProvenanceRecord owner must be a non-empty string")
        from pops.model.ownership import OwnerPath
        canonical = OwnerPath.from_data(data["owner_data"])
        if str(canonical) != data["owner"]:
            raise ValueError("ProvenanceRecord owner string/data disagree")
        if not isinstance(data["origins"], list):
            raise TypeError("ProvenanceRecord origins must be a list")
        if not isinstance(data["parents"], list):
            raise TypeError("ProvenanceRecord parents must be a list")
        result = object.__new__(cls)
        result._initialize(
            primary=SourceSpan.from_data(data["primary"]), owner=data["owner"],
            owner_data=data["owner_data"], authoring_api=data["authoring_api"],
            origins=tuple(SourceSpan.from_data(item) for item in data["origins"]),
            parents=tuple(data["parents"]), phase=data["phase"],
            transformation=data["transformation"],
        )
        if result.id != data["id"] or result.to_data() != dict(data):
            raise ValueError("ProvenanceRecord id or data is not canonical")
        return result

    @classmethod
    def derive(
        cls, records: Sequence[ProvenanceRecord], *, transformation: str,
        owner: Any | None = None, authoring_api: str | None = None,
    ) -> ProvenanceRecord:
        values = tuple(records)
        if not values or any(not isinstance(item, cls) for item in values):
            raise TypeError("ProvenanceRecord.derive requires one or more ProvenanceRecord values")
        first = values[0]
        origins = []
        seen_spans = set()
        for record in values:
            for span in record.origins:
                key = _canonical_blob(span.to_data())
                if key not in seen_spans:
                    seen_spans.add(key)
                    origins.append(span)
        parents = tuple(dict.fromkeys(record.id for record in values))
        if owner is None:
            from pops.model.ownership import OwnerPath
            owner = OwnerPath.from_data(_thaw_data(first.owner_data))
        phase = "lowering" if transformation == "lower" else "transform"
        return cls(
            primary=first.primary, owner=owner,
            authoring_api=authoring_api or first.authoring_api,
            origins=origins, parents=parents, phase=phase, transformation=transformation,
        )


__all__ = ["ProvenanceRecord", "SourceSpan", "callable_span", "source_span"]
