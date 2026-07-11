"""Content-addressed, collision-safe component registration."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import Any

from pops.identity import Identity, canonical_bytes, make_identity

from .component_protocols import FACET_PROTOCOLS


COMPONENT_MANIFEST_SCHEMA_VERSION = 1


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be an integer >= 1" % where)
    return value


def _component_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("component_id must be a non-empty canonical string")
    return value


def _freeze(value: Any, *, where: str = "manifest") -> Any:
    """Freeze the strict canonical value language without changing its meaning."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        # Ask the canonical encoder to enforce int64 and Unicode constraints now.
        canonical_bytes(value)
        return value
    if isinstance(value, float):
        raise TypeError("%s cannot contain floats" % where)
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("%s map keys must be strings" % where)
            frozen[key] = _freeze(item, where="%s.%s" % (where, key))
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, where="%s[]" % where) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError("%s cannot contain unordered sets" % where)
    raise TypeError("%s contains opaque %s" % (where, type(value).__name__))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    """Strict immutable identity input for one registered component."""

    component_id: str
    version: int
    facets: tuple[str, ...] = ()
    content: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = COMPONENT_MANIFEST_SCHEMA_VERSION
    content_digest: Identity = field(init=False, repr=False)
    _bytes: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _component_id(self.component_id))
        object.__setattr__(self, "version", _positive_int(self.version, where="version"))
        schema = _positive_int(self.schema_version, where="schema_version")
        if schema != COMPONENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported component manifest schema_version %d (expected %d)"
                % (schema, COMPONENT_MANIFEST_SCHEMA_VERSION)
            )
        if not isinstance(self.facets, (list, tuple)):
            raise TypeError("component manifest facets must be a list or tuple")
        facets = tuple(self.facets)
        if any(not isinstance(facet, str) or not facet for facet in facets):
            raise TypeError("component manifest facets must be non-empty strings")
        if len(set(facets)) != len(facets):
            raise ValueError("component manifest facets must be unique")
        unknown = sorted(set(facets) - set(FACET_PROTOCOLS))
        if unknown:
            raise ValueError("unknown component facet(s): %s" % ", ".join(unknown))
        facets = tuple(sorted(facets))
        if not isinstance(self.content, Mapping):
            raise TypeError("component manifest content must be a string-keyed mapping")
        frozen_content = _freeze(self.content, where="component manifest content")
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "content", frozen_content)
        payload = self.to_data(include_digest=False)
        encoded = canonical_bytes(payload)
        object.__setattr__(self, "_bytes", encoded)
        object.__setattr__(
            self, "content_digest", make_identity("component-manifest", payload,
                                                   schema_version=schema)
        )

    @property
    def digest(self) -> str:
        """Printable canonical digest token (the typed value is ``content_digest``)."""
        return self.content_digest.token

    @property
    def content_bytes(self) -> bytes:
        return self._bytes

    def to_bytes(self) -> bytes:
        return self.content_bytes

    def to_data(self, *, include_digest: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "component_id": self.component_id,
            "version": self.version,
            "facets": list(self.facets),
            "content": _thaw(self.content),
        }
        if include_digest:
            data["digest"] = self.content_digest.token
        return data

    @classmethod
    def from_data(cls, data: Any) -> ComponentManifest:
        required = {
            "schema_version", "component_id", "version", "facets", "content", "digest",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError(
                "ComponentManifest data must contain exactly %s" % sorted(required)
            )
        result = cls(
            component_id=data["component_id"],
            version=data["version"],
            facets=data["facets"],
            content=data["content"],
            schema_version=data["schema_version"],
        )
        supplied = Identity.from_token(data["digest"])
        if supplied != result.content_digest:
            raise ValueError("ComponentManifest digest does not match its canonical content")
        if result.to_data() != dict(data):
            raise ValueError("ComponentManifest data is not in canonical form")
        return result


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    manifest: ComponentManifest
    component: Any = field(compare=False, repr=False)

    @property
    def component_id(self) -> str:
        return self.manifest.component_id


@dataclass(frozen=True, slots=True)
class ComponentRegistrySnapshot:
    revision: int
    records: tuple[ComponentRecord, ...]
    frozen: bool = False
    _by_id: Mapping[str, ComponentRecord] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_by_id", MappingProxyType({record.component_id: record for record in self.records})
        )

    def resolve(self, component_id: Any) -> Any:
        try:
            return self._by_id[component_id].component
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def manifest(self, component_id: Any) -> ComponentManifest:
        try:
            return self._by_id[component_id].manifest
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def __len__(self) -> int:
        return len(self.records)


class ComponentRegistry:
    """Atomic registry with deterministic collision and freeze semantics."""

    __slots__ = ("_by_id", "_order", "_revision", "_frozen")

    def __init__(self) -> None:
        self._by_id: dict[str, ComponentRecord] = {}
        self._order: list[str] = []
        self._revision = 0
        self._frozen = False

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register(self, component: Any, manifest: ComponentManifest | None = None) -> Any:
        """Register a conforming component; an identical repeat is a no-op.

        A manifest may be supplied explicitly.  Otherwise the external component
        exposes ``component_manifest`` as either an attribute or a zero-argument
        method.  All validation completes before registry state is changed.
        """
        if self.frozen:
            raise RuntimeError("ComponentRegistry is frozen; cannot register components")
        if manifest is None:
            manifest = getattr(component, "component_manifest", None)
            if callable(manifest):
                manifest = manifest()
        if not isinstance(manifest, ComponentManifest):
            raise TypeError(
                "component registration requires a ComponentManifest, explicitly or through "
                "component_manifest"
            )
        self._validate_facets(component, manifest)
        previous = self._by_id.get(manifest.component_id)
        if previous is not None:
            if previous.manifest.content_bytes == manifest.content_bytes:
                return previous.component
            raise ValueError(
                "component identity collision for %r: existing digest %s, incoming digest %s"
                % (manifest.component_id, previous.manifest.digest, manifest.digest)
            )
        record = ComponentRecord(manifest, component)
        self._by_id[manifest.component_id] = record
        self._order.append(manifest.component_id)
        self._revision += 1
        return component

    @staticmethod
    def _validate_facets(component: Any, manifest: ComponentManifest) -> None:
        methods = {
            "requirement": ("requirements", ()),
            "lowering": ("lower", (object(),)),
            "stencil": ("stencil", ()),
            "stability": ("stability", ()),
            "provider": ("providers", ()),
            "effects": ("effects", ()),
            "restart": ("restart", ()),
            "report": ("report", ()),
            "fallible_evaluation": ("evaluate", (object(),)),
        }
        malformed = []
        for facet in manifest.facets:
            if not isinstance(component, FACET_PROTOCOLS[facet]):
                malformed.append(facet)
                continue
            method_name, probe_args = methods[facet]
            method = getattr(component, method_name, None)
            if not callable(method):
                malformed.append(facet)
                continue
            try:
                inspect.signature(method).bind(*probe_args)
            except (TypeError, ValueError):
                malformed.append(facet)
        if malformed:
            raise TypeError(
                "component %r does not conform to advertised facet(s): %s"
                % (manifest.component_id, ", ".join(malformed))
            )

    def resolve(self, component_id: Any) -> Any:
        try:
            return self._by_id[component_id].component
        except KeyError:
            known = ", ".join(self._order) or "<none>"
            raise KeyError(
                "unknown component %r (registered: %s)" % (component_id, known)
            ) from None

    def manifest(self, component_id: Any) -> ComponentManifest:
        try:
            return self._by_id[component_id].manifest
        except KeyError:
            raise KeyError("unknown component %r" % (component_id,)) from None

    def snapshot(self) -> ComponentRegistrySnapshot:
        if not self.frozen:
            raise RuntimeError(
                "ComponentRegistry must be frozen before snapshot capture"
            )
        return ComponentRegistrySnapshot(
            self.revision, tuple(self._by_id[item] for item in self._order), self.frozen
        )

    def freeze(self) -> ComponentRegistry:
        self._frozen = True
        return self

    def __contains__(self, component_id: Any) -> bool:
        return component_id in self._by_id

    def __len__(self) -> int:
        return len(self._order)


__all__ = [
    "COMPONENT_MANIFEST_SCHEMA_VERSION", "ComponentManifest", "ComponentRecord",
    "ComponentRegistrySnapshot", "ComponentRegistry",
]
