"""Canonical cross-language component manifest and extension contract."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

from pops.identity import Identity, canonical_bytes, make_identity

from ._component_manifest_schema import (
    ComponentExtensionSchema,
    ComponentManifestError,
    ComponentVersion,
    _canonical_string,
    _component_uri,
    _default_determinism,
    _default_precision,
    _default_restart,
    _default_target,
    _determinism,
    _entry_points,
    _exact_mapping,
    _extensions,
    _freeze,
    _interfaces,
    _positive_int,
    _precision,
    _refuse,
    _restart,
    _semantic_set,
    _string_set,
    _target,
    _thaw,
)
from ._generated_component_schema import (
    COMPONENT_DIGEST_FIELDS,
    COMPONENT_MANIFEST_SCHEMA_VERSION,
    COMPONENT_MANIFEST_TOP_LEVEL_FIELDS,
)


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    """Complete immutable semantic contract for a source or native component."""

    uri: str
    component_type: str
    version: ComponentVersion | str | tuple[int, int, int] | Mapping[str, int]
    facets: tuple[str, ...] = ()
    signature: Mapping[str, Any] = field(default_factory=dict)
    reads: tuple[Any, ...] = ()
    writes: tuple[Any, ...] = ()
    parameters: tuple[Any, ...] = ()
    interfaces: tuple[Any, ...] = ()
    requirements: tuple[Any, ...] = ()
    capabilities: tuple[Any, ...] = ()
    effects: tuple[Any, ...] = ()
    layouts: tuple[Any, ...] = ()
    clocks: tuple[Any, ...] = ()
    target: Mapping[str, Any] = field(default_factory=_default_target)
    determinism: Mapping[str, Any] = field(default_factory=_default_determinism)
    restart: Mapping[str, Any] = field(default_factory=_default_restart)
    precision: Mapping[str, Any] = field(default_factory=_default_precision)
    conservation: tuple[Any, ...] = ()
    entry_points: Mapping[str, str] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    extension_schemas: InitVar[Mapping[Any, ComponentExtensionSchema] | None] = None
    schema_version: int = COMPONENT_MANIFEST_SCHEMA_VERSION
    semantic_digest: Identity = field(init=False, repr=False)
    manifest_digest: Identity = field(init=False, repr=False)
    _semantic_bytes: bytes = field(init=False, repr=False, compare=False)
    _manifest_bytes: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self, extension_schemas: Any) -> None:
        schema = _positive_int(self.schema_version, path="schema_version")
        if schema != COMPONENT_MANIFEST_SCHEMA_VERSION:
            _refuse("unsupported_schema_version", "schema_version",
                    f"unsupported ComponentManifest schema_version {schema}; "
                    f"expected {COMPONENT_MANIFEST_SCHEMA_VERSION}")
        object.__setattr__(self, "uri", _component_uri(self.uri))
        object.__setattr__(self, "component_type", _canonical_string(
            self.component_type, path="component_type", pattern=r"[a-z][a-z0-9_.-]*"))
        object.__setattr__(self, "version", ComponentVersion.from_value(self.version))
        object.__setattr__(self, "facets", _string_set(self.facets, path="facets"))
        object.__setattr__(self, "signature", _freeze(self.signature, path="signature"))
        for name in (
            "reads", "writes", "parameters", "requirements", "capabilities",
            "effects", "layouts", "clocks", "conservation",
        ):
            object.__setattr__(self, name, _semantic_set(getattr(self, name), path=name))
        object.__setattr__(self, "target", _target(self.target))
        object.__setattr__(self, "determinism", _determinism(self.determinism))
        object.__setattr__(self, "restart", _restart(self.restart))
        object.__setattr__(self, "precision", _precision(self.precision))
        object.__setattr__(self, "entry_points", _entry_points(self.entry_points))
        object.__setattr__(self, "interfaces", _interfaces(
            self.interfaces, self.facets, self.entry_points))
        object.__setattr__(self, "extensions", _extensions(self.extensions, extension_schemas))

        semantic = self._semantic_data()
        full = self._manifest_data()
        semantic_bytes = canonical_bytes(semantic)
        manifest_bytes = canonical_bytes(full)
        object.__setattr__(self, "_semantic_bytes", semantic_bytes)
        object.__setattr__(self, "_manifest_bytes", manifest_bytes)
        object.__setattr__(self, "semantic_digest", make_identity(
            "component-semantics", semantic, schema_version=schema))
        object.__setattr__(self, "manifest_digest", make_identity(
            "component-manifest", full, schema_version=schema))

    @property
    def component_id(self) -> str:
        return f"{self.uri}@{self.version}"

    @property
    def digest(self) -> str:
        return self.manifest_digest.token

    @property
    def semantic_bytes(self) -> bytes:
        return self._semantic_bytes

    @property
    def manifest_bytes(self) -> bytes:
        return self._manifest_bytes

    def _base_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "uri": self.uri,
            "component_type": self.component_type,
            "version": cast(ComponentVersion, self.version).to_data(),
            "facets": list(self.facets),
            "signature": _thaw(self.signature),
            "reads": _thaw(self.reads),
            "writes": _thaw(self.writes),
            "parameters": _thaw(self.parameters),
            "interfaces": _thaw(self.interfaces),
            "requirements": _thaw(self.requirements),
            "capabilities": _thaw(self.capabilities),
            "effects": _thaw(self.effects),
            "layouts": _thaw(self.layouts),
            "clocks": _thaw(self.clocks),
            "target": _thaw(self.target),
            "determinism": _thaw(self.determinism),
            "restart": _thaw(self.restart),
            "precision": _thaw(self.precision),
            "conservation": _thaw(self.conservation),
            "entry_points": _thaw(self.entry_points),
        }

    def _semantic_data(self) -> dict[str, Any]:
        data = self._base_data()
        semantic_extensions = {
            key: _thaw(value) for key, value in self.extensions.items()
            if value["kind"] == "semantic"
        }
        if semantic_extensions:
            data["semantic_extensions"] = semantic_extensions
        return data

    def _manifest_data(self) -> dict[str, Any]:
        data = self._base_data()
        data["extensions"] = _thaw(self.extensions)
        return data

    def to_bytes(self) -> bytes:
        return self.manifest_bytes

    def semantic_data(self) -> dict[str, Any]:
        return self._semantic_data()

    def manifest_data(self) -> dict[str, Any]:
        return self._manifest_data()

    def to_data(self) -> dict[str, Any]:
        data = self._manifest_data()
        data["digests"] = {
            "semantic": self.semantic_digest.token,
            "manifest": self.manifest_digest.token,
        }
        return data

    @classmethod
    def from_data(
        cls, data: Any, *,
        extension_schemas: Mapping[Any, ComponentExtensionSchema] | None = None,
    ) -> ComponentManifest:
        row = _exact_mapping(data, set(COMPONENT_MANIFEST_TOP_LEVEL_FIELDS),
                             path="ComponentManifest")
        digests = _exact_mapping(row["digests"], set(COMPONENT_DIGEST_FIELDS),
                                 path="ComponentManifest.digests")
        kwargs = {key: row[key] for key in COMPONENT_MANIFEST_TOP_LEVEL_FIELDS
                  if key not in {"digests"}}
        result = cls(**kwargs, extension_schemas=extension_schemas)
        try:
            supplied_semantic = Identity.from_token(digests["semantic"])
        except (TypeError, ValueError):
            _refuse("invalid_digest", "digests.semantic",
                    "digests.semantic is not a canonical PoPS identity token",
                    evidence=digests["semantic"])
        try:
            supplied_manifest = Identity.from_token(digests["manifest"])
        except (TypeError, ValueError):
            _refuse("invalid_digest", "digests.manifest",
                    "digests.manifest is not a canonical PoPS identity token",
                    evidence=digests["manifest"])
        if supplied_semantic != result.semantic_digest:
            _refuse("semantic_digest_mismatch", "digests.semantic",
                    "ComponentManifest semantic digest does not match canonical semantics")
        if supplied_manifest != result.manifest_digest:
            _refuse("manifest_digest_mismatch", "digests.manifest",
                    "ComponentManifest digest does not match canonical content")
        if result.to_data() != dict(data):
            _refuse("noncanonical_manifest", "ComponentManifest",
                    "ComponentManifest data is not in canonical form")
        return result

    def require_target(self, platform: Any) -> None:
        """Refuse an unsupported platform with machine-readable evidence."""
        row = _exact_mapping(platform, {"dimension", "scalar", "device", "features"},
                             path="platform")
        requested = {
            "dimension": _positive_int(row["dimension"], path="platform.dimension"),
            "scalar": _canonical_string(row["scalar"], path="platform.scalar"),
            "device": _canonical_string(row["device"], path="platform.device"),
            "features": _string_set(row["features"], path="platform.features"),
        }
        variants = self.target["variants"]
        if not variants:
            _refuse("target_capability_unspecified", "target.variants",
                    f"component {self.component_id} does not declare target variants",
                    evidence={"component": self.component_id, "requested": requested})
        core_matches = [variant for variant in variants if all(
            variant[name] == requested[name] for name in ("dimension", "scalar", "device"))]
        for variant in core_matches:
            if set(variant["features"]) <= set(requested["features"]):
                return
        supported = [_thaw(variant) for variant in variants]
        if core_matches:
            missing = min(
                (sorted(set(variant["features"]) - set(requested["features"]))
                 for variant in core_matches),
                key=lambda values: (len(values), values),
            )
            _refuse("missing_target_features", "target.variants",
                    f"platform lacks required component features: {missing}",
                    evidence={"component": self.component_id, "requested": requested,
                              "matching_variants": [_thaw(item) for item in core_matches],
                              "missing": missing})
        _refuse("unsupported_target_combination", "target.variants",
                f"component {self.component_id} does not support the requested target combination",
                evidence={"component": self.component_id, "requested": requested,
                          "supported": supported})


__all__ = [
    "ComponentExtensionSchema", "ComponentManifest", "ComponentManifestError",
    "ComponentVersion",
]
