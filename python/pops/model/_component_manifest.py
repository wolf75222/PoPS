"""Canonical cross-language component manifest and extension contract."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import re
from types import MappingProxyType
from typing import Any, NoReturn
from urllib.parse import urlsplit

from pops.identity import Identity, canonical_bytes, make_identity

from ._generated_component_schema import (
    COMPONENT_DIGEST_FIELDS,
    COMPONENT_EXTENSION_KINDS,
    COMPONENT_INTERFACE_SPECS,
    COMPONENT_MANIFEST_SCHEMA_VERSION,
    COMPONENT_MANIFEST_TOP_LEVEL_FIELDS,
    COMPONENT_TARGET_FIELDS,
)


_INTERFACE_SPECS = MappingProxyType({row["name"]: MappingProxyType(dict(row))
                                    for row in COMPONENT_INTERFACE_SPECS})


class ComponentManifestError(ValueError):
    """Structured refusal at the component-manifest trust boundary."""

    def __init__(self, code: str, path: str, message: str, *, evidence: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.evidence = evidence

    def to_data(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": str(self),
            "evidence": _thaw(self.evidence),
        }


def _refuse(code: str, path: str, message: str, *, evidence: Any = None) -> NoReturn:
    raise ComponentManifestError(code, path, message, evidence=evidence)


def _positive_int(value: Any, *, path: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _refuse("invalid_integer", path, f"{path} must be an integer >= {minimum}", evidence=value)
    return value


def _canonical_string(value: Any, *, path: str, pattern: str | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _refuse("invalid_string", path, f"{path} must be a non-empty canonical string",
                evidence=value)
    if pattern is not None and re.fullmatch(pattern, value) is None:
        _refuse("invalid_string", path, f"{path} has a non-canonical spelling", evidence=value)
    canonical_bytes(value)
    return value


def _freeze(value: Any, *, path: str) -> Any:
    """Freeze the shared Python/C++ canonical value vocabulary."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        try:
            canonical_bytes(value)
        except (TypeError, ValueError, OverflowError) as exc:
            _refuse("invalid_canonical_value", path, f"{path}: {exc}", evidence=value)
        return value
    if isinstance(value, float):
        _refuse("float_not_canonical", path,
                f"{path} cannot contain binary floats; use an exact integer or decimal string",
                evidence=value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _refuse("non_string_key", path, f"{path} mapping keys must be strings",
                        evidence=key)
            frozen[key] = _freeze(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, path=f"{path}[]") for item in value)
    _refuse("opaque_value", path, f"{path} contains opaque {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _exact_mapping(value: Any, fields: set[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse("expected_mapping", path, f"{path} must be a mapping", evidence=value)
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown or missing:
        _refuse(
            "semantic_field_mismatch", path,
            f"{path} field mismatch: missing={missing}, unknown={unknown}",
            evidence={"missing": missing, "unknown": unknown},
        )
    return value


def _string_set(value: Any, *, path: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _refuse("expected_sequence", path, f"{path} must be a list or tuple", evidence=value)
    result = tuple(_canonical_string(item, path=f"{path}[]") for item in value)
    result = tuple(sorted(result, key=lambda item: (len(canonical_bytes(item)),
                                                    canonical_bytes(item))))
    if not allow_empty and not result:
        _refuse("empty_sequence", path, f"{path} must not be empty")
    if len(result) != len(set(result)):
        _refuse("duplicate_value", path, f"{path} must contain unique values", evidence=result)
    return result


def _semantic_set(value: Any, *, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        _refuse("expected_sequence", path, f"{path} must be a list or tuple", evidence=value)
    rows = [_freeze(item, path=f"{path}[]") for item in value]
    rows.sort(key=lambda row: (len(canonical_bytes(row)), canonical_bytes(row)))
    encoded = [canonical_bytes(row) for row in rows]
    if len(encoded) != len(set(encoded)):
        _refuse("duplicate_value", path, f"{path} contains duplicate semantic rows")
    return tuple(rows)


@dataclass(frozen=True, slots=True, order=True)
class ComponentVersion:
    major: int
    minor: int = 0
    patch: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "major", _positive_int(self.major, path="version.major",
                                                        allow_zero=True))
        object.__setattr__(self, "minor", _positive_int(self.minor, path="version.minor",
                                                        allow_zero=True))
        object.__setattr__(self, "patch", _positive_int(self.patch, path="version.patch",
                                                        allow_zero=True))

    def to_data(self) -> dict[str, int]:
        return {"major": self.major, "minor": self.minor, "patch": self.patch}

    @classmethod
    def from_value(cls, value: Any) -> ComponentVersion:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value)
            if match is None:
                _refuse("invalid_version", "version", "version must be MAJOR.MINOR.PATCH",
                        evidence=value)
            return cls(*(int(piece) for piece in match.groups()))
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return cls(*value)
        row = _exact_mapping(value, {"major", "minor", "patch"}, path="version")
        return cls(row["major"], row["minor"], row["patch"])

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class ComponentExtensionSchema:
    """Versioned exact-field schema for one semantic extension namespace."""

    uri: str
    version: int
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _component_uri(self.uri, path="extension_schema.uri"))
        object.__setattr__(self, "version", _positive_int(
            self.version, path="extension_schema.version"))
        required = _string_set(self.required_fields, path="extension_schema.required_fields")
        optional = _string_set(self.optional_fields, path="extension_schema.optional_fields")
        overlap = sorted(set(required) & set(optional))
        if overlap:
            _refuse("extension_schema_overlap", "extension_schema",
                    f"extension schema fields are both required and optional: {overlap}")
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "optional_fields", optional)

    @property
    def key(self) -> tuple[str, int]:
        return self.uri, self.version

    def normalize(self, value: Any, *, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            _refuse("expected_mapping", path, f"{path} must be a mapping")
        unknown = sorted(set(value) - set(self.required_fields) - set(self.optional_fields))
        missing = sorted(set(self.required_fields) - set(value))
        if unknown or missing:
            _refuse("semantic_extension_field_mismatch", path,
                    f"{path} field mismatch: missing={missing}, unknown={unknown}",
                    evidence={"schema_uri": self.uri, "schema_version": self.version,
                              "missing": missing, "unknown": unknown})
        return _freeze(value, path=path)


def _component_uri(value: Any, *, path: str = "uri") -> str:
    uri = _canonical_string(value, path=path)
    try:
        parsed = urlsplit(uri)
    except ValueError:
        _refuse("invalid_component_uri", path,
                f"{path} must be an absolute namespaced URI with an authority", evidence=value)
    if (re.match(r"^[a-z][a-z0-9+.-]*://", uri) is None or not parsed.netloc
            or any(character.isspace() for character in uri)):
        _refuse("invalid_component_uri", path,
                f"{path} must be an absolute namespaced URI with an authority", evidence=value)
    if parsed.query or parsed.fragment:
        _refuse("invalid_component_uri", path,
                f"{path} must not contain a query or fragment", evidence=value)
    return uri


def _target(value: Any) -> Mapping[str, Any]:
    row = _exact_mapping(value, set(COMPONENT_TARGET_FIELDS), path="target")
    variants = row["variants"]
    if not isinstance(variants, (list, tuple)):
        _refuse("expected_sequence", "target.variants", "target.variants must be a sequence")
    normalized = []
    for index, variant in enumerate(variants):
        path = f"target.variants[{index}]"
        fields = _exact_mapping(
            variant, {"dimension", "scalar", "device", "features"}, path=path)
        normalized.append(MappingProxyType({
            "dimension": _positive_int(fields["dimension"], path=f"{path}.dimension"),
            "scalar": _canonical_string(fields["scalar"], path=f"{path}.scalar"),
            "device": _canonical_string(fields["device"], path=f"{path}.device"),
            "features": _string_set(fields["features"], path=f"{path}.features"),
        }))
    normalized.sort(key=lambda item: (len(canonical_bytes(_thaw(item))),
                                      canonical_bytes(_thaw(item))))
    encoded = [canonical_bytes(_thaw(item)) for item in normalized]
    if len(encoded) != len(set(encoded)):
        _refuse("duplicate_value", "target.variants", "target.variants contains duplicates")
    return MappingProxyType({"variants": tuple(normalized)})


def _determinism(value: Any) -> Mapping[str, Any]:
    row = _exact_mapping(value, {"classification", "scope"}, path="determinism")
    classification = _canonical_string(row["classification"], path="determinism.classification")
    valid = {"unspecified", "bitwise", "reproducible", "statistical", "nondeterministic"}
    if classification not in valid:
        _refuse("invalid_determinism", "determinism.classification",
                f"determinism.classification must be one of {sorted(valid)}",
                evidence=classification)
    return MappingProxyType({
        "classification": classification,
        "scope": _string_set(row["scope"], path="determinism.scope"),
    })


def _restart(value: Any) -> Mapping[str, Any]:
    row = _exact_mapping(value, {"mode", "schema_uri", "schema_version"}, path="restart")
    mode = _canonical_string(row["mode"], path="restart.mode")
    if mode not in {"stateless", "stateful", "unsupported"}:
        _refuse("invalid_restart_mode", "restart.mode", "invalid restart mode", evidence=mode)
    schema_uri = row["schema_uri"]
    schema_version = row["schema_version"]
    if mode == "stateful":
        schema_uri = _component_uri(schema_uri, path="restart.schema_uri")
        schema_version = _positive_int(schema_version, path="restart.schema_version")
    else:
        if schema_uri != "" or schema_version != 0:
            _refuse("restart_schema_without_state", "restart",
                    "stateless/unsupported restart must use an empty schema_uri and version 0")
    return MappingProxyType({
        "mode": mode, "schema_uri": schema_uri, "schema_version": schema_version,
    })


def _precision(value: Any) -> Mapping[str, Any]:
    row = _exact_mapping(value, {"inputs", "accumulation", "outputs"}, path="precision")
    return MappingProxyType({
        "inputs": _string_set(row["inputs"], path="precision.inputs"),
        "accumulation": _canonical_string(row["accumulation"], path="precision.accumulation"),
        "outputs": _string_set(row["outputs"], path="precision.outputs"),
    })


def _entry_points(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        _refuse("expected_mapping", "entry_points", "entry_points must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        key = _canonical_string(key, path="entry_points key",
                                pattern=r"[a-z][a-z0-9_.-]*")
        result[key] = _canonical_string(item, path=f"entry_points.{key}")
    return MappingProxyType(result)


def _interfaces(value: Any, facets: tuple[str, ...],
                entry_points: Mapping[str, str]) -> tuple[Any, ...]:
    """Normalize exact small-interface bindings and enforce facet closure.

    A component never relies on method-name guessing.  Every advertised facet has one binding:
    ``method`` and ``value`` bind a named member on a source component, while ``entry_point`` binds
    a declared native/AOT entry point.  The same declaration drives Python and native adapters.
    """
    rows = _semantic_set(value, path="interfaces")
    normalized: list[Mapping[str, str]] = []
    names: set[str] = set()
    for index, item in enumerate(rows):
        path = f"interfaces[{index}]"
        row = _exact_mapping(item, {"name", "mode", "binding"}, path=path)
        name = _canonical_string(row["name"], path=f"{path}.name",
                                 pattern=r"[a-z][a-z0-9_]*")
        if name not in _INTERFACE_SPECS:
            _refuse("unknown_component_interface", f"{path}.name",
                    f"unknown component interface {name!r}", evidence=name)
        if name in names:
            _refuse("duplicate_component_interface", f"{path}.name",
                    f"component interface {name!r} is declared more than once")
        names.add(name)
        mode = _canonical_string(row["mode"], path=f"{path}.mode")
        if mode not in {"method", "value", "entry_point"}:
            _refuse("invalid_interface_mode", f"{path}.mode",
                    "interface mode must be method, value, or entry_point", evidence=mode)
        binding = _canonical_string(row["binding"], path=f"{path}.binding",
                                    pattern=r"[A-Za-z_][A-Za-z0-9_]*")
        if mode == "entry_point" and binding not in entry_points:
            _refuse("missing_interface_entry_point", f"{path}.binding",
                    f"interface {name!r} binds undeclared entry point {binding!r}",
                    evidence={"declared": sorted(entry_points)})
        normalized.append(MappingProxyType({
            "name": name, "mode": mode, "binding": binding,
        }))
    facet_names = set(facets)
    if names != facet_names:
        _refuse(
            "interface_facet_mismatch", "interfaces",
            "facets and interface declarations must name the same exact set",
            evidence={"missing": sorted(facet_names - names),
                      "undeclared_facets": sorted(names - facet_names)},
        )
    normalized.sort(key=lambda row: (len(canonical_bytes(row)), canonical_bytes(row)))
    return tuple(normalized)


def _extension_registry(value: Any) -> dict[tuple[str, int], ComponentExtensionSchema]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _refuse("invalid_extension_registry", "extension_schemas",
                "extension_schemas must be a mapping")
    result: dict[tuple[str, int], ComponentExtensionSchema] = {}
    for supplied_key, schema in value.items():
        if not isinstance(schema, ComponentExtensionSchema):
            _refuse("invalid_extension_schema", "extension_schemas",
                    "extension_schemas values must be ComponentExtensionSchema objects")
        if supplied_key not in (schema.key, f"{schema.uri}@{schema.version}"):
            _refuse("extension_schema_key_mismatch", "extension_schemas",
                    f"extension schema key {supplied_key!r} does not identify {schema.key!r}")
        result[schema.key] = schema
    return result


def _extensions(value: Any, schemas: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse("expected_mapping", "extensions", "extensions must be a mapping")
    registry = _extension_registry(schemas)
    result: dict[str, Any] = {}
    for namespace, extension in value.items():
        namespace = _component_uri(namespace, path="extensions namespace")
        if not isinstance(extension, Mapping):
            _refuse("expected_mapping", f"extensions.{namespace}", "extension must be a mapping")
        kind = extension.get("kind")
        if kind not in COMPONENT_EXTENSION_KINDS:
            _refuse("unknown_extension_kind", f"extensions.{namespace}.kind",
                    f"extension kind must be one of {COMPONENT_EXTENSION_KINDS}", evidence=kind)
        if kind == "documentary":
            row = _exact_mapping(extension, {"kind", "data"}, path=f"extensions.{namespace}")
            result[namespace] = MappingProxyType({
                "kind": kind,
                "data": _freeze(row["data"], path=f"extensions.{namespace}.data"),
            })
            continue
        row = _exact_mapping(extension, {"kind", "schema_uri", "schema_version", "data"},
                             path=f"extensions.{namespace}")
        schema_uri = _component_uri(row["schema_uri"],
                                    path=f"extensions.{namespace}.schema_uri")
        schema_version = _positive_int(row["schema_version"],
                                       path=f"extensions.{namespace}.schema_version")
        schema = registry.get((schema_uri, schema_version))
        if schema is None:
            _refuse(
                "unknown_semantic_extension_schema", f"extensions.{namespace}",
                "semantic extension has no registered versioned schema",
                evidence={"schema_uri": schema_uri, "schema_version": schema_version},
            )
        result[namespace] = MappingProxyType({
            "kind": kind, "schema_uri": schema_uri, "schema_version": schema_version,
            "data": schema.normalize(row["data"], path=f"extensions.{namespace}.data"),
        })
    return MappingProxyType(dict(sorted(result.items())))


def _default_target() -> dict[str, Any]:
    return {"variants": []}


def _default_determinism() -> dict[str, Any]:
    return {"classification": "unspecified", "scope": []}


def _default_restart() -> dict[str, Any]:
    return {"mode": "stateless", "schema_uri": "", "schema_version": 0}


def _default_precision() -> dict[str, Any]:
    return {"inputs": [], "accumulation": "unspecified", "outputs": []}


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
            "version": self.version.to_data(),
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
