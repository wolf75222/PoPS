#!/usr/bin/env python3
"""Generate the Python/C++ component catalog from one versioned declaration.

The checked-in products are intentional: package imports and C++ consumers do not need a
generator at runtime.  ``--check`` is the CI non-drift gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pprint
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "schemas" / "component_catalog.v2.json"
PY_SCHEMA = ROOT / "python" / "pops" / "model" / "_generated_component_schema.py"
PY_ROUTES = ROOT / "python" / "pops" / "runtime" / "_generated_component_routes.py"
CPP_CATALOG = ROOT / "include" / "pops" / "runtime" / "config" / "generated_component_catalog.hpp"
CPP_ACCESSORS = ROOT / "include" / "pops" / "runtime" / "config" / "generated_route_accessors.inc"

MANIFEST_SEMANTIC_FIELDS = (
    "schema_version", "uri", "component_type", "version", "facets", "signature", "reads",
    "writes", "parameters", "interfaces", "requirements", "capabilities", "effects",
    "layouts", "clocks", "target", "determinism", "restart", "precision", "conservation",
    "entry_points",
)
MANIFEST_TOP_LEVEL_FIELDS = MANIFEST_SEMANTIC_FIELDS + ("extensions", "digests")
TARGET_FIELDS = ("variants",)
DIGEST_FIELDS = ("semantic", "manifest")


class CatalogError(ValueError):
    pass


def _exact(value: Any, fields: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{where} must be an object")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown or missing:
        raise CatalogError(f"{where} field mismatch: missing={missing}, unknown={unknown}")
    return value


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise CatalogError(f"{where} must be a C++ identifier")
    return value


def _strings(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise CatalogError(f"{where} must be a list of canonical non-empty strings")
    if len(value) != len(set(value)):
        raise CatalogError(f"{where} contains duplicates")
    return value


def _catalog_digests(data: dict[str, Any]) -> tuple[str, str]:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    semantic = copy.deepcopy(data)
    for family in semantic["route_families"]:
        for route in family["routes"]:
            route.pop("limitations", None)
            route["metadata"].pop("summary", None)
    semantic_canonical = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), hashlib.sha256(semantic_canonical).hexdigest()


def _load_catalog() -> tuple[dict[str, Any], str, str]:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    _exact(data, {
        "catalog_schema_version", "component_manifest_schema_version",
        "route_registry_version", "capability_vocabulary_version", "manifest_schema",
        "route_component_defaults", "route_families",
    }, "component catalog")
    if data["catalog_schema_version"] != 1:
        raise CatalogError("unsupported component catalog schema_version")
    for name in (
        "component_manifest_schema_version", "route_registry_version",
        "capability_vocabulary_version",
    ):
        if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] < 1:
            raise CatalogError(f"{name} must be an integer >= 1")
    if data["component_manifest_schema_version"] != 2:
        raise CatalogError("this generator implements ComponentManifest schema version 2")
    schema = _exact(data["manifest_schema"], {
        "semantic_fields", "top_level_fields", "target_fields", "digest_fields",
        "extension_kinds",
    }, "manifest_schema")
    for field in schema:
        _strings(schema[field], f"manifest_schema.{field}")
    if tuple(schema["semantic_fields"]) != MANIFEST_SEMANTIC_FIELDS:
        raise CatalogError("manifest_schema.semantic_fields does not match implemented schema v2")
    if tuple(schema["top_level_fields"]) != MANIFEST_TOP_LEVEL_FIELDS:
        raise CatalogError("manifest_schema.top_level_fields does not match implemented schema v2")
    if tuple(schema["target_fields"]) != TARGET_FIELDS:
        raise CatalogError("manifest_schema.target_fields does not match implemented schema v2")
    if tuple(schema["digest_fields"]) != DIGEST_FIELDS:
        raise CatalogError("manifest_schema.digest_fields does not match implemented schema v2")
    if schema["extension_kinds"] != ["documentary", "semantic"]:
        raise CatalogError("extension_kinds must be documentary, semantic")

    defaults = _exact(data["route_component_defaults"], {
        "version", "facets", "signature", "reads", "writes", "parameters", "interfaces",
        "effects", "layouts", "clocks", "target", "determinism", "restart", "precision",
        "conservation", "extensions",
    }, "route_component_defaults")
    version = _exact(defaults["version"], {"major", "minor", "patch"},
                     "route_component_defaults.version")
    for name, value in version.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CatalogError(f"route_component_defaults.version.{name} must be an integer >= 0")
    for name in (
        "facets", "reads", "writes", "parameters", "interfaces", "effects", "layouts",
        "clocks", "conservation",
    ):
        if not isinstance(defaults[name], list):
            raise CatalogError(f"route_component_defaults.{name} must be a list")
    if not isinstance(defaults["signature"], dict):
        raise CatalogError("route_component_defaults.signature must be an object")
    target = _exact(defaults["target"], set(TARGET_FIELDS), "route_component_defaults.target")
    variants = target["variants"]
    if not isinstance(variants, list) or not variants:
        raise CatalogError("route_component_defaults.target.variants must be a non-empty list")
    normalized_variants = []
    for index, variant in enumerate(variants):
        variant = _exact(variant, {"dimension", "scalar", "device", "features"},
                         f"route_component_defaults.target.variants[{index}]")
        dimension = variant["dimension"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise CatalogError("route target variant dimension must be an integer >= 1")
        for name in ("scalar", "device"):
            value = variant[name]
            if not isinstance(value, str) or not value or value != value.strip():
                raise CatalogError(f"route target variant {name} must be canonical text")
        _strings(variant["features"],
                 f"route_component_defaults.target.variants[{index}].features")
        normalized_variants.append(json.dumps(variant, sort_keys=True, separators=(",", ":")))
    if len(normalized_variants) != len(set(normalized_variants)):
        raise CatalogError("route_component_defaults.target.variants contains duplicates")
    determinism = _exact(defaults["determinism"], {"classification", "scope"},
                         "route_component_defaults.determinism")
    if determinism["classification"] not in {
        "unspecified", "bitwise", "reproducible", "statistical", "nondeterministic",
    }:
        raise CatalogError("route_component_defaults.determinism.classification is invalid")
    _strings(determinism["scope"], "route_component_defaults.determinism.scope")
    restart = _exact(defaults["restart"], {"mode", "schema_uri", "schema_version"},
                     "route_component_defaults.restart")
    if restart != {"mode": "stateless", "schema_uri": "", "schema_version": 0}:
        raise CatalogError("builtin route defaults must be stateless")
    precision = _exact(defaults["precision"], {"inputs", "accumulation", "outputs"},
                       "route_component_defaults.precision")
    _strings(precision["inputs"], "route_component_defaults.precision.inputs")
    _strings(precision["outputs"], "route_component_defaults.precision.outputs")
    if not isinstance(precision["accumulation"], str) or not precision["accumulation"]:
        raise CatalogError("route_component_defaults.precision.accumulation must be non-empty")
    if defaults["extensions"] != {}:
        raise CatalogError("builtin route defaults cannot inject undeclared extensions")

    families = data["route_families"]
    if not isinstance(families, list) or not families:
        raise CatalogError("route_families must be a non-empty list")
    if len(families) > 255:
        raise CatalogError("route_families exceeds the uint8 RouteFamily wire space")
    seen_families: set[str] = set()
    for family_index, family in enumerate(families):
        family = _exact(family, {"name", "cpp_enum", "cpp_table", "routes"},
                        f"route_families[{family_index}]")
        name = family["name"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise CatalogError(f"route_families[{family_index}].name is not canonical")
        if name in seen_families:
            raise CatalogError(f"duplicate route family {name!r}")
        seen_families.add(name)
        _identifier(family["cpp_enum"], f"{name}.cpp_enum")
        _identifier(family["cpp_table"], f"{name}.cpp_table")
        routes = family["routes"]
        if not isinstance(routes, list) or not routes:
            raise CatalogError(f"{name}.routes must be non-empty")
        if len(routes) > 255:
            raise CatalogError(f"{name}.routes uses reserved wire id 255")
        tokens: set[str] = set()
        aliases: set[str] = set()
        cpp_ids: set[str] = set()
        for index, route in enumerate(routes):
            route = _exact(route, {
                "token", "wire_id", "cpp_id", "native_entry", "requirements",
                "limitations", "aliases", "metadata",
            }, f"{name}.routes[{index}]")
            token = route["token"]
            if not isinstance(token, str) or re.fullmatch(r"[a-z][a-z0-9_]*", token) is None:
                raise CatalogError(f"{name}.routes[{index}].token is not canonical")
            if token in tokens or token in aliases:
                raise CatalogError(f"duplicate route token {name}.{token}")
            tokens.add(token)
            if isinstance(route["wire_id"], bool) or not isinstance(route["wire_id"], int) \
                    or route["wire_id"] != index:
                raise CatalogError(f"{name}.{token} wire_id must equal its stable position {index}")
            cpp_id = _identifier(route["cpp_id"], f"{name}.{token}.cpp_id")
            if cpp_id in cpp_ids:
                raise CatalogError(f"duplicate C++ route id {family['cpp_enum']}::{cpp_id}")
            cpp_ids.add(cpp_id)
            if not isinstance(route["native_entry"], str) or not route["native_entry"] \
                    or route["native_entry"] != route["native_entry"].strip():
                raise CatalogError(f"{name}.{token}.native_entry must be canonical and non-empty")
            _strings(route["requirements"], f"{name}.{token}.requirements")
            _strings(route["limitations"], f"{name}.{token}.limitations")
            for alias in _strings(route["aliases"], f"{name}.{token}.aliases"):
                if re.fullmatch(r"[a-z][a-z0-9_]*", alias) is None:
                    raise CatalogError(f"route alias {name}.{alias} is not canonical")
                if alias in tokens or alias in aliases:
                    raise CatalogError(f"duplicate route alias {name}.{alias}")
                aliases.add(alias)
            if not isinstance(route["metadata"], dict):
                raise CatalogError(f"{name}.{token}.metadata must be an object")
            metadata_fields = {
                "riemann": {"needs_wave_speeds", "needs_hllc_struct", "needs_roe_diss", "polar_ok"},
                "limiter": {"n_ghost"},
                "transport": {"n_vars", "polar_ok", "parameters", "summary"},
                "source": {"min_vars", "parameters", "summary"},
                "elliptic": {"parameters", "summary"},
            }.get(name, set())
            _exact(route["metadata"], metadata_fields, f"{name}.{token}.metadata")
            if "parameters" in route["metadata"]:
                _strings(route["metadata"]["parameters"], f"{name}.{token}.metadata.parameters")
            for key in ("n_ghost", "n_vars", "min_vars"):
                if key in route["metadata"] and (
                    isinstance(route["metadata"][key], bool)
                    or not isinstance(route["metadata"][key], int)
                    or route["metadata"][key] < 1
                ):
                    raise CatalogError(f"{name}.{token}.metadata.{key} must be an integer >= 1")
            for key in ("polar_ok", "needs_wave_speeds", "needs_hllc_struct", "needs_roe_diss"):
                if key in route["metadata"] and not isinstance(route["metadata"][key], bool):
                    raise CatalogError(f"{name}.{token}.metadata.{key} must be boolean")
            if "summary" in route["metadata"] and (
                not isinstance(route["metadata"]["summary"], str)
                or not route["metadata"]["summary"]
            ):
                raise CatalogError(f"{name}.{token}.metadata.summary must be non-empty")

    required_specializations = {"riemann", "limiter", "transport", "source", "elliptic"}
    missing_specializations = sorted(required_specializations - seen_families)
    if missing_specializations:
        raise CatalogError(f"component catalog misses generated typed views {missing_specializations}")

    full_digest, semantic_digest = _catalog_digests(data)
    return data, full_digest, semantic_digest


def _py_literal(value: Any) -> str:
    return pprint.pformat(value, width=100, sort_dicts=False)


def _render_schema(catalog: dict[str, Any], digest: str, semantic_digest: str | None = None) -> str:
    semantic_digest = semantic_digest or digest
    schema = catalog["manifest_schema"]
    lines = [
        '"""Generated by scripts/generate_component_catalog.py; DO NOT EDIT."""',
        "from __future__ import annotations",
        "",
        f"COMPONENT_CATALOG_SCHEMA_VERSION = {catalog['catalog_schema_version']}",
        f"COMPONENT_MANIFEST_SCHEMA_VERSION = {catalog['component_manifest_schema_version']}",
        f"COMPONENT_CATALOG_SHA256 = {digest!r}",
        f"COMPONENT_CATALOG_SEMANTIC_SHA256 = {semantic_digest!r}",
    ]
    for name, source_name in (
        ("COMPONENT_MANIFEST_SEMANTIC_FIELDS", "semantic_fields"),
        ("COMPONENT_MANIFEST_TOP_LEVEL_FIELDS", "top_level_fields"),
        ("COMPONENT_TARGET_FIELDS", "target_fields"),
        ("COMPONENT_DIGEST_FIELDS", "digest_fields"),
        ("COMPONENT_EXTENSION_KINDS", "extension_kinds"),
    ):
        lines.append(f"{name} = {tuple(schema[source_name])!r}")
    lines.extend(("", "__all__ = [name for name in globals() if name.startswith('COMPONENT_')]", ""))
    return "\n".join(lines)


def _render_routes(catalog: dict[str, Any], digest: str,
                   semantic_digest: str | None = None) -> str:
    semantic_digest = semantic_digest or digest
    tables: dict[str, tuple[Any, ...]] = {}
    aliases: dict[str, dict[str, str]] = {}
    metadata: dict[str, dict[str, dict[str, Any]]] = {}
    cpp: dict[str, dict[str, Any]] = {}
    brick_rows: list[dict[str, Any]] = []
    for family in catalog["route_families"]:
        name = family["name"]
        tables[name] = tuple((
            row["token"], row["native_entry"], tuple(row["requirements"]),
            tuple(row["limitations"]),
        ) for row in family["routes"])
        aliases[name] = {
            alias: row["token"] for row in family["routes"] for alias in row["aliases"]
        }
        metadata[name] = {row["token"]: row["metadata"] for row in family["routes"]}
        cpp[name] = {
            "enum": family["cpp_enum"], "table": family["cpp_table"],
            "ids": tuple(row["cpp_id"] for row in family["routes"]),
        }
        if name in {"transport", "source", "elliptic"}:
            for row in family["routes"]:
                meta = row["metadata"]
                brick_rows.append({
                    "category": name,
                    "id": row["token"],
                    "route_index": row["wire_id"],
                    "native_entry": row["native_entry"],
                    "parameters": tuple(meta["parameters"]),
                    "n_vars": meta.get("n_vars", meta.get("min_vars", -1)),
                    "polar_ok": bool(meta.get("polar_ok", False)),
                    "requirements": tuple(row["requirements"]),
                    "limitations": tuple(row["limitations"]),
                    "summary": meta["summary"],
                })
    signature = f"v{catalog['route_registry_version']}:{semantic_digest}"
    values = {
        "COMPONENT_CATALOG_SCHEMA_VERSION": catalog["catalog_schema_version"],
        "COMPONENT_MANIFEST_SCHEMA_VERSION": catalog["component_manifest_schema_version"],
        "ROUTE_REGISTRY_VERSION": catalog["route_registry_version"],
        "CAPABILITY_VOCAB_VERSION": catalog["capability_vocabulary_version"],
        "COMPONENT_CATALOG_SHA256": digest,
        "COMPONENT_CATALOG_SEMANTIC_SHA256": semantic_digest,
        "ROUTE_REGISTRY_SIGNATURE": signature,
        "ROUTE_TABLES": tables,
        "ROUTE_ALIASES": aliases,
        "ROUTE_METADATA": metadata,
        "ROUTE_CPP_BINDINGS": cpp,
        "ROUTE_COMPONENT_DEFAULTS": catalog["route_component_defaults"],
        "BRICK_CATALOG_ROWS": tuple(brick_rows),
    }
    lines = [
        '"""Generated by scripts/generate_component_catalog.py; DO NOT EDIT."""',
        "from __future__ import annotations",
        "",
    ]
    for name, value in values.items():
        lines.append(f"{name} = {_py_literal(value)}")
        lines.append("")
    lines.append("__all__ = [name for name in globals() if name.startswith(('ROUTE_', 'COMPONENT_', 'CAPABILITY_', 'BRICK_'))]")
    lines.append("")
    return "\n".join(lines)


def _cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _camel(value: str) -> str:
    return "".join(piece.capitalize() for piece in value.split("_"))


def _render_cpp(catalog: dict[str, Any], digest: str,
                semantic_digest: str | None = None) -> str:
    semantic_digest = semantic_digest or digest
    families = catalog["route_families"]
    signature = f"v{catalog['route_registry_version']}:{semantic_digest}"
    out = [
        "#pragma once",
        "",
        "// Generated by scripts/generate_component_catalog.py; DO NOT EDIT.",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace pops {",
        "",
        "struct RouteInfo {",
        "  int index;",
        "  const char* token;",
        "  const char* native_entry;",
        "  const char* requirements;",
        "  const char* limitations;",
        "};",
        "",
        "enum class RouteFamily : std::uint8_t {",
    ]
    for index, family in enumerate(families):
        out.append(f"  k{_camel(family['name'])} = {index},")
    out.extend(("};", "", "constexpr const char* route_family_name(RouteFamily family) {", "  switch (family) {"))
    for family in families:
        out.append(f"    case RouteFamily::k{_camel(family['name'])}: return {_cpp_string(family['name'])};")
    out.extend(("  }", '  return "unknown";', "}", ""))

    for family in families:
        enum = family["cpp_enum"]
        out.append(f"enum class {enum} : int {{")
        for row in family["routes"]:
            out.append(f"  {row['cpp_id']} = {row['wire_id']},")
        out.extend(("};", f"inline constexpr RouteInfo {family['cpp_table']}[] = {{"))
        for row in family["routes"]:
            requirements = ",".join(row["requirements"])
            limitations = ",".join(row["limitations"])
            out.append(
                f"  {{{row['wire_id']}, {_cpp_string(row['token'])}, "
                f"{_cpp_string(row['native_entry'])}, {_cpp_string(requirements)}, "
                f"{_cpp_string(limitations)}}},"
            )
        route_csv = "|".join(row["token"] for row in family["routes"])
        out.extend((
            "};",
            f"inline constexpr const char* k{_camel(family['name'])}RouteTokensCsv = "
            f"{_cpp_string(route_csv)};",
            "",
        ))

    out.extend((
        "struct RouteAliasInfo {",
        "  RouteFamily family;",
        "  const char* alias;",
        "  int canonical_index;",
        "};",
        "inline constexpr RouteAliasInfo kRouteAliases[] = {",
    ))
    for family in families:
        for row in family["routes"]:
            for alias in row["aliases"]:
                out.append(
                    f"  {{RouteFamily::k{_camel(family['name'])}, {_cpp_string(alias)}, "
                    f"{row['wire_id']}}},"
                )
    out.extend(("};", ""))

    by_name = {family["name"]: family for family in families}
    limiter = by_name["limiter"]
    out.extend((
        "struct LimiterTag { const char* name; int n_ghost; };",
        "inline constexpr LimiterTag kLimiters[] = {",
    ))
    for row in limiter["routes"]:
        out.append(f"  {{{_cpp_string(row['token'])}, {row['metadata']['n_ghost']}}},")
    out.extend(("};", ""))

    riemann = by_name["riemann"]
    out.extend((
        "struct RiemannTag {",
        "  const char* name; bool needs_wave_speeds; bool needs_hllc_struct;",
        "  bool needs_roe_diss; bool polar_ok;",
        "};",
        "inline constexpr RiemannTag kRiemanns[] = {",
    ))
    for row in riemann["routes"]:
        m = row["metadata"]
        flags = ["true" if m[key] else "false" for key in (
            "needs_wave_speeds", "needs_hllc_struct", "needs_roe_diss", "polar_ok")]
        out.append(f"  {{{_cpp_string(row['token'])}, {', '.join(flags)}}},")
    out.extend(("};", ""))

    transport = by_name["transport"]
    out.extend((
        "struct TransportTag { const char* name; int n_vars; bool polar_ok; const char* summary; };",
        "inline constexpr TransportTag kTransports[] = {",
    ))
    for row in transport["routes"]:
        m = row["metadata"]
        polar = "true" if m["polar_ok"] else "false"
        out.append(
            f"  {{{_cpp_string(row['token'])}, {m['n_vars']}, {polar}, "
            f"{_cpp_string(m['summary'])}}},"
        )
    out.extend(("};", ""))

    source = by_name["source"]
    out.extend((
        "struct SourceTag { const char* name; int min_vars; const char* summary; };",
        "inline constexpr SourceTag kSources[] = {",
    ))
    for row in source["routes"]:
        m = row["metadata"]
        out.append(f"  {{{_cpp_string(row['token'])}, {m['min_vars']}, {_cpp_string(m['summary'])}}},")
        for alias in row["aliases"]:
            out.append(
                f"  {{{_cpp_string(alias)}, {m['min_vars']}, "
                f"{_cpp_string('alias of ' + repr(row['token']))}}},"
            )
    out.extend(("};", ""))

    elliptic = by_name["elliptic"]
    out.extend((
        "struct EllipticTag { const char* name; const char* summary; };",
        "inline constexpr EllipticTag kElliptics[] = {",
    ))
    for row in elliptic["routes"]:
        out.append(f"  {{{_cpp_string(row['token'])}, {_cpp_string(row['metadata']['summary'])}}},")
    out.extend(("};", ""))

    out.extend((
        "struct BrickCatalogEntry {",
        "  const char* id; const char* category; int route_index; const char* native_entry;",
        "  const char* parameters; const char* parameters_json; int n_vars; bool polar_ok;",
        "  const char* requirements; const char* requirements_json;",
        "  const char* limitations; const char* limitations_json; const char* summary;",
        "};",
        "inline constexpr BrickCatalogEntry kBrickCatalog[] = {",
    ))
    for category in ("transport", "source", "elliptic"):
        for row in by_name[category]["routes"]:
            metadata = row["metadata"]
            n_vars = metadata.get("n_vars", metadata.get("min_vars", -1))
            polar = "true" if metadata.get("polar_ok", False) else "false"
            values = (
                row["token"], category, row["wire_id"], row["native_entry"],
                ",".join(metadata["parameters"]),
                json.dumps(metadata["parameters"], ensure_ascii=True, separators=(",", ":")),
                n_vars, polar,
                ",".join(row["requirements"]),
                json.dumps(row["requirements"], ensure_ascii=True, separators=(",", ":")),
                ",".join(row["limitations"]),
                json.dumps(row["limitations"], ensure_ascii=True, separators=(",", ":")),
                metadata["summary"],
            )
            out.append(
                f"  {{{_cpp_string(values[0])}, {_cpp_string(values[1])}, {values[2]}, "
                f"{_cpp_string(values[3])}, {_cpp_string(values[4])}, {_cpp_string(values[5])}, "
                f"{values[6]}, {values[7]}, {_cpp_string(values[8])}, {_cpp_string(values[9])}, "
                f"{_cpp_string(values[10])}, {_cpp_string(values[11])}, {_cpp_string(values[12])}}},"
            )
    out.extend(("};", ""))

    schema = catalog["manifest_schema"]
    out.extend((
        f"inline constexpr int kComponentCatalogSchemaVersion = {catalog['catalog_schema_version']};",
        f"inline constexpr int kComponentManifestSchemaVersion = {catalog['component_manifest_schema_version']};",
        f"inline constexpr int kRouteRegistryVersion = {catalog['route_registry_version']};",
        f"inline constexpr int kCapabilityVocabularyVersion = {catalog['capability_vocabulary_version']};",
        f"inline constexpr const char* kComponentCatalogSha256 = {_cpp_string(digest)};",
        f"inline constexpr const char* kComponentCatalogSemanticSha256 = "
        f"{_cpp_string(semantic_digest)};",
        f"inline constexpr const char* kRouteRegistrySignature = {_cpp_string(signature)};",
        "inline constexpr const char* kComponentManifestSemanticFields[] = {",
    ))
    for field in schema["semantic_fields"]:
        out.append(f"  {_cpp_string(field)},")
    out.extend(("};", "inline constexpr const char* kComponentManifestTopLevelFields[] = {"))
    for field in schema["top_level_fields"]:
        out.append(f"  {_cpp_string(field)},")
    out.extend(("};", "inline constexpr const char* kComponentTargetFields[] = {"))
    for field in schema["target_fields"]:
        out.append(f"  {_cpp_string(field)},")
    out.extend(("};", "inline constexpr const char* kComponentDigestFields[] = {"))
    for field in schema["digest_fields"]:
        out.append(f"  {_cpp_string(field)},")
    out.extend(("};", "", "}  // namespace pops", ""))
    return "\n".join(out)


def _render_accessors(catalog: dict[str, Any], digest: str) -> str:
    lines = [
        f"// Generated from component catalog {digest}; DO NOT EDIT.",
        "// POPS_DEFINE_ROUTE_ACCESSORS must be defined by the including behavior header.",
    ]
    for family in catalog["route_families"]:
        lines.append(
            f"POPS_DEFINE_ROUTE_ACCESSORS({family['name']}, {family['cpp_enum']}, "
            f"{family['cpp_table']}, k{_camel(family['name'])});"
        )
    lines.append("")
    return "\n".join(lines)


def _update(path: Path, content: str, *, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    if check:
        print(f"generated component artifact diverged: {path.relative_to(ROOT)}", file=sys.stderr)
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog, digest, semantic_digest = _load_catalog()
    products = {
        PY_SCHEMA: _render_schema(catalog, digest, semantic_digest),
        PY_ROUTES: _render_routes(catalog, digest, semantic_digest),
        CPP_CATALOG: _render_cpp(catalog, digest, semantic_digest),
        CPP_ACCESSORS: _render_accessors(catalog, digest),
    }
    changed = any([
        _update(path, content, check=args.check) for path, content in products.items()
    ])
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
