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
PY_INTERFACES = ROOT / "python" / "pops" / "_generated_component_interfaces.py"
CPP_CATALOG = ROOT / "include" / "pops" / "runtime" / "config" / "generated_component_catalog.hpp"
CPP_ACCESSORS = ROOT / "include" / "pops" / "runtime" / "config" / "generated_route_accessors.inc"
CPP_COMPONENT_ABI = ROOT / "include" / "pops" / "runtime" / "config" / "generated_component_abi.hpp"
CPP_PYBIND_INVOKERS = (ROOT / "python" / "bindings" / "core" / "init" /
                       "generated_component_invokers.inc")

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
        "interface_vocabulary", "native_interface_abi_version", "native_common_abi_version",
        "tagging_program_abi",
        "native_interface_abis", "boundary_handle_native_routes",
        "route_family_native_interfaces", "route_family_interfaces",
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

    vocabulary = data["interface_vocabulary"]
    if not isinstance(vocabulary, list) or not vocabulary:
        raise CatalogError("interface_vocabulary must be a non-empty list")
    interface_names: set[str] = set()
    for index, interface in enumerate(vocabulary):
        interface = _exact(interface, {"name", "method", "required_args"},
                           f"interface_vocabulary[{index}]")
        name = interface["name"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise CatalogError(f"interface_vocabulary[{index}].name is not canonical")
        if name in interface_names:
            raise CatalogError(f"duplicate component interface {name!r}")
        interface_names.add(name)
        _identifier(interface["method"], f"interface_vocabulary[{index}].method")
        required_args = interface["required_args"]
        if isinstance(required_args, bool) or not isinstance(required_args, int) \
                or required_args < 0:
            raise CatalogError(
                f"interface_vocabulary[{index}].required_args must be an integer >= 0")

    for name, label in (
        ("native_interface_abi_version", "interface"),
        ("native_common_abi_version", "common"),
    ):
        if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] != 1:
            raise CatalogError(
                f"unsupported native component {label} ABI version")
    tagging = _exact(data["tagging_program_abi"], {
        "version", "leaf_opcodes", "logical_opcodes", "candidate_outputs",
        "indicator_stencil_routes", "maximum_stencil_terms",
        "maximum_instruction_count", "non_finite_policy", "persistent_hysteresis",
    }, "tagging_program_abi")
    if tagging["version"] != 1 or tagging["persistent_hysteresis"] is not False:
        raise CatalogError(
            "tagging_program_abi v1 requires explicit non-persistent hysteresis")
    if tagging["non_finite_policy"] != "reject":
        raise CatalogError(
            "tagging_program_abi v1 requires fail-closed non-finite rejection")
    opcode_ids: set[int] = set()
    for family in ("leaf_opcodes", "logical_opcodes"):
        values = tagging[family]
        if not isinstance(values, dict) or not values:
            raise CatalogError(f"tagging_program_abi.{family} must be a non-empty mapping")
        for name, opcode in values.items():
            _identifier(name, f"tagging_program_abi.{family} opcode")
            if isinstance(opcode, bool) or not isinstance(opcode, int) \
                    or opcode < 1 or opcode > 127 or opcode in opcode_ids:
                raise CatalogError(f"tagging_program_abi.{family}.{name} has an invalid id")
            opcode_ids.add(opcode)
    if tagging["candidate_outputs"] != [
        "refine_candidates", "coarsen_candidates",
        "refine_equalities", "coarsen_equalities",
    ]:
        raise CatalogError("tagging_program_abi candidate outputs are not canonical")
    routes = tagging["indicator_stencil_routes"]
    if not isinstance(routes, list) or not routes \
            or len(routes) != len(set(routes)) \
            or any(not isinstance(route, str) or not route for route in routes):
        raise CatalogError(
            "tagging_program_abi indicator_stencil_routes must be unique strings")
    maximum_terms = tagging["maximum_stencil_terms"]
    if isinstance(maximum_terms, bool) or not isinstance(maximum_terms, int) \
            or maximum_terms < 1:
        raise CatalogError("tagging_program_abi maximum_stencil_terms must be >= 1")
    maximum = tagging["maximum_instruction_count"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise CatalogError("tagging_program_abi maximum_instruction_count must be >= 1")
    native_abis = data["native_interface_abis"]
    if not isinstance(native_abis, list) or not native_abis:
        raise CatalogError("native_interface_abis must be a non-empty list")
    native_names: set[str] = set()
    native_ids: set[int] = set()
    native_uris: set[str] = set()
    native_tables: set[str] = set()
    for index, native in enumerate(native_abis):
        native = _exact(native, {
            "id", "name", "uri", "version", "cpp_table", "hot_path", "facets", "operations",
        }, f"native_interface_abis[{index}]")
        abi_id = native["id"]
        if isinstance(abi_id, bool) or not isinstance(abi_id, int) or abi_id < 0:
            raise CatalogError(
                f"native_interface_abis[{index}].id must be an integer >= 0")
        if abi_id in native_ids:
            raise CatalogError(f"duplicate native component interface id {abi_id}")
        native_ids.add(abi_id)
        name = native["name"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise CatalogError(f"native_interface_abis[{index}].name is not canonical")
        if name in native_names:
            raise CatalogError(f"duplicate native component interface {name!r}")
        native_names.add(name)
        uri = native["uri"]
        if not isinstance(uri, str) or not uri.startswith("pops://interfaces/"):
            raise CatalogError(f"native_interface_abis[{index}].uri is not a PoPS interface URI")
        if uri in native_uris:
            raise CatalogError(f"duplicate native component interface URI {uri!r}")
        native_uris.add(uri)
        if isinstance(native["version"], bool) or not isinstance(native["version"], int) \
                or native["version"] < 1:
            raise CatalogError(f"native_interface_abis[{index}].version must be >= 1")
        table = _identifier(native["cpp_table"], f"native_interface_abis[{index}].cpp_table")
        if name == "field_solver" and (native["version"] != 2 or
                                        table != "PopsFieldSolverApiV2"):
            raise CatalogError(
                "field_solver must declare the indivisible PopsFieldSolverApiV2 interface")
        if name == "field_topology" and (native["version"] != 2 or
                                          table != "PopsFieldTopologyApiV2"):
            raise CatalogError(
                "field_topology must declare the indivisible PopsFieldTopologyApiV2 interface")
        if table in native_tables:
            raise CatalogError(f"duplicate native component interface table {table!r}")
        native_tables.add(table)
        if not isinstance(native["hot_path"], bool):
            raise CatalogError(f"native_interface_abis[{index}].hot_path must be boolean")
        facets = _strings(native["facets"], f"native_interface_abis[{index}].facets")
        unknown_facets = sorted(set(facets) - interface_names)
        if unknown_facets:
            raise CatalogError(
                f"native_interface_abis[{index}].facets are unknown: {unknown_facets}")
        operations = _strings(
            native["operations"], f"native_interface_abis[{index}].operations")
        if not operations or any(re.fullmatch(r"[a-z][a-z0-9_]*", op) is None
                                 for op in operations):
            raise CatalogError(
                f"native_interface_abis[{index}].operations must be canonical identifiers")

    boundary_routes = data["boundary_handle_native_routes"]
    if not isinstance(boundary_routes, dict) or not boundary_routes:
        raise CatalogError("boundary_handle_native_routes must be a non-empty object")
    native_operations = {
        row["name"]: frozenset(row["operations"]) for row in native_abis
    }
    for kind, route in boundary_routes.items():
        if not isinstance(kind, str) or re.fullmatch(r"[a-z][a-z0-9_]*", kind) is None:
            raise CatalogError(
                f"boundary_handle_native_routes key {kind!r} is not canonical")
        route = _exact(
            route, {"interface", "operation"},
            f"boundary_handle_native_routes.{kind}")
        interface = route["interface"]
        operation = route["operation"]
        if interface not in native_operations:
            raise CatalogError(
                f"boundary_handle_native_routes.{kind} names unknown interface {interface!r}")
        if operation not in native_operations[interface]:
            raise CatalogError(
                f"boundary_handle_native_routes.{kind} operation {operation!r} is not "
                f"exported by {interface!r}")

    family_native_interfaces = data["route_family_native_interfaces"]
    if not isinstance(family_native_interfaces, dict):
        raise CatalogError("route_family_native_interfaces must be an object")
    for family, name in family_native_interfaces.items():
        if name is not None and name not in native_names:
            raise CatalogError(
                f"route_family_native_interfaces.{family} names unknown interface {name!r}")

    family_interfaces = data["route_family_interfaces"]
    if not isinstance(family_interfaces, dict):
        raise CatalogError("route_family_interfaces must be an object")
    for family, declarations in family_interfaces.items():
        if not isinstance(family, str) or re.fullmatch(r"[a-z][a-z0-9_]*", family) is None:
            raise CatalogError(f"route_family_interfaces key {family!r} is not canonical")
        if not isinstance(declarations, list) or not declarations:
            raise CatalogError(f"route_family_interfaces.{family} must be a non-empty list")
        declared: set[str] = set()
        for index, declaration in enumerate(declarations):
            declaration = _exact(declaration, {"name", "mode", "binding"},
                                 f"route_family_interfaces.{family}[{index}]")
            name = declaration["name"]
            if name not in interface_names:
                raise CatalogError(
                    f"route_family_interfaces.{family}[{index}] names unknown interface {name!r}")
            if name in declared:
                raise CatalogError(
                    f"route_family_interfaces.{family} declares {name!r} more than once")
            declared.add(name)
            if declaration["mode"] not in {"method", "value", "entry_point"}:
                raise CatalogError(
                    f"route_family_interfaces.{family}[{index}].mode is invalid")
            binding = declaration["binding"]
            if not isinstance(binding, str) or re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*", binding) is None:
                raise CatalogError(
                    f"route_family_interfaces.{family}[{index}].binding is invalid")

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
        cpp_ids: set[str] = set()
        for index, route in enumerate(routes):
            route = _exact(route, {
                "token", "wire_id", "cpp_id", "native_entry", "requirements",
                "limitations", "aliases", "metadata",
            }, f"{name}.routes[{index}]")
            token = route["token"]
            if not isinstance(token, str) or re.fullmatch(r"[a-z][a-z0-9_]*", token) is None:
                raise CatalogError(f"{name}.routes[{index}].token is not canonical")
            if token in tokens:
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
            if _strings(route["aliases"], f"{name}.{token}.aliases"):
                raise CatalogError(
                    f"{name}.{token}.aliases must be empty; final route IDs have one spelling")
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
    if set(family_interfaces) != seen_families:
        raise CatalogError(
            "route_family_interfaces must cover every route family exactly: missing=%s, unknown=%s"
            % (sorted(seen_families - set(family_interfaces)),
               sorted(set(family_interfaces) - seen_families)))
    if set(family_native_interfaces) != seen_families:
        raise CatalogError(
            "route_family_native_interfaces must cover every route family exactly: "
            "missing=%s, unknown=%s"
            % (sorted(seen_families - set(family_native_interfaces)),
               sorted(set(family_native_interfaces) - seen_families)))

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
        f"COMPONENT_INTERFACE_SPECS = {_py_literal(tuple(catalog['interface_vocabulary']))}",
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
    metadata: dict[str, dict[str, dict[str, Any]]] = {}
    cpp: dict[str, dict[str, Any]] = {}
    brick_rows: list[dict[str, Any]] = []
    for family in catalog["route_families"]:
        name = family["name"]
        tables[name] = tuple((
            row["token"], row["native_entry"], tuple(row["requirements"]),
            tuple(row["limitations"]),
        ) for row in family["routes"])
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
        "ROUTE_METADATA": metadata,
        "ROUTE_CPP_BINDINGS": cpp,
        "ROUTE_COMPONENT_DEFAULTS": catalog["route_component_defaults"],
        "ROUTE_FAMILY_INTERFACES": catalog["route_family_interfaces"],
        "COMPONENT_INTERFACE_SPECS": tuple(catalog["interface_vocabulary"]),
        "ROUTE_FAMILY_NATIVE_INTERFACES": catalog["route_family_native_interfaces"],
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


def _render_native_interfaces(catalog: dict[str, Any], digest: str,
                              semantic_digest: str) -> str:
    rows = tuple({
        "id": row["id"],
        "name": row["name"],
        "uri": row["uri"],
        "version": row["version"],
        "cpp_table": row["cpp_table"],
        "hot_path": row["hot_path"],
        "facets": tuple(row["facets"]),
        "operations": tuple(row["operations"]),
    } for row in catalog["native_interface_abis"])
    boundary_routes = {
        kind: (row["interface"], row["operation"])
        for kind, row in catalog["boundary_handle_native_routes"].items()
    }
    return "\n".join((
        '\"\"\"Generated by scripts/generate_component_catalog.py; DO NOT EDIT.\"\"\"',
        "from __future__ import annotations",
        "",
        f"NATIVE_COMPONENT_ABI_VERSION = {catalog['native_interface_abi_version']}",
        f"NATIVE_COMPONENT_COMMON_ABI_VERSION = {catalog['native_common_abi_version']}",
        f"NATIVE_COMPONENT_CATALOG_SHA256 = {digest!r}",
        f"NATIVE_COMPONENT_CATALOG_SEMANTIC_SHA256 = {semantic_digest!r}",
        f"NATIVE_TAGGING_PROGRAM_ABI = {_py_literal(catalog['tagging_program_abi'])}",
        f"NATIVE_COMPONENT_INTERFACES = {_py_literal(rows)}",
        "NATIVE_COMPONENT_INTERFACE_BY_NAME = {row['name']: row for row in NATIVE_COMPONENT_INTERFACES}",
        "NATIVE_COMPONENT_INTERFACE_BY_URI = {row['uri']: row for row in NATIVE_COMPONENT_INTERFACES}",
        f"NATIVE_COMPONENT_BOUNDARY_HANDLE_ROUTES = {_py_literal(boundary_routes)}",
        "",
        "__all__ = [name for name in globals() if name.startswith('NATIVE_COMPONENT_')]",
        "",
    ))


def _render_component_abi(catalog: dict[str, Any], digest: str) -> str:
    """Render the closed C/POD execution ABI shared by builtin and package conformers."""
    enum_rows = "\n".join(
        "  POPS_NATIVE_INTERFACE_%s_V%d = %d," % (
            row["name"].upper(), row["version"], row["id"])
        for row in catalog["native_interface_abis"]
    )
    tagging = catalog["tagging_program_abi"]
    tagging_opcode_rows = "\n".join(
        "  POPS_TAGGING_%s_V1 = %d," % (name.upper(), opcode)
        for family in ("leaf_opcodes", "logical_opcodes")
        for name, opcode in tagging[family].items()
    )
    tagging_leaf_cases = " ".join(
        "case POPS_TAGGING_%s_V1:" % name.upper()
        for name in tagging["leaf_opcodes"]
    )
    tagging_logical_cases = " ".join(
        "case POPS_TAGGING_%s_V1:" % name.upper()
        for name in tagging["logical_opcodes"]
    )
    tagging_stencil_route_rows = "\n".join(
        '#define POPS_TAGGING_STENCIL_ROUTE_%s "%s"'
        % (route.upper(), route)
        for route in tagging["indicator_stencil_routes"]
    )
    table_size_rows = "\n".join(
        "    case POPS_NATIVE_INTERFACE_%s_V%d: return sizeof(%s);"
        % (row["name"].upper(), row["version"], row["cpp_table"])
        for row in catalog["native_interface_abis"]
    )
    table_name_rows = "\n".join(
        "    case POPS_NATIVE_INTERFACE_%s_V%d: return \"%s\";"
        % (row["name"].upper(), row["version"], row["cpp_table"])
        for row in catalog["native_interface_abis"]
    )
    return f'''#pragma once

// Generated by scripts/generate_component_catalog.py; DO NOT EDIT.
// clang-format off
// The ABI crosses no C++ standard-library or backend-owned type. Hot interfaces are batch calls;
// discovery and symbol resolution happen once during installation, never inside a scientific loop.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <pops/runtime/dynamic/abi_key.hpp>

extern "C" {{
#endif

#define POPS_COMPONENT_API_SYMBOL_V1 "pops_component_interface_v1"
#define POPS_COMPONENT_CATALOG_SHA256_V1 "{digest}"
#define POPS_COMPONENT_PROTOCOL_ABI_V1 {catalog['native_interface_abi_version']}u
#define POPS_COMPONENT_COMMON_ABI_V1 {catalog['native_common_abi_version']}u

typedef enum PopsNativeInterfaceIdV1 {{
{enum_rows}
}} PopsNativeInterfaceIdV1;

typedef enum PopsTaggingOpcodeV1 {{
{tagging_opcode_rows}
}} PopsTaggingOpcodeV1;
#define POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1 {tagging['maximum_instruction_count']}u
#define POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 {tagging['maximum_stencil_terms']}u
#define POPS_TAGGING_NO_STENCIL_V1 ((size_t)-1)
#define POPS_TAGGING_NON_FINITE_REJECT_V1 1
{tagging_stencil_route_rows}

static inline int pops_tagging_opcode_is_leaf_v1(int32_t opcode) {{
  switch (opcode) {{ {tagging_leaf_cases} return 1; default: return 0; }}
}}
static inline int pops_tagging_opcode_is_logical_v1(int32_t opcode) {{
  switch (opcode) {{ {tagging_logical_cases} return 1; default: return 0; }}
}}

typedef enum PopsComponentActionV1 {{
  POPS_COMPONENT_CONTINUE_V1 = 0,
  POPS_COMPONENT_RETRY_STEP_V1 = 1,
  POPS_COMPONENT_REJECT_STEP_V1 = 2,
  POPS_COMPONENT_ABORT_RUN_V1 = 3
}} PopsComponentActionV1;

typedef struct PopsComponentStatusV1 {{
  uint32_t struct_size;
  int32_t code;
  PopsComponentActionV1 action;
  const char* reason;
}} PopsComponentStatusV1;

typedef enum PopsMemorySpaceV1 {{
  POPS_MEMORY_SPACE_HOST_V1 = 1,
  POPS_MEMORY_SPACE_DEVICE_V1 = 2,
  POPS_MEMORY_SPACE_MANAGED_V1 = 3
}} PopsMemorySpaceV1;
typedef enum PopsScalarTypeV1 {{
  POPS_SCALAR_FLOAT32_V1 = 1,
  POPS_SCALAR_FLOAT64_V1 = 2
}} PopsScalarTypeV1;
typedef enum PopsFieldCenteringV1 {{
  POPS_FIELD_CENTERING_CELL_V1 = 1,
  POPS_FIELD_CENTERING_FACE_V1 = 2,
  POPS_FIELD_CENTERING_NODE_V1 = 3,
  POPS_FIELD_CENTERING_EDGE_V1 = 4
}} PopsFieldCenteringV1;
typedef enum PopsFieldOwnershipV1 {{
  POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1 = 1,
  POPS_FIELD_OWNERSHIP_COMPONENT_BORROWED_V1 = 2,
  POPS_FIELD_OWNERSHIP_COMPONENT_OWNED_V1 = 3
}} PopsFieldOwnershipV1;

typedef struct PopsConstFieldViewV1 {{
  uint32_t struct_size;
  const void* data;
  int32_t dimension;
  size_t extents[3];
  ptrdiff_t axis_strides[3];
  size_t component_count;
  ptrdiff_t component_stride;
  PopsFieldCenteringV1 centering;
  uint32_t centering_axes;
  size_t ghost_lower[3];
  size_t ghost_upper[3];
  PopsScalarTypeV1 scalar_type;
  PopsMemorySpaceV1 memory_space;
  const char* layout_identity;
  const char* patch_identity;
  PopsFieldOwnershipV1 ownership;
}} PopsConstFieldViewV1;

typedef struct PopsFieldViewV1 {{
  uint32_t struct_size;
  void* data;
  int32_t dimension;
  size_t extents[3];
  ptrdiff_t axis_strides[3];
  size_t component_count;
  ptrdiff_t component_stride;
  PopsFieldCenteringV1 centering;
  uint32_t centering_axes;
  size_t ghost_lower[3];
  size_t ghost_upper[3];
  PopsScalarTypeV1 scalar_type;
  PopsMemorySpaceV1 memory_space;
  const char* layout_identity;
  const char* patch_identity;
  PopsFieldOwnershipV1 ownership;
}} PopsFieldViewV1;

typedef struct PopsConstByteViewV1 {{
  uint32_t struct_size;
  const uint8_t* data;
  size_t size;
}} PopsConstByteViewV1;

typedef struct PopsByteViewV1 {{
  uint32_t struct_size;
  uint8_t* data;
  size_t size;
}} PopsByteViewV1;

typedef struct PopsInt32ViewV1 {{
  uint32_t struct_size;
  int32_t* data;
  size_t size;
}} PopsInt32ViewV1;

typedef struct PopsConstInt32ViewV1 {{
  uint32_t struct_size;
  const int32_t* data;
  size_t size;
}} PopsConstInt32ViewV1;

typedef struct PopsLogicalTimeV1 {{
  uint32_t struct_size;
  const char* clock_identity;
  int64_t tick;
  int32_t level;
  int32_t substep;
  int32_t stage;
  int64_t fraction_numerator;
  int64_t fraction_denominator;
  double dt;
  double physical_time;
}} PopsLogicalTimeV1;

typedef enum PopsPrecisionV1 {{
  POPS_PRECISION_FLOAT16_V1 = 1,
  POPS_PRECISION_BFLOAT16_V1 = 2,
  POPS_PRECISION_FLOAT32_V1 = 3,
  POPS_PRECISION_FLOAT64_V1 = 4
}} PopsPrecisionV1;
typedef struct PopsExecutionContextV1 {{
  uint32_t struct_size;
  uint32_t context_version;
  const char* execution_identity;
  PopsMemorySpaceV1 memory_space;
  const char* backend_identity;
  const char* device_identity;
  PopsScalarTypeV1 scalar_type;
  PopsPrecisionV1 storage_precision;
  PopsPrecisionV1 compute_precision;
  PopsPrecisionV1 accumulation_precision;
  PopsPrecisionV1 reduction_precision;
  uint64_t stream_handle;
  const char* stream_identity;
  int64_t communicator_f_handle;
  int64_t communicator_datatype_f_handle;
  const char* communicator_identity;
  const char* communicator_datatype_identity;
}} PopsExecutionContextV1;

typedef struct PopsComponentPrepareRequestV1 {{
  uint32_t struct_size;
  const char* parameters_json;
  const char* target_json;
  PopsExecutionContextV1 execution;
}} PopsComponentPrepareRequestV1;

typedef int32_t (*PopsComponentPrepareFnV1)(
    const PopsComponentPrepareRequestV1*, void**, PopsComponentStatusV1*);
typedef void (*PopsComponentDestroyFnV1)(void*);

typedef struct PopsComponentTableHeaderV1 {{
  uint32_t struct_size;
  uint32_t abi_version;
  PopsNativeInterfaceIdV1 interface_id;
  uint32_t interface_version;
  PopsComponentPrepareFnV1 prepare;
  PopsComponentDestroyFnV1 destroy;
}} PopsComponentTableHeaderV1;

typedef struct PopsNumericalFluxRequestV1 {{
  uint32_t struct_size;
  PopsConstFieldViewV1 left;
  PopsConstFieldViewV1 right;
  PopsConstFieldViewV1 normals;
  const double* face_measures;
  PopsLogicalTimeV1 logical_time;
  PopsExecutionContextV1 execution;
}} PopsNumericalFluxRequestV1;
typedef struct PopsNumericalFluxResultV1 {{
  uint32_t struct_size;
  PopsFieldViewV1 normal_flux;
  double* stability_bounds;
  PopsComponentActionV1* actions;
  PopsComponentStatusV1 status;
}} PopsNumericalFluxResultV1;
typedef int32_t (*PopsEvaluateFacesFnV1)(
    void*, const PopsNumericalFluxRequestV1*, PopsNumericalFluxResultV1*);
typedef struct PopsNumericalFluxApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsEvaluateFacesFnV1 evaluate_faces;
}} PopsNumericalFluxApiV1;

typedef enum PopsBoundaryRegionKindV1 {{
  POPS_BOUNDARY_FACE_V1 = 1,
  POPS_BOUNDARY_EDGE_V1 = 2,
  POPS_BOUNDARY_CORNER_V1 = 3
}} PopsBoundaryRegionKindV1;
typedef struct PopsBoundaryRegionV1 {{
  uint32_t struct_size;
  PopsBoundaryRegionKindV1 kind;
  int32_t dimension;
  int32_t codimension;
  size_t axis_count;
  const int32_t* axes;
  const int32_t* sides;
  const char* region_identity;
}} PopsBoundaryRegionV1;
typedef struct PopsQualifiedConstFieldV1 {{
  uint32_t struct_size;
  uint32_t present;
  const char* qualified_id;
  PopsConstFieldViewV1 values;
}} PopsQualifiedConstFieldV1;
typedef struct PopsQualifiedFieldV1 {{
  uint32_t struct_size;
  const char* qualified_id;
  PopsFieldViewV1 values;
}} PopsQualifiedFieldV1;
typedef struct PopsQualifiedScalarV1 {{
  uint32_t struct_size;
  const char* qualified_id;
  double value;
}} PopsQualifiedScalarV1;
typedef struct PopsGhostBoundaryRequestV1 {{
  uint32_t struct_size;
  const char* producer_identity;
  const char* state_identity;
  const char* ghost_identity;
  PopsConstFieldViewV1 interior;
  PopsFieldViewV1 ghosts;
  PopsConstFieldViewV1 coordinates;
  PopsBoundaryRegionV1 region;
  size_t dependency_count;
  const PopsQualifiedConstFieldV1* dependencies;
  size_t parameter_count;
  const PopsQualifiedScalarV1* parameters;
  PopsLogicalTimeV1 logical_time;
  PopsExecutionContextV1 execution;
}} PopsGhostBoundaryRequestV1;
typedef int32_t (*PopsApplyRegionBatchFnV1)(
    void*, const PopsGhostBoundaryRequestV1*, PopsComponentStatusV1*);
typedef struct PopsGhostBoundaryApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsApplyRegionBatchFnV1 apply_region_batch;
}} PopsGhostBoundaryApiV1;

typedef struct PopsFieldBoundaryRequestV1 {{
  uint32_t struct_size;
  const char* closure_identity;
  PopsBoundaryRegionV1 region;
  PopsConstFieldViewV1 coordinates;
  size_t state_count;
  const PopsQualifiedConstFieldV1* states;
  size_t direction_count;
  const PopsQualifiedConstFieldV1* directions;
  size_t field_count;
  const PopsQualifiedConstFieldV1* fields;
  size_t parameter_count;
  const PopsQualifiedScalarV1* parameters;
  size_t output_count;
  PopsQualifiedFieldV1* outputs;
  PopsQualifiedConstFieldV1 rate;
  PopsQualifiedConstFieldV1 nonlinear_iterate;
  int32_t level;
  PopsLogicalTimeV1 logical_time;
  PopsExecutionContextV1 execution;
}} PopsFieldBoundaryRequestV1;
typedef int32_t (*PopsFieldBoundaryEvalFnV1)(
    void*, const PopsFieldBoundaryRequestV1*, PopsComponentStatusV1*);
typedef struct PopsFieldBoundaryClosureApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsFieldBoundaryEvalFnV1 residual;
  PopsFieldBoundaryEvalFnV1 jvp;
}} PopsFieldBoundaryClosureApiV1;

typedef struct PopsTaggingAxisStencilV1 {{
  uint32_t struct_size;
  int32_t axis;
  int32_t derivative_order;
  int32_t formal_order;
  size_t ghost_lower;
  size_t ghost_upper;
  size_t term_count;
  const int32_t* offsets;
  const double* coefficients;
}} PopsTaggingAxisStencilV1;
typedef struct PopsTaggingStencilV1 {{
  uint32_t struct_size;
  const char* stencil_identity;
  const char* route;
  const char* norm;
  const char* scale;
  const char* boundary_mode;
  int32_t dimension;
  size_t axis_count;
  const PopsTaggingAxisStencilV1* axes;
}} PopsTaggingStencilV1;
typedef struct PopsTaggingLeafV1 {{
  uint32_t struct_size;
  size_t state_index;
  size_t component;
  int32_t opcode;
  double threshold;
  size_t stencil_index;
}} PopsTaggingLeafV1;
typedef struct PopsTaggingProgramV1 {{
  uint32_t struct_size;
  const char* program_identity;
  size_t stencil_count;
  const PopsTaggingStencilV1* stencils;
  size_t leaf_count;
  const PopsTaggingLeafV1* leaves;
  size_t refine_instruction_count;
  const int32_t* refine_opcodes;
  const int32_t* refine_arguments;
  size_t coarsen_instruction_count;
  const int32_t* coarsen_opcodes;
  const int32_t* coarsen_arguments;
  int32_t minimum_cycles;
  int32_t equality_policy;
  int32_t conflict_policy;
  int32_t non_finite_policy;
}} PopsTaggingProgramV1;
typedef struct PopsTaggerRequestV1 {{
  uint32_t struct_size;
  size_t state_count;
  const PopsQualifiedConstFieldV1* states;
  PopsTaggingProgramV1 program;
  int64_t patch_lower[3];
  int64_t domain_lower[3];
  int64_t domain_upper[3];
  double cell_size[3];
  uint32_t periodic_axes;
  PopsByteViewV1 refine_candidates;
  PopsByteViewV1 coarsen_candidates;
  PopsByteViewV1 refine_equalities;
  PopsByteViewV1 coarsen_equalities;
  PopsLogicalTimeV1 logical_time;
  PopsExecutionContextV1 execution;
}} PopsTaggerRequestV1;
typedef int32_t (*PopsTagBatchFnV1)(void*, const PopsTaggerRequestV1*, PopsComponentStatusV1*);
typedef struct PopsTaggerApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsTagBatchFnV1 tag_batch;
}} PopsTaggerApiV1;

typedef struct PopsClusteringRequestV1 {{
  uint32_t struct_size;
  PopsConstByteViewV1 tags;
  const int64_t* extents;
  int32_t dimension;
  // `boxes` is box-major [lo_0..lo_(d-1), hi_0..hi_(d-1)]. Bounds are
  // inclusive and relative to the supplied tag region.
  int64_t* boxes;
  size_t box_capacity;
  size_t* box_count;
  PopsExecutionContextV1 execution;
}} PopsClusteringRequestV1;
typedef int32_t (*PopsClusterFnV1)(void*, const PopsClusteringRequestV1*, PopsComponentStatusV1*);
typedef struct PopsClusteringApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsClusterFnV1 cluster;
}} PopsClusteringApiV1;

typedef enum PopsTransferOperationV1 {{
  POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1 = 1
}} PopsTransferOperationV1;
typedef struct PopsTransferRequestV1 {{
  uint32_t struct_size;
  PopsConstFieldViewV1 source;
  PopsFieldViewV1 destination;
  const int32_t* refinement_ratio;
  int32_t dimension;
  PopsTransferOperationV1 operation;
  PopsExecutionContextV1 execution;
}} PopsTransferRequestV1;
typedef int32_t (*PopsTransferApplyFnV1)(void*, const PopsTransferRequestV1*, PopsComponentStatusV1*);
typedef struct PopsTransferApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsTransferApplyFnV1 apply;
}} PopsTransferApiV1;

typedef struct PopsFieldPatchMetadataV1 {{
  uint32_t struct_size;
  size_t global_patch_index;
  int32_t owner_rank;
  int32_t level;
  int32_t dimension;
  int64_t lower[3];
  int64_t upper[3];
  // Physical coordinate of the lower face at `lower`, not the global-domain origin.
  double physical_lower[3];
  double cell_spacing[3];
  PopsFieldCenteringV1 centering;
  uint32_t centering_axes;
  // Qualified source LayoutPlan identity; materialization is carried by global topology.
  const char* layout_identity;
  const char* patch_identity;
}} PopsFieldPatchMetadataV1;

typedef enum PopsFieldMaterialRepresentationV1 {{
  POPS_FIELD_MATERIAL_FULL_V1 = 1,
  POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1 = 2,
  POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1 = 3,
  POPS_FIELD_MATERIAL_IDS_V1 = 4,
  POPS_FIELD_MATERIAL_IDS_WITH_CUT_CELL_FRACTION_V1 = 5
}} PopsFieldMaterialRepresentationV1;

typedef struct PopsFieldGlobalTopologyV1 {{
  uint32_t struct_size;
  const char* topology_recipe_identity;
  // Qualified authoring LayoutPlan identity, preserved on every FieldView.
  const char* source_layout_identity;
  // Exact runtime materialization identity: source layout + geometry + boxes + owners + topology.
  const char* materialized_layout_identity;
  int32_t dimension;
  int64_t domain_lower[3];
  int64_t domain_upper[3];
  uint32_t periodic_axes;
  size_t patch_count;
  const PopsFieldPatchMetadataV1* patches;
}} PopsFieldGlobalTopologyV1;

typedef struct PopsFieldSolverTopologyLabelV2 {{
  uint32_t struct_size;
  int32_t id;
  const char* label;
  const char* provenance;
}} PopsFieldSolverTopologyLabelV2;

typedef struct PopsFieldSolverPatchV2 {{
  uint32_t struct_size;
  size_t metadata_index;
  PopsConstFieldViewV1 rhs;
  PopsFieldViewV1 solution;
  PopsConstFieldViewV1 coefficients;
  PopsConstByteViewV1 material_mask;
  PopsConstInt32ViewV1 component_labels;
}} PopsFieldSolverPatchV2;

typedef struct PopsFieldSolverRequestV2 {{
  uint32_t struct_size;
  PopsFieldGlobalTopologyV1 topology;
  size_t local_patch_count;
  const PopsFieldSolverPatchV2* local_patches;
  size_t topology_label_count;
  const PopsFieldSolverTopologyLabelV2* topology_labels;
  const char* topology_provenance;
  const char* topology_digest;
  const char* boundary_contract_json;
  double relative_tolerance;
  double absolute_tolerance;
  int32_t max_iterations;
  PopsExecutionContextV1 execution;
}} PopsFieldSolverRequestV2;

typedef enum PopsSolveStatusV2 {{
  POPS_SOLVE_SOLVED_V2 = 0,
  POPS_SOLVE_SINGULAR_V2 = 1,
  POPS_SOLVE_BREAKDOWN_V2 = 2,
  POPS_SOLVE_ITERATION_LIMIT_V2 = 3,
  POPS_SOLVE_INVALID_EVALUATION_V2 = 4,
  POPS_SOLVE_CAPABILITY_FAILURE_V2 = 5,
  POPS_SOLVE_INVALID_INPUT_V2 = 6,
  POPS_SOLVE_INCOMPATIBLE_RHS_V2 = 7
}} PopsSolveStatusV2;

typedef enum PopsSolveActionV2 {{
  POPS_SOLVE_ACTION_NONE_V2 = 0,
  POPS_SOLVE_ACTION_FAIL_RUN_V2 = 1,
  POPS_SOLVE_ACTION_REJECT_ATTEMPT_V2 = 2
}} PopsSolveActionV2;

typedef struct PopsSolveReportV2 {{
  uint32_t struct_size;
  PopsSolveStatusV2 status;
  PopsSolveActionV2 action;
  int32_t iterations;
  // residual_norm / reference_residual_norm, using denominator 1 only when the reference is zero.
  double relative_residual;
  // Exact ||R(x0)|| used by max(relative_tolerance * ||R(x0)||, absolute_tolerance).
  double reference_residual_norm;
  // Exact ||R(x_final)|| tested against the mixed convergence threshold.
  double residual_norm;
  const char* reason;
}} PopsSolveReportV2;
typedef int32_t (*PopsFieldSolveFnV2)(
    void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2*);
typedef struct PopsFieldSolverApiV2 {{
  PopsComponentTableHeaderV1 header;
  PopsFieldSolveFnV2 solve;
}} PopsFieldSolverApiV2;

typedef struct PopsWriterBoxV1 {{
  uint32_t struct_size;
  int32_t dimension;
  const int64_t* lower;
  const int64_t* upper;
}} PopsWriterBoxV1;
typedef struct PopsWriterGeometryV1 {{
  uint32_t struct_size;
  const char* layout_identity;
  const char* layout_kind;
  int32_t level;
  int32_t dimension;
  const double* origin;
  const double* spacing;
  const size_t* cell_shape;
  size_t box_count;
  const PopsWriterBoxV1* boxes;
  PopsConstByteViewV1 valid_cells;
  PopsConstByteViewV1 coverage;
  PopsConstFieldViewV1 cell_volumes;
}} PopsWriterGeometryV1;
typedef struct PopsWriterPieceV1 {{
  uint32_t struct_size;
  int32_t dimension;
  const int64_t* lower;
  const int64_t* upper;
  PopsConstFieldViewV1 values;
}} PopsWriterPieceV1;
typedef struct PopsWriterFieldV1 {{
  uint32_t struct_size;
  const char* field_identity;
  const char* reference_id;
  const char* component_manifest_identity;
  const char* layout_identity;
  int32_t level;
  const char* state_id;
  const char* centering;
  const char* units;
  size_t component_name_count;
  const char* const* component_names;
  int32_t dimension;
  const size_t* global_shape;
  size_t piece_count;
  const PopsWriterPieceV1* pieces;
}} PopsWriterFieldV1;
typedef struct PopsWriterDiagnosticV1 {{
  uint32_t struct_size;
  const char* diagnostic_identity;
  const char* reference_id;
  const char* component_manifest_identity;
  const char* layout_identity;
  int32_t level;
  const char* state_id;
  const char* reduction;
  double value;
  const char* units;
  const char* terms_json;
}} PopsWriterDiagnosticV1;
typedef struct PopsWriterRequestV1 {{
  uint32_t struct_size;
  size_t geometry_count;
  const PopsWriterGeometryV1* geometries;
  size_t field_count;
  const PopsWriterFieldV1* fields;
  size_t diagnostic_count;
  const PopsWriterDiagnosticV1* diagnostics;
  const char* metadata_json;
  const char* selection_identity;
  const char* temporary_path;
  const char* published_path;
  const char* snapshot_identity;
  PopsLogicalTimeV1 logical_time;
  PopsExecutionContextV1 execution;
}} PopsWriterRequestV1;
typedef struct PopsWriterReceiptV1 {{
  uint32_t struct_size;
  uint64_t bytes_written;
  const char* content_digest;
  PopsComponentStatusV1 status;
}} PopsWriterReceiptV1;
typedef int32_t (*PopsWriterVerifyFnV1)(void*, const PopsWriterRequestV1*, PopsWriterReceiptV1*);
typedef int32_t (*PopsWriterPublishFnV1)(void*, const PopsWriterRequestV1*, PopsWriterReceiptV1*);
typedef void (*PopsWriterCleanupFnV1)(void*, const PopsWriterRequestV1*);
typedef struct PopsWriterApiV1 {{
  PopsComponentTableHeaderV1 header;
  PopsWriterVerifyFnV1 verify;
  PopsWriterPublishFnV1 publish;
  PopsWriterCleanupFnV1 discard;
  PopsWriterCleanupFnV1 rollback;
}} PopsWriterApiV1;

typedef struct PopsFieldTopologyPatchV2 {{
  uint32_t struct_size;
  size_t metadata_index;
  PopsFieldMaterialRepresentationV1 material_representation;
  PopsConstByteViewV1 material_coverage;
  PopsConstFieldViewV1 cut_cell_volume_fraction;
  PopsConstInt32ViewV1 material_ids;
  PopsByteViewV1 material_mask;
  PopsInt32ViewV1 component_labels;
}} PopsFieldTopologyPatchV2;
typedef struct PopsFieldTopologyRequestV2 {{
  uint32_t struct_size;
  PopsFieldGlobalTopologyV1 topology;
  size_t local_patch_count;
  const PopsFieldTopologyPatchV2* local_patches;
  PopsExecutionContextV1 execution;
}} PopsFieldTopologyRequestV2;
typedef struct PopsTopologyLabelV2 {{
  uint32_t struct_size;
  int32_t id;
  const char* label;
  const char* provenance;
}} PopsTopologyLabelV2;
typedef struct PopsFieldTopologyResultV2 {{
  uint32_t struct_size;
  size_t label_count;
  const PopsTopologyLabelV2* labels;
  const char* provenance;
  const char* topology_digest;
  PopsComponentStatusV1 status;
}} PopsFieldTopologyResultV2;
typedef int32_t (*PopsPrepareTopologyFnV2)(
    void*, const PopsFieldTopologyRequestV2*, PopsFieldTopologyResultV2*);
typedef struct PopsFieldTopologyApiV2 {{
  PopsComponentTableHeaderV1 header;
  PopsPrepareTopologyFnV2 prepare_topology;
}} PopsFieldTopologyApiV2;

typedef struct PopsComponentInterfaceEntryV1 {{
  PopsNativeInterfaceIdV1 interface_id;
  uint32_t interface_version;
  uint32_t table_size;
  const void* table;
}} PopsComponentInterfaceEntryV1;

typedef struct PopsComponentApiV1 {{
  uint32_t struct_size;
  uint32_t protocol_abi;
  // Translation-unit-local native ABI identity. C++ providers use POPS_ABI_KEY_LITERAL; C
  // providers must emit the exact equivalent literal selected by their PoPS build.
  const char* abi_key;
  const char* catalog_sha256;
  const char* component_id;
  const char* semantic_identity;
  const char* manifest_identity;
  size_t interface_count;
  const PopsComponentInterfaceEntryV1* interfaces;
}} PopsComponentApiV1;

typedef const PopsComponentApiV1* (*PopsComponentApiGetterV1)(void);

#ifdef __cplusplus
}}  // extern "C"

namespace pops::component {{
inline constexpr size_t generated_native_interface_table_size(
    PopsNativeInterfaceIdV1 id) noexcept {{
  switch (id) {{
{table_size_rows}
  }}
  return 0;
}}
inline constexpr const char* generated_native_interface_table_name(
    PopsNativeInterfaceIdV1 id) noexcept {{
  switch (id) {{
{table_name_rows}
  }}
  return nullptr;
}}
}}  // namespace pops::component
#endif
// clang-format on
'''


def _render_pybind_invokers(catalog: dict[str, Any], digest: str) -> str:
    """Render the sole Python/native request marshaller from the ABI declaration."""
    interfaces = {row["name"]: row for row in catalog["native_interface_abis"]}
    writer = interfaces["writer"]
    transfer = interfaces["transfer"]
    writer_rows = "\n".join(
        "    if (operation == %s) return invoke_writer(api.%s, loaded, request, %s);"
        % (_cpp_string(operation), operation, _cpp_string(operation))
        for operation in writer["operations"]
    )
    source = r'''// Generated by scripts/generate_component_catalog.py from catalog @DIGEST@; DO NOT EDIT.
// This file is the sole Python/native request marshaller. init_component_loader.cpp only registers it.

#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::component::generated_pybind {
namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style>;
using MaskArray = py::array_t<bool, py::array::c_style>;
static_assert(sizeof(bool) == sizeof(std::uint8_t));

template <class Array>
Array required_array(const py::handle value, const char* where) {
  auto result = Array::ensure(value);
  if (!result)
    throw py::type_error(std::string(where) +
                         " must be a C-contiguous array of the exact dtype");
  return result;
}

inline PopsConstFieldViewV1 writer_field_view(
    const DoubleArray& values, const std::string& layout_identity,
    const std::string& patch_identity, const char* where) {
  if (values.ndim() != 2 && values.ndim() != 3)
    throw py::value_error(std::string(where) +
                          " must have shape (ny,nx) or (components,ny,nx)");
  const auto components = values.ndim() == 3
                              ? static_cast<std::size_t>(values.shape(0))
                              : 1u;
  const auto ny = static_cast<std::size_t>(values.shape(values.ndim() - 2));
  const auto nx = static_cast<std::size_t>(values.shape(values.ndim() - 1));
  if (components == 0 || ny == 0 || nx == 0)
    throw py::value_error(std::string(where) + " must be non-empty");
  const auto item = static_cast<py::ssize_t>(sizeof(double));
  return {sizeof(PopsConstFieldViewV1), values.data(), 2, {ny, nx, 1},
          {values.strides(values.ndim() - 2) / item,
           values.strides(values.ndim() - 1) / item, 0},
          components, values.ndim() == 3 ? values.strides(0) / item : 1,
          POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
          POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
          layout_identity.c_str(), patch_identity.c_str(),
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

struct ExecutionStorage {
  std::string execution_identity;
  std::string backend_identity;
  std::string device_identity;
  std::string stream_identity;
  std::string communicator_identity;
  std::string communicator_datatype_identity;
  PopsExecutionContextV1 value{};

  explicit ExecutionStorage(const py::dict& row)
      : execution_identity(py::cast<std::string>(row["execution_identity"])),
        backend_identity(py::cast<std::string>(row["backend_identity"])),
        device_identity(py::cast<std::string>(row["device_identity"])),
        stream_identity(py::cast<std::string>(row["stream_identity"])),
        communicator_identity(py::cast<std::string>(row["communicator_identity"])),
        communicator_datatype_identity(
            py::cast<std::string>(row["communicator_datatype_identity"])) {
    value = {
        sizeof(PopsExecutionContextV1), py::cast<std::uint32_t>(row["context_version"]),
        execution_identity.c_str(),
        static_cast<PopsMemorySpaceV1>(py::cast<std::int32_t>(row["memory_space"])),
        backend_identity.c_str(), device_identity.c_str(),
        static_cast<PopsScalarTypeV1>(py::cast<std::int32_t>(row["scalar_type"])),
        static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["storage_precision"])),
        static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["compute_precision"])),
        static_cast<PopsPrecisionV1>(
            py::cast<std::int32_t>(row["accumulation_precision"])),
        static_cast<PopsPrecisionV1>(py::cast<std::int32_t>(row["reduction_precision"])),
        py::cast<std::uint64_t>(row["stream_handle"]), stream_identity.c_str(),
        py::cast<std::int64_t>(row["communicator_f_handle"]),
        py::cast<std::int64_t>(row["communicator_datatype_f_handle"]),
        communicator_identity.c_str(), communicator_datatype_identity.c_str()};
    validate_execution_context(value);
  }
};

struct LogicalTimeStorage {
  std::string clock_identity;
  PopsLogicalTimeV1 value{};

  explicit LogicalTimeStorage(const py::dict& row)
      : clock_identity(py::cast<std::string>(row["clock_identity"])) {
    value = {sizeof(PopsLogicalTimeV1), clock_identity.c_str(),
             py::cast<std::int64_t>(row["tick"]),
             py::cast<std::int32_t>(row["level"]),
             py::cast<std::int32_t>(row["substep"]),
             py::cast<std::int32_t>(row["stage"]),
             py::cast<std::int64_t>(row["fraction_numerator"]),
             py::cast<std::int64_t>(row["fraction_denominator"]),
             py::cast<double>(row["dt"]), py::cast<double>(row["physical_time"])};
    validate_logical_time(value);
  }
};

struct WriterGeometryStorage {
  std::string layout_identity;
  std::string layout_kind;
  std::string patch_identity;
  std::vector<double> origin;
  std::vector<double> spacing;
  std::vector<std::size_t> cell_shape;
  std::vector<std::vector<std::int64_t>> lower;
  std::vector<std::vector<std::int64_t>> upper;
  std::vector<PopsWriterBoxV1> boxes;
  MaskArray valid_cells;
  MaskArray coverage;
  DoubleArray cell_volumes;
  PopsWriterGeometryV1 value{};

  explicit WriterGeometryStorage(const py::dict& row)
      : layout_identity(py::cast<std::string>(row["layout_identity"])),
        layout_kind(py::cast<std::string>(row["layout_kind"])),
        patch_identity(py::cast<std::string>(row["patch_identity"])),
        origin(py::cast<std::vector<double>>(row["origin"])),
        spacing(py::cast<std::vector<double>>(row["spacing"])),
        cell_shape(py::cast<std::vector<std::size_t>>(row["cell_shape"])),
        valid_cells(required_array<MaskArray>(row["valid_cells"], "Writer valid_cells")),
        coverage(required_array<MaskArray>(row["coverage"], "Writer coverage")),
        cell_volumes(required_array<DoubleArray>(row["cell_volumes"],
                                                 "Writer cell_volumes")) {
    const auto dimension = py::cast<std::int32_t>(row["dimension"]);
    if (dimension < 1 || dimension > 3 || origin.size() != dimension ||
        spacing.size() != dimension || cell_shape.size() != dimension)
      throw py::value_error("Writer geometry has inconsistent dimensioned axes");
    const auto raw_boxes = py::cast<py::list>(row["boxes"]);
    lower.reserve(raw_boxes.size());
    upper.reserve(raw_boxes.size());
    boxes.reserve(raw_boxes.size());
    for (const auto raw : raw_boxes) {
      const auto box = py::cast<py::dict>(raw);
      lower.push_back(py::cast<std::vector<std::int64_t>>(box["lower"]));
      upper.push_back(py::cast<std::vector<std::int64_t>>(box["upper"]));
      if (lower.back().size() != dimension || upper.back().size() != dimension)
        throw py::value_error("Writer box dimension differs from geometry");
      boxes.push_back({sizeof(PopsWriterBoxV1), dimension, lower.back().data(),
                       upper.back().data()});
    }
    value = {sizeof(PopsWriterGeometryV1), layout_identity.c_str(),
             layout_kind.c_str(), py::cast<std::int32_t>(row["level"]), dimension,
             origin.data(), spacing.data(), cell_shape.data(), boxes.size(), boxes.data(),
             {sizeof(PopsConstByteViewV1),
              reinterpret_cast<const std::uint8_t*>(valid_cells.data()),
              static_cast<std::size_t>(valid_cells.size())},
             {sizeof(PopsConstByteViewV1),
              reinterpret_cast<const std::uint8_t*>(coverage.data()),
              static_cast<std::size_t>(coverage.size())},
             writer_field_view(cell_volumes, layout_identity, patch_identity,
                               "Writer cell_volumes")};
  }
};

struct WriterPieceStorage {
  std::string patch_identity;
  std::vector<std::int64_t> lower;
  std::vector<std::int64_t> upper;
  DoubleArray values;
  PopsWriterPieceV1 value{};

  WriterPieceStorage(const py::dict& row, const std::string& layout_identity,
                     std::int32_t dimension)
      : patch_identity(py::cast<std::string>(row["patch_identity"])),
        lower(py::cast<std::vector<std::int64_t>>(row["lower"])),
        upper(py::cast<std::vector<std::int64_t>>(row["upper"])),
        values(required_array<DoubleArray>(row["values"], "Writer field piece")) {
    if (lower.size() != dimension || upper.size() != dimension)
      throw py::value_error("Writer field piece dimension differs from field");
    value = {sizeof(PopsWriterPieceV1), dimension, lower.data(), upper.data(),
             writer_field_view(values, layout_identity, patch_identity,
                               "Writer field piece")};
  }
};

struct WriterFieldStorage {
  std::string field_identity;
  std::string reference_id;
  std::string component_manifest_identity;
  std::string layout_identity;
  std::string state_id;
  std::string centering;
  std::string units;
  std::vector<std::string> component_names;
  std::vector<const char*> component_name_pointers;
  std::vector<std::size_t> global_shape;
  std::vector<WriterPieceStorage> piece_storage;
  std::vector<PopsWriterPieceV1> pieces;
  PopsWriterFieldV1 value{};

  explicit WriterFieldStorage(const py::dict& row)
      : field_identity(py::cast<std::string>(row["field_identity"])),
        reference_id(py::cast<std::string>(row["reference_id"])),
        component_manifest_identity(
            py::cast<std::string>(row["component_manifest_identity"])),
        layout_identity(py::cast<std::string>(row["layout_identity"])),
        state_id(py::cast<std::string>(row["state_id"])),
        centering(py::cast<std::string>(row["centering"])),
        units(py::cast<std::string>(row["units"])),
        component_names(py::cast<std::vector<std::string>>(row["component_names"])),
        global_shape(py::cast<std::vector<std::size_t>>(row["global_shape"])) {
    const auto dimension = py::cast<std::int32_t>(row["dimension"]);
    if (dimension < 1 || dimension > 3 || global_shape.size() != dimension)
      throw py::value_error("Writer field has inconsistent dimensioned shape");
    const auto raw_pieces = py::cast<py::list>(row["pieces"]);
    piece_storage.reserve(raw_pieces.size());
    pieces.reserve(raw_pieces.size());
    for (const auto raw : raw_pieces)
      piece_storage.emplace_back(py::cast<py::dict>(raw), layout_identity, dimension);
    for (const auto& piece : piece_storage) pieces.push_back(piece.value);
    component_name_pointers.reserve(component_names.size());
    for (const auto& name : component_names) component_name_pointers.push_back(name.c_str());
    value = {sizeof(PopsWriterFieldV1), field_identity.c_str(), reference_id.c_str(),
             component_manifest_identity.c_str(), layout_identity.c_str(),
             py::cast<std::int32_t>(row["level"]), state_id.c_str(), centering.c_str(),
             units.c_str(), component_name_pointers.size(), component_name_pointers.data(),
             dimension, global_shape.data(), pieces.size(), pieces.data()};
  }
};

struct WriterDiagnosticStorage {
  std::string diagnostic_identity;
  std::string reference_id;
  std::string component_manifest_identity;
  std::string layout_identity;
  std::string state_id;
  std::string reduction;
  std::string units;
  std::string terms_json;
  PopsWriterDiagnosticV1 value{};

  explicit WriterDiagnosticStorage(const py::dict& row)
      : diagnostic_identity(py::cast<std::string>(row["diagnostic_identity"])),
        reference_id(py::cast<std::string>(row["reference_id"])),
        component_manifest_identity(
            py::cast<std::string>(row["component_manifest_identity"])),
        layout_identity(py::cast<std::string>(row["layout_identity"])),
        state_id(py::cast<std::string>(row["state_id"])),
        reduction(py::cast<std::string>(row["reduction"])),
        units(py::cast<std::string>(row["units"])),
        terms_json(py::cast<std::string>(row["terms_json"])) {
    value = {sizeof(PopsWriterDiagnosticV1), diagnostic_identity.c_str(),
             reference_id.c_str(), component_manifest_identity.c_str(),
             layout_identity.c_str(), py::cast<std::int32_t>(row["level"]),
             state_id.c_str(), reduction.c_str(),
             py::cast<double>(row["value"]), units.c_str(), terms_json.c_str()};
  }
};

struct WriterRequestStorage {
  std::vector<WriterGeometryStorage> geometry_storage;
  std::vector<PopsWriterGeometryV1> geometries;
  std::vector<WriterFieldStorage> field_storage;
  std::vector<PopsWriterFieldV1> fields;
  std::vector<WriterDiagnosticStorage> diagnostic_storage;
  std::vector<PopsWriterDiagnosticV1> diagnostics;
  std::string metadata_json;
  std::string selection_identity;
  std::string temporary_path;
  std::string published_path;
  std::string snapshot_identity;
  ExecutionStorage execution;
  LogicalTimeStorage logical_time;
  PopsWriterRequestV1 value{};

  explicit WriterRequestStorage(const py::dict& request)
      : metadata_json(py::cast<std::string>(py::dict(request["snapshot"])["metadata_json"])),
        selection_identity(
            py::cast<std::string>(py::dict(request["snapshot"])["selection_identity"])),
        temporary_path(py::cast<std::string>(request["temporary_path"])),
        published_path(py::cast<std::string>(request["published_path"])),
        snapshot_identity(py::cast<std::string>(request["snapshot_identity"])),
        execution(py::cast<py::dict>(request["execution"])),
        logical_time(py::cast<py::dict>(request["logical_time"])) {
    const auto snapshot = py::cast<py::dict>(request["snapshot"]);
    const auto raw_geometries = py::cast<py::list>(snapshot["geometries"]);
    geometry_storage.reserve(raw_geometries.size());
    geometries.reserve(raw_geometries.size());
    for (const auto raw : raw_geometries)
      geometry_storage.emplace_back(py::cast<py::dict>(raw));
    for (const auto& row : geometry_storage) geometries.push_back(row.value);
    const auto raw_fields = py::cast<py::list>(snapshot["fields"]);
    field_storage.reserve(raw_fields.size());
    fields.reserve(raw_fields.size());
    for (const auto raw : raw_fields)
      field_storage.emplace_back(py::cast<py::dict>(raw));
    for (const auto& row : field_storage) fields.push_back(row.value);
    const auto raw_diagnostics = py::cast<py::list>(snapshot["diagnostics"]);
    diagnostic_storage.reserve(raw_diagnostics.size());
    diagnostics.reserve(raw_diagnostics.size());
    for (const auto raw : raw_diagnostics)
      diagnostic_storage.emplace_back(py::cast<py::dict>(raw));
    for (const auto& row : diagnostic_storage) diagnostics.push_back(row.value);
    value = {sizeof(PopsWriterRequestV1), geometries.size(), geometries.data(),
             fields.size(), fields.data(), diagnostics.size(), diagnostics.data(),
             metadata_json.c_str(), selection_identity.c_str(), temporary_path.c_str(),
             published_path.c_str(), snapshot_identity.c_str(), logical_time.value,
             execution.value};
    validate_writer_request(value);
  }
};

inline py::dict writer_receipt(PopsWriterReceiptV1 receipt, int code,
                               const char* operation) {
  const auto reason = receipt.status.reason == nullptr ? "" : receipt.status.reason;
  if (code != 0 || receipt.status.code != 0 ||
      receipt.status.action != POPS_COMPONENT_CONTINUE_V1)
    throw std::runtime_error(std::string("native Writer ") + operation +
                             " failed: " + reason);
  py::dict result;
  result["bytes_written"] = receipt.bytes_written;
  result["content_digest"] =
      receipt.content_digest == nullptr ? "" : receipt.content_digest;
  result["action"] = static_cast<int>(receipt.status.action);
  return result;
}

template <class Operation>
py::object invoke_writer(Operation operation, LoadedComponent& loaded,
                         const py::dict& request_data, const char* operation_name) {
  require_operation(operation != nullptr, operation_name);
  WriterRequestStorage request(request_data);
  void* state = loaded.prepared_state(
      POPS_NATIVE_INTERFACE_WRITER_V1, @WRITER_VERSION@, request.execution.value);
  if constexpr (std::is_same_v<Operation, PopsWriterCleanupFnV1>) {
    operation(state, &request.value);
    return py::none();
  } else {
    PopsWriterReceiptV1 receipt{
        sizeof(PopsWriterReceiptV1), 0, nullptr,
        {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
    const int code = operation(state, &request.value, &receipt);
    return writer_receipt(receipt, code, operation_name);
  }
}

inline py::object invoke_component_operation(
    LoadedComponent& loaded, const std::string& interface_uri,
    std::uint32_t interface_version, const std::string& operation,
    const py::dict& request) {
  if (interface_uri == @WRITER_URI@ && interface_version == @WRITER_VERSION@) {
    const auto& api = loaded.table<PopsWriterApiV1>(POPS_NATIVE_INTERFACE_WRITER_V1,
                                                    interface_version);
@WRITER_ROWS@
    throw py::value_error("Writer operation is not declared by the component catalog");
  }
  throw py::value_error("native component interface has no generated Python invoker");
}

struct TransferFieldStorage {
  std::string layout_identity;
  std::string patch_identity;
  DoubleArray values;
  PopsConstFieldViewV1 const_value{};
  PopsFieldViewV1 mutable_value{};

  TransferFieldStorage(const py::dict& row, bool writable)
      : layout_identity(py::cast<std::string>(row["layout_identity"])),
        patch_identity(py::cast<std::string>(row["patch_identity"])),
        values(required_array<DoubleArray>(row["values"], "Transfer field values")) {
    const auto dimension = py::cast<std::int32_t>(row["dimension"]);
    const auto extents = py::cast<std::array<std::size_t, 3>>(row["extents"]);
    const auto ghost_lower = py::cast<std::array<std::size_t, 3>>(row["ghost_lower"]);
    const auto ghost_upper = py::cast<std::array<std::size_t, 3>>(row["ghost_upper"]);
    if (dimension < 1 || dimension > 3 || values.ndim() != dimension + 1 ||
        values.shape(0) <= 0)
      throw py::value_error("Transfer array rank differs from its field dimension");
    std::array<std::ptrdiff_t, 3> strides{0, 0, 0};
    const auto item = static_cast<py::ssize_t>(sizeof(double));
    for (std::int32_t axis = 0; axis < 3; ++axis) {
      if (axis < dimension) {
        if (extents[axis] != static_cast<std::size_t>(values.shape(axis + 1)))
          throw py::value_error("Transfer array shape differs from declared extents");
        strides[axis] = values.strides(axis + 1) / item;
      } else if (extents[axis] != 1 || ghost_lower[axis] != 0 || ghost_upper[axis] != 0) {
        throw py::value_error("Transfer descriptor carries inactive-axis metadata");
      }
    }
    const_value = {
        sizeof(PopsConstFieldViewV1), values.data(), dimension,
        {extents[0], extents[1], extents[2]}, {strides[0], strides[1], strides[2]},
        static_cast<std::size_t>(values.shape(0)), values.strides(0) / item,
        static_cast<PopsFieldCenteringV1>(py::cast<std::int32_t>(row["centering"])),
        py::cast<std::uint32_t>(row["centering_axes"]),
        {ghost_lower[0], ghost_lower[1], ghost_lower[2]},
        {ghost_upper[0], ghost_upper[1], ghost_upper[2]},
        static_cast<PopsScalarTypeV1>(py::cast<std::int32_t>(row["scalar_type"])),
        static_cast<PopsMemorySpaceV1>(py::cast<std::int32_t>(row["memory_space"])),
        layout_identity.c_str(), patch_identity.c_str(),
        static_cast<PopsFieldOwnershipV1>(py::cast<std::int32_t>(row["ownership"]))};
    mutable_value = {sizeof(PopsFieldViewV1), writable ? values.mutable_data() : nullptr,
                     const_value.dimension,
                     {extents[0], extents[1], extents[2]},
                     {strides[0], strides[1], strides[2]}, const_value.component_count,
                     const_value.component_stride, const_value.centering,
                     const_value.centering_axes,
                     {ghost_lower[0], ghost_lower[1], ghost_lower[2]},
                     {ghost_upper[0], ghost_upper[1], ghost_upper[2]},
                     const_value.scalar_type, const_value.memory_space,
                     layout_identity.c_str(), patch_identity.c_str(), const_value.ownership};
  }
};

inline py::dict transfer_apply(
    LoadedComponent& loaded, const py::dict& source_data,
    const py::dict& destination_data, const std::vector<std::int32_t>& ratio,
    std::int32_t raw_operation, const py::dict& execution_data) {
  TransferFieldStorage source(source_data, false);
  TransferFieldStorage destination(destination_data, true);
  ExecutionStorage execution(execution_data);
  if (ratio.size() != static_cast<std::size_t>(source.const_value.dimension))
    throw py::value_error("Transfer ratio length differs from field dimension");
  const auto operation = static_cast<PopsTransferOperationV1>(raw_operation);
  PopsTransferRequestV1 request{
      sizeof(PopsTransferRequestV1), source.const_value, destination.mutable_value,
      ratio.data(), source.const_value.dimension, operation, execution.value};
  PopsComponentStatusV1 status{
      sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  const auto& api = loaded.table<PopsTransferApiV1>(POPS_NATIVE_INTERFACE_TRANSFER_V1,
                                                    @TRANSFER_VERSION@);
  void* state = loaded.prepared_state(
      POPS_NATIVE_INTERFACE_TRANSFER_V1, @TRANSFER_VERSION@, execution.value);
  const int code = apply_transfer(api, state, request, status);
  if (code != 0 || status.code != 0 || status.action != POPS_COMPONENT_CONTINUE_V1)
    throw std::runtime_error(status.reason == nullptr ? "native Transfer failed"
                                                      : status.reason);
  py::dict receipt;
  receipt["provider_component_id"] = loaded.api().component_id;
  receipt["provider_manifest_identity"] = loaded.api().manifest_identity;
  receipt["operation"] = static_cast<std::int32_t>(operation);
  receipt["source_element_count"] =
      field_point_count(source.const_value) * source.const_value.component_count;
  receipt["destination_element_count"] =
      field_point_count(destination.mutable_value) * destination.mutable_value.component_count;
  receipt["applied"] = true;
  return receipt;
}

template <class HandleClass>
void register_component_invokers(HandleClass& handle) {
  handle.def("_invoke_component_operation", &invoke_component_operation,
             py::arg("interface_uri"), py::arg("interface_version"),
             py::arg("operation"), py::arg("request"));
  handle.def("_transfer_apply", &transfer_apply, py::arg("source"),
             py::arg("destination"), py::arg("ratio"), py::arg("operation"),
             py::arg("execution_context"));
}

}  // namespace pops::component::generated_pybind
'''
    return (source.replace("@DIGEST@", digest)
            .replace("@WRITER_URI@", _cpp_string(writer["uri"]))
            .replace("@WRITER_VERSION@", str(writer["version"]) + "u")
            .replace("@WRITER_ROWS@", writer_rows)
            .replace("@TRANSFER_VERSION@", str(transfer["version"]) + "u"))


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
        "// clang-format off",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
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
        "enum class ComponentInterfaceId : std::uint8_t {",
    ]
    for index, interface in enumerate(catalog["interface_vocabulary"]):
        out.append(f"  k{_camel(interface['name'])} = {index},")
    out.extend((
        "};",
        "struct ComponentInterfaceInfo {",
        "  ComponentInterfaceId id; const char* name; const char* method; int required_args;",
        "};",
        "inline constexpr ComponentInterfaceInfo kComponentInterfaces[] = {",
    ))
    for interface in catalog["interface_vocabulary"]:
        out.append(
            f"  {{ComponentInterfaceId::k{_camel(interface['name'])}, "
            f"{_cpp_string(interface['name'])}, {_cpp_string(interface['method'])}, "
            f"{interface['required_args']}}},")
    out.extend((
        "};",
        "constexpr const ComponentInterfaceInfo* find_component_interface(std::string_view name) {",
        "  for (const auto& interface : kComponentInterfaces)",
        "    if (name == interface.name)",
        "      return &interface;",
        "  return nullptr;",
        "}",
        "",
        "enum class RouteFamily : std::uint8_t {",
    ))
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
    out.extend(("};", "", "}  // namespace pops", "// clang-format on", ""))
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
        PY_INTERFACES: _render_native_interfaces(catalog, digest, semantic_digest),
        CPP_CATALOG: _render_cpp(catalog, digest, semantic_digest),
        CPP_ACCESSORS: _render_accessors(catalog, digest),
        CPP_COMPONENT_ABI: _render_component_abi(catalog, digest),
        CPP_PYBIND_INVOKERS: _render_pybind_invokers(catalog, digest),
    }
    changed = any([
        _update(path, content, check=args.check) for path, content in products.items()
    ])
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
