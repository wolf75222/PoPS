"""ADC-679: one declaration generates every Python/C++ route registry surface."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "schemas" / "component_catalog.v2.json"
GENERATOR = ROOT / "scripts" / "generate_component_catalog.py"
PYTHON_ROUTES = ROOT / "python" / "pops" / "runtime" / "_generated_component_routes.py"
CPP_CATALOG = ROOT / "include" / "pops" / "runtime" / "config" / "generated_component_catalog.hpp"
CPP_ACCESSORS = ROOT / "include" / "pops" / "runtime" / "config" / "generated_route_accessors.inc"
ROUTE_API = ROOT / "include" / "pops" / "runtime" / "config" / "route_ids.hpp"
DISPATCH_API = ROOT / "include" / "pops" / "runtime" / "config" / "dispatch_tags.hpp"
MODEL_API = ROOT / "include" / "pops" / "runtime" / "dynamic" / "model_registry.hpp"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checked_in_products_are_exact_generator_output():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_catalog_rows_are_the_generated_python_and_cpp_rows():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    generated = _load(PYTHON_ROUTES, "_generated_component_routes_contract")
    cpp = CPP_CATALOG.read_text(encoding="utf-8")

    expected = {}
    for family in catalog["route_families"]:
        rows = tuple((
            row["token"], row["native_entry"], tuple(row["requirements"]),
            tuple(row["limitations"]),
        ) for row in family["routes"])
        expected[family["name"]] = rows
        assert f"enum class {family['cpp_enum']}" in cpp
        assert f"RouteInfo {family['cpp_table']}[]" in cpp
        for row in family["routes"]:
            assert f"{row['cpp_id']} = {row['wire_id']}" in cpp
    assert generated.ROUTE_TABLES == expected
    assert generated.COMPONENT_CATALOG_SHA256 in cpp
    assert generated.ROUTE_REGISTRY_SIGNATURE in cpp


def test_runtime_headers_consume_generated_catalog_instead_of_mirroring_rows():
    for path in (ROUTE_API, DISPATCH_API, MODEL_API):
        source = path.read_text(encoding="utf-8")
        assert "generated_component_catalog.hpp" in source
    assert "inline constexpr RouteInfo kRiemannRoutes[]" not in ROUTE_API.read_text(encoding="utf-8")
    assert "inline constexpr LimiterTag kLimiters[]" not in DISPATCH_API.read_text(encoding="utf-8")
    assert "inline constexpr TransportTag kTransports[]" not in MODEL_API.read_text(encoding="utf-8")


def test_one_catalog_row_generates_both_language_surfaces():
    generator = _load(GENERATOR, "_component_catalog_generator_contract")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    modified = copy.deepcopy(catalog)
    family = next(item for item in modified["route_families"] if item["name"] == "wall")
    family["routes"].append({
        "token": "contract_probe",
        "wire_id": len(family["routes"]),
        "cpp_id": "kContractProbe",
        "native_entry": "pops::ContractProbe",
        "requirements": [],
        "limitations": [],
        "aliases": [],
        "metadata": {},
    })
    python_product = generator._render_routes(modified, "0" * 64)
    cpp_product = generator._render_cpp(modified, "0" * 64)
    assert "contract_probe" in python_product
    assert "kContractProbe" in cpp_product


def test_one_new_family_generates_its_cpp_accessors_without_route_ids_edit():
    generator = _load(GENERATOR, "_component_catalog_generator_family_contract")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["route_families"].append({
        "name": "contract_family",
        "cpp_enum": "ContractFamilyRouteId",
        "cpp_table": "kContractFamilyRoutes",
        "routes": [{
            "token": "probe",
            "wire_id": 0,
            "cpp_id": "kProbe",
            "native_entry": "pops::ContractProbe",
            "requirements": [],
            "limitations": [],
            "aliases": [],
            "metadata": {},
        }],
    })
    accessor_product = generator._render_accessors(catalog, "0" * 64)
    assert (
        "POPS_DEFINE_ROUTE_ACCESSORS(contract_family, ContractFamilyRouteId, "
        "kContractFamilyRoutes, kContractFamily)"
    ) in accessor_product
    assert "generated_route_accessors.inc" in ROUTE_API.read_text(encoding="utf-8")
    assert CPP_ACCESSORS.read_text(encoding="utf-8") == generator._render_accessors(
        json.loads(CATALOG.read_text(encoding="utf-8")),
        _load(PYTHON_ROUTES, "_generated_accessors_digest").COMPONENT_CATALOG_SHA256,
    )


def test_no_unknown_fields_can_hide_in_catalog_rows():
    generator = _load(GENERATOR, "_component_catalog_generator_validation")
    try:
        generator._exact({"known": 1, "surprise": 2}, {"known"}, "probe")
    except generator.CatalogError as error:
        assert "unknown=['surprise']" in str(error)
    else:
        raise AssertionError("unknown semantic catalog field was accepted")


def test_documentation_changes_only_the_full_catalog_identity():
    generator = _load(GENERATOR, "_component_catalog_digest_contract")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    full, semantic = generator._catalog_digests(catalog)

    documentary = copy.deepcopy(catalog)
    transport = next(row for row in documentary["route_families"]
                     if row["name"] == "transport")
    transport["routes"][0]["metadata"]["summary"] = "documentation-only probe"
    documentary_full, documentary_semantic = generator._catalog_digests(documentary)
    assert documentary_full != full
    assert documentary_semantic == semantic

    behavior = copy.deepcopy(catalog)
    behavior["route_families"][0]["routes"][0]["native_entry"] = "pops::OtherFlux"
    behavior_full, behavior_semantic = generator._catalog_digests(behavior)
    assert behavior_full != full
    assert behavior_semantic != semantic


def test_native_behavior_contains_no_hand_written_generated_route_lists():
    generated = _load(PYTHON_ROUTES, "_generated_route_list_fence")
    pairs = []
    for rows in generated.ROUTE_TABLES.values():
        tokens = [row[0] for row in rows]
        pairs.extend(
            re.compile(re.escape(left) + r"\s*\|\s*" + re.escape(right))
            for left, right in zip(tokens, tokens[1:], strict=False)
        )
    string_literal = re.compile(r'"((?:[^"\\]|\\.)*)"')
    offenders = []
    for root in (ROOT / "include", ROOT / "python" / "bindings"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".hpp", ".h", ".cpp", ".cc"}:
                continue
            if path == CPP_CATALOG:
                continue
            source = re.sub(r"//.*?$|/\*.*?\*/", "", path.read_text(
                encoding="utf-8", errors="replace"), flags=re.MULTILINE | re.DOTALL)
            for literal in string_literal.findall(source):
                if any(pattern.search(literal) for pattern in pairs):
                    offenders.append((path.relative_to(ROOT).as_posix(), literal))
    assert not offenders, (
        "native route refusals must enumerate generated tables at runtime:\n"
        + "\n".join("  %s: %r" % item for item in offenders)
    )
