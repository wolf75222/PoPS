"""ADC-679: builtin brick inspection is a generated view, never a second registry."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "schemas" / "component_catalog.v2.json"
GENERATED_PY = ROOT / "python" / "pops" / "runtime" / "_generated_component_routes.py"
GENERATED_HPP = ROOT / "include" / "pops" / "runtime" / "config" / "generated_component_catalog.hpp"
PYTHON_VIEW = ROOT / "python" / "pops" / "runtime" / "brick_catalog.py"
CPP_VIEW = ROOT / "include" / "pops" / "runtime" / "builders" / "factory" / "brick_catalog.hpp"


def _load_generated():
    spec = importlib.util.spec_from_file_location("_brick_generated_contract", GENERATED_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _expected_rows():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = []
    for family in catalog["route_families"]:
        if family["name"] not in {"transport", "source", "elliptic"}:
            continue
        for route in family["routes"]:
            metadata = route["metadata"]
            rows.append({
                "category": family["name"],
                "id": route["token"],
                "route_index": route["wire_id"],
                "native_entry": route["native_entry"],
                "parameters": tuple(metadata["parameters"]),
                "n_vars": metadata.get("n_vars", metadata.get("min_vars", -1)),
                "polar_ok": metadata.get("polar_ok", False),
                "requirements": tuple(route["requirements"]),
                "limitations": tuple(route["limitations"]),
                "summary": metadata["summary"],
            })
    return tuple(rows)


def test_generated_brick_rows_are_an_exact_catalog_projection():
    generated = _load_generated()
    assert generated.BRICK_CATALOG_ROWS == _expected_rows()


def test_inspection_wrappers_do_not_declare_component_rows():
    py_source = PYTHON_VIEW.read_text(encoding="utf-8")
    cpp_source = CPP_VIEW.read_text(encoding="utf-8")
    assert "from ._generated_component_routes import" in py_source
    assert "BRICK_CATALOG_ROWS =" not in py_source
    assert "generated_component_catalog.hpp" in cpp_source
    assert "BrickCatalogEntry kBrickCatalog" not in cpp_source
    assert "BrickCatalogEntry kBrickCatalog" in GENERATED_HPP.read_text(encoding="utf-8")


def test_no_hand_coded_component_token_list_in_native_errors():
    """Public refusals must derive valid values from generated tables."""
    generated = _load_generated()
    ids_by_category = {}
    for row in generated.BRICK_CATALOG_ROWS:
        ids_by_category.setdefault(row["category"], []).append(row["id"])
    pairs = [
        re.compile(re.escape(left) + r"\s*\|\s*" + re.escape(right))
        for ids in ids_by_category.values()
        for left, right in zip(ids, ids[1:], strict=False)
    ]
    string_literal = re.compile(r'"((?:[^"\\]|\\.)*)"')
    offenders = []
    for root in (ROOT / "include", ROOT / "python" / "bindings"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".hpp", ".h", ".cpp", ".cc"}:
                continue
            if path == GENERATED_HPP:
                continue
            source = re.sub(r"//.*?$|/\*.*?\*/", "", path.read_text(
                encoding="utf-8", errors="replace"), flags=re.MULTILINE | re.DOTALL)
            for literal in string_literal.findall(source):
                if any(pattern.search(literal) for pattern in pairs):
                    offenders.append((path.relative_to(ROOT).as_posix(), literal))
    assert not offenders, (
        "component-token lists in native errors must be generated at runtime:\n"
        + "\n".join("  %s: %r" % item for item in offenders)
    )
