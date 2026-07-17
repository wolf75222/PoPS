"""ADC-681 fences against central scientific concrete-class dispatch."""
from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PYTHON_BOUNDARY = (
    ROOT / "python/pops/model/component_adapters.py",
    ROOT / "python/pops/model/component_registry.py",
)


def test_component_trust_boundary_never_classifies_the_scientific_component_type():
    forbidden_imports = (
        "pops.numerics", "pops.mesh", "pops.solvers", "pops.fields",
        "pops.runtime._engine_descriptors",
    )
    for path in PYTHON_BOUNDARY:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(name in source for name in forbidden_imports), path
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "isinstance" or not node.args:
                continue
            first = node.args[0]
            assert not isinstance(first, ast.Name) or first.id != "component", (
                "%s centrally dispatches on a scientific component class at line %d"
                % (path, node.lineno)
            )


def test_native_registry_has_no_rtti_or_untyped_capability_escape_hatch():
    header = (ROOT / "include/pops/runtime/config/component_interfaces.hpp").read_text(
        encoding="utf-8")
    behavior = re.sub(r"//.*?$|/\*.*?\*/", "", header,
                      flags=re.MULTILINE | re.DOTALL)
    for forbidden in ("dynamic_cast", "typeid(", "provides(any", "void* component"):
        assert forbidden not in behavior
    for interface in (
        "Requirement", "Lowering", "Stencil", "Stability", "Provider", "Effects",
        "Restart", "Report", "FallibleEvaluation", "Format",
    ):
        assert "concept %s" % interface in header
