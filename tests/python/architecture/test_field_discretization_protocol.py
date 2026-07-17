"""The field-plan extension seam is structural, never a concrete-class switch."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BOUNDARIES = (
    ROOT / "python/pops/problem/_declaration_registries.py",
    ROOT / "python/pops/codegen/field_install.py",
)


def test_field_registration_and_lowering_do_not_dispatch_on_concrete_plan_classes():
    for path in BOUNDARIES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "isinstance" or len(node.args) < 2:
                continue
            classifier = ast.unparse(node.args[1])
            assert "FieldDiscretization" not in classifier, (
                "%s centrally dispatches a field plan at line %d" % (path, node.lineno)
            )
