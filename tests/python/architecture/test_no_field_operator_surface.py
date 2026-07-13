"""Source-level lock for the final physical/numerical field architecture."""
from __future__ import annotations

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"
_RETIRED_MODEL_VERBS = {"solve_field", "field_problem", "vector_field"}


def _module(relative_path: str) -> ast.Module:
    path = POPS / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_methods(relative_path: str, class_name: str) -> set[str]:
    tree = _module(relative_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError("class %s not found in %s" % (class_name, relative_path))


def test_final_field_operator_and_discretization_types_are_explicit() -> None:
    assert _class_methods("fields/operator.py", "FieldOperator")
    assert _class_methods("fields/discretization.py", "FieldDiscretization")


def test_model_authors_physics_and_case_binds_numerics() -> None:
    assert "field_operator" in _class_methods(
        "physics/_board_elliptic.py", "_EllipticAuthoringMixin"
    )
    assert "field" in _class_methods("problem/problem.py", "Case")


def test_retired_model_field_verbs_have_no_implementation() -> None:
    implemented: set[str] = set()
    physics = POPS / "physics"
    for path in (physics / "board.py", *sorted(physics.glob("_board_*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        implemented.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

    assert implemented.isdisjoint(_RETIRED_MODEL_VERBS)
