"""Every current and future ``pops.ir.Expr`` node inherits one data-model contract.

This is intentionally source-only: the architecture lane can enforce the hierarchy before
the native extension is built.  The transitive discovery means adding an Expr subclass in
any ``python/pops/**/*.py`` file automatically brings it under this test.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _class_bases(node: ast.ClassDef) -> set[str]:
    bases: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.add(base.id)
        elif isinstance(base, ast.Attribute):
            bases.add(base.attr)
    return bases


def _class_definitions(package_dir: Path) -> dict[str, list[ast.ClassDef]]:
    definitions: dict[str, list[ast.ClassDef]] = {}
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                definitions.setdefault(node.name, []).append(node)
    return definitions


def test_every_expr_subclass_transitively_inherits_immutable_symbolic(repo_root: Path):
    package_dir = repo_root / "python" / "pops"
    definitions = _class_definitions(package_dir)

    expr = definitions["Expr"][0]
    assert "ImmutableSymbolic" in _class_bases(expr)

    expr_family = {"Expr"}
    changed = True
    while changed:
        changed = False
        for name, nodes in definitions.items():
            if name in expr_family:
                continue
            if any(_class_bases(node) & expr_family for node in nodes):
                expr_family.add(name)
                changed = True

    # Pin representative leaves from each IR family while discovery covers all the rest.
    assert {
        "Const", "Compare", "Equation", "TimeDerivative", "Divergence",
        "EllipticSum", "EigWitness", "RuntimeParamRef", "SourceTermExpr",
        "LocalLinearOperatorExpr",
    } <= expr_family
    assert len(expr_family) >= 25

    forbidden_methods = {"__bool__", "__hash__", "__setattr__", "__delattr__"}
    for name in sorted(expr_family - {"Expr"}):
        for node in definitions[name]:
            explicit_methods = {
                stmt.name for stmt in node.body
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name in forbidden_methods
            }
            explicit_hashes = [
                stmt for stmt in node.body
                if isinstance(stmt, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "__hash__"
                    for target in (
                        stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    )
                )
            ]
            assert not explicit_methods and not explicit_hashes, (
                f"{name} overrides symbolic safety barriers {sorted(explicit_methods)}; "
                "Expr nodes must remain immutable, non-hashable, and non-truthy")


def test_immutable_symbolic_defines_all_four_python_safety_barriers(repo_root: Path):
    path = repo_root / "python" / "pops" / "ir" / "symbolic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    base = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImmutableSymbolic"
    )
    methods = {node.name for node in base.body if isinstance(node, ast.FunctionDef)}
    hash_assignments = [
        node for node in base.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__hash__"
                for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]

    assert {"__bool__", "__setattr__", "__delattr__"} <= methods
    assert hash_assignments, "ImmutableSymbolic must set __hash__ = None"
