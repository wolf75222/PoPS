"""Final cut-over fences: one authoring surface and no importable legacy facade."""
from __future__ import annotations

import ast
from pathlib import Path

import pops


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "python" / "pops"


def _removed_names() -> tuple[str, ...]:
    # Keep the retired spellings out of normative source scans performed by this test itself.
    return (
        "Pro" + "blem",
        "Runtime" + "Policies",
        "Output" + "Policy",
        "Checkpoint" + "Policy",
        "Sys" + "tem",
        "Amr" + "System",
        "Model" + "Spec",
    )


def test_retired_root_and_runtime_exports_are_absent() -> None:
    from pops import runtime

    for name in _removed_names():
        assert name not in pops.__all__
        assert not hasattr(pops, name)
    for name in _removed_names()[-3:]:
        assert name not in runtime.__all__
        assert not hasattr(runtime, name)


def test_removed_public_modules_do_not_exist() -> None:
    for relative in (
        "case.py",
        "dsl.py",
        "output/policies.py",
        "output/runtime_policies.py",
        "runtime_policies.py",
    ):
        assert not (PACKAGE / relative).exists()


def test_case_has_one_registration_spelling_per_authority() -> None:
    case = pops.Case("canonical")
    assert hasattr(case, "block") and not hasattr(case, "add_block")
    assert hasattr(case, "field") and not hasattr(case, "add_field")
    assert hasattr(case, "program") and not hasattr(case, "time")
    assert hasattr(case, "consumers") and not hasattr(case, "output")


def test_program_has_one_runtime_branch_spelling() -> None:
    source = (PACKAGE / "time" / "program_authoring.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    authoring = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ProgramAuthoring")
    methods = {
        node.name for node in authoring.body if isinstance(node, ast.FunctionDef)}
    assert "branch" in methods
    assert "if_" not in methods


def test_time_presets_are_factories_for_ordinary_programs() -> None:
    from pops.lib import time

    assert callable(time.SSPRK2)
    assert not hasattr(time, "ssprk2")
