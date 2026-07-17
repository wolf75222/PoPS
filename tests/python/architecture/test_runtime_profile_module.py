"""Profiling has one cohesive internal owner and no circular helper module."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "python" / "pops" / "runtime"
PROFILE = RUNTIME / "_profile.py"


def test_profile_model_and_summary_have_one_direct_owner():
    assert PROFILE.is_file()
    assert not (RUNTIME / "_profile_summary.py").exists()

    tree = ast.parse(PROFILE.read_text(), str(PROFILE))
    classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert {"Profile", "PerformanceSummary", "_Unavailable"} <= classes

    forbidden = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {
                "pops.runtime._profile", "pops.runtime._profile_summary"}:
            forbidden.append((node.lineno, node.module))
    assert not forbidden


def test_runtime_sources_do_not_reference_the_retired_summary_module():
    offenders = []
    for path in RUNTIME.rglob("*.py"):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if module == "pops.runtime._profile_summary" \
                    or "pops.runtime._profile_summary" in names:
                offenders.append("%s:%d" % (path.name, node.lineno))
    assert not offenders
