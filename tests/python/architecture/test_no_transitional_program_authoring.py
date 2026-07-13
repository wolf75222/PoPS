"""Final Program authoring has one typed route and no hidden compatibility syntax."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def _tracked_python_files(*roots: str) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--", *roots],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        ROOT / line
        for line in result.stdout.splitlines()
        if line.endswith(".py")
    )


def test_retired_program_authoring_modules_and_inference_are_absent():
    assert not (ROOT / "python/pops/time/program_space_resolution.py").exists()
    source = (ROOT / "python/pops/time/program_core.py").read_text(encoding="utf-8")
    assert "_default_state_spaces" not in source
    assert "isinstance(state, StateHandle)" in source


def test_program_rhs_has_one_required_typed_terms_signature():
    path = ROOT / "python/pops/time/program_rhs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rhs = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "rhs"
    )
    assert rhs.args.kwarg is None
    names = [argument.arg for argument in rhs.args.kwonlyargs]
    assert names == ["terms"]
    assert rhs.args.kw_defaults == [None], "terms= must be explicit, not a hidden default"


def test_tracked_source_and_tests_never_use_retired_program_tokens():
    retired_rhs = "_rhs_" + "legacy"
    retired_resolver = "resolve_registered_" + "operator"
    private_projection = "_rhs_" + "primitive"
    offenders: list[str] = []
    for path in _tracked_python_files("python/pops", "tests/python"):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(ROOT))
        if retired_rhs in text or retired_resolver in text:
            offenders.append(relative)
        if relative.startswith("tests/python/") and private_projection in text:
            offenders.append(relative)
        if relative.startswith("tests/python/") and (
            "._call(\"" in text or "._call('" in text
        ):
            offenders.append(relative)
    assert not sorted(set(offenders)), (
        "retired/free-name Program authoring remains in: %s" % sorted(set(offenders))
    )


def test_normative_examples_never_call_private_program_builders():
    offenders = [
        str(path.relative_to(ROOT))
        for path in _tracked_python_files("examples/final")
        if "._call(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders
