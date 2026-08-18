"""Phase 0 dummy analytic case (in-memory manufactured cosine; no solver)."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "dummy_analytic"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
RUNNER = REPO_ROOT / "scripts" / "run_verification.py"
CHECKER = REPO_ROOT / "scripts" / "check_verification_manifest.py"
MANIFEST = REPO_ROOT / "verification" / "manifest.toml"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dummy_analytic_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("POPS_NATIVE_DIM", None)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_write_dummy_report_writes_four_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_dummy_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()


def test_written_summary_validates_against_report_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_dummy_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"


def test_coverage_one_run_and_no_failures(tmp_path: Path):
    analyze = _load_case_module("analyze")
    analyze.write_dummy_report(tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["coverage"]["cases_run"] == 1
    assert loaded["failures"] == []


def test_manufactured_pair_linf_is_finite():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    u_exact, volumes = exact.exact_sample()
    u_num, _ = run.numerical_sample()
    result = reference_errors(u_num, u_exact, volumes)
    assert math.isfinite(result.linf)


def test_case_modules_do_not_mention_pops_run():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "pops.run" not in text


def test_repo_pr_plan_selects_ph00(tmp_path: Path):
    output = tmp_path / "out"
    result = _run(
        RUNNER,
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    ids = [case["id"] for case in plan["cases"]]
    assert "PH-00" in ids
    assert result.stdout.strip() == f"planned {len(ids)} cases"


def test_repo_pr_plan_refuses_max_nodes_three(tmp_path: Path):
    output = tmp_path / "out"
    result = _run(
        RUNNER,
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "3",
        "--output",
        str(output),
    )
    assert result.returncode == 1
    assert "two-node" in result.stderr.lower() or "two node" in result.stderr.lower()
    assert not (output / "plan.json").exists()


def test_repo_manifest_still_checks_clean():
    result = _run(CHECKER)
    assert result.returncode == 0, result.stderr
