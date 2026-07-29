"""Source-only integrity checks for the executable M2 temporal gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/gates/m2_temporal_execution.toml"
RUNNER = ROOT / "scripts/run_m2_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_m2_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m2_manifest_references_only_real_mandatory_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors, "M2 gate matrix is incomplete:\n  " + "\n  ".join(errors)
    assert len(data["check"]) == 28


def test_m2_final_gate_has_no_deferred_requirement():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    assert data["deferred"] == []
    assert {row["issue"] for row in data["check"]} == {
        "ADC-648", "ADC-661", "ADC-662", "ADC-663", "ADC-664", "ADC-665", "ADC-666",
        "ADC-667", "ADC-668",
    }


def test_m2_native_pytest_execution_rejects_every_skip_or_xfail(tmp_path, monkeypatch):
    runner = _load_runner()
    report = tmp_path / "pytest.xml"
    skipped_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite tests="1" skipped="1">'
        '<testcase classname="m2" name="proof"><skipped type="pytest.skip"/></testcase>'
        "</testsuite></testsuites>"
    )
    report.write_text(skipped_xml, encoding="utf-8")
    assert runner._pytest_skip_count(report) == 1

    source = RUNNER.read_text(encoding="utf-8")
    assert 'environment["POPS_REQUIRE_NATIVE_TESTS"] = "1"' in source
    assert '"xfail_strict=true"' in source
    assert "skipped/xfail proof(s); every proof is mandatory" in source

    def successful_pytest_with_a_skip(command, *, cwd, env, check):
        assert cwd == ROOT
        assert env["POPS_REQUIRE_NATIVE_TESTS"] == "1"
        assert check is False
        junit = Path(command[command.index("--junitxml") + 1])
        junit.write_text(skipped_xml, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", successful_pytest_with_a_skip)
    with pytest.raises(RuntimeError, match="reported 1 skipped/xfail proof"):
        runner._run_pytest(["tests/python/unit/time/test_exact_program_graph.py::proof"])


def test_m2_restart_refusal_and_program_only_routes_use_real_exact_proofs():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    checks = data["check"]
    restart_refusals = {
        row.get("nodeid")
        for row in checks
        if row["requirement"] == "restart" and row["polarity"] == "refusal"
    }
    assert restart_refusals == {
        "tests/python/integration/io/test_time_history_checkpoint.py"
        "::test_uniform_restart_refuses_a_different_compiled_program"
    }
    temporal_routes = {
        row["nodeid"]
        for row in checks
        if row["requirement"] == "program_only_temporal_routes"
    }
    assert temporal_routes == {
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_system_temporal_facades_dispatch_only_through_an_installed_program",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_amr_temporal_facades_use_amr_runtime_only_as_the_spatial_engine",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_static_system_temporal_driver_is_test_only",
        "tests/python/architecture/test_program_only_temporal_facades.py"
        "::test_unlowerable_semantic_tests_remain_real_manifest_tests_without_fe_bridge",
    }
