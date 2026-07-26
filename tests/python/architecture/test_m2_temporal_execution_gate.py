"""Source-only integrity checks for the executable M2 temporal gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

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
    assert len(data["check"]) == 27
    assert [
        row["nodeid"] for row in data["check"] if row["target"] == "example"
    ] == [
        "tests/python/integration/bindings/test_m1_scalar_advection_pipeline.py::"
        "test_scalar_advection_final_example_runs_outputs_and_bit_identical_restart",
        "tests/python/examples/final/test_imex_amr_final_example.py::"
        "test_example_runs_and_every_scientific_format_reopens",
    ]


def test_m2_ctest_selector_rename_fails_source_integrity(tmp_path):
    tampered = tmp_path / "m2.toml"
    tampered.write_text(
        MANIFEST.read_text(encoding="utf-8").replace(
            "ResidualOperator\\\\.RealIndex1DaeChecksConsistentInitialization",
            "ResidualOperator\\\\.MissingDaeProof",
        ),
        encoding="utf-8",
    )

    _data, errors = _load_runner().validate_manifest(tampered)

    assert any("references missing GoogleTest" in error for error in errors)


def test_m2_gate_refuses_false_completion_while_blockers_remain():
    runner = _load_runner()
    data, errors = runner.validate_manifest(MANIFEST)
    assert not errors
    assert not runner.is_complete(data)
    assert {row["requirement"] for row in data["deferred"]} == {
        "normalized_program_execution",
        "native_solve_outcome_fault_matrix",
        "atomic_rejection_side_effects",
        "strict_temporal_continuation",
        "native_multiblock_implicit_phase",
        "refined_hierarchy_native_ordering",
        "legacy_temporal_route_retirement",
    }
    assert {row["issue"] for row in data["check"]} == {
        "ADC-648",
        "ADC-661",
        "ADC-662",
        "ADC-663",
        "ADC-664",
        "ADC-665",
        "ADC-666",
        "ADC-667",
    }
    assert data["issues"] == [
        "ADC-648",
        "ADC-661",
        "ADC-662",
        "ADC-663",
        "ADC-664",
        "ADC-665",
        "ADC-666",
        "ADC-667",
        "ADC-668",
        "ADC-700",
    ]
    assert {blocker for row in data["deferred"] for blocker in row["blocked_by"]} <= set(
        data["issues"]
    )


def test_m2_default_execution_fails_closed_but_source_integrity_remains_cheap():
    runner = _load_runner()

    assert runner.main(["--check-only"]) == 0
    assert runner.main([]) == 3


def test_m2_runtime_guard_rejects_skip_xfail_and_non_strict_xpass(tmp_path):
    runner = _load_runner()
    scenarios = {
        "skip": "def test_proof():\n    pytest.skip('unavailable')\n",
        "xfail": "@pytest.mark.xfail(reason='known gap')\ndef test_proof():\n    assert False\n",
        "xpass": (
            "@pytest.mark.xfail(reason='stale gap', strict=False)\n"
            "def test_proof():\n    assert True\n"
        ),
    }

    for name, body in scenarios.items():
        proof = tmp_path / ("test_%s.py" % name)
        proof.write_text("import pytest\n\n" + body, encoding="utf-8")
        with pytest.raises(subprocess.CalledProcessError):
            runner._run_mandatory_pytest([str(proof)])


def test_m2_native_examples_run_in_distinct_bounded_process_groups(monkeypatch):
    runner = _load_runner()
    calls = []

    class Process:
        pid = 17

        def wait(self, *, timeout):
            calls.append(("wait", timeout))
            return 0

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    runner._run_isolated_pytest(["first::test", "second::test"], timeout_seconds=17)

    launches = [row for row in calls if row[0] != "wait"]
    assert [command[-1] for command, _kwargs in launches] == [
        "first::test",
        "second::test",
    ]
    assert all(kwargs["start_new_session"] is True for _command, kwargs in launches)
    assert [row for row in calls if row[0] == "wait"] == [
        ("wait", 17),
        ("wait", 17),
    ]


def test_m2_example_timeout_terminates_the_complete_process_group(monkeypatch):
    runner = _load_runner()
    waits = []
    signals = []

    class Process:
        pid = 29

        def wait(self, *, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise runner.subprocess.TimeoutExpired(["pytest"], timeout)
            return -15

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    try:
        runner._run_isolated_pytest(["slow::test"], timeout_seconds=1)
    except RuntimeError as error:
        assert "timeout after 1s" in str(error)
    else:
        raise AssertionError("timed-out example did not fail the M2 battery")

    assert waits == [1, 5]
    assert signals == [(29, runner.signal.SIGTERM)]
