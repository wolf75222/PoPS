"""Source-only integrity checks for the executable M2 temporal gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path


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
    assert len(data["check"]) == 14


def test_m2_open_dependencies_are_deferred_not_claimed():
    data, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors
    deferred = {row["issue"]: row for row in data["deferred"]}
    assert set(deferred) == {"ADC-648", "ADC-667"}
    assert all(row["reason"] and row["close_condition"] for row in deferred.values())
    assert not ({row["issue"] for row in data["check"]} & set(deferred))
