"""Offline archive counter-tests: no PoPS import, JIT, or simulation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[4] / "docs/development/api_040"
SPEC = importlib.util.spec_from_file_location("m23_m05_checker", ROOT / "check_m23_m05_converged.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def copied(tmp_path):
    return Path(shutil.copytree(ROOT / "evidence/m23-m05-converged", tmp_path / "receipt"))


def reseal(bundle, path):
    manifest = bundle / "manifest.json"
    data = json.loads(manifest.read_text())
    data["files"][str(path.relative_to(bundle))] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    manifest.write_text(json.dumps(data))


def test_receipts_preserve_states_ledgers_and_historical_failures():
    result = CHECKER.verify_bundle(ROOT / "evidence/m23-m05-converged")
    assert result["npz"] == 26
    assert result["raw_ledgers"] == 10
    assert result["retained_failed_testcases"] == 4


def test_corrupt_saved_state_is_refused(tmp_path):
    bundle = copied(tmp_path)
    state = next(bundle.rglob("*.npz"))
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        CHECKER.verify_bundle(bundle)


def test_missing_rank_receipt_is_refused(tmp_path):
    bundle = copied(tmp_path)
    (bundle / "raw/installed-m23-native-mpi2/rank1.xml").unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        CHECKER.verify_bundle(bundle)


def test_historical_failure_cannot_be_erased_by_rehashing(tmp_path):
    bundle = copied(tmp_path)
    path = bundle / "raw/installed-m23-native-serial/pytest.xml"
    tree = ET.parse(path)
    for row in tree.getroot().iter("testcase"):
        failure = row.find("failure")
        if failure is not None:
            row.remove(failure)
    for suite in tree.getroot().iter("testsuite"):
        suite.set("failures", "0")
    tree.write(path)
    reseal(bundle, path)
    with pytest.raises(ValueError, match="unexpected archived status"):
        CHECKER.verify_bundle(bundle)


def test_nan_ledger_is_refused_even_after_manifest_rehash(tmp_path):
    bundle = copied(tmp_path)
    path = next((bundle / "raw/installed-m05-reviewed-serial").rglob("ledger_32.json"))
    data = json.loads(path.read_text())
    data[0]["numerical_flux"] = float("nan")
    path.write_text(json.dumps(data))
    reseal(bundle, path)
    with pytest.raises(ValueError, match="nonfinite JSON"):
        CHECKER.verify_bundle(bundle)


def test_python_refresh_cannot_be_attached_to_previous_source_window(tmp_path):
    bundle = copied(tmp_path)
    path = bundle / "source/run-source-commits.json"
    data = json.loads(path.read_text())
    data["installed-implicit-state-native-repaired"] = data["installed-m23-source"]
    path.write_text(json.dumps(data))
    reseal(bundle, path)
    with pytest.raises(ValueError, match="wrong receipt source commit"):
        CHECKER.verify_bundle(bundle)


def test_nested_artifact_identity_must_match_across_backends(tmp_path):
    bundle = copied(tmp_path)
    path = next((bundle / "raw/installed-m23-native-mpi2").rglob("receipt.json"))
    data = json.loads(path.read_text())
    data["records"][0]["run_report"]["artifact_identity"]["digest"]["bytes"] = "0" * 64
    path.write_text(json.dumps(data))
    reseal(bundle, path)
    with pytest.raises(ValueError, match="serial/MPI scientific artifact identity mismatch"):
        CHECKER.verify_bundle(bundle)


def test_cache_binary_cannot_be_added_by_rehashing(tmp_path):
    bundle = copied(tmp_path)
    path = bundle / "accidental.so"
    path.write_bytes(b"not a receipt")
    reseal(bundle, path)
    with pytest.raises(ValueError, match="binary cache payload forbidden"):
        CHECKER.verify_bundle(bundle)
