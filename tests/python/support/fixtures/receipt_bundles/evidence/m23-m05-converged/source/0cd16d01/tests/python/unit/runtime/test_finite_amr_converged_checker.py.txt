"""Archive checker counter-tests; no simulation, compiler, or PoPS import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[4] / "docs/development/api_040"
SPEC = importlib.util.spec_from_file_location("finite_receipt_checker", ROOT / "check_finite_amr_converged.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def copied(tmp_path):
    return Path(shutil.copytree(ROOT / "evidence/finite-amr-converged", tmp_path / "receipt"))


def reseal(bundle, path):
    manifest = bundle / "manifest.json"
    data = json.loads(manifest.read_text())
    data["files"][str(path.relative_to(bundle))] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    manifest.write_text(json.dumps(data))


def test_archived_receipt_counts_and_failures_are_preserved():
    result = CHECKER.verify_bundle(ROOT / "evidence/finite-amr-converged")
    assert result["saved_m09_states"] == 32
    assert result["retained_failed_xml_rows"] == 3


def test_corrupt_saved_state_is_refused(tmp_path):
    bundle = copied(tmp_path)
    state = next(bundle.rglob("*.npz"))
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        CHECKER.verify_bundle(bundle)


def test_missing_rank_receipt_is_refused(tmp_path):
    bundle = copied(tmp_path)
    (bundle / "raw/installed-finite-m09-mpi2/rank1.xml").unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        CHECKER.verify_bundle(bundle)


def test_removing_old_failure_cannot_be_hidden_by_rehashing(tmp_path):
    bundle = copied(tmp_path)
    path = bundle / "raw/native-w12-reviewed-ctest.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    for row in root.iter("testcase"):
        failure = row.find("failure")
        if failure is not None:
            row.remove(failure)
    root.set("failures", "0")
    tree.write(path)
    reseal(bundle, path)
    with pytest.raises(ValueError, match="unexpected test status"):
        CHECKER.verify_bundle(bundle)


def test_unexpected_cache_binary_is_refused(tmp_path):
    bundle = copied(tmp_path)
    path = bundle / "accidental.so"
    path.write_bytes(b"not a receipt")
    reseal(bundle, path)
    with pytest.raises(ValueError, match="binary/cache payload forbidden"):
        CHECKER.verify_bundle(bundle)
