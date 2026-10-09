"""Protocol/source tests only. No saved physical states or native receipts are fabricated."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/review/sol61_integral_feedback_offline_oracle.py"
spec = importlib.util.spec_from_file_location("sol61_feedback_offline", SCRIPT)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def test_cli_pending_contract_imports_no_pops_and_writes_no_runtime_states():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert all(not (name == "pops" or name.startswith("pops.")) for name in imports)
    assert "np.save" not in source and "np.zeros" not in source and "np.ones" not in source
    # Run under an import blocker as well as checking the static import inventory.
    harness = "import sys,importlib.abc,runpy\n" + '''
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname == "pops" or fullname.startswith("pops."): raise RuntimeError("PoPS forbidden")
sys.meta_path.insert(0,Deny())
''' + f"sys.argv=[{str(SCRIPT)!r},'--describe-contract']\nrunpy.run_path({str(SCRIPT)!r},run_name='__main__')\n"
    child = subprocess.run([sys.executable, "-I", "-c", harness], capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["scientific_status"] == "pending_receipts"
    assert [(row["kind"], row["cells"]) for row in result["expected_cases"]] == [
        ("restart", 8), ("restart", 16), ("retry", 8)]
    assert result["tolerance"] == 3e-13


@pytest.mark.parametrize("raw", ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'))
def test_strict_json_rejects_metadata_ambiguity(raw):
    with pytest.raises(ValueError):
        oracle.strict_json(raw)


def test_independent_cbor_matches_standard_vectors_and_distinguishes_bool():
    assert oracle.cbor(None).hex() == "f6"
    assert oracle.cbor(True).hex() == "f5"
    assert oracle.cbor(1).hex() == "01"
    assert oracle.cbor(-1000).hex() == "3903e7"
    assert oracle.cbor([1, 2, 3]).hex() == "83010203"
    assert oracle.cbor({"b": [2, 3], "a": 1}).hex() == "a26161016162820203"
    assert oracle.cbor({"aa": 1, "b": 2}).hex() == "a261620262616101"
    for invalid in (1., 1 << 63, -(1 << 63) - 1, {1: "not-string-key"}):
        with pytest.raises(ValueError):
            oracle.cbor(invalid)


@pytest.mark.parametrize("raw", (b"", b"POPSEX01" + bytes(24), b"POPSEX02", b"POPSEX02" + bytes(8),
                                  b"POPSEX02" + (1 << 63).to_bytes(8, "little")))
def test_ledger_protocol_rejects_unknown_truncated_or_unbounded_images(raw):
    with pytest.raises(ValueError):
        oracle.exchange_image(raw)


def test_protocol_empty_image_is_not_scientific_evidence():
    # Only a wire-format empty image, not a fabricated native state/checkpoint.
    assert oracle.exchange_image(b"POPSEX02" + bytes(24)) == dict(records=[], quantities={}, consumed=[])
    with pytest.raises(ValueError, match="trailing ledger bytes"):
        oracle.exchange_image(b"POPSEX02" + bytes(25))


def _frame(*, dt=.01, time=.01, tick=1, denominator=1, evaluation="source/evaluation"):
    def bits(value):
        return int.from_bytes(struct.pack("<d", value), "little")
    clock = "clock-test"
    context = (f"pops.exchange.frame.v1/{len(clock)}:{clock}/{tick}/0/0/0/0/{denominator}/"
               f"{bits(dt)}/{bits(time)}/{len(evaluation)}:{evaluation}")
    return dict(context=context, evaluation=evaluation)


def test_exact_frame_accepts_source_names_with_and_without_slashes():
    assert oracle.frame(_frame(), .01, 1) == "clock-test"
    assert oracle.frame(_frame(evaluation="plain"), .01, 1) == "clock-test"


@pytest.mark.parametrize("kwargs", (dict(dt=.010000000000000002), dict(time=0.), dict(tick=0),
                                    dict(denominator=2), dict(dt=float("nan"))))
def test_exact_frame_refuses_duration_clock_or_point_relabelling(kwargs):
    with pytest.raises(ValueError, match="runtime interval/duration"):
        oracle.frame(_frame(**kwargs), .01, 1)


def _assert_scientific_fixture_seams(source, frozen):
    def seams(text):
        tree = ast.parse(text)
        names = {"_case", "_oracle", "_receipt_check"}
        functions = {node.name: ast.dump(node, include_attributes=False)
                     for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
        assert set(functions) == names
        constants = [ast.dump(node, include_attributes=False) for node in tree.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Tuple) and [item.id for item in target.elts
                         if isinstance(item, ast.Name)] == ["DT", "GAMMA", "Q0", "TOL"]
                         for target in node.targets)]
        assert len(constants) == 1
        return functions, constants
    assert seams(source) == seams(frozen)


def test_source_fixture_contract_is_frozen_and_files_need_external_state_pins():
    # Authenticate the historical source, independently of later fixture extensions.
    # The archive checker authenticates the same fixture bytes against the owner seal.
    revision = "634cba3511fef957e65a21fcae3ab257f164b360"
    hashes = {
        "tests/python/integration/runtime/test_public_integral_feedback.py":
            "db1d5153eeb59c8e15e80f22837a9889c30e08f37dc99dab96a1e422fc12962d",
        "tests/python/support/integral_state_receipts.py":
            "bd8e4d7cf6bebfe1b816709433ce777e53c19514dd6900ab3293a9754879d4ca",
        "tests/python/unit/codegen/test_integral_candidate_capture.py":
            "f4fa9b22cfaf7c739dfd819e3699b742a86e4e26d852453d5e825811c0d0c16c"}
    frozen = {}
    for path, expected in hashes.items():
        raw = subprocess.run(["git", "show", revision + ":" + path], cwd=ROOT,
                             capture_output=True, check=True).stdout
        assert hashlib.sha256(raw).hexdigest() == expected
        frozen[path] = raw.decode()
    fixture, saver_path, _ = hashes
    source = (ROOT / fixture).read_text()
    _assert_scientific_fixture_seams(source, frozen[fixture])
    assert "@pytest.mark.parametrize(\"cells\",(8,16))" in source
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    assert any(isinstance(node.func, ast.Name) and node.func.id == "_case"
               and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 8
               and any(kw.arg == "proposed_dt" and isinstance(kw.value, ast.Constant)
                       and kw.value.value == .5 for kw in node.keywords) for node in calls)
    assert "record_property(\"dim\",2)" in source
    saver = (ROOT / saver_path).read_text()
    assert "checkpoint_sha256" in saver and "ledger_sha256" in saver
    assert "state_sha256" not in saver
    assert set(oracle.contract()["phase_pins"]) == {"receipt", "state", "checkpoint"}


def test_absent_receipts_fail_without_creating_a_pass_report(tmp_path):
    output = tmp_path / "not-a-result.json"
    child = subprocess.run([sys.executable, "-I", str(SCRIPT), "--pins", str(tmp_path / "absent.json"),
                            "--output", str(output)], capture_output=True, text=True, timeout=15)
    assert child.returncode != 0
    assert not output.exists()


def test_external_checksum_is_required_even_for_metadata_only(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text('{"not":"a runtime receipt"}')
    with pytest.raises(ValueError, match="SHA256 pin mismatch"):
        oracle.pinned_file(tmp_path, dict(path=path.name, sha256="0" * 64))
    with pytest.raises(ValueError, match="invalid external file pin"):
        oracle.pinned_file(tmp_path, dict(path=path.name))
