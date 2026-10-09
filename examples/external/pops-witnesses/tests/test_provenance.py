"""Detect accidental substitution relative to retained original execution records."""

import json

import numpy as np
import pytest

from pops_witnesses.provenance import digest, verify_capture, write_json
from pops_witnesses.reference import VARIANTS, check_arrays, reference


def fixture_capture(tmp_path):
    folder = tmp_path
    arrays = reference(VARIANTS["baseline"])
    np.savez(folder / "actual-arrays.npz", **arrays)
    np.savez(folder / "actual-bind-inputs.npz", q=arrays["q0"])
    compiler = folder / "compiler" / "one"
    compiler.mkdir(parents=True)
    binary = folder / "binary.so"
    binary.write_bytes(b"test fixture, not native proof")
    value = digest(binary)
    write_json(compiler / "invocation.json", {"returncode": 0, "output_sha256": value})
    write_json(
        folder / "capture.json",
        {
            "core_before": {},
            "core_after": {},
            "loaded_paths": ["/fixture/binary.so"],
            "loaded_binaries": [
                {
                    "original_path": "/fixture/binary.so",
                    "retained": "binary.so",
                    "sha256": value,
                    "role": "program",
                }
            ],
        },
    )
    start = {
        "event": "start",
        "pid": 7,
        "job_id": "fixture",
        "input_sha256": digest(folder / "actual-bind-inputs.npz"),
    }
    finish = {
        "event": "finish",
        "pid": 7,
        "job_id": "fixture",
        "accepted_steps": 1,
        "output_sha256": digest(folder / "actual-arrays.npz"),
        "receipt_sha256": digest(folder / "capture.json"),
        "files": {"binary.so": value},
    }
    (folder / "execution.jsonl").write_text(json.dumps(start) + "\n" + json.dumps(finish) + "\n")
    return folder


def test_original_integrity_reader(tmp_path):
    verify_capture(fixture_capture(tmp_path))


def test_oracle_substitution_can_pass_math_but_fails_original_link(tmp_path):
    folder = fixture_capture(tmp_path)
    arrays = reference(VARIANTS["baseline"])
    # Still within every mathematical budget, but these are freshly produced oracle bytes.
    arrays["qfinal"][0, 0, 0] = np.nextafter(arrays["qfinal"][0, 0, 0], np.inf)
    check_arrays(arrays, VARIANTS["baseline"])
    np.savez(folder / "actual-arrays.npz", **arrays)
    with pytest.raises(ValueError, match="original output replaced"):
        verify_capture(folder)


def test_loaded_binary_must_match_real_compiler_output(tmp_path):
    folder = fixture_capture(tmp_path)
    write_json(
        folder / "compiler/one/invocation.json",
        {"returncode": 0, "output_sha256": "wrong-compiler-output"},
    )
    with pytest.raises(ValueError, match="loaded binary differs"):
        verify_capture(folder)


def test_incomplete_execution_fails(tmp_path):
    folder = fixture_capture(tmp_path)
    events = (folder / "execution.jsonl").read_text().splitlines()
    (folder / "execution.jsonl").write_text(events[0] + "\n")
    with pytest.raises(ValueError, match="incomplete execution"):
        verify_capture(folder)


def test_native_report_byte_payload_is_preserved(tmp_path):
    target = tmp_path / "bytes.json"
    write_json(target, {"payload": bytes((0, 255, 10))})
    encoded = json.loads(target.read_text())["payload"]
    assert encoded["type"] == "bytes" and encoded["encoding"] == "hex"
    assert bytes.fromhex(encoded["value"]) == bytes((0, 255, 10))
