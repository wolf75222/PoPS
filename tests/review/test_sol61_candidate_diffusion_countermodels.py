"""Synthetic protocol/math units, never fabricated Native positives."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import pytest

PATH = Path(__file__).with_name("sol61_candidate_diffusion_real_countermodels.py")
spec = importlib.util.spec_from_file_location("candidate_negative_units", PATH)
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)


def test_wrong_candidate_frozen_equation_changes_the_manufactured_load():
    q, alpha = h.c.base.prescribed(8, 3)
    wrong = h.wrong_frozen_load(q, alpha)
    original, _ = h.c.base.original(q, alpha, rational=True)
    assert np.linalg.norm(original - wrong) / np.linalg.norm(original) > 1e-3


def test_synthetic_cp_reseals_every_array_with_exact_type_and_shape():
    def identity(domain):
        return dict(domain=domain, schema_version=1, algorithm="sha256", hexdigest="0" * 64)

    payload = dict(
        t=np.asarray(0.01),
        macro_step=np.asarray(1, dtype=np.int64),
        abi_key=np.asarray("synthetic-protocol-only"),
        sample=np.arange(6, dtype=np.float64).reshape(2, 3),
    )
    manifest = dict(
        schema_version=1,
        runtime_kind="uniform",
        semantic_identity=identity("semantic"),
        artifact_identity=identity("artifact"),
        bind_identity=identity("bind"),
        run_identity=identity("run"),
        clock=dict(time=(0.01).hex(), macro_step=1),
        arrays={},
        restart_identity=identity("restart"),
    )
    payload["pops_checkpoint_manifest"] = np.asarray(json.dumps(manifest))
    payload["pops_restart_identity"] = np.asarray("not-native")
    h.reseal_cp(payload)
    actual = h.c.strict_json(payload["pops_checkpoint_manifest"].item())
    assert actual["arrays"]["sample"] == h.c.wire.typed_array(payload["sample"])
    import io

    stream = io.BytesIO()
    np.savez(stream, **payload)
    h.c.wire.envelope(stream.getvalue(), "accepted", "synthetic-protocol-only")
    payload["sample"][0, 0] += 1
    stream = io.BytesIO()
    np.savez(stream, **payload)
    with pytest.raises(ValueError, match="typed array digest"):
        h.c.wire.envelope(stream.getvalue(), "accepted", "synthetic-protocol-only")


def diagnostic_payload():
    name = b"field_residual_7.rel_residual"
    images = [
        b"POPSDIA1"
        + struct.pack("<QQQQ", 64, rank, 2, 1)
        + struct.pack("<Q", len(name))
        + name
        + struct.pack("<d", 0.0)
        for rank in range(2)
    ]
    return dict(
        program_diagnostics_state=np.frombuffer(b"".join(images), dtype=np.uint8).copy(),
        program_diagnostics_offsets=np.asarray(
            [0, len(images[0]), sum(map(len, images))], dtype=np.int64
        ),
    )


def test_diagnostic_nonfinite_mutation_keeps_a_valid_opaque_packet():
    p = diagnostic_payload()
    h.edit_diagnostic(p, 1)
    tables = h.c.diagnostic_images(p, 2)
    assert tables[1][b"field_residual_7.rel_residual"] == struct.pack("<Q", 0x7FF8000000000021)
    assert tables[0][b"field_residual_7.rel_residual"] == struct.pack("<d", 0.0)


def test_diagnostic_rank_mutation_is_an_actual_codec_owner_refusal():
    p = diagnostic_payload()
    h.edit_diagnostic(p, 1, bad_owner=True)
    with pytest.raises(ValueError, match="width/rank/count"):
        h.c.diagnostic_images(p, 2)


def test_harness_does_not_open_data_or_create_fake_positive_without_real_seals(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="two external seals"):
        h.run("must-not-be-opened", None, None, None, output)
    assert not output.exists()


def test_harness_is_pops_free_in_new_isolated_process(tmp_path):
    code = """import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='pops' or fullname.startswith('pops.'):raise RuntimeError('forbidden PoPS import')
sys.meta_path.insert(0,Guard());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(PATH.resolve())],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
