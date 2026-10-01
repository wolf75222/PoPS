"""Prepared real-native AMR12 refusal campaign; ROOT owns actual execution.

No alternate model, mock engine, fake device/OOM or synthetic accepted image.
The public evolved-stage AMR fixture supplies the physics and history observers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import struct

import numpy as np
import pops
import pytest

from pops._native_collectives import allgather_value
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_evolved_stage_amr import capture, same_images
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.evolved_stage_amr import DT, build
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context


def _snapshot(world, runtime, width, *, capture_image=True):
    engine = runtime._executor._s
    projection = capture(world, runtime, width)
    image = collective_call(world, engine.checkpoint_state_carriers) if capture_image else None
    diagnostics = collective_call(world, engine._checkpoint_program_diagnostics) if capture_image else None
    # Native diagnostic checkpoint capture also requires an accepted owner. The
    # readonly map still exposes the actual IEEE double values during rollback.
    diagnostic_bits = tuple((name, struct.pack("<d", value).hex())
        for name, value in projection[1][2])
    temporal = collective_call(world, engine.checkpoint_temporal_relations)
    with collective_check(world):
        if capture_image:
            assert type(image) is bytes and image[:8] == b"POPSCAR1"
        if capture_image:
            assert type(diagnostics) is bytes and diagnostics[:8] == b"POPSDIA1"
    return {"projection":projection, "state_bytes":image,
            "diagnostics_bytes":diagnostics, "diagnostic_bits":diagnostic_bits, "temporal":temporal}


def _same(world, expected, actual):
    with collective_check(world):
        same_images(expected["projection"], actual["projection"])
        # Capture deliberately refuses an active transaction. Native full-grown
        # carrier manifests in projection metadata still compare byte hashes.
        if expected["state_bytes"] is not None and actual["state_bytes"] is not None:
            assert expected["state_bytes"] == actual["state_bytes"], "full grown state bytes"
        assert expected["diagnostic_bits"] == actual["diagnostic_bits"], "native diagnostic IEEE bits"
        if expected["diagnostics_bytes"] is not None and actual["diagnostics_bytes"] is not None:
            assert expected["diagnostics_bytes"] == actual["diagnostics_bytes"], "accepted diagnostic archive"
        assert expected["temporal"] == actual["temporal"], "temporal relation contracts"


def _refused(world, operation, *, boundary):
    _, failures = collective_attempt(world, operation)
    with collective_check(world):
        # A missing binding, import failure or assertion does not receive the refusal.
        assert len(failures) == world.size and all(failures), failures
        assert all(row[0] in ("ValueError", "TypeError", "RuntimeError") for row in failures), failures
        assert all(boundary in row[1] for row in failures), failures
    return failures


def _record_snapshot(directory, world, name, snapshot):
    """Pin actual native bytes and observations without reconstructing a codec."""
    files = {}
    for key in ("state_bytes", "diagnostics_bytes"):
        if snapshot[key] is None:
            continue
        path = directory/(name+"-rank%d-" % world.rank+key+".bin")
        path.write_bytes(snapshot[key])
        files[key] = {"path":str(path.resolve()), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    arrays = []
    for level, row in enumerate(snapshot["projection"][0]):
        path = directory/(name+"-rank%d-level%d.npz" % (world.rank, level))
        np.savez(path, **row)
        arrays.append({"path":str(path.resolve()), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"files":files, "levels":arrays, "metadata":snapshot["projection"][1],
            "diagnostic_bits":snapshot["diagnostic_bits"], "temporal":snapshot["temporal"]}


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_native_amr12_state_carrier_refusals_and_exact_rollback(
    isolated_native_cache, tmp_path, record_property,
):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    from pops.runtime._checkpoint_state_carriers import (
        STATE_CARRIERS_KEY, prepare_checkpoint_state_carriers,
    )
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert world.size in (1, 2), "this receipt is deliberately Serial/MPI2 only"
    # Reuse the actual physical model, constants and composite AMR stage.
    cells, width = 8, 2
    case, layout = collective_call(world, lambda: build(cells, width))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="AMR12 independent carrier refusals",
        compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context":context}))
    engine = runtime._executor._s
    with collective_check(world):
        for name in ("checkpoint_state_carriers", "validate_checkpoint_state_carriers",
                     "restore_checkpoint_state_carriers", "begin_restart_transaction",
                     "commit_restart_transaction", "finalize_restart_transaction", "rollback_restart_transaction"):
            assert callable(getattr(engine, name)), name
    for step in (1, 2):
        collective_call(world, lambda step=step: pops.run(runtime, t_end=step*DT, max_steps=1, console=False))
    baseline = _snapshot(world, runtime, width)
    with collective_check(world):
        assert baseline["projection"][1][-1][:2] == (2*DT, 2)
        assert len(baseline["projection"][0]) == 2
    image = baseline["state_bytes"]
    digests = collective_call(world, lambda: allgather_value(world, hashlib.sha256(image).hexdigest()))
    with collective_check(world):
        assert len(set(digests)) == 1, "capture must be a canonical all-source archive"
    collective_call(world, lambda: engine.validate_checkpoint_state_carriers(image))
    _same(world, baseline, _snapshot(world, runtime, width))
    refusals = {}
    # Exact bytes/type/shape preflights enter the real native consensus on all ranks.
    for name, payload, reason in (
        ("native-truncated", image[:-1], "state carrier payload exceeds bytes"),
        ("native-type", bytearray(image), "state carrier checkpoint payload must be exact bytes"),
        ("native-array-shape", np.frombuffer(image, dtype=np.uint8).reshape(1, -1),
         "state carrier checkpoint payload must be exact bytes"),
    ):
        injected = payload if world.rank == 0 else image
        refusals[name] = _refused(world, lambda injected=injected:
            engine.validate_checkpoint_state_carriers(injected), boundary=reason)
        _same(world, baseline, _snapshot(world, runtime, width))
    # Transport dtype/shape are refused by the real checkpoint boundary, not a mock.
    raw = np.frombuffer(image, dtype=np.uint8)
    for name, payload in (("transport-dtype", raw.astype(np.int16)),
                          ("transport-shape", raw.reshape(1, -1))):
        injected = payload if world.rank == 0 else raw
        refusals[name] = _refused(world, lambda injected=injected:
            prepare_checkpoint_state_carriers(runtime, engine, {STATE_CARRIERS_KEY:injected}),
            boundary="one-dimensional uint8")
        _same(world, baseline, _snapshot(world, runtime, width))
    if world.size == 2:
        # One genuine rank gets a structurally valid but different byte image.
        # The last archive byte belongs to the last IEEE word, not a schema field.
        divergent = image[:-1]+bytes((image[-1]^1,)) if world.rank == 0 else image
        refusals["rank-zero-divergent-archive"] = _refused(world,
            lambda: engine.validate_checkpoint_state_carriers(divergent), boundary="differs between destination ranks")
        _same(world, baseline, _snapshot(world, runtime, width))
    refusals["restore-outside-transaction"] = _refused(world,
        lambda: engine.restore_checkpoint_state_carriers(image), boundary="requires active native restart transaction")
    _same(world, baseline, _snapshot(world, runtime, width))

    # Genuine native transaction; alter real valid state on rank0 only. No codec
    # bytes are fabricated, and every rank enters the restoration collective.
    collective_call(world, engine.begin_restart_transaction)
    mutated = None
    try:
        refusals["capture-during-transaction"] = _refused(world,
            engine.checkpoint_state_carriers, boundary="requires prepared accepted state")
        _same(world, baseline, _snapshot(world, runtime, width, capture_image=False))
        before = np.asarray(collective_call(world,
            lambda: runtime.block_level_state_global("Q0", 0))).copy()
        local = np.nextafter(before, np.inf) if world.rank == 0 else before
        collective_call(world, lambda: engine.set_block_level_state("Q0", 0, local))
        mutated = _snapshot(world, runtime, width, capture_image=False)
        changed = collective_call(world, lambda: allgather_value(world,
            baseline["projection"][1][1] != mutated["projection"][1][1]))
        with collective_check(world):
            assert changed[0], "rank0 must own actual mutated coarse state storage"
            assert before.tobytes() != mutated["projection"][0][0]["Q0"].tobytes(), (
                "injection must change actual valid cells, not only a fixture marker")
        refusals["valid-state-contradicts-scientific-projection"] = _refused(world,
            lambda: engine.restore_checkpoint_state_carriers(image),
            boundary="valid cells contradict scientific state projection")
        _same(world, mutated, _snapshot(world, runtime, width, capture_image=False))
    finally:
        collective_call(world, engine.rollback_restart_transaction)
    rolled_back = _snapshot(world, runtime, width)
    _same(world, baseline, rolled_back)
    # A successful in-transaction restore of the authentic accepted image must
    # also preserve every byte. This direct codec test does not bypass the full
    # public restart history/auxiliary authority to claim commit/finalize coverage.
    collective_call(world, engine.begin_restart_transaction)
    try:
        collective_call(world, lambda: engine.restore_checkpoint_state_carriers(image))
        _same(world, baseline, _snapshot(world, runtime, width, capture_image=False))
    finally:
        collective_call(world, engine.rollback_restart_transaction)
    _same(world, baseline, _snapshot(world, runtime, width))
    refusals["restore-after-rollback"] = _refused(world,
        lambda: engine.restore_checkpoint_state_carriers(image), boundary="requires active native restart transaction")
    _same(world, baseline, _snapshot(world, runtime, width))

    directory = collective_directory(world, tmp_path/"amr12-carrier-failures")
    with collective_check(world):
        snapshots = {"baseline":_record_snapshot(directory, world, "baseline", baseline),
                     "contradicted":_record_snapshot(directory, world, "contradicted", mutated),
                     "rolled-back":_record_snapshot(directory, world, "rolled-back", rolled_back)}
        fixture = Path(__file__).resolve()
        native_path = Path(native.__file__).resolve()
        receipt = {"schema":"sol61.amr12.native-carrier-failures@1", "dimension":2,
            "rank":world.rank, "size":world.size, "cells":cells, "width":width,
            "fixture":{"path":str(fixture), "sha256":hashlib.sha256(fixture.read_bytes()).hexdigest()},
            "native":{"path":str(native_path), "sha256":hashlib.sha256(native_path.read_bytes()).hexdigest()},
            "artifact":artifact.artifact_identity.token, "platform":artifact.platform_manifest.to_data(),
            "snapshots":snapshots, "refusals":refusals,
            "rank_divergence_executed":world.size == 2,
            "limits":["Direct codec refused/rolled back within genuine native restart transaction.",
                      "Accepted POPSCAR1/POPSDIA1 images; active transactions compare native full-carrier hashes and diagnostic IEEE bits.",
                      "No GPU/device failure/OOM injection, regrid or topology redistribution.",
                      "Full public commit/finalize coverage belongs to the existing AMR restart fixture."]}
        path = directory/("receipt-rank%d.json" % world.rank)
        path.write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    record_property("amr12_carrier_failure_receipt", str(path))
    record_property("native_dimension", 2)
    record_property("mpi_size", world.size)
