"""Installed PoPS kinetic transport; ROOT runs Serial/MPI after SDK rebuild.

The source/math tests live separately. No synthetic state is a native receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_evolved_original_stage import checkpoint_provenance, compare_checkpoint_replay
from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.m19_freestreaming import CASES, FINAL_TIME, FRAME, MODULE, build
from tests.python.support.native_execution_context import artifact_execution_context
from tests.review.sol61_m19_freestreaming_oracle import cell_centers, initial, moments, receive

SCHEMA = "pops.m19-freestreaming-native-fixture@1"
PHASES = ("initial", "accepted", "continuous", "restored", "replay")


def _bytes(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        chunks, position = [], 0
        while position < before.st_size:
            chunk = os.read(fd, min(1 << 20, before.st_size - position))
            if not chunk:
                raise ValueError("freestreaming origin was truncated")
            chunks.append(chunk)
            position += len(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("freestreaming origin changed during bounded read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _pin(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(_bytes(path)).hexdigest()}


def _root(world):
    return world is None or int(world.rank) == 0


def _velocity_snapshot(runtime):
    """Use the exact Python wrapper key; the binding owns ranked native gathering."""
    from pops.model.provider_pack import ComponentKey
    key = ComponentKey(str(MODULE.owner_path.canonical()), "aux",
                       "velocity_coordinate", "velocity_coordinate")
    return runtime._executor.auxiliary_component(key)


def _snapshot(world, runtime, directory, phase, nx, nv, *, auxiliary=False):
    values = state_snapshots(runtime, world, ("kinetic",))[0]
    clock = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
    boxes = collective_call(world, lambda: runtime.local_boxes("kinetic"))
    if world is not None:
        from pops._native_collectives import allgather_value
        boxes = collective_call(world, lambda: allgather_value(world, boxes))
    else:
        boxes = (boxes,)
    velocity = None
    if auxiliary:
        velocity = collective_call(world, lambda: _velocity_snapshot(runtime))
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory/(phase+"-checkpoint")))
    checkpoint_pin = collective_call(world, lambda: _pin(checkpoint))
    authority = (collective_call(world, lambda: checkpoint_provenance(runtime, checkpoint))
                 if phase in ("accepted", "continuous", "replay") else None)
    if phase == "restored":
        collective_call(world, lambda: _authenticate_restored(runtime, checkpoint))
    with collective_check(world):
        assert values.shape == (1, nx, nv) and np.isfinite(values).all()
        if auxiliary:
            velocity = np.asarray(velocity).copy()
            assert velocity.shape == (nx, nv)
            np.testing.assert_allclose(velocity, np.broadcast_to(cell_centers(nx, nv)[1], (nx, nv)),
                                       rtol=0, atol=1e-14)
        if _root(world):
            path = directory/(phase+"-state.npz")
            data = {"population": values, "time": np.float64(clock[0]), "step": np.int64(clock[1])}
            if auxiliary:
                data["velocity_coordinate_native"] = velocity
            np.savez(path, **data)
            row = {"phase": phase, "state": _pin(path), "checkpoint": checkpoint_pin,
                   "time_hex": float(clock[0]).hex(), "macro_step": clock[1],
                   "boxes_by_rank": boxes, "moments_cell_midpoint": moments(values, nx, nv),
                   "auxiliary_captured": auxiliary, "checkpoint_provenance": authority}
            (directory/(phase+"-receipt.json")).write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    return values, clock, checkpoint, checkpoint_pin, authority


def _authenticate_restored(runtime, path):
    from io import BytesIO
    from pops.runtime._checkpoint_manifest import MANIFEST_KEY, authenticate_checkpoint_payload
    with np.load(BytesIO(_bytes(path)), allow_pickle=False) as stored:
        manifest = json.loads(str(stored[MANIFEST_KEY]))
        return authenticate_checkpoint_payload(runtime, stored, runtime_kind=manifest["runtime_kind"]).token


def _compare_restored_payload(accepted, restored):
    """Compare every saved physical/cache/history entry, including empty entries."""
    from io import BytesIO
    from pops.runtime._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY
    payloads = []
    for path in (accepted, restored):
        with np.load(BytesIO(_bytes(path)), allow_pickle=False) as stored:
            payloads.append({key: stored[key].copy() for key in stored.files})
    left, right = payloads
    assert left.keys() == right.keys()
    keys = sorted(left.keys() - {IDENTITY_KEY, MANIFEST_KEY})
    for key in keys:
        assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape, key
        assert left[key].tobytes(order="C") == right[key].tobytes(order="C"), key
    return {"exact_payload": True, "keys": keys}


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("nx,nv", CASES)
def test_native_m19_signed_freestreaming_exact_restart(nx, nv, tmp_path, record_property,
                                                       isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native = select_native_dimension(2)
    world = native.mpi_world() if native_mpi_communicator(native) == "MPI_COMM_WORLD" else None
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), "installed package required"
    case, layout, quadrature, dt = collective_call(world, lambda: build(nx, nv))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    if world is None:
        artifact = collective_call(world, lambda: pops.compile(resolved))
    else:
        artifact = compile_resolved_plan_once(world, resolved,
            route="m19-signed-freestreaming-%d-%d" % (nx, nv), compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    # The Case InitialConditionPlan is the sole initialization authority.
    runtime = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context": context}))
    directory = collective_directory(world, tmp_path/"m19-freestreaming")
    rows = {}
    rows["initial"] = _snapshot(world, runtime, directory, "initial", nx, nv)
    seed = rows["initial"][0]
    with collective_check(world):
        np.testing.assert_allclose(seed, initial(nx, nv), rtol=0, atol=3e-12)
        assert rows["initial"][1] == (0., 0)
    first = collective_call(world, lambda: pops.run(runtime, t_end=float(FINAL_TIME), max_steps=nx//2))
    rows["accepted"] = _snapshot(world, runtime, directory, "accepted", nx, nv, auxiliary=True)
    with collective_check(world):
        assert first.accepted_steps == nx//2 and first.rejected_steps == 0
        assert rows["accepted"][1] == (float(FINAL_TIME), nx//2)
        receive(rows["accepted"][0], seed, nx, nv, float(dt), nx//2)
    second = collective_call(world, lambda: pops.run(runtime, t_end=float(2*FINAL_TIME), max_steps=nx//2))
    rows["continuous"] = _snapshot(world, runtime, directory, "continuous", nx, nv, auxiliary=True)
    restored_context = collective_call(world, lambda: artifact_execution_context(artifact))
    restored = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context": restored_context}))
    collective_call(world, lambda: restored.restart(rows["accepted"][2]))
    rows["restored"] = _snapshot(world, restored, directory, "restored", nx, nv, auxiliary=True)
    replay = collective_call(world, lambda: pops.run(restored, t_end=float(2*FINAL_TIME), max_steps=nx//2))
    rows["replay"] = _snapshot(world, restored, directory, "replay", nx, nv, auxiliary=True)
    with collective_check(world):
        assert second.accepted_steps == replay.accepted_steps == nx//2
        assert second.rejected_steps == replay.rejected_steps == 0
        assert rows["restored"][1] == rows["accepted"][1]
        assert rows["replay"][1] == rows["continuous"][1] == (float(2*FINAL_TIME), nx)
        assert rows["accepted"][0].tobytes() == rows["restored"][0].tobytes()
        assert rows["continuous"][0].tobytes() == rows["replay"][0].tobytes()
        for phase, steps in (("continuous", nx), ("restored", nx//2), ("replay", nx)):
            receive(rows[phase][0], seed, nx, nv, float(dt), steps)
        # Authenticate CP immediately at capture and again after all observations.
        assert len({Path(row[2]).resolve() for row in rows.values()}) == len(PHASES)
        for row in rows.values():
            assert _pin(row[2]) == row[3], "native checkpoint changed after capture"
        checkpoint_equivalence = collective_call(world, lambda: compare_checkpoint_replay(
            {phase: rows[phase][2] for phase in ("accepted", "continuous", "replay")},
            {phase: rows[phase][4] for phase in ("accepted", "continuous", "replay")}))
        restored_equivalence = collective_call(world, lambda: _compare_restored_payload(
            rows["accepted"][2], rows["restored"][2]))
        if _root(world):
            binaries, programs, model_sources, model_manifests = [], [], [], []
            components = [("block-"+row.name, row.model) for row in artifact.blocks]
            components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
            for ordinal, (name, component) in enumerate(components):
                binary = Path(component.so_path)
                sidecar = Path(str(binary)+".pops-artifact.json")
                assert sidecar.is_file(), "actual component sidecar required"
                binaries.append({"component": name, "binary": _pin(binary), "sidecar": _pin(sidecar)})
                if name.startswith("block-"):
                    source = component._generated_cpp
                    assert source is None or type(source) is str
                    # The current model driver may not retain its translation unit. Record
                    # that gap explicitly instead of regenerating a falsely compiled source.
                    retained = None
                    if source is not None:
                        path = directory/("model-%d.cpp" % ordinal)
                        path.write_text(source)
                        retained = _pin(path)
                    model_sources.append({"component": name, "cpp": retained})
                    assert component.module_manifest is not None
                    path = directory/("model-%d.manifest.json" % ordinal)
                    path.write_text(json.dumps(component.module_manifest.to_data(),
                                               indent=2, sort_keys=True, allow_nan=False)+"\n")
                    model_manifests.append({"component": name, "manifest": _pin(path)})
                if name.startswith("program-"):
                    assert component._generated_cpp is not None, "retain compiler source, never re-emit"
                    cpp = Path(component.dump_cpp(directory/("program-%d.cpp" % ordinal)))
                    ir = Path(component.dump_ir(directory/("program-%d.ir.json" % ordinal)))
                    programs.append({"component": name, "cpp": _pin(cpp), "ir": _pin(ir),
                                     "program_hash": component.program_hash})
            from tests.python.support import m19_freestreaming as physical_source
            from tests.review import sol61_m19_freestreaming_oracle as scientific_source
            receipt = {"schema": SCHEMA, "dimension": 2, "rank": 0,
                "size": 1 if world is None else int(world.size), "nx": nx, "nv": nv,
                "native_axes": {"velocity": 0, "position": 1},
                "frame": FRAME.to_dict(), "velocity_quadrature": quadrature.to_data(),
                "dt": [dt.numerator, dt.denominator], "final_time": [FINAL_TIME.numerator, FINAL_TIME.denominator],
                "physics": "ddt(f)=-div((0,v*f)); periodic position; velocity flux zero",
                "discretization": "ConservativeCellAverage/FirstOrder/HLLExplicitPair(v,v)",
                "temporal": "SSPRK2", "phases": {phase: _pin(directory/(phase+"-receipt.json")) for phase in PHASES},
                "native": _pin(native.__file__), "package": _pin(pops.__file__),
                "checkpoint_equivalence": checkpoint_equivalence, "restored_payload_equivalence": restored_equivalence,
                "artifact_identity": artifact.artifact_identity.token,
                "bind_identity": runtime.bind_identity.token, "platform": artifact.platform_manifest.to_data(),
                "binaries": binaries, "programs": programs, "model_sources": model_sources,
                "model_manifests": model_manifests,
                "sources": [_pin(__file__), _pin(physical_source.__file__), _pin(scientific_source.__file__)],
                "qualification": {"signed_freestreaming": True, "exact_restart_and_replay": True,
                    "vlasov_poisson": False, "bgk": False, "aggregate_payload_recomposition": False}}
            (directory/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")
        for key, value in (("m19_freestreaming_receipt", str(directory/"receipt.json")),
                           ("dimension", 2), ("rank", 0 if world is None else int(world.rank)),
                           ("size", 1 if world is None else int(world.size)),
                           ("artifact_identity", artifact.artifact_identity.token)):
            record_property(key, value)
