"""Installed-native original Q evolution, independent FV load and exact replay.

ROOT owns compilation/execution. These fixtures do not qualify Marshak or M13.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_mms import (
    ACCEPTANCE, CONTROLS, FD_STEP, build, check_original, manufactured_data,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context


def checkpoint_restart_authority(runtime):
    """Persist the live pre-restart owner envelope; never infer it from replay."""
    from pops.identity import Identity
    rows, seen = [], set()
    def visit(owner, path):
        assert id(owner) not in seen
        seen.add(id(owner))
        values = []
        for name in ("_last_run_identity", "_restart_lineage_identity"):
            identity = getattr(owner, name)
            assert identity is None or (type(identity) is Identity and identity.domain == "run")
            values.append(None if identity is None else identity.token)
        rows.append({"owner_path": list(path), "previous_owner_run": values[0],
                     "previous_owner_lineage": values[1]})
        children = getattr(owner, "_engines", {})
        assert isinstance(children, dict) and all(type(key) is str for key in children)
        for key in sorted(children):
            visit(children[key], (*path, key))
    visit(runtime._executor, ())
    return {"contract": "pops.evolved-stage-restart-authority@1", "owners": rows}


def authenticate_replay_continuation(accepted, replay, authority):
    """Independent consumer reconstruction of the public restart epoch contract."""
    from pops.identity import Identity, make_identity
    assert type(authority) is dict and set(authority) == {"contract", "owners", "restored_source_run"}
    assert authority["contract"] == "pops.evolved-stage-restart-authority@1"
    assert authority["restored_source_run"] == accepted.run_identity.token
    rows = authority["owners"]
    assert type(rows) is list and rows
    canonical, paths = [], []
    for row in rows:
        assert type(row) is dict and set(row) == {"owner_path", "previous_owner_run", "previous_owner_lineage"}
        path = row["owner_path"]
        assert type(path) is list and all(type(key) is str for key in path)
        paths.append(tuple(path))
        entry = {"owner_path": tuple(path)}
        for name in ("previous_owner_run", "previous_owner_lineage"):
            token = row[name]
            identity = None if token is None else Identity.from_token(token)
            assert identity is None or (identity.domain == "run" and identity.token == token)
            entry[name] = None if identity is None else identity.to_data()
        canonical.append(entry)
    assert paths[0] == () and paths == sorted(set(paths))
    assert all(not path or path[:-1] in paths for path in paths)
    expected = make_identity("run", {"continuation": "checkpoint_restart_epoch@1",
        "source_run_identity": accepted.run_identity.to_data(), "owner_authorities": canonical})
    assert replay.continuation_identity == expected
    assert replay.run_identity != accepted.run_identity
    return expected


def checkpoint_provenance(runtime, path, *, restart_authority=None):
    """Authenticate the file against its live creator before another run/restart."""
    from pops.runtime._checkpoint_manifest import (
        MANIFEST_KEY, authenticate_checkpoint_payload,
    )
    with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as stored:
        manifest = json.loads(str(stored[MANIFEST_KEY]))
        restart = authenticate_checkpoint_payload(runtime, stored, runtime_kind=manifest["runtime_kind"])
    run = runtime.last_run_manifest
    assert run is not None and run.run_identity == runtime.last_run_identity
    semantic, artifact, bind = runtime._checkpoint_identities()
    result = {"semantic": semantic.token, "artifact": artifact.token, "bind": bind.token,
        "run": runtime.last_run_identity.token, "run_manifest": run.to_dict(),
        "last_restart": None if runtime.last_restart_identity is None else runtime.last_restart_identity.token,
        "restart": restart.token, "clock": {"time": runtime.time().hex(), "macro_step": runtime.macro_step()}}

    if restart_authority is not None:
        result["restart_authority"] = restart_authority
    return result


def compare_checkpoint_replay_v2(paths, provenance):
    """Exact physical checkpoint equality with authenticated continuation identities."""
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY, MANIFEST_KEY, _identity_from_json, inspect_checkpoint_payload_integrity,
    )
    from pops.runtime._run_manifest import RunManifest
    payloads, manifests, runs = {}, {}, {}
    for phase in ("accepted", "continuous", "replay"):
        with np.load(BytesIO(bounded_bytes(paths[phase])), allow_pickle=False) as stored:
            payload = {key: stored[key].copy() for key in stored.files}
        raw = json.loads(str(payload[MANIFEST_KEY]))
        manifest, restart = inspect_checkpoint_payload_integrity(payload, runtime_kind=raw["runtime_kind"])
        authority = provenance[phase]
        run = RunManifest.from_dict(authority["run_manifest"])
        assert run.run_identity.token == authority["run"]
        assert _identity_from_json(manifest["run_identity"]) == run.run_identity
        assert run.bind_identity.token == authority["bind"]
        for name in ("semantic", "artifact", "bind"):
            assert _identity_from_json(manifest[name+"_identity"]).token == authority[name]
        assert restart.token == authority["restart"]
        assert manifest["clock"] == authority["clock"]
        payloads[phase], manifests[phase], runs[phase] = payload, manifest, run
    assert runs["accepted"].continuation_identity is None
    assert runs["continuous"].continuation_identity is None
    assert provenance["accepted"]["last_restart"] is None
    assert provenance["continuous"]["last_restart"] is None
    authenticate_replay_continuation(runs["accepted"], runs["replay"],
                                    provenance["replay"]["restart_authority"])
    assert provenance["replay"]["last_restart"] == provenance["accepted"]["restart"]
    for attribute in ("bind_identity", "start_time", "start_macro_step", "controls"):
        assert getattr(runs["continuous"], attribute) == getattr(runs["replay"], attribute)
    assert runs["continuous"].start_time.hex() == provenance["accepted"]["clock"]["time"]
    assert runs["continuous"].start_macro_step == provenance["accepted"]["clock"]["macro_step"]
    left, right = payloads["continuous"], payloads["replay"]
    assert left.keys() == right.keys()
    for key in left.keys()-{MANIFEST_KEY, IDENTITY_KEY}:
        assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape, key
        assert left[key].tobytes(order="C") == right[key].tobytes(order="C"), key
    def comparable(manifest):
        return {key: value for key, value in manifest.items()
            if key not in {"run_identity", "restart_identity"}}
    assert comparable(manifests["continuous"]) == comparable(manifests["replay"])
    return {"contract": "pops.evolved-stage-checkpoint-equivalence@2",
        "exact_payload_and_manifest": True, "provenance": provenance}



def compare_checkpoint_replay(paths, provenance):
    """Exact physical checkpoint equality with authenticated continuation identities."""
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY, MANIFEST_KEY, _identity_from_json, inspect_checkpoint_payload_integrity,
    )
    from pops.runtime._run_manifest import RunManifest
    payloads, manifests, runs = {}, {}, {}
    for phase in ("accepted", "continuous", "replay"):
        with np.load(BytesIO(bounded_bytes(paths[phase])), allow_pickle=False) as stored:
            payload = {key: stored[key].copy() for key in stored.files}
        raw = json.loads(str(payload[MANIFEST_KEY]))
        manifest, restart = inspect_checkpoint_payload_integrity(payload, runtime_kind=raw["runtime_kind"])
        authority = provenance[phase]
        run = RunManifest.from_dict(authority["run_manifest"])
        assert run.run_identity.token == authority["run"]
        assert _identity_from_json(manifest["run_identity"]) == run.run_identity
        assert run.bind_identity.token == authority["bind"]
        for name in ("semantic", "artifact", "bind"):
            assert _identity_from_json(manifest[name+"_identity"]).token == authority[name]
        assert restart.token == authority["restart"]
        assert manifest["clock"] == authority["clock"]
        payloads[phase], manifests[phase], runs[phase] = payload, manifest, run
    assert runs["accepted"].continuation_identity is None
    assert runs["continuous"].continuation_identity is None
    assert provenance["accepted"]["last_restart"] is None
    assert provenance["continuous"]["last_restart"] is None
    assert runs["replay"].continuation_identity == runs["accepted"].run_identity
    assert provenance["replay"]["last_restart"] == provenance["accepted"]["restart"]
    for attribute in ("bind_identity", "start_time", "start_macro_step", "controls"):
        assert getattr(runs["continuous"], attribute) == getattr(runs["replay"], attribute)
    assert runs["continuous"].start_time.hex() == provenance["accepted"]["clock"]["time"]
    assert runs["continuous"].start_macro_step == provenance["accepted"]["clock"]["macro_step"]
    left, right = payloads["continuous"], payloads["replay"]
    assert left.keys() == right.keys()
    for key in left.keys()-{MANIFEST_KEY, IDENTITY_KEY}:
        assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape, key
        assert left[key].tobytes(order="C") == right[key].tobytes(order="C"), key
    def comparable(manifest):
        return {key: value for key, value in manifest.items()
            if key not in {"run_identity", "restart_identity"}}
    assert comparable(manifests["continuous"]) == comparable(manifests["replay"])
    return {"contract": "pops.evolved-stage-checkpoint-equivalence@1",
        "exact_payload_and_manifest": True, "provenance": provenance}


def capture(world, runtime, cells, width, dt, step):
    image = {}
    for name, ncomp in ((*(("Q%d" % i, 1) for i in range(width)), ("forcing", width))):
        data = collective_call(world, lambda name=name: runtime.state_global(name))
        with collective_check(world):
            image[name] = np.asarray(data).reshape(ncomp, cells, cells).copy()
    history = []
    names = ["T%d" % i for i in range(width)] + (["z"] if width == 2 else [])
    for name in names:
        slots = []
        for lag in range(2):
            value = collective_call(world, lambda name=name, lag=lag: runtime.history_global(name, lag))
            with collective_check(world):
                slots.append(np.asarray(value).reshape(cells, cells).copy())
        depth = collective_call(world, lambda name=name: runtime.history_depth(name))
        initialized = collective_call(world, lambda name=name: runtime._executor.history_initialized(name))
        fill = collective_call(world, lambda name=name: runtime._executor.history_fill_count(name))
        durations = tuple(collective_call(world, lambda name=name, lag=lag:
            runtime.history_slot_dt(name, lag)) for lag in range(2))
        sample = collective_call(world, lambda name=name: bytes(runtime._executor.history_sample_identity(name)))
        with collective_check(world):
            assert depth == 2 and initialized and fill == min(step, 2)
            assert durations == (dt, dt)
            header = b"POPSHID1"+struct.pack("<Q", len(name))+name.encode()+struct.pack("<qQ", -1, depth)
            expected_sample = header+b"".join(struct.pack("<QQQQ", 2,
                int.from_bytes(struct.pack("<d", start), "little"),
                int.from_bytes(struct.pack("<d", dt), "little"), 1)
                for start in (0., (step-1)*dt))
            assert sample == expected_sample
            image[name] = slots[1]
            image[name+"-previous"] = slots[0]
            history.append((name, depth, initialized, fill, durations, sample,
                            tuple(row.tobytes() for row in slots)))
    carriers = collective_call(world, lambda: bytes(runtime._executor.capture_auxiliary_checkpoint_accepted_state()))
    diagnostics = collective_call(world, lambda: tuple(sorted(runtime._executor.program_diagnostics().items())))
    lifecycle = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
    return image, (carriers, diagnostics, tuple(history), lifecycle)


def same_images(left, right):
    assert left[1] == right[1]
    assert left[0].keys() == right[0].keys()
    for name in left[0]:
        np.testing.assert_array_equal(left[0][name], right[0][name])


def check_saved(path, previous, initial, target, width, dt, candidate):
    with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as saved:
        # Detach accepted data, then mark them immutable for the oracle. No
        # runtime candidate, evaluated material or internal residual is an input.
        values = {name: saved[name].copy() for name in saved.files}
    for array in values.values():
        array.setflags(write=False)
    t = np.stack(tuple(values["T%d" % i] for i in range(width)))
    q = np.concatenate(tuple(values["Q%d" % i] for i in range(width)))
    np.testing.assert_array_equal(values["forcing"], initial["forcing"])
    checks = check_original(t, q, previous, values["forcing"], dt,
                            candidate_diffusion=candidate, target=target)
    if width == 2:
        error = float(np.max(np.abs(values["z"]-.25*t[0]-.5*t[1])))
        assert error <= ACCEPTANCE
        checks["auxiliary_constraint_linf"] = error
    return checks


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("cells", (8, 16))
@pytest.mark.parametrize("dt", (.01, .02))
@pytest.mark.parametrize("width,candidate", ((1, False), (2, True)))
def test_public_evolved_original_stage_saved_and_exact_replay(
        isolated_native_cache, tmp_path, record_property, cells, dt, width, candidate):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout, _ = collective_call(world, lambda: build(cells, width, dt, candidate_diffusion=candidate))
    initial, initial_t, target, _, volumes = collective_call(world, lambda:
        manufactured_data(cells, width, dt, candidate_diffusion=candidate))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="evolved-original-stage-Uniform",
                                          compile_artifact=pops.compile)
    def bind():
        context = artifact_execution_context(artifact)
        return pops.bind(artifact, initial_state={name: value.copy() for name, value in initial.items()},
                         resources={"execution_context": context})
    runtime = collective_call(world, bind)
    directory = collective_directory(world, tmp_path/"evolved-original-stage")
    for name, expected in initial.items():
        actual = collective_call(world, lambda name=name: runtime.state_global(name))
        with collective_check(world):
            np.testing.assert_array_equal(np.asarray(actual).reshape(expected.shape), expected)
    collective_call(world, lambda: pops.run(runtime, t_end=dt, max_steps=1, console=False))
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory/"accepted-checkpoint"))
    checkpoint_hashes = {"accepted": collective_call(world, lambda:
        hashlib.sha256(bounded_bytes(checkpoint)).hexdigest())}
    checkpoint_authorities = {"accepted": collective_call(world, lambda: checkpoint_provenance(runtime, checkpoint))}
    accepted = capture(world, runtime, cells, width, dt, 1)
    collective_call(world, lambda: pops.run(runtime, t_end=2*dt, max_steps=1, console=False))
    continuous_checkpoint = collective_call(world, lambda: runtime.checkpoint(directory/"continuous-checkpoint"))
    checkpoint_hashes["continuous"] = collective_call(world, lambda:
        hashlib.sha256(bounded_bytes(continuous_checkpoint)).hexdigest())
    checkpoint_authorities["continuous"] = collective_call(world, lambda: checkpoint_provenance(runtime, continuous_checkpoint))
    continuous = capture(world, runtime, cells, width, dt, 2)
    restored = collective_call(world, bind)
    restart_authority = collective_call(world, lambda: checkpoint_restart_authority(restored))
    collective_call(world, lambda: restored.restart(checkpoint))
    restart_authority["restored_source_run"] = collective_call(world, lambda: restored.last_run_identity.token)
    reloaded = capture(world, restored, cells, width, dt, 1)
    with collective_check(world):
        same_images(reloaded, accepted)
    collective_call(world, lambda: pops.run(restored, t_end=2*dt, max_steps=1, console=False))
    replay_checkpoint = collective_call(world, lambda: restored.checkpoint(directory/"replay-checkpoint"))
    checkpoint_hashes["replay"] = collective_call(world, lambda:
        hashlib.sha256(bounded_bytes(replay_checkpoint)).hexdigest())
    checkpoint_authorities["replay"] = collective_call(world, lambda: checkpoint_provenance(restored, replay_checkpoint, restart_authority=restart_authority))
    replay = capture(world, restored, cells, width, dt, 2)
    checkpoint_files = (("accepted", checkpoint), ("continuous", continuous_checkpoint), ("replay", replay_checkpoint))
    with collective_check(world):
        same_images(replay, continuous)
        assert accepted[1][-1] == (dt, 1) and continuous[1][-1] == (2*dt, 2)
        for index in range(width):
            np.testing.assert_array_equal(continuous[0]["T%d-previous" % index], accepted[0]["T%d" % index])
        if world.rank == 0:
            checkpoint_equivalence = compare_checkpoint_replay_v2(dict(checkpoint_files), checkpoint_authorities)
            initial_path = directory/"initial.npz"
            np.savez(initial_path, **initial, initial_temperature=initial_t, first_target=target, cell_volumes=volumes)
            phases = {}
            q0 = np.concatenate(tuple(initial["Q%d" % i] for i in range(width)))
            q1 = np.concatenate(tuple(accepted[0]["Q%d" % i] for i in range(width)))
            for phase, image, previous, expected in (("accepted", accepted, q0, target),
                    ("reloaded", reloaded, q0, target), ("continuous", continuous, q1, None),
                    ("replay", replay, q1, None)):
                path = directory/(phase+".npz")
                np.savez(path, **image[0], time=image[1][-1][0], step=image[1][-1][1])
                phases[phase] = {"path": str(path), "sha256": hashlib.sha256(bounded_bytes(path)).hexdigest(),
                    "checks": check_saved(path, previous, initial, expected, width, dt, candidate),
                    "diagnostics": dict(image[1][1])}
            components = [("block-"+row.name, row.model) for row in artifact.blocks]
            components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
            binaries, sources, program_irs = [], [], []
            for index, (name, component) in enumerate(components):
                binary = Path(component.so_path)
                binaries.append({"component": name, "path": str(binary),
                    "sha256": hashlib.sha256(bounded_bytes(binary)).hexdigest(),
                    "compile_command": getattr(component, "compile_command", None)})
                from pops.codegen.compile_provenance import artifact_sidecar_path
                sidecar = Path(artifact_sidecar_path(str(binary)))
                assert sidecar.is_file(), "the actual DSO identity sidecar must survive reception"
                binaries[-1]["sidecar"] = {"path": str(sidecar),
                    "sha256": hashlib.sha256(bounded_bytes(sidecar)).hexdigest()}
                if name.startswith("program-"):
                    assert isinstance(component.compile_command, str) and component.compile_command
                    assert component._generated_cpp is not None
                    source = Path(component.dump_cpp(directory/("program-%d.cpp" % index)))
                    sources.append({"path": str(source), "sha256": hashlib.sha256(bounded_bytes(source)).hexdigest()})
                    ir_path = Path(component.dump_ir(directory/("program-%d.ir.json" % index)))
                    program_irs.append({"component": name, "path": str(ir_path),
                        "sha256": hashlib.sha256(bounded_bytes(ir_path)).hexdigest(),
                        "program_hash": component.program_hash})
            checkpoints = {phase: {"path": str(path),
                "sha256": hashlib.sha256(bounded_bytes(path)).hexdigest()} for phase, path in checkpoint_files}
            assert all(row["sha256"] == checkpoint_hashes[phase] for phase, row in checkpoints.items()), \
                "native checkpoint was overwritten"
            checkpoint_paths = {Path(row["path"]).resolve() for row in checkpoints.values()}
            observation_paths = {Path(row["path"]).resolve() for row in phases.values()}
            observation_paths.add(initial_path.resolve())
            assert checkpoint_paths.isdisjoint(observation_paths), "checkpoints and observations must use distinct files"
            receipt = {"fixture_schema": "pops.evolved-stage-native-fixture@2",
                "artifact": artifact.artifact_identity.token, "dimension": 2, "rank": world.rank,
                "size": world.size, "cells": cells, "dt": dt, "evolved_states": width,
                "unknowns": width+(width == 2), "candidate_diffusion": candidate,
                "newton": CONTROLS, "fd_step": FD_STEP, "acceptance": ACCEPTANCE,
                "projection": "piecewise_constant_cell", "volume_sum": float(np.sum(volumes)),
                "native": {"path": str(native.__file__), "sha256": hashlib.sha256(bounded_bytes(native.__file__)).hexdigest()},
                "platform": artifact.platform_manifest.to_data(), "binaries": binaries, "sources": sources,
                "program_irs": program_irs,
                "initial": {"path": str(initial_path), "sha256": hashlib.sha256(bounded_bytes(initial_path)).hexdigest()},
                "phases": phases, "checkpoints": checkpoints,
                "checkpoint_equivalence": checkpoint_equivalence,
                "exact_restart_and_replay": True}
            (directory/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        for name, value in (("artifact_identity", artifact.artifact_identity.token), ("dimension", 2),
                ("rank", world.rank), ("size", world.size), ("evidence_path", str(directory)),
                ("evolved_stage_receipt", str(directory/"receipt.json"))):
            record_property(name, value)
