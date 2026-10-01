"""Installed-native original FieldProblem D(State) MMS and exact replay.

These tests require ROOT's rebuilt SDK. Source-only tests below execute no DSO.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.captured_diffusion_mms import (
    CONTROLS, DT, FD_STEP, HISTORY_MAX_LAG, RESIDUAL_TOL, SOLUTION_TOL, build, initial_data, matrices, original_action,
)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context


def bounded_bytes(path):
    """Read the fstat extent on one fd; authenticate it again before hashing."""
    fd = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        chunks, position = [], 0
        while position < before.st_size:
            part = os.pread(fd, min(1024*1024, before.st_size-position), position)
            if not part:
                raise ValueError("receipt file shortened during bounded read")
            chunks.append(part)
            position += len(part)
        after = os.fstat(fd)
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        return b"".join(chunks)
    finally:
        os.close(fd)


def capture(world, runtime, cells, width, *, step, first_solution=None):
    states = {}
    for name, ncomp in (("response", width), ("forcing", width), ("material", 1)):
        data = collective_call(world, lambda name=name: runtime.state_global(name))
        with collective_check(world):
            states[name] = np.asarray(data).reshape(ncomp, cells, cells).copy()
    fields, history = [], []
    for index in range(width):
        name = "q%d" % index
        slots = []
        for slot in range(HISTORY_MAX_LAG+1):
            data = collective_call(world, lambda name=name, slot=slot: runtime.history_global(name, slot))
            with collective_check(world):
                slots.append(np.asarray(data).reshape(cells, cells).copy())
        depth = collective_call(world, lambda name=name: runtime.history_depth(name))
        initialized = collective_call(world, lambda name=name: runtime._executor.history_initialized(name))
        fill = collective_call(world, lambda name=name: runtime._executor.history_fill_count(name))
        durations = tuple(collective_call(world, lambda name=name, slot=slot: runtime.history_slot_dt(name, slot))
                          for slot in range(HISTORY_MAX_LAG+1))
        sample = collective_call(world, lambda name=name: bytes(runtime._executor.history_sample_identity(name)))
        with collective_check(world):
            # depth in store_history declares the maximum lag, so its ring has
            # two physical slots. End-of-step rotation places the latest store
            # in slot 1 and recycles the first accepted store into slot 0.
            assert depth == HISTORY_MAX_LAG+1 and initialized and fill == min(step, depth)
            assert durations == (DT, DT)
            previous = slots[1] if step == 1 else first_solution[index]
            assert slots[0].tobytes() == previous.tobytes()
            header = b"POPSHID1"+struct.pack("<Q", len(name))+name.encode()+struct.pack("<qQ", -1, depth)
            expected_sample = header+b"".join(struct.pack("<QQQQ", 2,
                int.from_bytes(struct.pack("<d", start), "little"),
                int.from_bytes(struct.pack("<d", DT), "little"), 1)
                for start in (0., (step-1)*DT))
            assert sample == expected_sample
            fields.append(slots[1])
            history.append((name, depth, initialized, fill, np.asarray(durations).tobytes(),
                            sample, tuple(value.tobytes() for value in slots)))
    with collective_check(world):
        states["solution"] = np.stack(fields)
    # Uniform exposes the sealed POPSAUX2 accepted image; the rank-local
    # carrier manifest accessor belongs to the AMR engine.
    carriers = collective_call(world, lambda: bytes(
        runtime._executor.capture_auxiliary_checkpoint_accepted_state()))
    diagnostics = collective_call(world, lambda: tuple(sorted(runtime._executor.program_diagnostics().items())))
    lifecycle = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
    return states, (carriers, diagnostics, tuple(history), lifecycle)


def check_saved(saved, expected_initial, target, time):
    # All scientific inputs are values reloaded from this actual native image.
    with np.load(BytesIO(bounded_bytes(saved)), allow_pickle=False) as image:
        q, alpha, forcing, response = (image[key].copy() for key in (
            "solution", "material", "forcing", "response"))
    np.testing.assert_array_equal(alpha, expected_initial["material"])
    np.testing.assert_array_equal(forcing, expected_initial["forcing"])
    action, spatial = original_action(q, alpha[0])
    relative = float(np.linalg.norm(action-forcing)/np.linalg.norm(forcing))
    error = float(np.max(np.abs(q-target)))
    consumer_error = float(np.max(np.abs(response/time-q)))
    assert error < SOLUTION_TOL
    assert relative < RESIDUAL_TOL
    assert consumer_error < SOLUTION_TOL
    assert np.max(np.abs(spatial)) > .01
    return {"solution_linf": error, "original_residual_relative_l2": relative,
            "consumer_linf": consumer_error, "spatial_action_linf": float(np.max(np.abs(spatial)))}


def same_images(left, right):
    assert left[1] == right[1]
    assert left[0].keys() == right[0].keys()
    for key in left[0]:
        np.testing.assert_array_equal(left[0][key], right[0][key])


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("width,order", [(1, (0,)), (3, (2, 0, 1))])
def test_public_captured_diffusion_nonconstant_saved_and_exact_replay(
        isolated_native_cache, tmp_path, record_property, width, order):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    cells = 16
    case, layout, _, _ = collective_call(world, lambda: build(cells, width, order))
    initial, target, _ = collective_call(world, lambda: initial_data(cells, width))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="captured-D-original-Uniform",
                                          compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_state={key: value.copy() for key, value in initial.items()}, resources={"execution_context": context}))
    directory = collective_directory(world, tmp_path/"captured-D-MMS")
    for name in ("forcing", "material"):
        actual = collective_call(world, lambda name=name: runtime.state_global(name))
        with collective_check(world):
            np.testing.assert_array_equal(np.asarray(actual).reshape(initial[name].shape), initial[name])

    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = capture(world, runtime, cells, width, step=1)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory/"accepted"))
    collective_call(world, lambda: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    continuous = capture(world, runtime, cells, width, step=2, first_solution=accepted[0]["solution"])
    continuous_checkpoint = collective_call(world, lambda: runtime.checkpoint(directory/"continuous"))
    restored_context = collective_call(world, lambda: artifact_execution_context(artifact))
    restored = collective_call(world, lambda: pops.bind(artifact,
        initial_state={key: value.copy() for key, value in initial.items()}, resources={"execution_context": restored_context}))
    collective_call(world, lambda: restored.restart(checkpoint))
    reloaded = capture(world, restored, cells, width, step=1)
    with collective_check(world):
        same_images(reloaded, accepted)
    collective_call(world, lambda: pops.run(restored, t_end=2*DT, max_steps=1, console=False))
    replay = capture(world, restored, cells, width, step=2, first_solution=reloaded[0]["solution"])
    replay_checkpoint = collective_call(world, lambda: restored.checkpoint(directory/"replay"))
    with collective_check(world):
        same_images(replay, continuous)
        assert accepted[1][-1] == (DT, 1) and continuous[1][-1] == (2*DT, 2)
        if world.rank == 0:
            phases = {}
            np.savez(directory/"initial.npz", **initial, target=target)
            for phase, image, time in (("accepted", accepted, DT), ("continuous", continuous, 2*DT),
                                       ("reloaded", reloaded, DT), ("replay", replay, 2*DT)):
                path = directory/(phase+".npz")
                np.savez(path, **image[0], time=time, step=image[1][-1][1])
                phases[phase] = {"npz": str(path), "sha256": hashlib.sha256(bounded_bytes(path)).hexdigest(),
                                 "checks": check_saved(path, initial, target, time)}
            # Preserve exact compiler-owned source and every actual linked DSO.
            components = [("block-"+row.name, row.model) for row in artifact.blocks]
            components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
            binaries, sources = [], []
            for component_index, (name, component) in enumerate(components):
                binary = Path(component.so_path)
                binaries.append({"component": name, "path": str(binary),
                                 "sha256": hashlib.sha256(bounded_bytes(binary)).hexdigest(),
                                 "compile_command": getattr(component, "compile_command", None)})
                if name.startswith("program-"):
                    assert component._generated_cpp is not None, "compiler source must be retained, never re-emitted"
                    path = Path(component.dump_cpp(directory/("program-%d.cpp" % component_index)))
                    sources.append({"component": name, "path": str(path),
                                    "sha256": hashlib.sha256(bounded_bytes(path)).hexdigest()})
            checkpoints = {phase: {"path": str(path), "sha256": hashlib.sha256(bounded_bytes(path)).hexdigest()}
                           for phase, path in (("accepted", checkpoint), ("continuous", continuous_checkpoint),
                                               ("replay", replay_checkpoint))}
            receipt = {"kind": "actual-native-captured-D-original-MMS",
                "fixture_schema": "pops.captured-diffusion-native-fixture@2", "artifact": artifact.artifact_identity.token,
                "dimension": 2, "rank": world.rank, "size": world.size, "cells": cells, "width": width,
                "order": order, "face_policy": "pops.field.face-mean.arithmetic@1", "newton": CONTROLS,
                "fd_step": FD_STEP, "solution_tolerance": SOLUTION_TOL, "residual_tolerance": RESIDUAL_TOL,
                "native": {"path": str(native.__file__), "sha256": hashlib.sha256(bounded_bytes(native.__file__)).hexdigest()},
                "platform": artifact.platform_manifest.to_data(), "binaries": binaries, "sources": sources,
                "initial_npz": str(directory/"initial.npz"),
                "initial_sha256": hashlib.sha256(bounded_bytes(directory/"initial.npz")).hexdigest(),
                "phases": phases, "checkpoints": checkpoints,
                "exact_restart_and_replay": True}
            (directory/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        for name, value in (("artifact_identity", artifact.artifact_identity.token), ("dimension", 2),
                            ("rank", world.rank), ("size", world.size), ("evidence_path", str(directory)),
                            ("captured_diffusion_receipt", str(directory/"receipt.json"))):
            record_property(name, value)


@pytest.mark.parametrize("width,order", [(1, (0,)), (3, (2, 0, 1))])
def test_captured_diffusion_mms_source_and_independent_action(width, order):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    for cells in (8, 16):
        initial, target, spatial = initial_data(cells, width)
        actual, measured_spatial = original_action(target, initial["material"][0])
        np.testing.assert_array_equal(actual, initial["forcing"])
        np.testing.assert_array_equal(spatial, measured_spatial)
        np.testing.assert_allclose(spatial.sum(axis=(1, 2)), 0, atol=2e-13)
        # A uniform substitute for D or a constant q must change this forcing.
        uniform_action, _ = original_action(target, np.zeros_like(initial["material"][0]))
        assert np.max(np.abs(actual-uniform_action)) > 10000*RESIDUAL_TOL
        # Independent Fourier-symbol check of the constant-D Cartesian stencil.
        coordinate = (np.arange(cells)+.5)/cells
        y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
        modes = np.stack(tuple(4*cells**2*(
            np.sin(np.pi*(column+1)/cells)**2*.025*np.cos(2*np.pi*(column+1)*x)
            + np.sin(np.pi/cells)**2*.02*np.sin(2*np.pi*y)) for column in range(width)))
        diffusion, reaction = matrices(width)
        symbol = np.einsum("ij,jyx->iyx", diffusion, modes)
        np.testing.assert_allclose(uniform_action-(np.einsum("ij,jyx->iyx", reaction, target)+.2*target**3),
                                   symbol, rtol=0, atol=2e-13)
    case, layout, program, token = build(16, width, order)
    validate_nonlinear_field_request(program, token)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    emitted = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                               target="system")
    assert "apply_general_field<pops::kNativeDimension, %d, %d, true>" % (width, width**2) in emitted
    assert "nonfinite_original_field_residual" in emitted
    assert "field expression inputs require exact layout/distribution identity" in emitted
    assert token.attrs["contract"] == "pops.spatial-field-residual@2"
    assert token.inputs[1].attrs["coefficient_admissibility"] == "finite_general"
    assert token.inputs[1].inputs == token.inputs[2:2+token.attrs["capture_count"]]
    assert len(token.inputs[1].attrs["field_dependencies"]) == 1
    diffusion, _ = matrices(width)
    if width == 3:
        assert np.linalg.det(diffusion) == 0 and np.any(diffusion < 0)
        assert not np.array_equal(diffusion, diffusion.T)
    with pytest.raises(ValueError, match="missing exact qualified equation capture"):
        build(16, width, order, omit_material=True)
