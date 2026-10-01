"""Genuine installed spatial-map witnesses; ROOT executes Serial/MPI reception.

The map is an observation, not an aggregation/diffusion PDE or energy flux.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.spatial_interaction_receipts import (
    CELLS, DT, GAMMA, WORKSPACE, build, initial_values, capture, arrays, same, digest,
)


def bind(world, artifact, values):
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    return collective_call(world, lambda: pops.bind(artifact, initial_state={"density": values.copy()},
                                                   resources={"execution_context": context}))


def prepare(world, width, adaptive, failure):
    case, layout, selected, _ = collective_call(world, lambda: build(width, adaptive=adaptive, failure=failure))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="direct native spatial interaction",
                                          compile_artifact=pops.compile)
    return artifact, selected


def world_and_native():
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    return world, native


def checkpoint(world, runtime, directory, phase):
    path = collective_call(world, lambda: runtime.checkpoint(directory/(phase+"-checkpoint")))
    return {"path": str(path), "sha256": collective_call(world, lambda: digest(path))}


def save_origins(artifact, directory):
    from pops.codegen.compile_provenance import ARTIFACT_SIDECAR_SUFFIX
    components = [("block-"+row.name, row.model) for row in artifact.blocks]
    components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
    result = []
    for index, (name, component) in enumerate(components):
        path = Path(component.so_path)
        sidecar = Path(str(path)+ARTIFACT_SIDECAR_SUFFIX)
        assert path.is_file() and sidecar.is_file()
        row = {"component": name, "binary": str(path), "binary_sha256": digest(path),
               "sidecar": str(sidecar), "sidecar_sha256": digest(sidecar)}
        if name.startswith("program-"):
            assert component._generated_cpp is not None, "retained compiler source required"
            cpp = Path(component.dump_cpp(directory/("program-%d.cpp" % index)))
            ir = Path(component.dump_ir(directory/("program-%d.ir.json" % index)))
            row.update(cpp=str(cpp), cpp_sha256=digest(cpp), ir=str(ir), ir_sha256=digest(ir),
                       program_hash=component.program_hash)
        result.append(row)
    return result


def save(world, native, artifact, directory, images, checkpoints, *, width, adaptive, selected, failure=None, failure_errors=()):
    with collective_check(world):
        if world.rank == 0:
            phases = {}
            for name, image in images.items():
                path = directory/(name+".npz")
                np.savez(path, **arrays(image))
                phases[name] = {"path": str(path), "sha256": digest(path),
                               "role": "SOURCE_SNAPSHOT_ONLY" if name == "initial" else
                                   "REJECTED_SOURCE_SNAPSHOT" if name == "rejected" else "ACTUAL_NATIVE_OUTPUT_AND_NEXT_SOURCE"}
            assert len({row["path"] for row in checkpoints.values()}) == len(checkpoints)
            for row in checkpoints.values():
                assert digest(row["path"]) == row["sha256"], "checkpoint overwritten after capture"
            assert {row["path"] for row in phases.values()}.isdisjoint(row["path"] for row in checkpoints.values())
            receipt = {"schema": "pops.spatial-interaction-native-fixture@1", "dimension": 2,
                "width": width, "components": selected, "cells": CELLS, "adaptive": adaptive,
                "dt": DT, "gamma": GAMMA, "workspace": 1 if failure == "budget" else WORKSPACE, "failure": failure,
                "rank": world.rank, "size": world.size, "phases": phases, "checkpoints": checkpoints,
                "failure_errors": failure_errors,
                "fixture_sources": [{"path": str(path), "sha256": digest(path)} for path in
                    (Path(__file__).resolve(), Path(sys.modules[build.__module__].__file__).resolve())],
                "artifact_identity": artifact.artifact_identity.token,
                "platform": artifact.platform_manifest.to_data(), "components_origins": save_origins(artifact, directory),
                "native": {"path": str(native.__file__), "sha256": digest(native.__file__)},
                "consumption_anchors": {} if failure else {
                    "accepted": "initial", "continuous": "accepted", "reloaded": "initial", "replay": "reloaded"},
                "history_source_contract": "pops.spatial-interaction-history-source@1",
                "raw_coverage_semantics": "true_is_covered", "coordinate_authority": "bound-normalized-native-Cartesian",
                "association": "fixture-recorded actual execution; ROOT owner approval pending",
                "aggregate_cryptographic_binding": False, "m26_pde_qualification": False}
            (directory/"receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("width,adaptive", ((1, False), (3, False), (3, True)),
                         ids=("scalar-cutcell", "signed-vector-permuted", "partial-amr-cutcell"))
def test_public_spatial_interaction_saved_history_and_exact_replay(
        isolated_native_cache, tmp_path, record_property, width, adaptive):
    del isolated_native_cache
    world, native = world_and_native()
    artifact, selected = prepare(world, width, adaptive, None)
    runtime = bind(world, artifact, initial_values(width))
    directory = collective_directory(world, tmp_path/"spatial-interaction")
    initial = capture(world, runtime, width, adaptive=adaptive, step=0)
    checkpoints = {"initial": checkpoint(world, runtime, directory, "initial")}
    with collective_check(world):
        # Native observers must really expose a fractional active cut cell.
        kappa = [np.asarray(piece["values"]) for level in initial["levels"]
                 for copy in level["copies"] for piece in copy["pieces"]["kappa"]]
        assert any(np.any((array > 0) & (array < 1)) for array in kappa)
        if adaptive:
            assert len(initial["levels"]) == 2
            geometry = initial["levels"][0]["geometry"]
            assert np.any(geometry["coverage"] & geometry["valid_cells"])
            assert np.any(~geometry["coverage"] & geometry["valid_cells"])
            assert any(np.any((np.asarray(piece["values"])[0] > 0) &
                ~geometry["coverage"][piece["lower"][0]:piece["upper"][0], piece["lower"][1]:piece["upper"][1]])
                for copy in initial["levels"][0]["copies"] for piece in copy["pieces"]["kappa"])
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = capture(world, runtime, width, adaptive=adaptive, step=1)
    checkpoints["accepted"] = checkpoint(world, runtime, directory, "accepted")
    collective_call(world, lambda: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    continuous = capture(world, runtime, width, adaptive=adaptive, step=2)
    checkpoints["continuous"] = checkpoint(world, runtime, directory, "continuous")
    restored = bind(world, artifact, initial_values(width))
    collective_call(world, lambda: restored.restart(checkpoints["accepted"]["path"]))
    reloaded = capture(world, restored, width, adaptive=adaptive, step=1)
    checkpoints["reloaded"] = checkpoint(world, restored, directory, "reloaded")
    with collective_check(world):
        same(accepted, reloaded)
    collective_call(world, lambda: pops.run(restored, t_end=2*DT, max_steps=1, console=False))
    replay = capture(world, restored, width, adaptive=adaptive, step=2)
    checkpoints["replay"] = checkpoint(world, restored, directory, "replay")
    with collective_check(world):
        same(continuous, replay)
        assert accepted["clock"] == (DT, 1) and continuous["clock"] == (2*DT, 2)
        for old, new in zip(accepted["levels"], continuous["levels"], strict=True):
            for first, second in zip(old["copies"], new["copies"], strict=True):
                a, b = first["history_values"], second["history_values"]
                np.testing.assert_array_equal(a["I_history"], a["I_accepted"])
                np.testing.assert_array_equal(b["I_history"], a["I_history"])
                np.testing.assert_allclose(b["I_accepted"], (1-GAMMA*DT)*a["I_accepted"], rtol=5e-12, atol=5e-12)
                assert np.max(np.abs(a["I_accepted"])) > .01
    save(world, native, artifact, directory,
         dict(initial=initial, accepted=accepted, continuous=continuous, reloaded=reloaded, replay=replay),
         checkpoints, width=width, adaptive=adaptive, selected=selected)
    for key, value in (("evidence_path", str(directory)), ("dimension", 2), ("rank", world.rank),
                       ("size", world.size), ("adaptive", adaptive), ("width", width),
                       ("fixture_schema", "pops.spatial-interaction-native-fixture@1")):
        record_property(key, value)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("failure", ("budget", "pole", "nonfinite"))
def test_public_spatial_interaction_refuses_without_publication(
        isolated_native_cache, tmp_path, record_property, failure):
    del isolated_native_cache
    world, native = world_and_native()
    artifact, selected = prepare(world, 1, False, failure)
    values = initial_values(1)
    if failure == "nonfinite":
        values[0, CELLS[1]//2, CELLS[0]//2] = 3.
    runtime = bind(world, artifact, values)
    directory = collective_directory(world, tmp_path/("spatial-interaction-"+failure))
    initial = capture(world, runtime, 1, adaptive=False, step=0)
    checkpoints = {"initial": checkpoint(world, runtime, directory, "initial")}
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    with collective_check(world):
        assert all(error is not None for error in errors), errors
        assert all("direct spatial interaction" in error[1] or "direct interaction" in error[1] for error in errors), errors
    rejected = capture(world, runtime, 1, adaptive=False, step=0)
    with collective_check(world):
        same(initial, rejected)
    checkpoints["rejected"] = checkpoint(world, runtime, directory, "rejected")
    save(world, native, artifact, directory, dict(initial=initial, rejected=rejected), checkpoints,
         width=1, adaptive=False, selected=selected, failure=failure, failure_errors=errors)
    for key, value in (("evidence_path", str(directory)), ("rank", world.rank), ("size", world.size), ("failure", failure),
                       ("fixture_schema", "pops.spatial-interaction-native-fixture@1")):
        record_property(key, value)
