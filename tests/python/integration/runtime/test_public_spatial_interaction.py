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

from tests.python.support.public_native_profile import compile_profile_plan_once as compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import rank_and_size
from tests.python.support.spatial_interaction_receipts import (
    CELLS, DT, GAMMA, WORKSPACE, build, bound_initial_values, capture, arrays, same, digest,
)


def bind(world, artifact, width, *, failure=None):
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    values = collective_call(world, lambda: bound_initial_values(
        artifact.plan.initial_condition_plan, width, failure=failure))
    return collective_call(world, lambda: pops.bind(artifact, initial_values=values,
                                                   resources={"execution_context": context}))


def prepare(world, width, adaptive, failure, *, amr_periodic=False):
    case, layout, selected, _ = collective_call(world, lambda: build(
        width, adaptive=adaptive, failure=failure, amr_periodic=amr_periodic))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="direct native spatial interaction",
                                          compile_artifact=pops.compile)
    return artifact, selected


def world_and_native():
    from tests.python.support.public_native_profile import native_profile
    native, world = native_profile(scope="public spatial interaction")
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
        if rank_and_size(world)[0] == 0:
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
                "rank": rank_and_size(world)[0], "size": rank_and_size(world)[1], "phases": phases, "checkpoints": checkpoints,
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
                "transport_boundary_profile": "periodic" if adaptive else "physical_no_flux",
                "interaction_domain_profile": "bounded_cartesian_cell_midpoint_without_periodic_images",
                "amr_qualification_scope": "periodic transport topology only; active nonperiodic EB is refused at bind" if adaptive else None,
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
    artifact, selected = prepare(world, width, adaptive, None, amr_periodic=adaptive)
    runtime = bind(world, artifact, width)
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
    restored = bind(world, artifact, width)
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
        for image in (initial, accepted, continuous, reloaded, replay):
            assert all(np.all(np.isfinite(np.asarray(piece["values"])))
                for level in image["levels"] for copy in level["copies"]
                for piece in copy["pieces"]["rho"]), "zero-flux source-only domain must stay finite"
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
    for key, value in (("evidence_path", str(directory)), ("dimension", 2), ("rank", rank_and_size(world)[0]),
                       ("size", rank_and_size(world)[1]), ("adaptive", adaptive), ("width", width),
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
    runtime = bind(world, artifact, 1, failure=failure)
    directory = collective_directory(world, tmp_path/("spatial-interaction-"+failure))
    initial = capture(world, runtime, 1, adaptive=False, step=0)
    if failure == "nonfinite":
        with collective_check(world):
            resident = [tuple(np.asarray(piece["values"]) for piece in pieces)
                for level in initial["levels"] for copy in level["copies"]
                for pieces in zip(copy["pieces"]["rho"], copy["pieces"]["active"],
                                  copy["pieces"]["kappa"], strict=True)]
            assert all(np.all(np.isfinite(rho)) for rho, _active, _kappa in resident)
            assert any(np.any((rho == 3.) & (active == 1) & (kappa > 0))
                       for rho, active, kappa in resident), "native active source pole is absent"
    checkpoints = {"initial": checkpoint(world, runtime, directory, "initial")}
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    with collective_check(world):
        assert all(error is not None for error in errors), errors
        assert all("direct spatial interaction" in error[1] or "direct interaction" in error[1] for error in errors), errors
        if failure == "nonfinite" and rank_and_size(world)[1] == 1:
            assert all("nonfinite active source" in error[1] for error in errors), errors
    rejected = capture(world, runtime, 1, adaptive=False, step=0)
    with collective_check(world):
        same(initial, rejected)
    checkpoints["rejected"] = checkpoint(world, runtime, directory, "rejected")
    save(world, native, artifact, directory, dict(initial=initial, rejected=rejected), checkpoints,
         width=1, adaptive=False, selected=selected, failure=failure, failure_errors=errors)
    for key, value in (("evidence_path", str(directory)), ("rank", rank_and_size(world)[0]), ("size", rank_and_size(world)[1]), ("failure", failure),
                       ("fixture_schema", "pops.spatial-interaction-native-fixture@1")):
        record_property(key, value)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_public_spatial_interaction_ignores_inactive_source_pole(
        isolated_native_cache, tmp_path, record_property):
    del isolated_native_cache
    world, native = world_and_native()
    artifact, _selected = prepare(world, 1, False, "nonfinite-inactive")
    runtime = bind(world, artifact, 1, failure="nonfinite-inactive")
    directory = collective_directory(world, tmp_path/"spatial-interaction-inactive-pole")
    initial = capture(world, runtime, 1, adaptive=False, step=0)
    with collective_check(world):
        source_pieces = [tuple(np.asarray(piece["values"]) for piece in pieces)
            for level in initial["levels"] for copy in level["copies"]
            for pieces in zip(copy["pieces"]["rho"], copy["pieces"]["active"],
                              copy["pieces"]["kappa"], strict=True)]
        assert all(np.all(np.isfinite(rho)) for rho, _active, _kappa in source_pieces)
        assert any(np.any((rho == 3.) & (active == 0) & (kappa == 0))
                   for rho, active, kappa in source_pieces)
        assert not any(np.any((rho == 3.) & (active == 1))
                       for rho, active, _kappa in source_pieces)
    checkpoints = {"initial": checkpoint(world, runtime, directory, "initial")}
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = capture(world, runtime, 1, adaptive=False, step=1)
    checkpoints["accepted"] = checkpoint(world, runtime, directory, "accepted")
    with collective_check(world):
        assert accepted["clock"] == (DT, 1)
        for old, new in zip(initial["levels"], accepted["levels"], strict=True):
            for first, second in zip(old["copies"], new["copies"], strict=True):
                for before, active, after in zip(first["pieces"]["rho"], first["pieces"]["active"],
                                                  second["pieces"]["rho"], strict=True):
                    rho, mask = np.asarray(before["values"]), np.asarray(active["values"])
                    rate = np.zeros_like(rho)
                    np.divide(rho, 3-rho, out=rate, where=mask == 1)
                    values = np.asarray(after["values"])
                    assert np.all(np.isfinite(values))
                    np.testing.assert_allclose(values, rho+DT*rate, rtol=5e-12, atol=5e-12)
                    np.testing.assert_array_equal(values[mask == 0], rho[mask == 0])
                assert np.all(np.isfinite(second["history_values"]["I_accepted"]))
        if rank_and_size(world)[0] == 0:
            phases = {}
            for name, image in (("initial", initial), ("accepted", accepted)):
                path = directory/(name+".npz")
                np.savez(path, **arrays(image))
                phases[name] = dict(path=str(path), sha256=digest(path))
            receipt = dict(schema="pops.spatial-source-inactive-pole-native-fixture@1",
                source_law="rho/(3-rho)", dt=DT, dimension=2, rank=0,
                size=rank_and_size(world)[1], phases=phases, checkpoints=checkpoints,
                native=dict(path=str(native.__file__), sha256=digest(native.__file__)),
                artifact_identity=artifact.artifact_identity.token,
                components_origins=save_origins(artifact, directory),
                fixture_sources=[dict(path=str(path), sha256=digest(path)) for path in
                    (Path(__file__).resolve(), Path(sys.modules[build.__module__].__file__).resolve())])
            (directory/"receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    record_property("evidence_path", str(directory))
    record_property("fixture_schema", "pops.spatial-source-inactive-pole-native-fixture@1")


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_public_spatial_interaction_nonperiodic_amr_refuses_at_bind(
        isolated_native_cache, tmp_path, record_property):
    del isolated_native_cache
    world, native = world_and_native()
    artifact, _selected = prepare(world, 3, True, None, amr_periodic=False)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    values = collective_call(world, lambda: bound_initial_values(artifact.plan.initial_condition_plan, 3))
    result, errors = collective_attempt(world, lambda: pops.bind(
        artifact, initial_values=values, resources={"execution_context": context}))
    with collective_check(world):
        assert result is None
        guard = "active AMR embedded transport has no EB-qualified physical-boundary provider"
        assert all(error is not None and guard in error[1] for error in errors), errors
    directory = collective_directory(world, tmp_path/"spatial-interaction-nonperiodic-amr-bind-refusal")
    with collective_check(world):
        if rank_and_size(world)[0] == 0:
            receipt = dict(schema="pops.spatial-amr-nonperiodic-bind-refusal@1",
                dimension=2, width=3, rank=0, size=rank_and_size(world)[1],
                transport_boundary_profile="physical_no_flux", embedded_boundary_profile="cutcell_zero_flux",
                interaction_domain_profile="bounded_cartesian_cell_midpoint_without_periodic_images",
                qualification_scope="nonperiodic active AMR EB intentionally unsupported; bind must fail before runtime publication",
                failure_errors=errors, artifact_identity=artifact.artifact_identity.token,
                platform=artifact.platform_manifest.to_data(), components_origins=save_origins(artifact, directory),
                native=dict(path=str(native.__file__), sha256=digest(native.__file__)),
                fixture_sources=[dict(path=str(path), sha256=digest(path)) for path in
                    (Path(__file__).resolve(), Path(sys.modules[build.__module__].__file__).resolve())])
            (directory/"receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    record_property("evidence_path", str(directory))
    record_property("fixture_schema", "pops.spatial-amr-nonperiodic-bind-refusal@1")
