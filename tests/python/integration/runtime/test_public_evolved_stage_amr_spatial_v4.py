"""Installed native spatial Q evolution; a separate periodic-strip qualification."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest
from tests.python.support.original_field_acceptance import selected_native_mixed_rule, STOP_RULE
from pops._native_collectives import allgather_value

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.integration.runtime.test_public_evolved_original_stage import checkpoint_provenance, compare_checkpoint_replay
from tests.python.support.amr_snapshots import level_valid_mask
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_amr_spatial import (
    ACCEPTANCE, CONTROLS, DENSE_BYTES, DT, FD_STEP, build, check_saved, check_initial, metric_arrays, carrier_patch_boxes,
)
from tests.python.integration.runtime.test_public_evolved_stage_amr import capture as base_capture, same_images
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context


def capture(world, runtime, width, *, histories=True):
    rows, metadata = base_capture(world, runtime, width, histories=histories)
    shape = collective_call(world, runtime.spatial_shape)
    boxes = collective_call(world, lambda: tuple(runtime.patch_boxes()))
    manifests = collective_call(world, lambda: allgather_value(world, metadata[1]))
    full_boxes = collective_call(world, lambda: carrier_patch_boxes(manifests))
    with collective_check(world):
        assert len(shape) == 2 and shape[0] == shape[1]
    for level, row in enumerate(rows):
        valid = collective_call(world, lambda level=level: level_valid_mask(runtime, level, refinement_ratio=2))
        with collective_check(world):
            row.update(metric_arrays(shape[0], level, np.asarray(valid)))
            row["carrier_patch_boxes"] = full_boxes.copy()
            row["native_base_shape"] = np.asarray(shape, dtype=np.int64)
            row["native_patch_boxes"] = np.asarray([(lev, *lo, *hi) for lev, lo, hi in boxes], dtype=np.int64)
    return rows, metadata


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("cells", (8, 16))
def test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4(isolated_native_cache, tmp_path, record_property, cells):
    width = 2
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout = collective_call(world, lambda: build(cells))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="nonconstant original evolved composite AMR stage", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    def bind():
        return pops.bind(artifact, resources={"execution_context":context})
    runtime = collective_call(world, bind)
    directory = collective_directory(world, tmp_path/"evolved-stage-amr-spatial")
    with collective_check(world):
        if world.rank == 0:
            components = [("block-"+row.name, row.model) for row in artifact.blocks]
            components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
            compilation = []
            fixture_sources = []
            from tests.python.support import evolved_stage_amr_spatial, evolved_stage_amr
            for index, source in enumerate((Path(__file__), Path(evolved_stage_amr_spatial.__file__), Path(evolved_stage_amr.__file__))):
                path = directory/("fixture-source-%d.py" % index)
                path.write_bytes(bounded_bytes(source))
                fixture_sources.append({"original_path":str(source), "path":str(path),
                    "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()})
            from pops.codegen.compile_provenance import artifact_sidecar_path
            for index, (name, component) in enumerate(components):
                binary = Path(component.so_path)
                sidecar = Path(artifact_sidecar_path(str(binary)))
                entry = {"component":name, "DSO":{"path":str(binary), "sha256":hashlib.sha256(bounded_bytes(binary)).hexdigest()},
                    "sidecar":{"path":str(sidecar), "sha256":hashlib.sha256(bounded_bytes(sidecar)).hexdigest()}}
                if name.startswith("program-"):
                    assert component.compile_command and component._generated_cpp is not None
                    entry["command"] = component.compile_command
                    for kind, dump in (("cpp", component.dump_cpp), ("ir.json", component.dump_ir)):
                        path = Path(dump(directory/("program-%d.%s" % (index, kind))))
                        entry[kind] = {"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()}
                    entry["program_hash"] = component.program_hash
                    program_ir = json.loads(bounded_bytes(Path(entry["ir.json"]["path"])))
                    solve = [node for node in program_ir["nodes"] if node["op"] == "solve_spatial_field"]
                    assert len(solve) == 1 and solve[0]["attrs"]["seed_index"] is None
                    request = solve[0]["attrs"]["solve_request"]
                    assert request["seed"] is None
                    seed_selection = request["initialization_identity"]
                compilation.append(entry)
    initial = capture(world, runtime, width, histories=False)
    registry_phases = {"initial": {"rows_by_rank": collective_call(world, lambda: allgather_value(world, initial[1][1]))}}
    paths, seals, authorities, phases = {}, {}, {}, {}
    for phase, owner in (("accepted", runtime), ("continuous", runtime), ("replay", None)):
        if phase == "replay":
            owner = collective_call(world, bind)
            collective_call(world, lambda owner=owner: owner.restart(paths["accepted"]))
            reloaded = capture(world, owner, width)
            with collective_check(world):
                same_images(reloaded, phases["accepted"])
            phases["reloaded"] = reloaded
            registry_phases["reloaded"] = {"rows_by_rank": collective_call(world, lambda: allgather_value(world, reloaded[1][1]))}
        end = DT if phase == "accepted" else 2*DT
        collective_call(world, lambda owner=owner, end=end: pops.run(owner, t_end=end, max_steps=1, console=False))
        path = collective_call(world, lambda owner=owner, phase=phase: owner.checkpoint(directory/(phase+"-checkpoint")))
        seals[phase] = collective_call(world, lambda path=path: hashlib.sha256(bounded_bytes(path)).hexdigest())
        authorities[phase] = collective_call(world, lambda owner=owner, path=path: checkpoint_provenance(owner, path))
        paths[phase] = path
        phases[phase] = capture(world, owner, width)
        registry_phases[phase] = {"rows_by_rank": collective_call(world, lambda phase=phase: allgather_value(world, phases[phase][1][1]))}
    with collective_check(world):
        same_images(phases["continuous"], phases["replay"])
        assert phases["accepted"][1][-1][:2] == (DT, 1)
        assert phases["continuous"][1][-1][:2] == (2*DT, 2)
        for row, old in zip(phases["continuous"][0], phases["accepted"][0], strict=True):
            for name in (*tuple("T%d" % i for i in range(width)), *(("z",) if width == 2 else ())):
                np.testing.assert_array_equal(row[name+"-previous"], old[name])
        if world.rank == 0:
            equivalent = compare_checkpoint_replay(paths, authorities)
            observations, stored_images = {}, {}
            for phase, image in (("initial", initial), *phases.items()):
                files, reloaded_rows = [], []
                for level, row in enumerate(image[0]):
                    path = directory/(phase+"-level%d.npz" % level)
                    np.savez(path, **row)
                    with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as stored:
                        reloaded_rows.append({key:stored[key].copy() for key in stored.files})
                    files.append({"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()})
                stored_images[phase] = reloaded_rows
                observations[phase] = {"levels":files, "metadata":image[1]}
            registry_path = directory/"carrier-registry.json"
            registry_path.write_text(json.dumps({"schema":"sol61.amr.carrier-registry@1",
                "dimension":2, "size":world.size, "phases":registry_phases}, sort_keys=True, indent=2)+"\n")
            registry_pin = {"path":str(registry_path.resolve()), "sha256":hashlib.sha256(bounded_bytes(registry_path)).hexdigest()}
            # Preserve every actual native capture even if a scientific guard fails.
            (directory/"observation-index.json").write_text(json.dumps({
                "status":"raw-native-captures-not-yet-qualified", "artifact":artifact.artifact_identity.token,
                "phases":observations, "carrier_registry":registry_pin}, sort_keys=True, indent=2)+"\n")
            for phase, image in (("initial", initial), *phases.items()):
                reloaded_rows = stored_images[phase]
                files = observations[phase]["levels"]
                reference_files = []
                if phase == "initial":
                    metrics = {"initial_Q_amounts":check_initial(reloaded_rows, cells).tolist()}
                else:
                    metrics, reference_rows = check_saved(reloaded_rows, cells, width,
                        None if phase in ("accepted", "reloaded") else phases["accepted"][0],
                        1 if phase in ("accepted", "reloaded") else 2, initial[0])
                    for level, row in enumerate(reference_rows):
                        path = directory/(phase+"-reference-level%d.npz" % level)
                        np.savez(path, **row)
                        reference_files.append({"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()})
                if phase != "initial":
                    original = selected_native_mixed_rule(image[1][2], CONTROLS["tolerance"])
                    assert metrics["original_F_weighted_l2"] <= CONTROLS["tolerance"]*max(1., original["reference_residual_norm"])
                    assert metrics["original_F_weighted_l2"] <= CONTROLS["tolerance"]
                    metrics["native_original_F_triplet"] = original
                    metrics["native_original_F_composite_relative"] = original["rel_residual"]
                for level, row in enumerate(reloaded_rows):
                    np.testing.assert_array_equal(row["forcing"], initial[0][level]["forcing"])
                observations[phase] = {"levels":files, "independent_references":reference_files, "checks":metrics, "metadata":image[1]}
            checkpoints = {phase:{"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()} for phase, path in paths.items()}
            assert all(row["sha256"] == seals[phase] for phase, row in checkpoints.items())
            assert {Path(row["path"]).resolve() for row in checkpoints.values()}.isdisjoint(
                {Path(row["path"]).resolve() for phase in observations.values()
                    for row in (*phase["levels"], *phase["independent_references"])})
            receipt = {"fixture_schema":"pops.evolved-stage-amr-spatial-native-fixture@4", "stop_rule":STOP_RULE, "independent_original_l2_threshold":CONTROLS["tolerance"], "seed_selection":seed_selection, "carrier_registry":registry_pin, "artifact":artifact.artifact_identity.token,
                "dimension":2, "rank":world.rank, "size":world.size, "cells":cells, "width":width,
                "qualification":"nonconstant periodic full-y strips; composite flux and nonlinear restriction",
                "initial_temperature":{"T0":[.15, .02, "cos(2*pi*x)"], "T1":[.25, .015, "sin(2*pi*x)"]},
                "metric_authority":{"kind":"derived-from-declared-Cartesian-unit-square-and-native-shape",
                    "lower":[0., 0.], "upper":[1., 1.], "ratio":2, "periodic_axes":["x", "y"],
                    "kappa":"declared no embedded boundary, exactly one; not a native getter",
                    "native_shape":initial[0][0]["native_base_shape"].tolist(),
                    "native_patch_boxes":initial[0][0]["native_patch_boxes"].tolist()},
                "history_protocol":{"wire":"POPSHID1", "raw_slots_after_publication":True,
                    "latest_slot":1, "previous_slot":0, "depth":2},
                "newton":CONTROLS, "fd_step":FD_STEP, "acceptance":ACCEPTANCE, "dt":DT,
                "realization":"FullResidualBasisLU@1", "max_dense_bytes":DENSE_BYTES,
                "native":{"path":str(native.__file__), "sha256":hashlib.sha256(bounded_bytes(native.__file__)).hexdigest()},
                "platform":artifact.platform_manifest.to_data(), "compilation":compilation,
                "fixture_sources":fixture_sources,
                "active_scalar_DOFs":3*sum(int(np.count_nonzero(row["active"])) for row in initial[0]),
                "phases":observations, "checkpoints":checkpoints, "checkpoint_equivalence":equivalent}
            (directory/"receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
        for name, value in (("artifact_identity", artifact.artifact_identity.token), ("dimension",2),
            ("rank",world.rank), ("size",world.size), ("evolved_stage_amr_spatial_receipt",str(directory/"receipt.json"))):
            record_property(name, value)
