"""Installed native composite Q evolution; ROOT owns the actual execution."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from pops._native_collectives import allgather_value
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.integration.runtime.test_public_evolved_original_stage import checkpoint_provenance, compare_checkpoint_replay
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_amr import (
    ACCEPTANCE, CONTROLS, DENSE_BYTES, DT, FD_STEP, build, check_saved, closed_data, published_history_image,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context


def capture(world, runtime, width, *, histories=True):
    levels = collective_call(world, runtime.n_levels)
    rows, metadata = [], []
    native = runtime._executor
    step = collective_call(world, runtime.macro_step) if histories else None
    names = tuple("T%d" % i for i in range(width))+(("z",) if width == 2 else ())
    for level in range(levels):
        row = {}
        for name in (*tuple("Q%d" % i for i in range(width)), "forcing"):
            row[name] = np.asarray(collective_call(world,
                lambda name=name, level=level: runtime.block_level_state_global(name, level))).copy()
        row["active"] = np.asarray(collective_call(world,
            lambda level=level: composite_active_mask(runtime, level, refinement_ratio=2))).copy()
        if histories:
            for name in names:
                depth = collective_call(world, lambda name=name: native.history_depth(name))
                initialized = collective_call(world, lambda name=name, level=level: native.history_initialized(name, level))
                fill = collective_call(world, lambda name=name, level=level: native.history_fill_count(name, level))
                with collective_check(world):
                    assert depth == 2 and initialized and fill == min(step, 2)
                raw_slots, durations = [], []
                for slot in range(depth):
                    data = collective_call(world, lambda name=name, level=level, slot=slot: native.history_global(name, level, slot))
                    duration = collective_call(world, lambda name=name, level=level, slot=slot: native.history_slot_dt(name, level, slot))
                    raw_slots.append(np.asarray(data).copy())
                    durations.append(float(duration))
                sample = collective_call(world, lambda name=name, level=level:
                    bytes(native.history_sample_identity(name, level)))
                with collective_check(world):
                    row.update(published_history_image(name, level, raw_slots, durations, sample, step))
                metadata.append((level, name, depth, initialized, fill,
                    tuple(value.hex() for value in durations), sample.hex()))
        rows.append(row)
    carriers = collective_call(world, lambda: tuple(tuple(row) for row in native.checkpoint_rank_local_carrier_manifest()))
    diagnostics = collective_call(world, lambda: tuple(sorted(native.program_diagnostics().items())))
    lifecycle = collective_call(world, lambda: (runtime.time(), runtime.macro_step(), tuple(runtime.patch_boxes())))
    return rows, (tuple(metadata), carriers, diagnostics, lifecycle)


def same_images(left, right):
    assert left[1] == right[1] and len(left[0]) == len(right[0])
    for a, b in zip(left[0], right[0], strict=True):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
            assert a[key].tobytes() == b[key].tobytes(), key


@pytest.mark.parametrize("cells", (8, 16))
@pytest.mark.parametrize("width", (1, 2))
def test_evolved_stage_amr_source_and_closed_load(cells, width):
    case, layout = build(cells, width)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert "AmrFieldRightPreconditioner::kFullResidualBasisLU" in cpp
    assert "PreparedAmrFieldResidual" in cpp
    assert cpp.count("ctx.store_global_field_history(") == 2*(width+(width == 2))
    assert resolved.time._serialize()["version"] == 16
    assert cpp.count("evolved_accumulation") >= width
    initial, target, q0, load = closed_data(width)
    from tests.python.support.evolved_stage_mms import accumulation
    assert np.max(np.abs(accumulation(target[:, None])[:, 0]-q0-DT*load)) < 1e-15
    assert np.max(np.abs(initial-target)) > .005
    if width == 2:
        assert load.sum() == 0


@pytest.mark.parametrize("mutation", ("clone", "foreign", "point", "physical", "alias", "metadata"))
def test_global_field_history_storage_source_authority(mutation):
    from pops.time._program.global_history_storage import storage_contract, validate_storage_node
    case, _ = build(8, 2)
    program = case._time_registry.program
    node = next(value for value in program._values if value.op == "store_history")
    source, owner = node.inputs[0], node.block
    original = storage_contract(program, source, owner)
    assert source.block is None and source.state_ref is None and original["ncomp"] == 1
    assert original["point"] == source.point
    if mutation == "clone":
        with pytest.raises(ValueError, match="unissued alias"):
            storage_contract(program, source, owner._with_owner(owner.owner_path))
    elif mutation == "foreign":
        other, _ = build(8, 2)
        foreign = next(value.block for value in other._time_registry.program._values if value.op == "store_history")
        with pytest.raises(ValueError, match="foreign Case"):
            storage_contract(program, source, foreign)
    elif mutation == "point":
        accepted = next(iter(program._time_states.values())).n.point
        altered = program._replace_value(source, point=accepted)
        with pytest.raises(ValueError, match="solve point"):
            storage_contract(program, altered, owner)
    elif mutation == "physical":
        object.__setattr__(source, "block", owner)
        altered = source
        with pytest.raises(ValueError, match="physical State ownership"):
            storage_contract(program, altered, owner)
    elif mutation == "alias":
        program._replace_value(source, name=source.name+"-new-authority")
        with pytest.raises(ValueError, match="current issued observation"):
            storage_contract(program, source, owner)
    else:
        data = dict(node.attrs)
        data["global_field_storage"] = dict(original, ncomp=2)
        altered = program._replace_value(node, attrs=data)
        with pytest.raises(ValueError, match="immutable authority"):
            validate_storage_node(program, altered)


def test_global_field_history_storage_explicit_scope_and_frame():
    from pops.time._program.global_history_storage import storage_contract
    case, _ = build(8, 1)
    program = case._time_registry.program
    node = next(value for value in program._values if value.op == "store_history")
    source = node.inputs[0]
    model = case._block_registry._blocks["Q0"]["model"]
    archive = case.block("observation-archive", model)
    with pytest.raises(ValueError, match="TimeState scope"):
        storage_contract(program, source, archive)
    declaration = next(iter(program._time_states.values())).state.declaration_ref
    from pops.time import Clock
    program.state(archive[declaration], clock=Clock("archive-clock", owner=program.owner_path))
    with pytest.raises(ValueError, match="observation clock"):
        storage_contract(program, source, archive)
    program.state(archive[declaration])
    contract = storage_contract(program, source, archive)
    assert contract["owner_block"] is archive and source.block is None
    # The storage-only block did not add a physical input to the original solve.
    assert all(value.block != archive for value in source.inputs[0].inputs[0].inputs[0].inputs)
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    frame = Rectangle("foreign-frame", lower=(0, 0), upper=(2, 1)).frame(Cartesian2D())
    other_model = pops.Model("other-frame-model", frame=frame)
    state = other_model.state("archive-value", components=("a", "b"))
    other_block = case.block("other-frame-storage", other_model)
    program.state(other_block[state])
    with pytest.raises(ValueError, match="physical frame"):
        storage_contract(program, source, other_block)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("cells", (8, 16))
@pytest.mark.parametrize("width", (1, 2))
def test_public_evolved_stage_amr_checkpoint_and_composite_Q(isolated_native_cache, tmp_path, record_property, cells, width):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout = collective_call(world, lambda: build(cells, width))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="original evolved composite AMR stage", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    def bind():
        return pops.bind(artifact, resources={"execution_context":context})
    runtime = collective_call(world, bind)
    directory = collective_directory(world, tmp_path/"evolved-stage-amr")
    initial = capture(world, runtime, width, histories=False)
    # All ranks publish their actual rank-local manifests while each phase is live.
    # Preserve native row/list order; the independent reader binds rank0 to metadata.
    registry_phases = {"initial": {"rows_by_rank": collective_call(world,
        lambda: allgather_value(world, initial[1][1]))}}
    paths, seals, authorities, phases = {}, {}, {}, {}
    for phase, owner in (("accepted", runtime), ("continuous", runtime), ("replay", None)):
        if phase == "replay":
            owner = collective_call(world, bind)
            collective_call(world, lambda owner=owner: owner.restart(paths["accepted"]))
            reloaded = capture(world, owner, width)
            with collective_check(world):
                same_images(reloaded, phases["accepted"])
            phases["reloaded"] = reloaded
            registry_phases["reloaded"] = {"rows_by_rank": collective_call(world,
                lambda: allgather_value(world, reloaded[1][1]))}
        end = DT if phase == "accepted" else 2*DT
        collective_call(world, lambda owner=owner, end=end: pops.run(owner, t_end=end, max_steps=1, console=False))
        path = collective_call(world, lambda owner=owner, phase=phase: owner.checkpoint(directory/(phase+"-checkpoint")))
        seals[phase] = collective_call(world, lambda path=path: hashlib.sha256(bounded_bytes(path)).hexdigest())
        authorities[phase] = collective_call(world, lambda owner=owner, path=path: checkpoint_provenance(owner, path))
        paths[phase] = path
        phases[phase] = capture(world, owner, width)
        registry_phases[phase] = {"rows_by_rank": collective_call(world,
            lambda phase=phase: allgather_value(world, phases[phase][1][1]))}
    with collective_check(world):
        same_images(phases["continuous"], phases["replay"])
        assert phases["accepted"][1][-1][:2] == (DT, 1)
        assert phases["continuous"][1][-1][:2] == (2*DT, 2)
        for row, old in zip(phases["continuous"][0], phases["accepted"][0], strict=True):
            for name in (*tuple("T%d" % i for i in range(width)), *(("z",) if width == 2 else ())):
                np.testing.assert_array_equal(row[name+"-previous"], old[name])
        if world.rank == 0:
            equivalent = compare_checkpoint_replay(paths, authorities)
            observations = {}
            for phase, image in (("initial", initial), *phases.items()):
                files, reloaded_rows = [], []
                for level, row in enumerate(image[0]):
                    path = directory/(phase+"-level%d.npz" % level)
                    np.savez(path, **row)
                    with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as stored:
                        reloaded_rows.append({key:stored[key].copy() for key in stored.files})
                    files.append({"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()})
                metrics = None if phase == "initial" else check_saved(reloaded_rows, cells, width,
                    None if phase in ("accepted", "reloaded") else phases["accepted"][0],
                    1 if phase in ("accepted", "reloaded") else 2, initial[0])
                if metrics is not None:
                    original = [(name, value) for name, value in image[1][2] if name.endswith(".rel_residual")]
                    assert len(original) == 1 and np.isfinite(original[0][1])
                    assert 0 <= original[0][1] <= CONTROLS["tolerance"]
                    metrics["native_original_F_composite_relative"] = original[0][1]
                for level, row in enumerate(reloaded_rows):
                    np.testing.assert_array_equal(row["forcing"], initial[0][level]["forcing"])
                observations[phase] = {"levels":files, "checks":metrics, "metadata":image[1]}
            components = [("block-"+row.name, row.model) for row in artifact.blocks]
            components += [("program-"+row.layout_id, row.program) for row in artifact.layout_programs]
            compilation = []
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
                compilation.append(entry)
            checkpoints = {phase:{"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()} for phase, path in paths.items()}
            assert all(row["sha256"] == seals[phase] for phase, row in checkpoints.items())
            assert {Path(row["path"]).resolve() for row in checkpoints.values()}.isdisjoint(
                {Path(row["path"]).resolve() for phase in observations.values() for row in phase["levels"]})
            registry_path = directory/"carrier-registry.json"
            registry_path.write_text(json.dumps({"schema":"sol61.amr.carrier-registry@1",
                "dimension":2, "size":world.size, "phases":registry_phases},
                sort_keys=True, indent=2)+"\n")
            registry_pin = {"path":str(registry_path.resolve()),
                "sha256":hashlib.sha256(bounded_bytes(registry_path)).hexdigest()}
            receipt = {"carrier_registry":registry_pin, "fixture_schema":"pops.evolved-stage-amr-native-fixture@2", "artifact":artifact.artifact_identity.token,
                "dimension":2, "rank":world.rank, "size":world.size, "cells":cells, "width":width,
                "history_protocol":{"wire":"POPSHID1", "raw_slots_after_publication":True,
                    "latest_slot":1, "previous_slot":0, "depth":2},
                "newton":CONTROLS, "fd_step":FD_STEP, "acceptance":ACCEPTANCE, "dt":DT,
                "realization":"FullResidualBasisLU@1", "max_dense_bytes":DENSE_BYTES,
                "native":{"path":str(native.__file__), "sha256":hashlib.sha256(bounded_bytes(native.__file__)).hexdigest()},
                "platform":artifact.platform_manifest.to_data(), "compilation":compilation,
                "phases":observations, "checkpoints":checkpoints, "checkpoint_equivalence":equivalent}
            (directory/"receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
        for name, value in (("artifact_identity", artifact.artifact_identity.token), ("dimension",2),
            ("rank",world.rank), ("size",world.size), ("evolved_stage_amr_receipt",str(directory/"receipt.json"))):
            record_property(name, value)
