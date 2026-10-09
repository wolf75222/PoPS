"""Installed native composite Q evolution; ROOT owns the actual execution."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.integration.runtime.test_public_evolved_original_stage import checkpoint_provenance, compare_checkpoint_replay_v2 as compare_checkpoint_replay, checkpoint_restart_authority
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_amr import (
    ACCEPTANCE, CONTROLS, DENSE_BYTES, DT, FD_STEP, build, check_saved, closed_data, published_history_image,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import (
    native_profile_world, world_rank, world_size, world_allgather,
)


from tests.python.integration.runtime.test_public_evolved_stage_amr import capture, same_images
from tests.python.support.evolved_stage_v_capture import retain_v_provenance, capture_initial_carriers, initial_carrier_authority

@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("cells", (8, 16))
@pytest.mark.parametrize("width", (1, 2))
def test_public_evolved_stage_amr_c25_and_initial_carrier(isolated_native_cache, tmp_path, record_property, cells, width):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native_profile_world(native)
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout = collective_call(world, lambda: build(cells, width))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout, compile_options={"model_source_policy":"require"}))
    artifact = (pops.compile(resolved) if world is None else
                compile_resolved_plan_once(world, resolved, route="original evolved composite AMR stage", compile_artifact=pops.compile))
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    def bind():
        return pops.bind(artifact, resources={"execution_context":context})
    directory = collective_directory(world, tmp_path/"evolved-stage-amr")
    compilation = collective_call(world, lambda: retain_v_provenance(artifact, native, directory, world_rank(world)))
    runtime = collective_call(world, bind)
    initial_authority = initial_carrier_authority(world, runtime, artifact)
    initial_carriers = capture_initial_carriers(world, runtime, directory, initial_authority)
    initial = capture(world, runtime, width, histories=False)
    # All ranks publish their actual rank-local manifests while each phase is live.
    # Preserve native row/list order; the independent reader binds rank0 to metadata.
    registry_phases = {"initial": {"rows_by_rank": collective_call(world,
        lambda: world_allgather(world, initial[1][1]))}}
    paths, seals, authorities, phases = {}, {}, {}, {}
    for phase, owner in (("accepted", runtime), ("continuous", runtime), ("replay", None)):
        if phase == "replay":
            owner = collective_call(world, bind)
            restart_authority = collective_call(world, lambda: checkpoint_restart_authority(owner))
            collective_call(world, lambda owner=owner: owner.restart(paths["accepted"]))
            restart_authority["restored_source_run"] = collective_call(world, lambda: owner.last_run_identity.token)
            reloaded = capture(world, owner, width)
            with collective_check(world):
                same_images(reloaded, phases["accepted"])
            phases["reloaded"] = reloaded
            registry_phases["reloaded"] = {"rows_by_rank": collective_call(world,
                lambda: world_allgather(world, reloaded[1][1]))}
        end = DT if phase == "accepted" else 2*DT
        collective_call(world, lambda owner=owner, end=end: pops.run(owner, t_end=end, max_steps=1, console=False))
        path = collective_call(world, lambda owner=owner, phase=phase: owner.checkpoint(directory/(phase+"-checkpoint")))
        seals[phase] = collective_call(world, lambda path=path: hashlib.sha256(bounded_bytes(path)).hexdigest())
        authorities[phase] = collective_call(world, lambda owner=owner, path=path: checkpoint_provenance(owner, path,
            restart_authority=restart_authority if phase == "replay" else None))
        paths[phase] = path
        phases[phase] = capture(world, owner, width)
        registry_phases[phase] = {"rows_by_rank": collective_call(world,
            lambda phase=phase: world_allgather(world, phases[phase][1][1]))}
    with collective_check(world):
        same_images(phases["continuous"], phases["replay"])
        assert phases["accepted"][1][-1][:2] == (DT, 1)
        assert phases["continuous"][1][-1][:2] == (2*DT, 2)
        for row, old in zip(phases["continuous"][0], phases["accepted"][0], strict=True):
            for name in (*tuple("T%d" % i for i in range(width)), *(("z",) if width == 2 else ())):
                np.testing.assert_array_equal(row[name+"-previous"], old[name])
        if world_rank(world) == 0:
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
            checkpoints = {phase:{"path":str(path), "sha256":hashlib.sha256(bounded_bytes(path)).hexdigest()} for phase, path in paths.items()}
            assert all(row["sha256"] == seals[phase] for phase, row in checkpoints.items())
            assert {Path(row["path"]).resolve() for row in checkpoints.values()}.isdisjoint(
                {Path(row["path"]).resolve() for phase in observations.values() for row in phase["levels"]})
            registry_path = directory/"carrier-registry.json"
            registry_path.write_text(json.dumps({"schema":"sol61.amr.carrier-registry@1",
                "dimension":2, "size":world_size(world), "phases":registry_phases},
                sort_keys=True, indent=2)+"\n")
            registry_pin = {"path":str(registry_path.resolve()),
                "sha256":hashlib.sha256(bounded_bytes(registry_path)).hexdigest()}
            receipt = {"carrier_registry":registry_pin, "fixture_schema":"pops.evolved-stage-amr-native-fixture@3", "initial_carriers":initial_carriers, "retained_provenance_before_bind":True, "artifact":artifact.artifact_identity.token,
                "dimension":2, "rank":world_rank(world), "size":world_size(world), "cells":cells, "width":width,
                "history_protocol":{"wire":"POPSHID1", "raw_slots_after_publication":True,
                    "latest_slot":1, "previous_slot":0, "depth":2},
                "newton":CONTROLS, "fd_step":FD_STEP, "acceptance":ACCEPTANCE, "dt":DT,
                "realization":"FullResidualBasisLU@1", "max_dense_bytes":DENSE_BYTES,
                "native":{"path":str(native.__file__), "sha256":hashlib.sha256(bounded_bytes(native.__file__)).hexdigest()},
                "platform":artifact.platform_manifest.to_data(), "compilation":compilation,
                "phases":observations, "checkpoints":checkpoints, "checkpoint_equivalence":equivalent}
            (directory/"receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
        for name, value in (("artifact_identity", artifact.artifact_identity.token), ("dimension",2),
            ("rank",world_rank(world)), ("size",world_size(world)), ("evolved_stage_amr_receipt",str(directory/"receipt.json"))):
            record_property(name, value)
