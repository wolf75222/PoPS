"""Prepared installed-native IR19 campaign; execution is owned by ROOT."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.integration.runtime.test_public_evolved_original_stage import checkpoint_provenance, compare_checkpoint_replay
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.completed_original_interaction import (
    ACCEPTANCE, CELLS, CONTROLS, KINDS, SCHEMA, WORKSPACE,
    build, capture, check, digest, origins, same, save_phase,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_amr import DENSE_BYTES, DT, FD_STEP


@pytest.mark.parametrize("width", (1,2), ids=("scalar", "coupled-partition-permuted"))
def test_completed_original_interaction_source_uses_real_accept_and_keeps_original_F(width):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.support.evolved_stage_amr import build as original_build
    original, _ = original_build(CELLS,width)
    case, layout, profile = build(width)
    old_program, program = original._time_registry.program, case._time_registry.program
    old_solve = next(node for node in old_program._values if node.op == "solve_spatial_field")
    solve = next(node for node in program._values if node.op == "solve_spatial_field")
    assert old_solve.attrs == solve.attrs
    source = next(node for node in program._values if node.id == profile["source_value_id"])
    assert source.block is source.space is source.state_ref is None
    assert profile["tuple_component"] == width-1
    assert source.name == profile["source_ssa_name"]
    assert set(node.attrs["diagnostic"] for node in program._values if node.op == "record_scalar") == {
        "ir19."+profile["source_name"]+"."+kind for kind in KINDS}
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved.time._serialize()["version"] == 19
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert "ctx.seal_original_field_source(" in cpp and "ctx.prepare_closed_original_interaction(" in cpp
    assert "accepted_original_candidate_generation" not in cpp  # Native owns this guard.
    assert "AmrFieldRightPreconditioner::kFullResidualBasisLU" in cpp
    assert cpp.count("ctx.reduce_closed_original_interaction(") == 8  # flat plus composite observer routes
    initial = resolved.initial_condition_plan
    assert len(initial.bindings) == width+1
    for row in initial.bindings:
        assert initial.canonical_subject(row.subject) is row.subject


def checkpoint(world, runtime, directory, phase):
    """Seal/authenticate the actual checkpoint before any observation write."""
    from pops.runtime._checkpoint_manifest import MANIFEST_KEY, authenticate_checkpoint_payload
    path = collective_call(world, lambda: runtime.checkpoint(directory/(phase+"-checkpoint")))
    checksum = collective_call(world, lambda: digest(path))
    def authenticate():
        with np.load(BytesIO(bounded_bytes(path)), allow_pickle=False) as stored:
            manifest = json.loads(str(stored[MANIFEST_KEY]))
            restart = authenticate_checkpoint_payload(runtime, stored, runtime_kind=manifest["runtime_kind"])
        return {"clock":manifest["clock"], "restart":restart.token,
            "semantic":manifest["semantic_identity"], "artifact":manifest["artifact_identity"],
            "bind":manifest["bind_identity"], "run":manifest["run_identity"]}
    authority = collective_call(world, authenticate)
    return {"path":str(path),"sha256":checksum,"authority":authority}


def test_reference_kernel_math_is_signed_nonsymmetric_and_uses_finest_measures(tmp_path):
    """Synthetic HOST math only; no native positive or forged runtime receipt."""
    from tests.python.support.completed_original_interaction import reference
    rows, geometries = [], []
    for extent in (4,8):
        y,x = np.meshgrid((np.arange(extent)+.5)/extent,(np.arange(extent)+.5)/extent,indexing="ij")
        active = np.zeros((extent,extent),dtype=np.bool_)
        active[:,extent//2 if extent == 4 else 0:(extent if extent == 4 else extent//2)] = True
        rows.append({"active":active,"T0":np.full((extent,extent),2.),"Q0":np.full((extent,extent),6.)})
        geometries.append({"centers":np.stack((x,y)),"cell_volumes":np.full((extent,extent),1./extent**2),
            "valid_cells":np.ones_like(active),"coverage":~active})
    image = {"rows":rows,"geometry":geometries,"source_role":"SYNTHETIC_HOST_REFERENCE_DISCRIMINANT"}
    computed,reductions = reference(image,"T0")
    for result,geometry in zip(computed,geometries,strict=True):
        x,y = geometry["centers"]
        np.testing.assert_allclose(result,x+y-1.5,rtol=0,atol=1e-15)
        assert np.max(np.abs(result-(1+y-4*x))) > .1  # transposed kernel differs
    assert reductions["min"] < 0 < reductions["max"]
    wrong,_ = reference(image,"Q0")
    assert any(np.max(np.abs(left-right)) > .1 for left,right in zip(computed,wrong,strict=True))
    paths = save_phase(tmp_path,"host-reference-only",image,computed)
    assert paths["I_array_role"] == "COMPUTED_REFERENCE_ONLY"


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("width", (1,2), ids=("scalar", "coupled-partition-permuted"))
def test_public_completed_original_interaction_exact_restart_and_saved_reductions(
        isolated_native_cache, tmp_path, record_property, width):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case, layout, profile = collective_call(world, lambda: build(width))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="completed original composite source IR19", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    def bind():
        # The helper has genuine declared Analytic ICs. No ad-hoc array/index
        # initialization replaces their canonical InitialConditionPlan Handles.
        plan = artifact.plan.initial_condition_plan
        assert len(plan.bindings) == width+1
        assert all(plan.canonical_subject(row.subject) is row.subject for row in plan.bindings)
        return pops.bind(artifact, resources={"execution_context":context})
    runtime = collective_call(world, bind)
    directory = collective_directory(world, tmp_path/"completed-original-interaction")
    checkpoints = {"initial":checkpoint(world,runtime,directory,"initial")}
    phases = {"initial":capture(world,runtime,width,initial=True)}
    provenance = {}
    for phase, owner in (("accepted",runtime),("continuous",runtime),("replay",None)):
        if phase == "replay":
            owner = collective_call(world, bind)
            collective_call(world, lambda owner=owner: owner.restart(checkpoints["accepted"]["path"]))
            checkpoints["reloaded"] = checkpoint(world,owner,directory,"reloaded")
            phases["reloaded"] = capture(world,owner,width)
            with collective_check(world):
                same(phases["accepted"],phases["reloaded"])
        end = DT if phase == "accepted" else 2*DT
        collective_call(world, lambda owner=owner,end=end: pops.run(owner,t_end=end,max_steps=1,console=False))
        checkpoints[phase] = checkpoint(world,owner,directory,phase)
        provenance[phase] = collective_call(world, lambda owner=owner,phase=phase:
            checkpoint_provenance(owner,checkpoints[phase]["path"]))
        phases[phase] = capture(world,owner,width)
    with collective_check(world):
        same(phases["continuous"],phases["replay"])
        for image in phases.values():
            for row,initial in zip(image["rows"],phases["initial"]["rows"],strict=True):
                np.testing.assert_array_equal(row["forcing"],initial["forcing"])
        assert phases["accepted"]["metadata"][-1][:2] == (DT,1)
        assert phases["continuous"]["metadata"][-1][:2] == (2*DT,2)
        for latest, previous in zip(phases["continuous"]["rows"],phases["accepted"]["rows"],strict=True):
            for name in (*tuple("T%d" % index for index in range(width)), *(("z",) if width == 2 else ())):
                np.testing.assert_array_equal(latest[name+"-previous"],previous[name])
        if world.rank == 0:
            import tests.python.support.evolved_stage_amr as original_helper
            equivalence = compare_checkpoint_replay({phase:row["path"] for phase,row in checkpoints.items()},provenance)
            observations = {}
            for phase, image in phases.items():
                expected, metrics = (None,None) if phase == "initial" else check(image,phases["initial"],
                    None if phase in ("accepted","reloaded") else phases["accepted"]["rows"],width,
                    1 if phase in ("accepted","reloaded") else 2,profile["source_name"])
                observations[phase] = save_phase(directory,phase,image,expected)
                observations[phase]["checks"] = metrics
            for row in checkpoints.values():
                assert digest(row["path"]) == row["sha256"]
            assert {Path(row["path"]).resolve() for row in checkpoints.values()}.isdisjoint(
                Path(part["path"]).resolve() for row in observations.values() for key,part in row.items() if key in ("wire","npz"))
            receipt = {"schema":SCHEMA,"artifact":artifact.artifact_identity.token,"dimension":2,
                "rank":world.rank,"size":world.size,"cells":CELLS,"width":width,"profile":profile,
                "dt":DT,"newton":CONTROLS,"fd_step":FD_STEP,"acceptance":ACCEPTANCE,
                "original_realization":"FullResidualBasisLU@1","max_dense_bytes":DENSE_BYTES,
                "interaction":"DirectSpatialInteraction@1","max_workspace_bytes":WORKSPACE,
                "source_contract":"pops.completed-original-field-source@1",
                "array_wire":"pops.spatial-interaction-fixture-array-wire@1",
                "history":{"wire":"POPSHID1","latest_slot":1,"previous_slot":0,"depth":2},
                "source_physical_owner":None,"storage_owner_only":profile["storage"],
                "native_I_array_available":False,"I_array_role":"COMPUTED_REFERENCE_ONLY",
                "native_I_reductions":KINDS,"entire_I_per_cell_qualified":False,
                "full_nonlocal_newton_qualified":False,"m26_pde_qualification":False,
                "native":{"path":str(native.__file__),"sha256":digest(native.__file__)},
                "platform":artifact.platform_manifest.to_data(),"compilation":origins(artifact,directory),
                "initial_condition_plan":artifact.plan.initial_condition_plan.canonical_identity(),
                "initial_condition_plan_identity":artifact.plan.initial_condition_plan.identity.token,
                "phases":observations,"checkpoints":checkpoints,"checkpoint_equivalence":equivalence,
                "fixture_sources":[{"path":str(path),"sha256":digest(path)} for path in (
                    Path(__file__).resolve(),Path(sys.modules[build.__module__].__file__).resolve(),
                    Path(original_helper.__file__).resolve())]}
            (directory/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
        for name,value in (("artifact_identity",artifact.artifact_identity.token),("dimension",2),
                ("rank",world.rank),("size",world.size),("completed_original_interaction_receipt",str(directory/"receipt.json")),
                ("fixture_schema",SCHEMA),("width",width)):
            record_property(name,value)
