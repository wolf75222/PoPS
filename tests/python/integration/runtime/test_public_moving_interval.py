"""Real Uniform1D ALE: original bodies, accepted nodes, saved inventory and restart.

Run against a freshly rebuilt native Dim1 SDK. No worker source test executes
this compiler/runtime fixture. POPS_ALE_RESOLUTIONS can select independent sizes.
"""
import json
import os
import numpy as np
import pops
import pytest
from pops.output import NPZ, ParallelMode
from tests.python.unit.codegen.test_moving_interval_codegen import declared_case
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt, state_snapshots
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.moving_interval_receipts import save_moving_snapshot, assert_exact_moving_snapshot
from tests.python.support.moving_interval_diagnostics import require_exact_moving_interval_refusal

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
def configured_resolutions():
    values=tuple(int(value) for value in os.environ.get("POPS_ALE_RESOLUTIONS","16,32").split(","))
    if len(set(values))<2 or len(set(values))!=len(values) or any(value<4 for value in values):
        raise ValueError("ALE reception requires at least two distinct resolutions >=4")
    return values


def world_context():
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator
    return _pops.mpi_world() if native_mpi_communicator(_pops)=="MPI_COMM_WORLD" else None


def declared_initial_case(**kwargs):
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.codegen.program_lowerability import all_ops
    case,layout=declared_case(**kwargs)
    state=next(value for value in all_ops(case._time) if value.op=="geometry_state").state_ref
    case.initials.add(InitialCondition(state=state,value=BindArray(),projection=ConservativeCellAverage()))
    return case,layout


def bind_collectively(world,artifact,initial):
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    with collective_check(world):
        subject=artifact.plan.initial_condition_plan.bindings[0].subject
        values=np.ascontiguousarray(initial)
    return collective_call(world,lambda:pops.bind(artifact,initial_values={subject:values},
                                                   resources={"execution_context":context}))


@pytest.mark.parametrize("components",[("a",),("a","b","c"),("c","a","b")])
@pytest.mark.parametrize("with_source",[False,True])
def test_declared_moving_public_chain(isolated_native_cache,native_cxx,kokkos_root,tmp_path,
                                     record_property,components,with_source):
    del isolated_native_cache,native_cxx,kokkos_root
    world=world_context()
    mode=ParallelMode.SERIAL if world is None else ParallelMode.ROOT
    root=world is None or int(world.rank)==0
    resolutions=collective_call(world,configured_resolutions)
    velocity=-.3 if components==("a","b","c") else .7
    configuration=(resolutions,components,with_source,velocity)
    if world is not None:
        from pops._native_collectives import allgather_value
        configurations=allgather_value(world,configuration)
        with collective_check(world):
            assert all(row==configuration for row in configurations),configurations
    base=collective_directory(world,tmp_path)
    with collective_check(world):
        record_property("rank",0 if world is None else int(world.rank))
        record_property("size",1 if world is None else int(world.size))
        record_property("moving_receipts",str(base/"actual-moving-evidence.json"))
    evidence=[]
    artifacts=[]
    for n in resolutions:
        case,layout=collective_call(world,lambda:declared_initial_case(
            components=components,cells=n,with_source=with_source,velocity=velocity,output_mode=mode))
        resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
        if world is None:
            artifact=collective_call(world,lambda:pops.compile(resolved))
        else:
            from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
            artifact=compile_resolved_plan_once(world,resolved,
                route="moving-public-%d-%s-%s"%(n,"".join(components),with_source),
                compile_artifact=pops.compile)
        with collective_check(world):
            artifact.verify()
            assert artifact.resolved_dimension==1
            artifacts.append(artifact.artifact_identity.token)
            record_property("artifact_identity_cells_%d"%n,artifact.artifact_identity.token)
            record_property("dimension_cells_%d"%n,artifact.resolved_dimension)
            initial=np.stack([np.full(n,{"a":2.,"b":3.,"c":4.}[name]) for name in components])
        runtime=bind_collectively(world,artifact,initial)
        directory=collective_directory(world,base/(str(n)+"-"+"".join(components)))
        with collective_check(world):
            from pops.codegen.program_lowerability import all_ops
            from pops.time.references import canonical_handle
            geometry=next(value for value in all_ops(artifact.program.program) if value.op=="geometry_state")
            identity="pops.moving:"+canonical_handle(geometry.state_ref).qualified_id
            physical_frame=geometry.space.frame
        actual_initial,=state_snapshots(runtime,world,("fluid",))
        initial_geometry=collective_call(world,lambda:runtime._executor._s._output_moving_geometry_snapshot(
            identity,geometry.space.frame))
        with collective_check(world):
            if root:
                np.testing.assert_array_equal(actual_initial,initial)
                assert initial_geometry["generation"]==0
                np.savez(directory/"initial-physical-state.npz",state=actual_initial,
                    node_coordinates=initial_geometry["node_coordinates"],cell_volumes=initial_geometry["cell_volumes"],
                    artifact_identity=artifact.artifact_identity.token,dimension=artifact.resolved_dimension,
                    physical_time=runtime.time(),macro_step=runtime.macro_step())
        save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"initial")
        collective_call(world,lambda:pops.run(runtime,t_end=.001,max_steps=1,console=False,output_dir=directory))
        save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"step1")
        collective_call(world,lambda:pops.run(runtime,t_end=.002,max_steps=1,console=False,output_dir=directory))
        checkpoint,accepted_snapshot=save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"accepted")
        accepted,=state_snapshots(runtime,world,("fluid",))
        mailbox=collective_call(world,runtime._executor._checkpoint_program_exchanges)
        with collective_check(world):
            assert mailbox.startswith(b"POPSEX04")
        collective_call(world,lambda:pops.run(runtime,t_end=.003,max_steps=1,console=False,output_dir=directory))
        _,continuous_snapshot=save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"continuous")
        final,=state_snapshots(runtime,world,("fluid",))
        final_mailbox=collective_call(world,runtime._executor._checkpoint_program_exchanges)
        restored=bind_collectively(world,artifact,initial)
        collective_call(world,lambda:restored.restart(checkpoint))
        _,restored_snapshot=save_moving_snapshot(world,restored,artifact,identity,geometry.space.frame,directory,"restored")
        replay_accepted,=state_snapshots(restored,world,("fluid",))
        replay_mailbox=collective_call(world,restored._executor._checkpoint_program_exchanges)
        with collective_check(world):
            np.testing.assert_array_equal(replay_accepted,accepted)
            assert replay_mailbox==mailbox
            if root: assert_exact_moving_snapshot(accepted_snapshot,restored_snapshot)
        collective_call(world,lambda:pops.run(restored,t_end=.003,max_steps=1,console=False,output_dir=directory/"replay"))
        _,replayed_snapshot=save_moving_snapshot(world,restored,artifact,identity,geometry.space.frame,directory,"replayed")
        replay_final,=state_snapshots(restored,world,("fluid",))
        replay_final_mailbox=collective_call(world,restored._executor._checkpoint_program_exchanges)
        with collective_check(world):
            np.testing.assert_array_equal(replay_final,final)
            assert replay_final_mailbox==final_mailbox
            if root: assert_exact_moving_snapshot(continuous_snapshot,replayed_snapshot)
        # Read only scientific output, never substitute the coordinate law or
        # the runtime's own proposed volumes for its saved accepted geometry.
        with collective_check(world):
            if root:
                with np.load(directory/"initial-physical-state.npz",allow_pickle=False) as image:
                    previous_nodes=image["node_coordinates"][:,0].copy()
                    previous_volumes=image["cell_volumes"].copy()
                saved=[]
                outputs_by_phase={}
                publications=[]
                for path in directory.rglob("*.npz"):
                    if "replay" in path.relative_to(directory).parts: continue
                    with np.load(path,allow_pickle=False) as candidate:
                        if "pops_output_manifest" not in candidate.files: continue
                    reopened=NPZ(mode).reopen(path)
                    publications.append((reopened.manifest["snapshot"]["clock"]["macro_step"],path,reopened))
                assert sorted(row[0] for row in publications)==[1,2,3]
                # Filenames contain a run-family digest before their step. Read
                # the authenticated accepted clock rather than lexical paths.
                for _,path,reopened in sorted(publications,key=lambda row:row[0]):
                    datasets=reopened.manifest["datasets"]
                    geometry=next(iter(datasets["geometries"].values()))
                    nodes=reopened.arrays[geometry["node_coordinates"]][:,0]
                    volumes=reopened.arrays[geometry["cell_volumes"]]
                    np.testing.assert_array_equal(np.diff(nodes),volumes)
                    displacements=nodes-previous_nodes
                    residual=(volumes-previous_volumes)-np.diff(displacements)
                    np.testing.assert_allclose(residual,0,rtol=0,atol=64*np.finfo(float).eps)
                    previous_nodes,previous_volumes=nodes.copy(),volumes.copy()
                    field=next(iter(datasets["fields"].values()))
                    pieces=sorted(field["pieces"],key=lambda piece:piece["lower"])
                    values=np.concatenate([reopened.arrays[piece["name"]] for piece in pieces],axis=1)
                    inventory=np.sum(values*volumes[None,:],axis=1)
                    saved.append((path,inventory,nodes))
                    phase={1:"step1",2:"accepted",3:"continuous"}[reopened.manifest["snapshot"]["clock"]["macro_step"]]
                    assert phase not in outputs_by_phase
                    outputs_by_phase[phase]=str(path)
                assert len(saved)==3,"each accepted interval must publish a real physical snapshot"
                for step,(_,inventory,nodes) in enumerate(saved,1):
                    np.testing.assert_allclose(inventory,initial[:,0]*(1+.05*.001 if with_source else 1)**step,
                                               rtol=4e-14,atol=4e-14)
                    assert np.max(np.abs(nodes-np.linspace(0,1,n+1)))>0
                replay_outputs=[]
                for path in (directory/"replay").rglob("*.npz"):
                    with np.load(path,allow_pickle=False) as candidate:
                        if "pops_output_manifest" in candidate.files: replay_outputs.append(path)
                assert len(replay_outputs)==1
                outputs_by_phase["replayed"]=str(replay_outputs[0])
                if not with_source: np.testing.assert_allclose(final,initial,rtol=3e-14,atol=3e-14)
                evidence.append({"kind":"restart","cells":n,"components":components,"source":with_source,"velocity":velocity,
                    "artifact":artifact.artifact_identity.token,"dimension":artifact.resolved_dimension,
                    "rank":0,"size":1 if world is None else int(world.size),
                    "saved_inventory":saved[-1][1].tolist(),"saved_paths":[str(row[0]) for row in saved],
                    "initial_state":str(directory/"initial-physical-state.npz"),
                    "checkpoint":str(checkpoint),"replay_exact":True,
                    "phases":["initial","step1","accepted","continuous","restored","replayed"],
                    "moving_identity":identity,"physical_frame":physical_frame,
                    "scientific_outputs":outputs_by_phase})
    with collective_check(world):
        receipt=base/"actual-moving-evidence.json"
        if root: receipt.write_text(json.dumps(evidence,indent=2))
        record_property("artifact_identity",json.dumps(artifacts))
        record_property("dimension",1)


@pytest.mark.parametrize("cells",(16,32))
def test_rejected_moving_interval_preserves_state_geometry_wire_and_safe_retry(
        isolated_native_cache,native_cxx,kokkos_root,tmp_path,record_property,cells):
    del isolated_native_cache,native_cxx,kokkos_root
    world=world_context()
    mode=ParallelMode.SERIAL if world is None else ParallelMode.ROOT
    root=world is None or int(world.rank)==0
    directory=collective_directory(world,tmp_path)
    # At both physical endpoints wg=0 and h*.7 exceeds 1/N for N16/N32.
    # The real face provider emits nonfinite rejected flux amounts; the exact
    # shared-face guard refuses before publication. Safe retry caps dt to .001.
    case,layout=collective_call(world,lambda:declared_initial_case(
        components=("c","a","b"),cells=cells,with_source=True,output_mode=mode,proposed_dt=.1))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    if world is None:
        artifact=collective_call(world,lambda:pops.compile(resolved))
    else:
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
        artifact=compile_resolved_plan_once(world,resolved,route="moving-retry-%d"%cells,
                                           compile_artifact=pops.compile)
    with collective_check(world):
        assert artifact.resolved_dimension==1
        from pops.codegen.program_lowerability import all_ops
        from pops.time.references import canonical_handle
        geometry=next(value for value in all_ops(artifact.program.program) if value.op=="geometry_state")
        identity="pops.moving:"+canonical_handle(geometry.state_ref).qualified_id
        initial=np.stack([np.full(cells,value) for value in (4.,2.,3.)])
    runtime=bind_collectively(world,artifact,initial)
    _,before=save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"before")
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.1,max_steps=1,console=False,output_dir=directory))
    with collective_check(world):
        require_exact_moving_interval_refusal(failures, size=1 if world is None else int(world.size))
        assert runtime.time()==0. and runtime.macro_step()==0
    _,rejected=save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"rejected")
    with collective_check(world):
        if root: assert_exact_moving_snapshot(before,rejected)
    collective_call(world,lambda:pops.run(runtime,t_end=.001,max_steps=1,console=False,output_dir=directory))
    _,retried=save_moving_snapshot(world,runtime,artifact,identity,geometry.space.frame,directory,"retried")
    with collective_check(world):
        if root:
            assert retried["generation"]==retried["step"]==1 and retried["time"]==.001
            inventory=np.sum(retried["state"]*retried["cell_volumes"][None,:],axis=1)
            np.testing.assert_allclose(inventory,initial[:,0]*(1+.05*.001),rtol=4e-14,atol=4e-14)
            publications=[]
            for path in directory.rglob("*.npz"):
                with np.load(path,allow_pickle=False) as candidate:
                    if "pops_output_manifest" in candidate.files: publications.append(path)
            assert len(publications)==1,"refused interval must not publish scientific output"
            (directory/"actual-moving-evidence.json").write_text(json.dumps([dict(
                kind="retry",cells=cells,components=["c","a","b"],source=True,velocity=.7,
                artifact=artifact.artifact_identity.token,dimension=1,size=1 if world is None else int(world.size),
                moving_identity=identity,physical_frame=geometry.space.frame,
                phases=["before","rejected","retried"],scientific_outputs={"retried":str(publications[0])},failures=failures)],indent=2))
        record_property("moving_receipts",str(directory/"actual-moving-evidence.json"))
        record_property("artifact_identity",artifact.artifact_identity.token)
        record_property("rank",0 if world is None else int(world.rank))
        record_property("size",1 if world is None else int(world.size))
        record_property("dimension",1)
