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
from tests.python.support.collective_checks import collective_call, collective_check, state_snapshots
from tests.python.support.integral_state_receipts import collective_directory

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
    configuration=(resolutions,components,with_source)
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
            components=components,cells=n,with_source=with_source,output_mode=mode))
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
        collective_call(world,lambda:pops.run(runtime,t_end=.002,max_steps=2,console=False,output_dir=directory))
        checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/"accepted-restart"))
        accepted,=state_snapshots(runtime,world,("fluid",))
        mailbox=collective_call(world,runtime._executor._checkpoint_program_exchanges)
        with collective_check(world):
            assert mailbox.startswith(b"POPSEX03")
        collective_call(world,lambda:pops.run(runtime,t_end=.003,max_steps=1,console=False,output_dir=directory))
        final,=state_snapshots(runtime,world,("fluid",))
        final_mailbox=collective_call(world,runtime._executor._checkpoint_program_exchanges)
        restored=bind_collectively(world,artifact,initial)
        collective_call(world,lambda:restored.restart(checkpoint))
        replay_accepted,=state_snapshots(restored,world,("fluid",))
        replay_mailbox=collective_call(world,restored._executor._checkpoint_program_exchanges)
        with collective_check(world):
            np.testing.assert_array_equal(replay_accepted,accepted)
            assert replay_mailbox==mailbox
        collective_call(world,lambda:pops.run(restored,t_end=.003,max_steps=1,console=False,output_dir=directory/"replay"))
        replay_final,=state_snapshots(restored,world,("fluid",))
        replay_final_mailbox=collective_call(world,restored._executor._checkpoint_program_exchanges)
        with collective_check(world):
            np.testing.assert_array_equal(replay_final,final)
            assert replay_final_mailbox==final_mailbox
        # Read only scientific output, never substitute the coordinate law or
        # the runtime's own proposed volumes for its saved accepted geometry.
        with collective_check(world):
            if root:
                with np.load(directory/"initial-physical-state.npz",allow_pickle=False) as image:
                    previous_nodes=image["node_coordinates"][:,0].copy()
                    previous_volumes=image["cell_volumes"].copy()
                saved=[]
                for path in sorted(directory.rglob("*.npz")):
                    if "replay" in path.relative_to(directory).parts: continue
                    with np.load(path,allow_pickle=False) as candidate:
                        if "pops_output_manifest" not in candidate.files: continue
                    reopened=NPZ(mode).reopen(path)
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
                assert len(saved)==3,"each accepted interval must publish a real physical snapshot"
                for step,(_,inventory,nodes) in enumerate(saved,1):
                    np.testing.assert_allclose(inventory,initial[:,0]*(1+.05*.001 if with_source else 1)**step,
                                               rtol=4e-14,atol=4e-14)
                    assert np.max(np.abs(nodes-np.linspace(0,1,n+1)))>0
                if not with_source: np.testing.assert_allclose(final,initial,rtol=3e-14,atol=3e-14)
                evidence.append({"cells":n,"components":components,"source":with_source,
                    "artifact":artifact.artifact_identity.token,"dimension":artifact.resolved_dimension,
                    "rank":0,"size":1 if world is None else int(world.size),
                    "saved_inventory":saved[-1][1].tolist(),"saved_paths":[str(row[0]) for row in saved],
                    "initial_state":str(directory/"initial-physical-state.npz"),
                    "checkpoint":str(checkpoint),"replay_exact":True})
    with collective_check(world):
        receipt=base/"actual-moving-evidence.json"
        if root: receipt.write_text(json.dumps(evidence,indent=2))
        record_property("artifact_identity",json.dumps(artifacts))
        record_property("dimension",1)
