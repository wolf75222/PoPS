"""Installed Native dispatch faults; CPU/world1 or real MPI world2, no Host parity claim."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from pops import interfaces
from pops.fields import ExternalFieldSolver
from tests.python.support.external_field_fault_component import fault_source, json_evidence
from tests.python.integration.native_loader.test_external_field_solver_runtime import _component, _topology_source, _program
from tests.python.integration._final_field_program import passive_field_model, resolve_periodic_field_program
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize('fault', ('callback_throw', 'table_preparation'))
def test_installed_external_field_dispatch_fault_rollback_retry(tmp_path, fault,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else int(world.rank)
    ranks=1 if world is None else int(world.size)
    assert ranks in (1,2)
    target=ranks-1
    directory=collective_directory(world,tmp_path/('external-field-'+fault))
    # Publish one authentic source package path before rank-local resolution.
    # A path is retained by the immutable source package and must agree across ranks.
    from pops.external import load
    package=directory/'components'
    collective_call(world,lambda:package.mkdir() if rank==0 else None)
    for name,interface,factory,parameters in (
        ('topology',interfaces.FieldTopology,_topology_source,()),
        ('solver',interfaces.FieldSolver,fault_source,({'name':'answer','kind':'runtime'},))):
        collective_call(world,lambda name=name,interface=interface,factory=factory,parameters=parameters:
            _component(package,name=name,interface=interface,source_factory=factory,
                       manifest_parameters=parameters,instance_parameters={'answer':7} if name=='solver' else {}) if rank==0 else None)
    topology=collective_call(world,lambda:load(package/'topology'/'topology.pops.json').require('topology',interface=interfaces.FieldTopology)())
    solver=collective_call(world,lambda:load(package/'solver'/'solver.pops.json').require('solver',interface=interfaces.FieldSolver)(answer=7))
    provider=ExternalFieldSolver(topology=topology,solver=solver,relative_tolerance=1e-11,absolute_tolerance=0.,max_iterations=23)
    resolved=resolve_periodic_field_program(passive_field_model('dispatch fault',coefficient=0.),_program,
        name='dispatch fault',block_name='material',target='system',n=8,field_solver=provider,components=(topology,solver))
    artifact=compile_resolved_plan_once(world,resolved,route='dispatch-'+fault,
        compile_artifact=pops.compile) if world is not None else pops.compile(resolved)
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    for index, component in enumerate(artifact.component_artifacts):
        collective_call(world,lambda index=index,component=component:(directory/('component-%d-rank%d.so'%(index,rank))).write_bytes(component.binary))
        collective_call(world,lambda index=index,component=component:(directory/('component-%d-rank%d.json'%(index,rank))).write_text(json.dumps(json_evidence(component.to_data()),allow_nan=False,sort_keys=True)))
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_state={'material':np.ones((1,8,8))},resources={'execution_context':artifact_execution_context(artifact)}))
    def open_control():
        installed=runtime.inspect().to_dict()['instance']['installed_components']
        row,=tuple(row for row in installed if row['component_id']==solver.component_manifest.component_id)
        matched,=tuple(component for component in artifact.component_artifacts if component.component_id==solver.component_manifest.component_id)
        assert Path(row['path']).read_bytes()==matched.binary
        (directory/('installed-component-rank%d.json'%rank)).write_text(json.dumps({'installed':json_evidence(row),'sha256':hashlib.sha256(matched.binary).hexdigest(),'after_bind_before_arm':True},allow_nan=False,sort_keys=True))
        library=ctypes.CDLL(row['path'])
        library.pops_test_field_fault_arm.argtypes=[ctypes.c_int]
        library.pops_test_field_fault_arm.restype=ctypes.c_int
        library.pops_test_field_callback_count.restype=ctypes.c_int
        library.pops_test_field_write_count.restype=ctypes.c_int
        return library
    library=collective_call(world,open_control)
    slot,=collective_call(world,runtime.field_provider_slots)
    def capture(label):
        storage=collective_call(world,runtime.observe_accepted_state_storage)
        collective_call(world,lambda:(directory/(label+'-rank%d-local.bin'%rank)).write_bytes(storage.rank_local))
        collective_call(world,lambda:(directory/(label+'-rank%d-complete.bin'%rank)).write_bytes(storage.complete))
        state=collective_call(world,lambda:np.asarray(runtime.state_global('material')).copy())
        collective_call(world,lambda:np.save(directory/(label+'-rank%d-state.npy'%rank),state,allow_pickle=False))
        potential=collective_call(world,lambda:np.asarray(runtime.field_potential_global(slot)).copy())
        collective_call(world,lambda:np.save(directory/(label+'-rank%d-field.npy'%rank),potential,allow_pickle=False))
        metadata={'schema':'sol61.external-field-dispatch-fault-state@1','rank':rank,'ranks':ranks,'clock':[runtime.time(),runtime.macro_step()],'storage_contract':storage.contract,'storage_dimension':storage.dimension,'storage_clock':[storage.time,storage.macro_step],'local_sha256':hashlib.sha256(storage.rank_local).hexdigest(),'complete_sha256':hashlib.sha256(storage.complete).hexdigest(),'consumer_cursors':runtime.consumer_cursors.to_data()}
        collective_call(world,lambda:(directory/(label+'-rank%d.json'%rank)).write_text(json.dumps(metadata,allow_nan=False,sort_keys=True)))
        return storage,state,potential,metadata
    before=capture('before')
    with collective_check(world):
        assert library.pops_test_field_callback_count()==0
        assert library.pops_test_field_fault_arm((1 if fault=='callback_throw' else 2) if rank==target else 0)==0
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=1e-4,max_steps=1,console=False))
    counts={'callbacks':library.pops_test_field_callback_count(),'writes':library.pops_test_field_write_count(),'rank':rank,'target':target,'fault':fault,'failures':failures}
    collective_call(world,lambda:(directory/('failure-rank%d.json'%rank)).write_text(json.dumps(counts,allow_nan=False,sort_keys=True)))
    # Restore only test-provider table authority before readonly captures/retry, never numerical state.
    collective_call(world,lambda:library.pops_test_field_fault_arm(0))
    after=capture('after')
    with collective_check(world):
        assert all(error and error[2] for error in failures), failures
        expected='execution failed collectively' if ranks==2 and fault=='callback_throw' else ('callback preparation failed collectively' if ranks==2 else ('after actual writes' if fault=='callback_throw' else 'interface table is truncated'))
        assert all(expected in error[1] for error in failures), failures
        assert counts['callbacks']==(1 if fault=='callback_throw' else 0)
        if fault=='callback_throw' and runtime.local_boxes('material'): assert counts['writes']>0
        assert before[0]==after[0]
        assert np.array_equal(before[1].view(np.uint64),after[1].view(np.uint64))
        assert np.array_equal(before[2].view(np.uint64),after[2].view(np.uint64))
        assert before[3]==after[3]
    result=collective_call(world,lambda:pops.run(runtime,t_end=1e-4,max_steps=1,console=False))
    retry_image=capture('retry')
    with collective_check(world):
        assert result.accepted_steps==1 and runtime.macro_step()==1 and runtime.time()==1e-4
        assert library.pops_test_field_callback_count()==counts['callbacks']+1
        assert np.array_equal(retry_image[1],before[1])
        assert np.all(retry_image[2]==7.)

