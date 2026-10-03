"""Prospective installed CPU/MPI witness: collection is not Native reception."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pops
import pytest
from examples.migration.scientific.api040_m18_w09 import make_case, NODES, WEIGHTS, CERTIFICATES
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _world
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt, state_snapshots
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]

@pytest.mark.parametrize('moment,diagnostic',[(.49,None),(.5,'upper_support_finite_dual_not_certified'),(.9,'upper_support_target_infeasible')])
def test_declared_quadrature_near_boundary_and_w09(isolated_native_cache,tmp_path,record_property,moment,diagnostic):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2); world = _world()
    rank = 0 if world is None else world.rank
    case,layout,subjects = make_case()
    resolved = pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    artifact = pops.compile(resolved) if world is None else compile_resolved_plan_once(world,resolved,route='declared finite quadrature domain',compile_artifact=pops.compile)
    directory = collective_directory(world,tmp_path/'entropy-domain')
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    prescribed = np.stack((np.ones((2,2)),np.full((2,2),moment)))
    runtime = collective_call(world,lambda:pops.bind(artifact,initial_values={subjects[0]:np.zeros_like(prescribed),subjects[1]:prescribed.copy()},resources={'execution_context':artifact_execution_context(artifact)}))
    def json_file(path,value):
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n')
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def capture(label):
        observation = collective_call(world,runtime.observe_accepted_state_storage)
        def raw():
            pins={}
            for name,payload in [('rank-local',observation.rank_local),('complete',observation.complete)]:
                path=directory/f'{label}.rank{rank}.{name}.bin';path.write_bytes(payload)
                pins[name]={'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()}
            pins['metadata']=json_file(directory/f'{label}.rank{rank}.json',{'contract':observation.contract,'time':runtime.time(),'macro_step':runtime.macro_step(),'history':runtime.consumer_cursors.to_data(),'dimension':observation.dimension})
            return pins
        pins=collective_call(world,raw)
        states=state_snapshots(runtime,world,('unknown','prescribed'))
        def valid():
            for name,array in zip(('dual','target'),states,strict=True):
                path=directory/f'{label}.rank{rank}.{name}.npy';np.save(path,array,allow_pickle=False)
                pins[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            json_file(directory/f'{label}.rank{rank}.pins.json',pins)
        collective_call(world,valid)
        return observation.complete,states,(runtime.time(),runtime.macro_step()),runtime.consumer_cursors.to_data()
    before=capture('before')
    result,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.01,max_steps=1,console=False))
    collective_call(world,lambda:json_file(directory/f'attempt.rank{rank}.json',{'schema':'pops.entropy-declared-domain-fixture@1','moment':moment,'failures':failures,'certificates':[c.to_data() for c in CERTIFICATES]}))
    after=capture('after')
    with collective_check(world):
        np.testing.assert_array_equal(after[1][1].reshape(2,2,2),prescribed)
        if diagnostic is not None:
            assert all(f is not None and f[0]=='RuntimeError' and diagnostic in f[1] for f in failures),failures
            assert before[0]==after[0] and before[2:]==after[2:]
            for a,b in zip(before[1],after[1],strict=True):assert a.tobytes()==b.tobytes()
        else:
            assert not any(failures),failures
            assert after[2]==(.01,1)
            dual=after[1][0].reshape(2,2,2)
            populations=np.asarray(WEIGHTS)[:,None,None]*np.exp(dual[0]+np.asarray(NODES)[:,None,None]*dual[1])
            assert np.isfinite(populations).all() and (populations>0).all()
            reconstructed=np.stack((populations.sum(axis=0),(np.asarray(NODES)[:,None,None]*populations).sum(axis=0)))
            np.testing.assert_allclose(reconstructed,prescribed,rtol=0,atol=2e-11)
            variation=.05*populations.min(axis=0)
            perturbed=populations+np.asarray((1.,-2.,1.))[:,None,None]*variation
            entropy=lambda p:(p*(np.log(p/np.asarray(WEIGHTS)[:,None,None])-1)).sum(axis=0)
            assert (perturbed>0).all() and (entropy(perturbed)>entropy(populations)).all()
            np.savez(directory/f'original-residual.rank{rank}.npz',populations=populations,residual=reconstructed-prescribed,entropy=entropy(populations),entropy_gap=entropy(perturbed)-entropy(populations))
    # Public CP9 retains accepted State/clock metadata; no declared History
    # variable exists in this algebraic closure. Do not invent one for the receipt.
    checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/'observed'))
    collective_call(world,lambda:json_file(directory/f'checkpoint.rank{rank}.json',{'path':str(checkpoint),'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}))
    record_property('entropy_domain_receipt',str(directory/f'attempt.rank{rank}.json'))
    record_property('rank',rank);record_property('size',1 if world is None else world.size)
