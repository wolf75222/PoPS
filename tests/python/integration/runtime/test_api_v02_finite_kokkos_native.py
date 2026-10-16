"""Actual Program DSO Kokkos dispatch; backend identity is observed, never inferred."""
import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from pops.time import FixedDt
from examples.api_v02.methods_T01_T12 import _storage, _initial
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _build(kind):
    case, layout, model, block, state, p, q, initial = _storage(
        "finite_kokkos_"+kind,("a","b"),(1.,2.),cells=(8,4),sampling="cell_average")
    q.n
    n=0 if kind=='empty' else 257
    domain=p.range_set(n,capacity=2**63-1,ownership='replicated')
    expected=initial.copy()
    if kind in ('arithmetic','gather','empty'):
        big=2**53+3
        values=p.map(domain,lambda b,i:b.product(
            exact=b.index_value(i)+big,
            value=b.branch(b.index_value(i)==0,lambda c:c.float64(7),
                lambda c:c.float64(1)/c.float64(c.index_value(i))),
            even=b.index_value(i)%2==0))
        if kind=='gather':
            values=p.map(domain,lambda b,i:values[b.index(domain,n-1-b.index_value(i))])
        total=p.reduce(values,p.product(k=p.int64(0),x=p.float64(0),position=p.int64(0)),
            lambda b,s,x:b.product(k=s['k']+(x['exact']-big)*(s['position']+1),
                x=s['x']+x['value']*b.float64(s['position']+1),position=s['position']+1),order='ascending')
        left=p.value('integer_result',p.float64(total['k'])*q.n)
        right=p.value('real_result',total['x']*q.n)
        selected=p.value('map_result',(left[0],right[1]))
        order=list(reversed(range(n))) if kind=='gather' else list(range(n))
        expected[0]*=sum((j+1)*k for j,k in enumerate(order))
        expected[1]*=sum((j+1)*(7. if k==0 else 1./k) for j,k in enumerate(order))
    else:
        def attempt(b):
            if kind=='overflow':
                return b.map(domain,lambda c,i:c.index_value(i)+(2**63-256))
            if kind=='bounds':
                a=b.map(domain,lambda c,i:c.index_value(i))
                return b.map(domain,lambda c,i:a[c.index(domain,c.index_value(i)+1)])
            return b.map(domain,lambda c,i:c.float64(1)/c.float64(256-c.index_value(i)))
        outcome=p.capture(attempt)
        selected=p.match(outcome,
            ok=lambda b,_:b.value('unexpected_success',0.*q.n),
            err=lambda b,error:b.branch((error['origin']==2),
                lambda c:c.branch(error['code']==3,lambda d:q.n,lambda d:d.value('wrong_error',0.*q.n)),
                lambda c:c.value('wrong_origin',0.*q.n)))
    p.commit(q.next,p.value('accepted',selected,at=q.next.point))
    p.step_strategy(FixedDt(.01));case.program(p)
    def check(runtime):
        actual=state_array(runtime,'method',initial.shape)
        if actual is None:return non_owner_metrics()
        error=float(np.max(np.abs(actual-expected)))
        return dict(passed=error<2e-12,metrics_owner=True,max_error=error)
    return CaseSpec(case,layout,_initial(case,block[state],initial),{'method':block[state]},
                    .01,.01,check,'actual finite Kokkos dispatch '+kind)


@pytest.mark.parametrize('kind',('arithmetic','gather','empty','overflow','bounds','nonfinite'))
def test_installed_finite_map_dispatch(kind,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='finite Kokkos '+kind)
    runtime,artifact,metrics=execute_case(lambda:_build(kind),world,'finite Kokkos '+kind)
    with collective_check(world):
        program=artifact.program
        assert program is not None
        path=Path(program.so_path).resolve();dso=ctypes.CDLL(str(path))
        get_count=dso.pops_finite_kokkos_launch_count;get_count.argtypes=[];get_count.restype=ctypes.c_uint64
        get_space=dso.pops_finite_kokkos_execution_space;get_space.argtypes=[];get_space.restype=ctypes.c_char_p
        space=get_space().decode();launches=get_count()
        assert launches==0 if kind=='empty' else launches>0
        host_count=dso.pops_finite_host_map_count;host_count.argtypes=[];host_count.restype=ctypes.c_uint64
        assert host_count()==0
        last_failure=dso.pops_finite_kokkos_last_failure_index
        last_failure.argtypes=[];last_failure.restype=ctypes.c_int64
        assert last_failure()==(256 if kind in ('overflow','bounds','nonfinite') else -1)
        if os.environ.get('POPS_REQUIRE_DEVICE')=='1':
            assert space.lower() in ('cuda','hip','sycl','openmptarget'),space
        get_hash=dso.pops_program_hash;get_hash.argtypes=[];get_hash.restype=ctypes.c_char_p
        assert get_hash().decode()==runtime._executor._s.installed_program_hash()
        record_property('artifact_identity',artifact.artifact_identity.hexdigest)
        record_property('program_dso_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
        record_property('kokkos_execution_space',space)
        record_property('actual_nonempty_map_launches',launches)
        record_property('host_map_invocations',host_count())
        record_property('first_failing_map_index',last_failure())
        record_property('metrics',str(metrics))
