"""Distributed equations dispatched by actual Kokkos kernels, with native MPI publication."""
import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from pops.time import FixedDt
from pops.algebra import Distributed
from examples.api_v02.methods_T01_T12 import _storage, _initial
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_check
from tests.python.support.public_native_profile import native_profile, rank_and_size

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _build(kind, ranks):
    case, layout, model, block, state, p, q, initial = _storage(
        "distributed_kokkos_"+kind,("a","b"),(1.,2.),cells=(8,4),sampling="cell_average")
    q.n
    n = 0 if kind == 'empty' else 5
    # Empty first rank (MPI), nonuniform local cardinalities, actual storage only.
    offsets = tuple(n*max(0,r-1)//max(1,ranks-1) for r in range(ranks+1)) if ranks>1 else (0,n)
    d = Distributed(p); domain=d.domain(offsets, capacity=2**63-1)
    expected=initial.copy()
    def mapped(body):
        return d.local_map(domain,lambda b,i:b.capture(lambda c:body(c,i)),byte_capacity=1<<20)
    base=p.require_success(mapped(lambda b,i:b.product(
        exact=Distributed(b).gid(i)+(2**53+3),
        value=b.branch(Distributed(b).gid(i)==0,lambda c:c.float64(7),
                       lambda c:c.float64(1)/c.float64(Distributed(c).gid(i))),
        owner=Distributed(b).rank())))
    if kind in ('arithmetic','import','empty','nested'):
        source=base
        if kind=='nested':
            species=p.range_set(2,capacity=2,ownership='replicated')
            requests=d.requests(domain,domain,lambda b,i:Distributed(b).index(domain,n-1-Distributed(b).gid(i)))
            container=p.map(species,lambda b,i:b.product(mass=b.float64(b.index_value(i)+1),
                                                       domain=domain,values=base,requests=requests))
            held=container[p.index(species,1)]
            # Shared domain/version authority; each rank still owns its distinct shard.
            source=d.import_values(held['values'],held['requests'],byte_capacity=1<<20)
        if kind=='import':
            requests=d.requests(domain,domain,lambda b,i:Distributed(b).index(domain,n-1-Distributed(b).gid(i)))
            source=d.import_values(base,requests,byte_capacity=1<<20)
        result=p.require_success(mapped(lambda b,i:b.product(
            exact=(Distributed(b).at_local(source,i)['exact']-(2**53+3))*(Distributed(b).gid(i)+1),
            value=Distributed(b).at_local(source,i)['value']*b.float64(Distributed(b).gid(i)+1))))
        total=d.reduce(result,p.product(k=p.int64(0),x=p.float64(0)),
            lambda b,s,x:b.product(k=s['k']+x['exact'],x=s['x']+x['value']),
            order='global_gid_ascending',byte_capacity=1<<20)
        left=p.value('integer_result',p.float64(total['k'])*q.n)
        right=p.value('real_result',total['x']*q.n)
        selected=p.value('map_result',(left[0],right[1]))
        order=list(reversed(range(n))) if kind in ('import','nested') else list(range(n))
        expected[0]*=sum((j+1)*k for j,k in enumerate(order))
        expected[1]*=sum((j+1)*(7. if k==0 else 1./k) for j,k in enumerate(order))
    else:
        def attempt(b,i):
            gid=Distributed(b).gid(i)
            if kind=='overflow':return gid+(2**63-4)
            if kind=='bounds':return Distributed(b).at_local(base,Distributed(b).index(domain,gid+1))['exact']
            return b.float64(1)/b.float64(4-gid)
        outcome=mapped(attempt)
        failed_gid = (offsets[2]-1 if ranks>1 else n-1) if kind=='bounds' else 4
        failed_rank = next(r for r in range(ranks) if offsets[r]<=failed_gid<offsets[r+1])
        def rejected(b,error):
            def user(c,payload):
                good=(payload['code']==3)
                good=c.branch(good,lambda e:payload['origin']==2,lambda e:e.int64(0)==1)
                good=c.branch(good,lambda e:error['gid']==failed_gid,lambda e:e.int64(0)==1)
                good=c.branch(good,lambda e:error['rank']==failed_rank,lambda e:e.int64(0)==1)
                return c.branch(good,lambda e:q.n,lambda e:e.value('wrong_error',0.*q.n))
            return b.match(error['reason'],user=user,runtime=lambda c,_:c.value('runtime_error',0.*q.n))
        selected=p.match(outcome,ok=lambda b,_:b.value('unexpected_success',0.*q.n),err=rejected)
    p.commit(q.next,p.value('accepted',selected,at=q.next.point))
    p.step_strategy(FixedDt(.01));case.program(p)
    def check(runtime):
        actual=state_array(runtime,'method',initial.shape)
        if actual is None:return non_owner_metrics()
        error=float(np.max(np.abs(actual-expected)))
        return dict(passed=error<2e-12,metrics_owner=True,max_error=error)
    return CaseSpec(case,layout,_initial(case,block[state],initial),{'method':block[state]},
                    .01,.01,check,'actual distributed Kokkos dispatch '+kind)


@pytest.mark.parametrize('kind',('arithmetic','import','empty','nested','overflow','bounds','nonfinite'))
def test_installed_distributed_map_dispatch(kind,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _,world=native_profile(dimension=2,scope='distributed Kokkos '+kind)
    rank,ranks=rank_and_size(world)
    runtime,artifact,metrics=execute_case(lambda:_build(kind,ranks),world,'distributed Kokkos '+kind)
    with collective_check(world):
        program=artifact.program;assert program is not None
        path=Path(program.so_path).resolve();dso=ctypes.CDLL(str(path))
        count=dso.pops_finite_kokkos_launch_count;count.argtypes=[];count.restype=ctypes.c_uint64
        space=dso.pops_finite_kokkos_execution_space;space.argtypes=[];space.restype=ctypes.c_char_p
        last=dso.pops_finite_kokkos_last_failure_index;last.argtypes=[];last.restype=ctypes.c_int64
        active=kind!='empty' and (ranks==1 or rank>0)
        assert (count()>0) == active
        gid=(5//max(1,ranks-1)-1 if ranks>1 else 4) if kind=='bounds' else 4
        assert last()==(gid if kind in ('overflow','bounds','nonfinite') else -1)
        backend=space().decode()
        if os.environ.get('POPS_REQUIRE_DEVICE')=='1':assert backend.lower() in ('cuda','hip','sycl','openmptarget'),backend
        get_hash=dso.pops_program_hash;get_hash.argtypes=[];get_hash.restype=ctypes.c_char_p
        assert get_hash().decode()==runtime._executor._s.installed_program_hash()
        record_property('program_dso_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
        record_property('artifact_identity',artifact.artifact_identity.hexdigest)
        record_property('kokkos_execution_space',backend)
        record_property('actual_nonempty_map_launches',count())
        record_property('first_failing_map_gid',last())
        record_property('metrics',str(metrics))
