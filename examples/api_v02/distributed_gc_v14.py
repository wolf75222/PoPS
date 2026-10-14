"""Repeated native distributed work with a fixed two-version vault budget.

This is a transport/lifetime reception on Uniform, not a physical model. Each
step creates two immutable distributed values, verifies their sums in compiled
algebra, then publishes a field update. Their Python descriptions do not
carry numerical data between steps; previous native temporaries must be freed
at a collective liveness frontier for the next step to fit the same budget.
"""
import numpy as np
from pops.time import FixedDt
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Int64, Result
from .common import CaseSpec, non_owner_metrics, state_array
from .distributed_program_checks import _all, _require_condition
from .methods_T01_T12 import _initial, _storage


def build_distributed_gc(*, ranks=2, steps=64):
    if ranks not in (2, 3) or type(steps) is not int or not 2 <= steps <= 256:
        raise ValueError("GC reception requires MPI2/MPI3 and 2..256 steps")
    case, layout, model, block, state, p, q, initial = _storage(
        "distributed_gc_v14_"+str(ranks), ("value",), (1.,),
        cells=(8, 4), sampling="cell_average")
    old=q.n
    d=Distributed(p,storage_bytes=65536,storage_versions=2)
    domain=d.domain((0,1,4) if ranks==2 else (0,1,1,4),capacity=4)
    source=p.require_success(d.local_map(domain,
        lambda b,i:b.variant(Result(Int64,Int64),"ok",d.gid(i)+1),byte_capacity=4096))
    doubled=p.require_success(d.local_map(domain,
        lambda b,i:b.variant(Result(Int64,Int64),"ok",2*d.at_local(source,i)),byte_capacity=4096))
    total=lambda value:d.reduce(value,p.int64(0),lambda b,s,x:s+x,
        order="global_gid_ascending",byte_capacity=4096)
    verified=_require_condition(p,_all(p,total(source)==10,total(doubled)==20),2.,code=14021)
    p.commit(q.next,p.value("verified_step_multiplier",verified*old,at=q.next.point))
    p.step_strategy(FixedDt(.01));case.program(p);subject=block[state]
    def invariants(runtime):
        actual=state_array(runtime,"method",initial.shape)
        if actual is None:return non_owner_metrics()
        error=float(np.max(np.abs(actual-(initial*2.**steps))))
        return {"passed":bool(np.isfinite(actual).all() and error==0.),
                "max_error":error,"verified_steps":steps,"storage_versions":2,
                "storage_bytes":65536,"metrics_owner":True}
    return CaseSpec(case,layout,_initial(case,subject,initial),{"method":subject},
        .01*steps,.01,invariants,
        str(steps)+" repeated compiled distributed transactions within two retained Array versions; "
        "collective GC, MPI empty shard; no PDE or AMR qualification")
