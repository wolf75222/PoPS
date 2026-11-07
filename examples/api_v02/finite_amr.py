"""Finite Python algebra inside every invocation of a true subcycled hierarchy.

The same ordered map/scan/reduce equations run on coarse and fine levels. Their
computed scalar enters the state update, so successful compilation alone cannot
make this receiver pass. This profile has no persistent Continuation leaves.
"""
from __future__ import annotations

import numpy as np
from pops import math as pm
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                      AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy,
                      Hysteresis, Tag)
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.params import RuntimeParam
from pops.time import FixedDt, RejectAttempt
from pops.time.expressions import ProgramExpression, ProgramScalar
from pops.time.variant_types import Int64, Result

from .common import CaseSpec, non_owner_metrics
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def authored_array_total(program, *, distributed=False, ranks=1):
    """Squares plus one, inclusive prefix sum, then sum: [1,3,8,18] -> 30."""
    if distributed:
        from pops.time._program.distributed_arrays import Distributed
        algebra = Distributed(program)
        domain = algebra.domain(tuple(4*rank//ranks for rank in range(ranks+1)), capacity=4)
        values = program.require_success(algebra.local_map(domain,
            lambda b, i: b.variant(Result(Int64, Int64), "ok", algebra.gid(i)*algebra.gid(i)+1),
            byte_capacity=65536))
        scanned = algebra.scan(values, program.int64(0),
            lambda b, s, x: b.product(state=s+x, value=s+x),
            order="global_gid_ascending", byte_capacity=65536)
        return algebra.reduce(scanned["values"], program.int64(0), lambda b, s, x: s+x,
                              order="global_gid_ascending", byte_capacity=65536)
    domain = program.range_set(4, capacity=4, ownership="replicated")
    values = program.map(domain, lambda b, i: b.index_value(i)*b.index_value(i)+1)
    # This inactive arm would fail if a region evaluator eagerly executed it.
    values = program.branch(program.boolean(True), lambda b: values,
        lambda b: b.map(domain, lambda c, i: c.int64(1)//c.int64(0)))
    scanned = program.scan(domain, program.int64(0),
        lambda b, s, i: b.product(state=s+values[i], value=s+values[i]), order="ascending")
    indices = program.map(domain, lambda b, i: b.index(domain, 3-b.index_value(i)))
    reversed_values = program.gather(scanned["values"], indices)
    return program.reduce(reversed_values, program.int64(0), lambda b, s, x: s+x,
                          order="ascending")


def build_finite_amr(*, cells=(16, 8), dt=.01, steps=3, ratio=2,
                     distributed=False, ranks=1, reject_parent=False):
    if ratio not in (1, 2) or type(ranks) is not int or ranks < 1:
        raise ValueError("this reception selects temporal ratios 1 or 2 and positive rank count")
    parts = _storage("finite_amr_algebra", ("q",), (1.,), cells=cells)
    case, _, model, block, state, program, q, initial = parts
    initial[..., cells[0]//2:] = 2.
    total = authored_array_total(program, distributed=distributed, ranks=ranks)
    coefficient = program.float64(total)/30.
    program.record_scalar("finite_total", program.float64(total))
    expression = q.n[0]+ProgramScalar(program.requested_dt())*ProgramScalar(coefficient)
    program.commit(q.next, program.value("accepted", ProgramExpression((expression,), q.n),
                                        at=q.next.point))
    if reject_parent:
        def reject(builder):
            value = builder.value("refluxed_candidate", q.n, at=q.next.point)
            builder.commit(q.next, builder.guard("finite_parent_reject", value,
                builder.min(value) > 100., action=RejectAttempt()))
        program.after_synchronization(reject)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("finite_refinement_threshold", default=1.5))
    frame = model.frame
    execution = (AMRExecution.subcycled((AMRClockRelation(0, 1, ratio),))
                 if ratio > 1 else AMRExecution.synchronous())
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells,
                                    periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state]) > case.value(threshold)),
                                 Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)

    def assess(runtime):
        from .method_amr_reception import composite_base_projection
        arrays = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        projection = composite_base_projection(arrays, runtime.patch_boxes(), cells=cells)
        if projection is None:
            return non_owner_metrics()
        error = float(np.max(abs(projection["means"]-initial[0]-runtime.time())))
        return dict(passed=runtime.n_levels() == 2 and error < 3.e-13,
                    metrics_owner=True, reference_error=error, levels=runtime.n_levels())
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    steps*dt, dt, assess,
                    "Finite map/scan/gather/reduce in native AMR level invocations; distributed="
                    +str(distributed)+"; temporal_ratio="+str(ratio))
