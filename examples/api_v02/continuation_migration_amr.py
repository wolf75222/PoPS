"""Transport and a complete assessed AMR forest, entirely authored in Python."""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                      AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy,
                      Hysteresis, Tag)
from pops.continuations import ContinuationSchema
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import RuntimeParam
from pops.time import FixedDt

from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _domain, _initial, _fv


from .continuation_forest_amr import _accepted


def build_continuation_migration_amr(*, cells=(16, 8), dt=.002, steps=3, ratio=2,
                                reject_parent=False, reject_last_child=False):
    if ratio not in (1, 2):
        raise ValueError("the reception selects temporal ratios 1 or 2")
    frame, _ = _domain("continuation_forest_box", cells)
    model = pops.Model("authored_forest_transport", frame=frame)
    state = model.state("amount", components=("u",))
    flux = model.flux("transport", state=state, frame=frame,
        components={axis: (speed*state[0],) for axis, speed in zip(frame.axes, (.7, .2), strict=True)},
        waves={axis: (speed,) for axis, speed in zip(frame.axes, (.7, .2), strict=True)})
    rate = model.rate("conservation", equation=pm.ddt(state) == -pm.div(flux))
    case = pops.Case("complete_continuation_forest")
    block = case.block("method", model)
    discretization = DiscretizationPlan()
    discretization.rates.add(rate, _fv(flux, state))
    case.numerics(discretization, block=block)
    p = pops.Program("authored_forest_steps")
    q = p.state(block[state])
    from pops.algebra import Distributed
    from pops.time.variant_types import Int64, Result
    algebra = Distributed(p)
    domain = algebra.balanced_domain(p.int64(7), capacity=7)
    history = p.require_success(algebra.local_map(domain,
        lambda b, i: b.variant(Result(Int64, Int64), "ok", algebra.gid(i)+b.int64((1 << 60)+17)),
        byte_capacity=1 << 20))
    initial = p.product(state=q.n, time=p.float64(0), step=p.int64(0),
                        counter=p.int64(0), history=history, history_alias=history)
    schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
                                time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    candidate = p.value("finite_volume_trial", current["state"]+p.dt*rate(current["state"]),
                        at=q.next.point)
    proposal = p.product(h=p.requested_dt())
    trial = p.product(state=candidate, time=current["time"]+proposal["h"],
                      step=current["step"]+p.int64(1), counter=current["counter"]+p.int64(1),
                      history=current["history"], history_alias=current["history_alias"])
    approved = _accepted(p, base, proposal, trial, p.product())

    def synchronize(b, base, proposal):
        entry = b.continuation_read(base)
        old = proposal["candidate"]
        distributed = Distributed(b)
        domain = distributed.domain_of(old["history"])
        retained = b.require_success(distributed.local_map(domain,
            lambda z, i: z.variant(Result(Int64, Int64), "ok",
                distributed.at_local(old["history"], i)+z.int64(1)), byte_capacity=1 << 20))
        result = b.product(state=proposal["synchronized"]["field_0"],
                           time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
                           counter=old["counter"], history=retained, history_alias=retained)
        if reject_last_child:
            # A genuine typed assessment failure at the final child must restore
            # earlier synchronized children and the global parent atomically.
            return b.branch(proposal["level"] == b.int64(1),
                lambda z: _accepted(z, base, proposal, result,
                    z.product(counter=old["counter"], history=retained), reject=True),
                lambda z: _accepted(z, base, proposal, result,
                    z.product(counter=old["counter"], history=retained)))
        return _accepted(b, base, proposal, result, b.product(counter=old["counter"], history=retained))

    def initialize(b):
        entry = b.attempt_start(q.n)
        mass = b.composite_integral("forest_initial_mass", entry[0], over=q.n)
        initial = b.product(time=b.float64(0), step=b.int64(0), counter=b.int64(0),
                            mass=mass)
        return initial, ContinuationSchema(initial, time=("time",), step=("step",))

    def observe(b):
        return b.product(mass=b.composite_integral("complete_forest_mass", q.n[0], over=q.n))

    def parent(b, base, proposal):
        entry = b.continuation_read(base)
        counters = b.reduce(proposal["children"], b.int64(0),
                            lambda z, total, child: total+child["counter"], order="ascending")
        result = b.product(time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
                           counter=entry["counter"]+counters, mass=proposal["observations"]["mass"])
        return _accepted(b, base, proposal, result, b.product(children=counters),
                         reject=reject_parent, conserve_mass=True)

    p.continuation_forest(approved, synchronize=synchronize, initialize=initialize,
                          parent=parent, observe=observe, level_capacity=8)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("forest_refinement_threshold", default=1.5))
    execution = (AMRExecution.subcycled((AMRClockRelation(0, 1, ratio),))
                 if ratio > 1 else AMRExecution.synchronous())
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)
    data = np.ones((1, cells[1], cells[0]))
    data[..., cells[0]//2:] = 2.

    def check(runtime):
        from .method_amr_reception import composite_base_projection
        images = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        projection = composite_base_projection(images, runtime.patch_boxes(), cells=cells)
        if projection is None:
            return non_owner_metrics()
        mass_error = float(abs(np.mean(projection["means"])-1.5))
        change = float(np.max(abs(projection["means"]-data[0])))
        return dict(passed=runtime.n_levels() == 2 and mass_error < 3.e-13 and change > 1.e-8,
                    metrics_owner=True, mass_error=mass_error, change=change)
    return CaseSpec(case, layout, _initial(case, block[state], data), {"method": block[state]},
                    steps*dt, dt, check,
                    "Python transport and aliased distributed histories across native MPI rematerialization")
