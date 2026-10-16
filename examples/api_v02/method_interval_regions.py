"""One issued interval captured by nested native loops and a typed result region."""
from __future__ import annotations

import numpy as np

from pops import math as pm
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
                      AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import FixedDt

from .common import CaseSpec, non_owner_metrics, state_array
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def build_interval_regions(*, amr=False, reject=False, dt=.03, cells=(8, 4)):
    case, layout, model, block, state, p, q, initial = _storage(
        "issued_interval_regions", ("q",), (1.,), cells=cells)
    initial[..., cells[0]//2:] = 2.
    plan = DiscretizationPlan()
    plan.states.add(state, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    good = p.interval()
    wrong = p.interval(duration=p.requested_dt()*2)
    current = p.value("region_initial", 1*q.n, at=q.next.point)

    def update(b, value, index, site):
        candidate = b.value("growth", (1+p.dt/9)*value, at=value.point)
        # The bad branch is present in both artifacts. Native execution selects
        # it only for the rejection fixture, at the second invocation of a loop.
        return b.branch(index == b.int64(1) if reject else index < b.int64(0),
            lambda z: wrong.bind(z, candidate, name=site+"_wrong_window"),
            lambda z: good.bind(z, candidate, name=site+"_exact_window"))

    current = p.fold(current, 3, lambda b, value, index: update(b, value, index, "fold"))
    current = p.require_success(p.capture(lambda b: b.fold(current, 3,
        lambda z, value, index: update(z, value, index, "capture"))))
    sequence = p.iterate(p.product(value=current, index=p.int64(0)),
        lambda b, carry: b.product(value=update(b, carry["value"], carry["index"], "iterate"),
                                   index=carry["index"]+b.int64(1)),
        lambda b, carry: carry["index"] == b.int64(3), max_steps=3)
    current = p.require_success(sequence)["state"]["value"]
    p.commit(q.next, current)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    if amr:
        transfer = AMRTransfer()
        transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam("interval_refinement_threshold", default=1.5))
        layout = AMR(grid=CartesianGrid(frame=model.frame, cells=cells,
                                       periodic=PeriodicAxes(model.frame.axes)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=0)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid.frozen(), transfer=transfer,
            execution=AMRExecution.synchronous())

    def check(runtime):
        if amr:
            from .method_amr_reception import composite_base_projection
            images = [runtime.block_level_state_global("method", level)
                      for level in range(runtime.n_levels())]
            projection = composite_base_projection(images, runtime.patch_boxes(), cells=cells)
            actual = None if projection is None else projection["means"][None]
        else:
            actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        expected = initial*(1+dt/9)**(9*runtime.macro_step())
        error = float(np.max(abs(actual-expected)))
        return dict(passed=error < 8.e-14 and (not amr or runtime.n_levels() == 2),
                    metrics_owner=True, state_error=error)

    return CaseSpec(case, layout, _initial(case, block[state], initial),
        {"method": block[state]}, dt, dt, check,
        "Exact Interval captures across branch, fold, capture and iterate")
