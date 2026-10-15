"""Assessed child scopes suspended by a nonlinear hierarchy field solve."""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.continuations import ContinuationSchema
from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import RuntimeParam
from pops.solvers import Newton
from pops.time import FailRun, FixedDt

from .common import CaseSpec, non_owner_metrics
from .continuation_forest_amr import _accepted
from .models_M01_M14 import _domain, _fv, _initial


def build_continuation_capture_amr(*, cells=(16, 8), dt=.002, steps=2, recover=False):
    frame, _ = _domain("captured_continuation_box", cells)
    model = pops.Model("captured_continuation_equations", frame=frame)
    state = model.state("amount", components=("u",))
    potential = model.field("potential")
    flux = model.flux("stationary", state=state, frame=frame,
        components={axis: (0*state[0],) for axis in frame.axes},
        waves={axis: (0*state[0],) for axis in frame.axes})
    rate = model.rate("unchanged_amount", equation=pm.ddt(state) == -pm.div(flux))
    case = pops.Case("assessed_capture_across_hierarchy_solve")
    block = case.block("method", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, _fv(flux, state))
    case.numerics(numerics, block=block)
    solver = Newton(tolerance=1e-10, max_iterations=20, linear_tolerance=1e-8,
                    linear_max_iterations=240, restart=60,
                    right_preconditioner="SpatialBasisJacobi@1")
    problem = FieldProblem("nonlinear_observation", unknowns=(potential,),
        equations=(Reaction(potential, 1+pm.ValueExpr(potential)**2)
                   - DivCoeffGrad(potential, .04) == state[0],),
        boundaries=(FieldBoundary(potential,
            bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),))
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=1e-6), boundaries=(), solver=solver))
    p = pops.Program("captured_children_with_full_parent")
    q = p.state(block[state])
    initial = p.product(state=q.n, time=p.float64(0), step=p.int64(0), counter=p.int64(0))
    schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
                                time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    proposal = p.product(h=p.requested_dt())

    def continuation(b, count):
        return b.product(state=b.value("same_amount", current["state"], at=q.next.point),
            time=current["time"]+proposal["h"], step=current["step"]+b.int64(1),
            counter=current["counter"]+b.int64(count))

    def trial(b):
        request = field.bind_program_inputs(program=b,
            values={block[state]: current["state"]}, at=q.n.point, solver=solver)
        observation = field.observe(b.solve(request, solver=solver).consume(action=FailRun()))
        maximum = b.max(observation[field[potential]])
        approved = _accepted(b, base, proposal, continuation(b, 1), b.product(maximum=maximum))
        if recover:
            # A recoverable authored failure after assessment must revoke the
            # provisional child instead of publishing its complete image.
            b.require_success(b.failure(b.float64(0), 17))
        return approved

    captured = p.continuation_capture(trial)
    approved = p.match(captured, ok=lambda b, value: value,
        err=lambda b, error: _accepted(b, base, proposal, continuation(b, 10),
                                      b.product(maximum=b.float64(-1))))

    def synchronize(b, base, proposal):
        entry = b.continuation_read(base)
        old = proposal["candidate"]
        result = b.product(state=proposal["synchronized"]["field_0"],
            time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
            counter=old["counter"])
        return _accepted(b, base, proposal, result, b.product(counter=old["counter"]))

    def initialize(b):
        image = b.product(time=b.float64(0), step=b.int64(0), counter=b.int64(0))
        return image, ContinuationSchema(image, time=("time",), step=("step",))

    def parent(b, base, proposal):
        entry = b.continuation_read(base)
        counters = b.reduce(proposal["children"], b.int64(0),
            lambda child, total, row: total+row["counter"], order="ascending")
        image = b.product(time=proposal["interval"]["end"],
            step=entry["step"]+b.int64(1), counter=counters)
        return _accepted(b, base, proposal, image, b.product())

    p.continuation_forest(approved, synchronize=synchronize, initialize=initialize,
                          parent=parent, level_capacity=8)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("capture_refinement_threshold", default=1.5))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    data = np.ones((1, cells[1], cells[0]))
    data[..., cells[0]//2:] = 2.

    def check(runtime):
        from .method_amr_reception import composite_base_projection
        images = [runtime.block_level_state_global("method", level) for level in range(runtime.n_levels())]
        projection = composite_base_projection(images, runtime.patch_boxes(), cells=cells)
        if projection is None:
            return non_owner_metrics()
        error = float(np.max(abs(projection["means"]-data[0])))
        return dict(passed=runtime.n_levels() == 2 and error < 1e-14,
                    metrics_owner=True, unchanged_state_error=error)

    return CaseSpec(case, layout, _initial(case, block[state], data), {"method": block[state]},
        steps*dt, dt, check, "Actual nonlinear hierarchy solve inside an assessed child scope")
