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


def build_continuation_capture_amr(*, cells=(16, 8), dt=.002, steps=2, recover=False,
                                   field_payload=False, illegal_read=False, independent_field=False):
    if independent_field and not field_payload:
        raise ValueError("independent retained observation requires the typed Field payload")
    if illegal_read and not field_payload:
        raise ValueError("foreign read witness requires the typed Field payload")
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
    seed = q.n
    if independent_field:
        seed_request = field.bind_program_inputs(program=p,
            values={block[state]: q.n}, at=q.n.point, solver=solver)
        seed = field.observe(p.solve(seed_request, solver=solver).consume(action=FailRun()))[field[potential]]
    retained = {"retained": seed} if field_payload else {}
    initial = p.product(state=q.n, time=p.float64(0), step=p.int64(0), counter=p.int64(0),
                        **({"previous": p.float64(0)} if independent_field else {}), **retained)
    schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
                                retained_fields=(("retained",),) if field_payload else (),
                                time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    proposal = p.product(h=p.requested_dt())
    previous_sample = (p.branch(p.level() == 1,
        lambda b: b.sum(current["retained"]), lambda b: b.float64(0)) if independent_field else None)

    def continuation(b, count, sample=None):
        retained = {"retained": current["retained"] if sample is None else sample} if field_payload else {}
        return b.product(state=b.value("same_amount", current["state"], at=q.next.point),
            time=current["time"]+proposal["h"], step=current["step"]+b.int64(1),
            counter=current["counter"]+b.int64(count),
            **({"previous": previous_sample} if independent_field else {}), **retained)

    def trial(b):
        input_state = current["state"]
        if independent_field:
            input_state = b.value("scaled_nonlinear_source",
                                  current["state"]*(1+50*current["time"]), at=q.n.point)
        request = field.bind_program_inputs(program=b,
            values={block[state]: input_state}, at=q.n.point, solver=solver)
        observation = field.observe(b.solve(request, solver=solver).consume(action=FailRun()))
        maximum = b.max(observation[field[potential]])
        approved = _accepted(b, base, proposal, continuation(b, 1), b.product(maximum=maximum))
        if field_payload:
            from pops.time.variant_types import Result, Variant, type_of
            payload = b.product(sample=observation[field[potential]] if independent_field else current["state"],
                                producer=b.level())
            errors = Variant(sample=type_of(payload))
            result = Result(type_of(approved), errors)
            selected = b.branch(b.level() == 1,
                lambda child: child.variant(result, "err", child.variant(errors, "sample", payload)),
                lambda child: child.variant(result, "ok", approved))
            return b.require_success(selected)
        if recover:
            # A recoverable authored failure after assessment must revoke the
            # provisional child instead of publishing its complete image.
            b.require_success(b.failure(b.float64(0), 17))
        return approved

    captured = p.continuation_capture(trial)
    def recover_field(b, payload):
        # Structural transport and checkpointing do not numerically read the
        # fine sample on coarse. Only its producer level may reduce it.
        def read(child):
            child.record_scalar("capsule_sample", child.sum(payload["sample"]))
            return child.int64(1)
        read_here = (read(b) if illegal_read else
            b.branch(b.level() == payload["producer"], read, lambda child: child.int64(0)))
        return _accepted(b, base, proposal, continuation(b, 10, payload["sample"]),
                          b.product(maximum=b.float64(read_here)))

    def recovery(b, error):
        if field_payload:
            def unexpected(child, payload):
                return _accepted(child, base, proposal, continuation(child, 100),
                                 child.product(maximum=child.float64(-99)))
            return b.match(error, sample=recover_field,
                           assessment_rejected=unexpected, runtime=unexpected)
        return _accepted(b, base, proposal, continuation(b, 10), b.product(maximum=b.float64(-1)))

    approved = p.match(captured, ok=lambda b, value: value, err=recovery)

    def synchronize(b, base, proposal):
        entry = b.continuation_read(base)
        old = proposal["candidate"]
        result = b.product(state=proposal["synchronized"]["field_0"],
            time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
            counter=old["counter"], **({"previous": old["previous"]} if independent_field else {}),
            **({"retained": old["retained"]} if field_payload else {}))
        return _accepted(b, base, proposal, result, b.product(counter=old["counter"],
            **({"previous": old["previous"]} if independent_field else {})))

    def initialize(b):
        image = b.product(time=b.float64(0), step=b.int64(0), counter=b.int64(0),
                          **({"previous": b.float64(0)} if independent_field else {}))
        return image, ContinuationSchema(image, time=("time",), step=("step",))

    def parent(b, base, proposal):
        entry = b.continuation_read(base)
        counters = b.reduce(proposal["children"], b.int64(0),
            lambda child, total, row: total+row["counter"], order="ascending")
        previous = (b.reduce(proposal["children"], b.float64(0),
            lambda child, total, row: total+row["previous"], order="ascending")
            if independent_field else None)
        image = b.product(time=proposal["interval"]["end"],
            step=entry["step"]+b.int64(1), counter=counters,
            **({"previous": previous} if independent_field else {}))
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
