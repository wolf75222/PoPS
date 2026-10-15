"""A local AMR clock driving an explicitly reconstructed composite elliptic load."""

import numpy as np
import pops
from pops import math as pm
from pops._ir.elliptic import Reaction
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import FieldSpace, PhysicalDimension, PhysicalSupport
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.solvers import CompositeFieldGMRES
from pops.time import TemporalEvaluation, FixedDt, FailRun, RejectAttempt, HistoryTransfer, every
from pops.time.expressions import ProgramScalar
from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _domain, _initial


def quadratic_evaluation(history, *, constant_control=False):
    def equations(samples, target):
        # Source0 is the macro-entry point; slot0 is recycled, lags1/2 are 2/3.
        selected = (0, 2, 3)
        times = tuple(samples[i].time for i in selected)
        mature = (
            (samples[2].available > 0)
            & (samples[3].available > 0)
            & (times[0] > times[1])
            & (times[1] > times[2])
        )
        weights = []
        for i in range(len(samples)):
            coefficient = 0.0
            if i in selected:
                coefficient = 1.0
                for j in selected:
                    if j != i:
                        coefficient = (
                            coefficient
                            * (target - samples[j].time)
                            / (samples[i].time - samples[j].time)
                        )
            weights.append(
                (1.0 if i == 0 else 0.0)
                if constant_control
                else pm.where(
                    mature,
                    lambda coefficient=coefficient: coefficient,
                    lambda i=i: 1.0 if i == 0 else 0.0,
                )
            )
        return target >= samples[0].time, tuple(weights)

    return TemporalEvaluation(history=history, body=equations)


def endpoint_output_transfer():
    """History intervals label invocation starts; this observation is at the end.

    The authored law reconstructs endpoint values at child endpoint times, while
    retaining the outgoing start/interval authority of every target slot.
    """

    def equations(samples, accepted_time, ratio):
        points = tuple(sample.time + sample.interval for sample in samples)
        h = samples[1].interval / ratio
        starts = (accepted_time - 3 * h, accepted_time - h, accepted_time - 2 * h)
        valid = (ratio > 0) & (points[1] > points[2]) & (points[2] > points[0])
        rows = []
        for start in starts:
            weights = []
            for i in range(3):
                weight = 1
                for j in range(3):
                    if i != j:
                        weight = weight * (start + h - points[j]) / (points[i] - points[j])
                weights.append(weight)
            rows.append((start, h, tuple(weights)))
        return valid, tuple(rows)

    return HistoryTransfer(slots=3, body=equations)


def build_temporal_field(
    cells=(16, 16),
    *,
    constant_control=False,
    reject_parent=False,
    regrid_observation=False,
    observation_transfer=True,
    hierarchy_companion=False,
    observation_byte_capacity=64 * 1024 * 1024,
):
    frame, _ = _domain("temporal_field_domain", cells)
    model = pops.Model("polynomial_field_load", frame=frame)
    unit = PhysicalDimension()
    U = model.state(
        "u",
        components=("quadratic", "clock", "observed"),
        units=(unit, unit, unit),
        support=PhysicalSupport((("x", "temporal_field_domain"), ("y", "temporal_field_domain"))),
    )
    phi = model.field("potential")
    case = pops.Case("authored_composite_temporal_evaluation")
    block = case.block("method", model, states=(U,))
    plan = DiscretizationPlan()
    plan.states.add(U, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    state_space = model.module.state_spaces()["u"]
    observed_space = FieldSpace(
        "potential",
        components=("phi",),
        representation="field",
        layout="cell",
        centering="cell",
        sampling="cell",
        frame=state_space.frame,
        clock=state_space.clock,
        support=state_space.support,
        units=(unit,),
    )
    problem = FieldProblem(
        "screened_source",
        unknowns=(phi,),
        equations=(-pm.laplacian(phi) + Reaction(phi, 1.0) == U[0],),
        boundaries=(
            FieldBoundary(phi, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),
        ),
        unknown_spaces={phi: observed_space},
        coordinate_units=(unit, unit),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            observation_axes=(0, 1),
            solver=CompositeFieldGMRES(max_iter=3000, restart=60, rel_tol=1.0e-12, abs_tol=1.0e-13),
        ),
    )
    p = pops.Program("local_clock_composite_field")
    q = p.state(block[U])
    if regrid_observation:
        from .method_history_transfer import regular_quadratic_flux_transfer

        p.keep_history(q, depth=2, interpolation=regular_quadratic_flux_transfer())
    else:
        p.keep_history(q, depth=2)
    policy = quadratic_evaluation(q.prev, constant_control=constant_control)
    target = q.n[1] + p.dt
    candidate = p.value("polynomial_endpoint", (1 + target**2, target, q.n[2]), at=q.next.point)
    observed = field.observe(
        p.solve(
            field, values={block[U]: candidate}, at=q.next.point, temporal={block[U]: policy}
        ).consume(action=FailRun()),
        byte_capacity=observation_byte_capacity,
    )[field[phi]]
    p.store_history(
        "observed",
        observed,
        depth=2 if regrid_observation else 1,
        owner_block=block,
        interpolation=endpoint_output_transfer()
        if regrid_observation and observation_transfer
        else None,
    )
    if hierarchy_companion:
        observed_sum_before = p.sum(observed)
        p.store_history("observed_before_barrier", observed, depth=1, owner_block=block)
        companion_input = p.value(
            "companion_endpoint", (3 + target**2, target, q.n[2]), at=q.next.point
        )
        synchronized = field.observe(
            p.solve(field, values={block[U]: companion_input}, at=q.next.point).consume(
                action=FailRun()
            ),
            byte_capacity=observation_byte_capacity,
        )[field[phi]]
        p.store_history("synchronized", synchronized, depth=1, owner_block=block)
        p.store_history("observed_after_barrier", observed, depth=1, owner_block=block)
        observed_sum_after = p.sum(observed)
        candidate = p.value(
            "retained_observation_reduction_difference",
            (candidate[0], candidate[1], ProgramScalar(observed_sum_after - observed_sum_before)),
            at=q.next.point,
        )
    p.commit(q.next, candidate)
    marker_model = pops.Model("temporal_mesh_marker", frame=frame)
    marker = marker_model.state("marker", components=("tag",))
    marker_block = case.block("marker", marker_model, states=(marker,))
    marker_plan = DiscretizationPlan()
    marker_plan.states.add(marker, StateStorage(ghost_depth=1))
    case.numerics(marker_plan, block=marker_block)
    marked = p.state(marker_block[marker])
    p.commit(
        marked.next,
        p.value(
            "static_marker",
            (marked.n[0] + p.dt,) if regrid_observation else 1 * marked.n,
            at=marked.next.point,
        ),
    )
    if reject_parent:

        def guard(builder):
            candidate = builder.value("post_sync_state", q.n, at=q.next.point)
            clock = builder.value("post_sync_clock", (candidate[1],) * 3)
            candidate = builder.guard(
                "reject_after_composite_solve",
                candidate,
                builder.max(clock) < 0.181,
                action=RejectAttempt(),
            )
            builder.commit(q.next, candidate)

        p.after_synchronization(guard)
    p.step_strategy(FixedDt(0.04))
    case.program(p)
    subjects = {"method": block[U], "marker": marker_block[marker]}
    transfer = AMRTransfer()
    for subject in subjects.values():
        transfer.state(subject, StateTransfer())
    threshold = case.param(RuntimeParam("static_refinement", default=0.5))
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(pm.ValueExpr(marker_block[marker]) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(4, clock=p.clock))
        if regrid_observation
        else AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.synchronous()
        if hierarchy_companion
        else AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    x, y = np.meshgrid(
        (np.arange(cells[0]) + 0.5) / cells[0], (np.arange(cells[1]) + 0.5) / cells[1]
    )
    values = _initial(case, block[U], np.stack((1 + 0 * x, 0 * x, 1 + 0 * x)))
    central = ((x > 0.21) & (x < 0.53) & (y > 0.17) & (y < 0.69)).astype(float)
    marker_values = (
        np.where(
            central > 0, 0.7, np.where((x > 0.6) & (x < 0.8) & (y > 0.3) & (y < 0.6), 0.4, 0.0)
        )
        if regrid_observation
        else central
    )
    values.update(_initial(case, marker_block[marker], marker_values[None]))

    def check(runtime):
        states = [
            runtime.block_level_state_global("method", level) for level in range(runtime.n_levels())
        ]
        if states[0] is None:
            return non_owner_metrics()
        fields = [
            runtime.history_global("observed", level, 1) for level in range(runtime.n_levels())
        ]
        errors = []
        counts = []
        for level, (state, field) in enumerate(zip(states, fields, strict=True)):
            a = np.asarray(state).reshape(3, cells[1] * 2**level, cells[0] * 2**level)
            live = a[1] > 0
            counts.append(int(live.sum()))
            phi = np.asarray(field).reshape(a.shape[1:])
            errors.append(float(np.max(abs(phi[live] - (1 + a[1][live] ** 2)))))
        error = max(errors)
        return dict(
            passed=error < 3.0e-9, metrics_owner=True, field_error=error, level_counts=counts
        )

    return CaseSpec(
        case,
        layout,
        values,
        subjects,
        0.18,
        0.04,
        check,
        "2:1 partial AMR; source polynomial exact at local endpoints; Python quadratic temporal "
        "reconstruction feeds the real composite screened operator; shortened final interval",
    )
