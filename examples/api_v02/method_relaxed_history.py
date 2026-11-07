"""Authored interval-history reconstruction after whole-macro AMR relaxation.

The history matrix retains a linearly reconstructed midpoint and the genuine
macro-entry State on every level. Fine provisional rotations cannot erase that
entry authority. This is an explicit dense-history law, not a claim that linear
interpolation is the exact nonlinear trajectory.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                     AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy,
                     Hysteresis, Tag)
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import PhysicalDimension
from pops.numerics import DiscretizationPlan, ProgramRealization
from pops.numerics.stencil_flux import StencilFlux
from pops.params import RuntimeParam
from pops.realization import Realize, Use
from pops.time import ComputedDt, FixedDt, RejectAttempt

from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _domain, _initial


def build_t10_history(*, cells=(32, 16), temporal_ratio=1, relaxed=True,
                         reject_parent=False, invalid_history=False):
    """Whole-macro RRK: no fine history is used as the macro-entry state.

    Requires the generic attempt-entry/composite-pair/finalization contracts.
    The unrelaxed branch is the same native candidate map and face quadrature;
    it is a comparison oracle for the affine finalization, not another runtime.
    """
    if temporal_ratio not in (1, 2):
        raise ValueError("reception requests synchronous or two-substep AMR")
    # Each cell owns two incident face-frequency contributions. At the finest
    # dx=1/64 and h=1 this gives 2*h*speed/dx=.256, below every unchanged
    # predictor/quadrature admission budget. Rotation still takes the full h=1.
    speed = .002
    frame, _ = _domain("whole_macro_relaxation", cells)
    model = pops.Model("rotation_and_passive_transport", frame=frame)
    state = model.state("state", components=("x", "y", "z"), sampling="cell_average",
                        units=(PhysicalDimension(),)*3)
    flux = model.flux("transport", state=state, frame=frame,
        components={frame.axes[0]: (0., 0., speed*state[2]), frame.axes[1]: (0., 0., 0.)},
        waves={frame.axes[0]: (0., 0., speed), frame.axes[1]: (0., 0., 0.)})
    source = model.source("rotation", on=state, value=(-state[1], state[0], 0.))
    equation = model.rate("coupled_balance", equation=pm.ddt(state) == -pm.div(flux)+source)
    case = pops.Case("whole_composite_macro_relaxation")
    block = case.block("method", model)
    face_use = Use(equation.occurrences[0], "mesh_face", "requested_step", "inventory", "transport")
    source_use = Use(equation.occurrences[1], "cell", "requested_step", "inventory", "rotation")
    discretization = DiscretizationPlan()
    discretization.rates.add(equation, ProgramRealization(uses=(face_use, source_use), ghost_depth=1))
    case.numerics(discretization, block=block)
    def face(sample, spacing, axis):
        return ((0., 0., speed*sample((-1, 0), 2)), speed/spacing[0]) if axis == 0 else ((0., 0., 0.), 0.)
    stencil = StencilFlux(dimension=2, components=3, radius=1, body=face)
    program = pops.Program("Heun_then_composite_affine_finalization")
    q = program.state(block[state])
    program.keep_history(q, depth=2)
    predictor_point = program.stage("predictor", c=1)
    endpoint = q.next.point
    f0 = program.stencil_flux_divergence("begin_faces", q.n, flux=stencil)
    program.store_history("retained_transport", f0, depth=2)
    r0 = program.rate("begin_rotation", (-q.n[1], q.n[0], 0.), of=q.n)
    predictor = program.value("predictor", q.n+program.dt*f0+program.dt*r0, at=predictor_point)
    f1 = program.stencil_flux_divergence("end_faces", predictor, flux=stencil)
    r1 = program.rate("end_rotation", (-predictor[1], predictor[0], 0.), of=predictor)
    face_amount = program.value("face_quadrature", .5*program.dt*f0+.5*program.dt*f1, at=endpoint)
    source_amount = program.value("source_quadrature", .5*program.dt*r0+.5*program.dt*r1, at=endpoint)
    # Explicit integrated contributions bind the complete authored stage graph;
    # one physical Use owns the quadrature, rather than being repeated per stage.
    outputs = Realize("Heun_quadrature", owns=(face_use, source_use),
        body=lambda builder, face, source: {"face": face, "source": source},
        bindings={"face": face_amount, "source": source_amount},
        outputs={face_use: ("face",), source_use: ("source",)}).evaluate(program)
    candidate = program.value("unrelaxed", q.n+outputs[face_use]+outputs[source_use], at=endpoint)
    program.commit(q.next, candidate)
    if relaxed:
        def finalize(builder):
            old = builder.attempt_start(q.n)
            metric = (1., 1., 0.)
            old_energy = builder.composite_pair(old, old, weights=metric)
            mixed = builder.composite_pair(old, q.n, weights=metric)
            new_energy = builder.composite_pair(q.n, q.n, weights=metric)
            squared_increment = new_energy-2*mixed+old_energy
            gamma = builder.branch(squared_increment > 0,
                lambda b: -2*(mixed-old_energy)/squared_increment,
                lambda b: old_energy*0+1)
            builder.guard("positive_finite_macro_factor", q.n, gamma > 0, action=RejectAttempt())
            builder.record_scalar("rrk.gamma", gamma)
            builder.record_scalar("rrk.energy_before", old_energy)
            builder.record_scalar("rrk.energy_provisional", new_energy)
            # The generic transaction owns this exact affine combination for all
            # states AND their accepted measures. The engine never computes gamma.
            from pops.time import HistoryTransfer
            # Source order is entry ring[0:3], provisional ring[0:3], then
            # immutable native State points at the macro entry and accepted end.
            # The endpoints have interval0; each target explicitly declares its
            # outgoing interval. Slot0 is recycled storage, lags1/2 are retained.
            def history_rows(samples, accepted_time, ratio, macro_start,
                             source_duration, accepted_duration, scale):
                half = .5*accepted_duration
                entry = (0., 0., 0., 0., 0., 0., 1., 0.)
                middle = (0., 0., 0., 0., 0., 0., .5, .5)
                interval = -half if invalid_history else half
                return (scale > 0, ((macro_start, interval, entry),
                    (macro_start+half, interval, middle), (macro_start, interval, entry)))
            retained = HistoryTransfer(source_slots=8, target_slots=3,
                parameters=("macro_start", "source_duration", "accepted_duration", "scale"),
                body=history_rows)
            # A stored RHS is not State: no endpoint may be invented for it.
            # This explicit constant reconstruction keeps the latest provisional
            # derivative (candidate lag1, source column4) at authored target times.
            # Its exact flux basis must follow the same matrix as its values.
            def rhs_rows(samples, accepted_time, ratio, macro_start,
                         source_duration, accepted_duration, scale):
                half = .5*accepted_duration
                latest = (0., 0., 0., 0., 1., 0.)
                return (scale > 0, ((macro_start, half, latest),
                    (macro_start+half, half, latest), (macro_start, half, latest)))
            retained_rhs = HistoryTransfer(source_slots=6, target_slots=3,
                parameters=("macro_start", "source_duration", "accepted_duration", "scale"),
                body=rhs_rows)
            builder.accept_increment(scale=gamma, duration=gamma*builder.requested_dt(),
                                     histories={"method.state": retained,
                                                "retained_transport": retained_rhs})
            midpoint = builder.value("accepted_history_midpoint_x", (q.prev(1).value[0],)*3)
            builder.record_scalar("rrk.history_midpoint_x", builder.min(midpoint))
            transported = builder.history("retained_transport", 1, space=f0.space,
                                          block=f0.block, state_ref=f0.state_ref)
            rhs_value = builder.value("accepted_rhs_sample", (transported[2],)*3)
            builder.record_scalar("rrk.retained_rhs_minimum", builder.min(rhs_value))
            if reject_parent:
                builder.guard("reject_relaxed_parent", q.n, gamma < 0, action=RejectAttempt())
        program.after_synchronization(finalize)
        program.step_strategy(ComputedDt(1., max_rejections=0, endpoint_ulps=4))
    else:
        program.step_strategy(FixedDt(1.))
    case.program(program)
    transfer = AMRTransfer(); transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=.65))
    execution = (AMRExecution.synchronous() if temporal_ratio == 1 else
                 AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state])["z"] > case.value(threshold)), Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = np.stack((np.ones_like(x), np.zeros_like(x),
                         .3+.5*np.exp(-90*(x-.45)**2)*(1+.2*np.cos(2*np.pi*y))))
    def assess(runtime):
        arrays = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        if arrays[0] is None:
            return non_owner_metrics()
        return dict(passed=runtime.n_levels() == 2 and runtime.macro_step() == 1,
                    metrics_owner=True, frontier_only=True)
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
        1., 1., assess, "T10: global volume metric, macro-entry snapshot, one factor on the complete "
        "refluxed increment and published face measures; requested stage dates retained")
