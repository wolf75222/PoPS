"""One authored relaxation of the entire refluxed AMR macro increment.

The first two components rotate. The third transports a nonconstant periodic
profile, so the common scalar also scales genuine face fluxes and reflux. The
chosen quadratic observable ignores the third component; this is a method
reception, not an assertion that arbitrary transport preserves quadratic energy.
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


def build_t10_composite(*, cells=(32, 16), temporal_ratio=1, relaxed=True,
                         reject_parent=False, requested_dt=1., t_end=1.,
                         clip_to_frontier=False):
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
    predictor_point = program.stage("predictor", c=1)
    endpoint = q.next.point
    f0 = program.stencil_flux_divergence("begin_faces", q.n, flux=stencil)
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
            new_energy = builder.composite_pair(q.n, q.n, weights=metric)
            # This witness declares the rotation polynomial of its own Heun map.
            # Evaluate R(h)-1 before addition to the state: subtracting two
            # rounded endpoint fields loses the O(h**2) radial increment near
            # the run frontier. The core only integrates these Python equations.
            h = builder.requested_dt()
            norm = old[0]*old[0]+old[1]*old[1]
            radial, squared = [], []
            for level, count in ((0, 1), (1, temporal_ratio)):
                real = -.5*h*h if count == 1 else -.5*h*h+h*h*h*h/64.
                imag = h if count == 1 else h-h*h*h/8.
                selected = builder.int64(level)
                radial.append(builder.composite_integral("radial_increment_%d" % level,
                    norm*real, over=q.n, level=selected))
                squared.append(builder.composite_integral("squared_increment_%d" % level,
                    norm*(real*real+imag*imag), over=q.n, level=selected))
            mixed_increment = radial[0]+radial[1]
            squared_increment = squared[0]+squared[1]
            gamma = builder.branch(squared_increment > 0,
                lambda b: -2*mixed_increment/squared_increment,
                lambda b: squared_increment*0+1)
            builder.guard("positive_finite_macro_factor", q.n, gamma > 0, action=RejectAttempt())
            builder.record_scalar("rrk.gamma", gamma)
            builder.record_scalar("rrk.energy_before", old_energy)
            builder.record_scalar("rrk.energy_provisional", new_energy)
            # The generic transaction owns this exact affine combination for all
            # states AND their accepted measures. The engine never computes gamma.
            builder.accept_increment(scale=gamma, duration=gamma*builder.requested_dt())
            if reject_parent:
                builder.guard("reject_relaxed_parent", q.n, gamma < 0, action=RejectAttempt())
        program.after_synchronization(finalize)
        program.step_strategy(ComputedDt(requested_dt, max_rejections=0, endpoint_ulps=4,
                                         clip_to_frontier=clip_to_frontier))
    else:
        program.step_strategy(FixedDt(requested_dt))
    case.program(program)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=.65))
    execution = (AMRExecution.synchronous() if temporal_ratio == 1 else
                 AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state])["z"] > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = np.stack((np.ones_like(x), np.zeros_like(x),
                         .3+.5*np.exp(-90*(x-.45)**2)*(1+.2*np.cos(2*np.pi*y))))
    def assess(runtime):
        arrays = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        if arrays[0] is None:
            return non_owner_metrics()
        if clip_to_frontier:
            from .relaxed_composite_reference import trajectory_metrics
            metrics = trajectory_metrics(arrays, runtime.patch_boxes(), cells=cells,
                ratio=temporal_ratio, requested=requested_dt, end=t_end,
                initial_passive=float(initial[2].mean()))
            metrics["time_error"] = abs(runtime.time()-metrics["reference_time"])
            metrics["accepted_steps"] = runtime.macro_step()
            metrics["passed"] = bool(metrics["passed"] and metrics["time_error"] < 2.e-13
                and runtime.macro_step() == len(metrics["reference_requests"]))
            return metrics
        return dict(passed=runtime.n_levels() == 2 and runtime.macro_step() == 1,
                    metrics_owner=True, frontier_only=True)
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
        t_end, requested_dt, assess, "T10: global volume metric, macro-entry snapshot, one factor on the complete "
        "refluxed increment and published face measures; requested stage dates retained")
