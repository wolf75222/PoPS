"""Quartic relaxation authored as arbitrary composite integrands and Newton algebra.

The invariant of x'=-y**3, y'=x**3 is x**4+y**4. A separate transported
component supplies real AMR face measures. No invariant or root equation is
known by the runtime. The complete provisional AMR map is then relaxed once.
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


def build_t10_quartic(*, cells=(32, 16), temporal_ratio=1, relaxed=True,
                       invalid_integrand=False):
    """Whole-macro RRK: no fine history is used as the macro-entry state.

    Requires the generic attempt-entry/composite-pair/finalization contracts.
    The unrelaxed branch is the same native candidate map and face quadrature;
    it is a comparison oracle for the affine finalization, not another runtime.
    """
    if temporal_ratio not in (1, 2):
        raise ValueError("reception requests synchronous or two-substep AMR")
    speed = .002  # Two incident faces retain the unchanged native CFL guards.
    frame, _ = _domain("quartic_macro_relaxation", cells)
    model = pops.Model("quartic_flow_and_transport", frame=frame)
    state = model.state("state", components=("x", "y", "z"), sampling="cell_average",
                        units=(PhysicalDimension(),)*3)
    flux = model.flux("transport", state=state, frame=frame,
        components={frame.axes[0]: (0., 0., speed*state[2]), frame.axes[1]: (0., 0., 0.)},
        waves={frame.axes[0]: (0., 0., speed), frame.axes[1]: (0., 0., 0.)})
    source = model.source("rotation", on=state, value=(-state[1]**3, state[0]**3, 0.))
    equation = model.rate("coupled_balance", equation=pm.ddt(state) == -pm.div(flux)+source)
    case = pops.Case("quartic_composite_macro_relaxation")
    block = case.block("method", model)
    face_use = Use(equation.occurrences[0], "mesh_face", "requested_step", "inventory", "transport")
    source_use = Use(equation.occurrences[1], "cell", "requested_step", "inventory", "rotation")
    discretization = DiscretizationPlan()
    discretization.rates.add(equation, ProgramRealization(uses=(face_use, source_use), ghost_depth=1))
    case.numerics(discretization, block=block)
    def face(sample, spacing, axis):
        return ((0., 0., speed*sample((-1, 0), 2)), speed/spacing[0]) if axis == 0 else ((0., 0., 0.), 0.)
    stencil = StencilFlux(dimension=2, components=3, radius=1, body=face)
    program = pops.Program("Heun_then_authored_quartic_root")
    q = program.state(block[state])
    predictor_point = program.stage("predictor", c=1)
    endpoint = q.next.point
    f0 = program.stencil_flux_divergence("begin_faces", q.n, flux=stencil)
    r0 = program.rate("begin_rotation", (-q.n[1]**3, q.n[0]**3, 0.), of=q.n)
    predictor = program.value("predictor", q.n+program.dt*f0+program.dt*r0, at=predictor_point)
    f1 = program.stencil_flux_divergence("end_faces", predictor, flux=stencil)
    r1 = program.rate("end_rotation", (-predictor[1]**3, predictor[0]**3, 0.), of=predictor)
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
            dx, dy = q.n[0]-old[0], q.n[1]-old[1]
            initial = builder.composite_integral("quartic_initial", old[0]**4+old[1]**4, over=q.n)
            gamma = builder.scalar_expression("initial_root", 1.)
            # Newton solves the nontrivial root of (H(U0+g*d)-H(U0))/g.
            # Every runtime scalar g is captured by the generated cell integrand.
            # This bounded receiver requires a positive root near one and checks
            # the original invariant; it makes no global convergence assertion.
            for iteration in range(8):
                x, y = old[0]+gamma*dx, old[1]+gamma*dy
                energy = builder.composite_integral("quartic_trial_%d" % iteration,
                                                    x**4+y**4, over=q.n)
                slope = builder.composite_integral("quartic_slope_%d" % iteration,
                                                    4*x**3*dx+4*y**3*dy, over=q.n)
                difference = energy-initial
                quotient = difference/gamma
                derivative = (gamma*slope-difference)/(gamma*gamma)
                builder.guard("quartic_root_simple_%d" % iteration, q.n,
                              abs(derivative) > 1.e-12, action=RejectAttempt())
                gamma = gamma-quotient/derivative
                builder.guard("quartic_root_positive_%d" % iteration, q.n,
                              gamma > .25, action=RejectAttempt())
            x, y = old[0]+gamma*dx, old[1]+gamma*dy
            accepted = builder.composite_integral("quartic_accepted", x**4+y**4, over=q.n)
            builder.guard("quartic_original_residual", q.n,
                          abs(accepted-initial) < 2.e-12, action=RejectAttempt())
            if invalid_integrand:
                builder.composite_integral("invalid_active_expression", pm.sqrt(-old[0]), over=q.n)
            builder.record_scalar("rrk.gamma", gamma)
            builder.record_scalar("rrk.quartic_before", initial)
            builder.record_scalar("rrk.quartic_after", accepted)
            builder.accept_increment(scale=gamma, duration=gamma*builder.requested_dt())
        program.after_synchronization(finalize)
        program.step_strategy(ComputedDt(.5, max_rejections=0, endpoint_ulps=4))
    else:
        program.step_strategy(FixedDt(.5))
    case.program(program)
    transfer = AMRTransfer(); transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=.65))
    execution = (AMRExecution.synchronous() if temporal_ratio == 1 else
                 AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state])["z"] > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = np.stack((np.ones_like(x), .2*np.ones_like(x),
                         .3+.5*np.exp(-90*(x-.45)**2)*(1+.2*np.cos(2*np.pi*y))))
    def assess(runtime):
        arrays = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        if arrays[0] is None:
            return non_owner_metrics()
        return dict(passed=runtime.n_levels() == 2 and runtime.macro_step() == 1,
                    metrics_owner=True, frontier_only=True)
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
         1., .5, assess, "T10: arbitrary Python composite integrands, nonquadratic invariant root, "
        "one affine publication of the complete refluxed increment and its face measures")
