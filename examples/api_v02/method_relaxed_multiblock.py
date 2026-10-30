"""One affine accepted increment across two native coupled AMR block towers.

``numerical_flux`` is a Python-authored TraceFlux component. The actual native
interface scheduler evaluates both traces once; no callback is implemented in
this example. The scalar .8 isolates transaction/measure composition from the
quadratic and quartic root algorithms received in the other T10 examples.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                     AMRTagging, AMRTransfer, Buffer, ConflictPolicy,
                     EqualityPolicy, Hysteresis, Tag)
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.boundaries import BlockInterfaceSide, ConservativeInterface
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import ComputedDt, FixedDt, RejectAttempt
from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _initial


def transport_trace_flux(speed=.002):
    """Both traces and the normal enter the Python equation compiled by PoPS."""
    from pops.numerics import TraceFlux
    return TraceFlux(lambda left, right, normal, time, dt:
        (.5*speed*normal[0]*(left[0]+right[0])
         -.5*abs(speed*normal[0])*(right[0]-left[0]),),
        dimension=2, components=1,
        stability=lambda left, right, normal, time, dt: abs(speed*normal[0]))


def build_t10_multiblock(numerical_flux, *, cells=(32, 16), temporal_ratio=2,
                        relaxed=True, reject_parent=False):
    speed, h = .002, 1.
    frame = Rectangle("relaxed_two_block_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("two_block_transport", frame=frame)
    state = model.state("density", components=("q",), sampling="cell_average")
    flux = model.flux("transport", state=state, frame=frame,
        components={frame.axes[0]: (speed*state[0],), frame.axes[1]: (0.,)},
        waves={frame.axes[0]: (speed,), frame.axes[1]: (0.,)})
    rate = model.rate("balance", equation=pm.ddt(state) == -pm.div(flux))
    case = pops.Case("two_block_affine_publication")
    left, right = case.block("left", model), case.block("right", model)
    subjects = left[state], right[state]
    finite_volume = FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    plans = []
    boundaries = frame.boundaries
    for subject in subjects:
        plan = DiscretizationPlan()
        plan.rates.add(rate, finite_volume)
        plan.boundaries.add(TransportBoundarySet({
            boundaries.x_min: Inflow(state=subject, value=(0.,), trace="prescribed"),
            boundaries.x_max: Outflow(state=subject),
            boundaries.y_min: Outflow(state=subject),
            boundaries.y_max: Outflow(state=subject),
        }))
        plans.append(plan)
    ConservativeInterface("between_blocks",
        left=BlockInterfaceSide(subjects[0], boundaries.x_max),
        right=BlockInterfaceSide(subjects[1], boundaries.x_min),
        numerical_flux=numerical_flux, permutation=(0,), right_normal_translation=1.).attach(*plans)
    for block, plan in zip((left, right), plans, strict=True):
        case.numerics(plan, block=block)
    p = pops.Program("SSPRK2_coupled_affine_publication")
    states = tuple(p.state(subject) for subject in subjects)
    current = tuple(state.n for state in states)
    start, end = p.stage("start", c=0), p.stage("end", c=1)
    k0 = tuple(p.value("begin_%d" % i, rate(value), at=start) for i, value in enumerate(current))
    stage = tuple(p.value("stage_%d" % i, value+p.dt*rhs, at=end)
                  for i, (value, rhs) in enumerate(zip(current, k0, strict=True)))
    k1 = tuple(p.value("end_%d" % i, rate(value), at=end) for i, value in enumerate(stage))
    for i, state in enumerate(states):
        p.commit(state.next, p.value("candidate_%d" % i,
            current[i]+.5*p.dt*k0[i]+.5*p.dt*k1[i], at=state.next.point))
    if relaxed:
        def finalize(builder):
            factor = builder.scalar_expression("accepted_increment_scale", .8)
            builder.accept_increment(scale=factor, duration=factor*builder.requested_dt())
            if reject_parent:
                builder.guard("reject_two_block_parent", current[0], factor < 0,
                              action=RejectAttempt())
        p.after_synchronization(finalize)
        p.step_strategy(ComputedDt(h, max_rejections=0))
    else:
        p.step_strategy(FixedDt(h))
    case.program(p)
    transfer = AMRTransfer()
    for subject in subjects: transfer.state(subject, StateTransfer())
    from pops.params import RuntimeParam
    threshold = case.param(RuntimeParam("refine_threshold", default=.5))
    tags = tuple(Tag(pm.ValueExpr(subject)["q"] > case.value(threshold)) for subject in subjects)
    if temporal_ratio not in (1, 2):
        raise ValueError("receiver requires synchronous or 2:1 temporal composition")
    execution = (AMRExecution.synchronous() if temporal_ratio == 1 else
                 AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(*tags, Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = {}
    for i, subject in enumerate(subjects):
        values = (.1+(.8+.1*i)*np.exp(-70*(x-(.9 if i == 0 else .1))**2)
                  *(1+.1*np.cos(2*np.pi*y)))[None]
        initial.update(_initial(case, subject, values))
    def assess(runtime):
        value = runtime.block_level_state_global("left", 0)
        if value is None: return non_owner_metrics()
        return dict(passed=runtime.n_levels() == 2 and runtime.macro_step() == 1,
                    metrics_owner=True, frontier_only=True)
    return CaseSpec(case, layout, initial, dict(zip(("left", "right"), subjects)),
        h, h, assess, "T10: two-trace Python flux, two AMR block towers, affine accepted "
        "state/time/coarse-fine/interface measures; source evaluations remain authenticated")
