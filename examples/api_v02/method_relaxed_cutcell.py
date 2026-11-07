"""Variable-frequency rotation and RRK in an actual cut-cell physical measure."""
import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
    AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.boundary import ZeroFlux
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.mesh.geometry import Disc, EmbeddedBoundary
from pops.mesh.masks import CutCell
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import ComputedDt, FixedDt, RejectAttempt
from .common import CaseSpec
from .models_M01_M14 import _domain, _initial


def build_t10_cutcell(*, temporal_ratio=2, relaxed=True, reject_parent=False, cells=(16, 12)):
    if temporal_ratio not in (1, 2):
        raise ValueError("reception requires ratio1 or ratio2")
    frame, _ = _domain("physical_cutcell_rotation", cells)
    model = pops.Model("variable_rotation", frame=frame)
    state = model.state("q", components=("x", "y", "frequency"), sampling="cell_average")
    source = model.source("rotation", on=state, value=(-state[2]*state[1], state[2]*state[0], 0.))
    equation = model.rate("rotation_ode", equation=pm.ddt(state) == source)
    case = pops.Case("relaxed_cutcell_measure")
    block = case.block("method", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(equation, StateStorage())
    case.numerics(numerics, block=block)
    p = pops.Program("Heun_RRK_with_physical_volume")
    q = p.state(block[state])
    first = equation(q.n)
    predictor = p.value("predictor", q.n+p.dt*first, at=q.next.point)
    candidate = p.value("Heun", q.n+.5*p.dt*(first+equation(predictor)), at=q.next.point)
    p.commit(q.next, candidate)
    if relaxed:
        def finalize(b):
            old = b.attempt_start(q.n)
            e0 = b.composite_pair(old, old, weights=(1., 1., 0.))
            cross = b.composite_pair(old, q.n, weights=(1., 1., 0.))
            e1 = b.composite_pair(q.n, q.n, weights=(1., 1., 0.))
            denominator = e1-2*cross+e0
            gamma = -2*(cross-e0)/denominator
            volume = b.composite_integral("physical_volume", 1., over=q.n)
            energy = b.composite_integral("same_quadratic_energy", old[0]**2+old[1]**2, over=q.n)
            b.record_scalar("rrk.gamma", gamma)
            b.record_scalar("metric.volume", volume)
            b.record_scalar("metric.energy", energy)
            b.record_scalar("metric.pair", e0)
            b.accept_increment(scale=gamma, duration=gamma*b.requested_dt())
            if reject_parent:
                b.guard("reject_cutcell_acceptance", q.n, gamma < 0, action=RejectAttempt())
        p.after_synchronization(finalize)
        p.step_strategy(ComputedDt(1., max_rejections=0))
    else:
        p.step_strategy(FixedDt(1.))
    case.program(p)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_frequency", default=1.13))
    execution = (AMRExecution.synchronous() if temporal_ratio == 1 else
                 AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        embedded_boundary=EmbeddedBoundary(Disc(center=(.5, .5), radius=.46), CutCell(), ZeroFlux()),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state])[2] > case.value(threshold)), Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=execution)
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = np.stack((1.+.2*x, .1*y, .55+x+.15*y))
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
        1., 1., lambda runtime: {"passed": runtime.n_levels() == 2},
        "physical DeltaV*kappa, active mask and finest ownership; fixed metric throughout the macro attempt")
