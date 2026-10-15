"""Authored face correction on a real recursively subcycled AMR hierarchy.

The detector is composed with its provisional stencil in Python. Both incident
provisional cell averages are evaluated from the same authenticated old-state
halo; no ghost value of an uncomputed provisional field is inferred.
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
from pops.time import FixedDt, RejectAttempt

from .common import CaseSpec, non_owner_metrics
from .models_M01_M14 import _domain, _initial


def composed_correction_flux(*, selected=True):
    def body(sample, spacing, axis, duration):
        if axis != 0:
            return (0.,), 0.

        def high(face):
            left, right = sample((face-1, 0), 0), sample((face, 0), 0)
            return (left+right)/2-duration*(right-left)/(2*spacing[0])

        def provisional(cell):
            return sample((cell, 0), 0)-duration*(high(cell+1)-high(cell))/spacing[0]

        flux = high(0)
        if selected:
            troubled = (provisional(-1) < 0.) | (provisional(0) < 0.)
            flux = pm.where(troubled, lambda: sample((-1, 0), 0), lambda: flux)
        return (flux,), 1/spacing[0]
    return StencilFlux(dimension=2, components=1, radius=2, body=body,
                       parameters=("duration",))


def build_t12_amr(*, cells=(32, 16), cfl=.2, steps=3, selected=True,
                  reject_parent=False):
    if not 0 < cfl <= .25 or type(steps) is not int or steps < 1:
        raise ValueError("reception requires 0<CFL<=1/4 and positive step count")
    frame, _ = _domain("corrected_face_mesh", cells)
    model = pops.Model("authored_face_balance", frame=frame)
    state = model.state("amount", components=("q",), sampling="cell_average",
                        units=(PhysicalDimension(),))
    flux = model.flux("transport", state=state, frame=frame,
        components={frame.axes[0]: (state[0],), frame.axes[1]: (0.,)},
        waves={frame.axes[0]: (1.,), frame.axes[1]: (0.,)})
    rate = model.rate("balance", equation=pm.ddt(state) == -pm.div(flux))
    case = pops.Case("AMR_composed_face_correction")
    block = case.block("method", model, states=(state,))
    use = Use(rate.occurrences[0], "mesh_face", "accepted_step", "cell_average")
    discretization = DiscretizationPlan()
    discretization.rates.add(rate, ProgramRealization(uses=(use,), ghost_depth=2))
    case.numerics(discretization, block=block)
    program = pops.Program("predict_detect_replace_reflux")
    q = program.state(block[state])
    h = program.requested_dt()
    # This first face basis is deliberately evaluated and left unaccepted.
    provisional = program.stencil_flux_divergence("provisional", q.n,
        flux=composed_correction_flux(selected=False), parameters=(h,))
    program.record_scalar("provisional_rate_min", program.min(provisional))
    selected_flux = composed_correction_flux(selected=selected)
    corrected = Realize("accepted_face", owns=(use,),
        body=lambda builder, old, duration: builder.stencil_flux_divergence("selected", old,
            flux=selected_flux, parameters=(duration,)),
        bindings={"old": q.n, "duration": h}, outputs={use: ()}).evaluate(program)[use]
    candidate = program.value("accepted", q.n+program.dt*corrected, at=q.next.point)
    program.commit(q.next, candidate)
    if reject_parent:
        def after_sync(builder):
            value = builder.value("synchronized", q.n, at=q.next.point)
            checked = builder.guard("reject_after_reflux", value,
                builder.min(value) > 2., action=RejectAttempt())
            builder.commit(q.next, checked)
        program.after_synchronization(after_sync)
    dt = cfl/cells[0]
    program.step_strategy(FixedDt(dt))
    case.program(program)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=.4))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells,
                                   periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    initial = np.full((1, cells[1], cells[0]), .15)
    initial[..., cells[0]//4:cells[0]//4+6] = [.15, .06, .005, 1., .65, .3]

    def assess(runtime):
        from .method_amr_reception import composite_base_projection
        levels = runtime.n_levels()
        mass = runtime.integral("method", component=0)
        arrays = [runtime.block_level_state_global("method", i) for i in range(levels)]
        boxes = [(int(level), tuple(map(int, lo)), tuple(map(int, hi)))
                 for level, lo, hi in runtime.patch_boxes()]
        projection = composite_base_projection(arrays, boxes, cells=cells)
        if projection is None:
            return non_owner_metrics()
        actual = projection["means"]
        defect = abs(float(mass)-float(initial.mean()))
        change = float(np.max(abs(actual-initial[0])))
        return dict(passed=bool(levels == 2 and defect < 2.e-12 and change > 1.e-5
                               and np.isfinite(actual).all()),
            metrics_owner=True, composite_mass_defect=defect, composite_change=change,
            active_counts=projection["active_counts"], levels=levels,
            composite_minimum=float(actual.min()))
    return CaseSpec(case, layout, _initial(case, block[state], initial),
        {"method": block[state]}, steps*dt, dt, assess,
        "Composed two-incident provisional detector, one selected face, real 2:1 AMR "
        "subcycling and accepted native FluxExpression reflux; selected="+str(selected))
