"""Native stencil consumers of source-owned state and physical-field extensions.

These are reception compositions: the same face equation reads the native state,
its explicit midpoint sampling, or a consumed screened-Poisson observation. A
separate static marker makes refinement independent of the physical data.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops._ir.elliptic import Reaction
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                     AMRTagging, AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy,
                     Hysteresis, Tag)
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.fields import (CellMidpoint, FieldBoundary, FieldDiscretization,
                         FieldProblem, bcs)
from pops.fields.methods import CellCenteredSecondOrder
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import FieldSpace, PhysicalDimension, PhysicalSupport
from pops.numerics import DiscretizationPlan, ProgramRealization, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.params import RuntimeParam
from pops.realization import Realize, Use
from pops.solvers import CG, CompositeFieldGMRES
from pops.time import FailRun, FixedDt

from .common import CaseSpec, non_owner_metrics, state_array
from .models_M01_M14 import _domain, _initial


def upwind_field_flux(*, sampled):
    def body(sample, spacing, axis):
        if axis == 1:
            return (0.,), 0.
        value = sample((-1, 0), 0, field=0) if sampled else sample((-1, 0), 0)
        return (value,), 1/spacing[0]
    return StencilFlux(dimension=2, components=1, radius=1, body=body,
                       field_components=(1,) if sampled else ())


def outflow_ssprk2(initial, *, dt, steps):
    """Independent array oracle, with the left ghost equal to the first cell."""
    def rhs(value):
        left = np.concatenate((value[..., :1], value[..., :-1]), axis=-1)
        return -(value-left)*value.shape[-1]
    result = np.array(initial, dtype=float, copy=True)
    for _ in range(steps):
        first = result+dt*rhs(result)
        result = .5*result+.5*(first+dt*rhs(first))
    return result


def build_stencil_extensions(*, cells=(24, 16), sample="cell", amr=True,
                             periodic=True, offset=1., amplitude=.07, steps=2):
    if sample not in ("state", "cell", "field"):
        raise ValueError("sample must identify state, midpoint cells or a consumed field")
    if sample == "field" and not periodic:
        raise ValueError("this field receipt uses the periodic screened-Poisson problem")
    if type(steps) is not int or steps < 1:
        raise ValueError("positive reception step count required")
    frame, layout = _domain("stencil_extension_domain", cells,
                            periodic=(0, 1) if periodic else (1,))
    unit = PhysicalDimension()
    model = pops.Model("stencil_extension_balance", frame=frame)
    state = model.state("quantity", components=("q",), sampling="cell_average", units=(unit,),
        support=PhysicalSupport((("x", "stencil_extension_domain"), ("y", "stencil_extension_domain"))))
    physical = model.flux("transport", state=state, frame=frame,
        components={frame.axes[0]:(state[0],), frame.axes[1]:(0.,)},
        waves={frame.axes[0]:(1.,), frame.axes[1]:(0.,)})
    rate = model.rate("balance", equation=pm.ddt(state) == -pm.div(physical))
    uses = {label:Use(rate.occurrences[0], "mesh_face", label, "cell_average")
            for label in ("start", "end")}
    case = pops.Case("authored_stencil_extensions")
    block = case.block("method", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, ProgramRealization(uses=tuple(uses.values()), ghost_depth=1))
    if not periodic:
        plan.boundaries.add(TransportBoundarySet({
            frame.boundaries.x_min: Outflow(state=block[state]),
            frame.boundaries.x_max: Outflow(state=block[state]),
        }, periodic=PeriodicAxes((frame.axes[1],))))
    case.numerics(plan, block=block)
    field = potential = None
    if sample == "field":
        potential = model.field("potential")
        state_space = model.module.state_spaces()["quantity"]
        observed_space = FieldSpace("potential", components=("phi",), representation="field",
            layout="cell", centering="cell", sampling="cell", frame=state_space.frame,
            clock=state_space.clock, support=state_space.support, units=(unit,))
        problem = FieldProblem("screened_Poisson", unknowns=(potential,),
            equations=(-pm.laplacian(potential)+Reaction(potential, 1.) == state[0],),
            boundaries=(FieldBoundary(potential, bcs.BoundaryCondition(
                bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
            unknown_spaces={potential: observed_space}, coordinate_units=(unit, unit))
        solver = (CompositeFieldGMRES(max_iter=2000, restart=60, rel_tol=1.e-12, abs_tol=1.e-13)
                  if amr else CG(max_iter=2000, rel_tol=1.e-12, abs_tol=1.e-13))
        field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(),
            boundaries=(), observation_axes=(0, 1), solver=solver))
    program = pops.Program("sampled_stencil_ssprk2",
        history_field_metadata_budget=(8*1024*1024, 16*1024*1024))
    current = program.state(block[state])
    flux = upwind_field_flux(sampled=sample != "state")

    def evaluate(builder, value, point, label):
        use = uses[label]
        if sample == "cell":
            sampled = builder.cell_field(value, quadrature=CellMidpoint(), name=label+"_sample")
        elif sample == "field":
            observed = field.observe(builder.solve(field, values={block[state]: value},
                at=point).consume(action=FailRun()))
            sampled = observed[field[potential]]
            builder.store_history(label+"_potential", sampled, depth=1, owner_block=block)
        else:
            sampled = None
        builder.record_scalar(label+"_source_min", builder.min(value))
        if sampled is None:
            body = lambda b, old: b.stencil_flux_divergence(label+"_flux", old, flux=flux)
            bindings = {"old":value}
        else:
            body = lambda b, old, field: b.stencil_flux_divergence(label+"_flux", old,
                flux=flux, fields=(field,))
            bindings = {"old":value, "field":sampled}
        return Realize(label+"_face", owns=(use,), body=body,
            bindings=bindings, outputs={use:()}).evaluate(builder)[use]

    first = evaluate(program, current.n, current.n.point, "start")
    end_point = program.stage("end_evaluation", c=1)
    provisional = program.value("stage", current.n+program.dt*first, at=end_point)
    second = evaluate(program, provisional, end_point, "end")
    accepted = program.value("accepted", .5*current.n+.5*(provisional+program.dt*second),
                             at=current.next.point)
    program.commit(current.next, accepted)
    dt = .15/cells[0]
    program.step_strategy(FixedDt(dt))
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = (offset+amplitude*np.sin(2*np.pi*x)*np.cos(2*np.pi*y))[None]
    values = _initial(case, block[state], initial)
    subjects = {"method":block[state]}
    if amr:
        marker_model = pops.Model("static_mesh_marker", frame=frame)
        marker = marker_model.state("marker", components=("tag",), sampling="cell_average", units=(unit,))
        marker_block = case.block("marker", marker_model, states=(marker,))
        marker_plan = DiscretizationPlan()
        marker_plan.states.add(marker, StateStorage(ghost_depth=1))
        if not periodic:
            marker_plan.boundaries.add(TransportBoundarySet({
                frame.boundaries.x_min: Outflow(state=marker_block[marker]),
                frame.boundaries.x_max: Outflow(state=marker_block[marker]),
            }, periodic=PeriodicAxes((frame.axes[1],))))
        case.numerics(marker_plan, block=marker_block)
        marked = program.state(marker_block[marker])
        program.commit(marked.next, program.value("static_marker", 1*marked.n, at=marked.next.point))
        # Asymmetric rectangle includes both odd/even fine-cell tangential indices.
        tagging = ((x > .21) & (x < .53) & (y > .17) & (y < .69)).astype(float)[None]
        values.update(_initial(case, marker_block[marker], tagging))
        subjects["marker"] = marker_block[marker]
        transfer = AMRTransfer()
        for subject in subjects.values():
            transfer.state(subject, StateTransfer())
        threshold = case.param(RuntimeParam("static_refinement", default=.5))
        grid = CartesianGrid(frame=frame, cells=cells,
            periodic=PeriodicAxes(frame.axes if periodic else (frame.axes[1],)))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(pm.ValueExpr(marker_block[marker]) > case.value(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid.frozen(), transfer=transfer,
            execution=AMRExecution.synchronous() if sample == "field" else
                      AMRExecution.subcycled((AMRClockRelation(0,1,2),)))
    case.program(program)

    def assess(runtime):
        if amr:
            from .method_amr_reception import composite_base_projection
            arrays = [runtime.block_level_state_global("method", level) for level in range(runtime.n_levels())]
            projection = composite_base_projection(arrays, runtime.patch_boxes(), cells=cells)
            mass = runtime.integral("method", component=0)
            if projection is None:
                return non_owner_metrics()
            actual = projection["means"][None]
            counts = projection["active_counts"]
        else:
            actual = state_array(runtime, "method", initial.shape)
            if actual is None:
                return non_owner_metrics()
            mass = float(actual.mean()); counts = [cells[0]*cells[1]]
        metrics = dict(metrics_owner=True, finite=bool(np.isfinite(actual).all()),
            mass_defect=abs(float(mass)-float(initial.mean())),
            change=float(np.max(abs(actual-initial))), active_counts=counts)
        if amplitude == 0:
            metrics["constant_error"] = float(np.max(abs(actual-offset)))
        if not amr and not periodic:
            expected = outflow_ssprk2(initial, dt=dt, steps=steps)
            metrics["reference_error"] = float(np.max(abs(actual-expected)))
        metrics["passed"] = bool(metrics["finite"]
            and (not periodic or metrics["mass_defect"] < 3.e-11)
            and (not amr or all(counts))
            and metrics.get("constant_error", 0) < 3.e-11
            and metrics.get("reference_error", 0) < 3.e-13)
        return metrics
    return CaseSpec(case, layout, values, subjects, steps*dt, dt, assess,
        "Exact source-owned "+sample+" stencil extension on "+("partial AMR" if amr else "Uniform")
        +(" periodic" if periodic else " outflow")+"; two authored SSPRK stages, no runtime callback")
