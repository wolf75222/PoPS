"""Public physical diffusion and a screened Field evaluated at its predictor."""

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import pops
from pops import math
from pops.analytic import time as analytic_time
from pops.domain import Rectangle
from pops.fields import (
    AnalyticAux,
    AuxiliaryBoundary,
    CellCenteredSecondOrder,
    FieldDiscretization,
    FieldOutput,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.solvers.elliptic import CartesianCG
from pops.solvers.tolerances import AbsoluteFloor, Relative
from pops.time import FailRun, FixedDt, RejectAttempt, GuardRole, ErrorControlledDt
from pops.params import RuntimeParam

# Declared physical inputs, before observing any runtime result.
CELLS = (16, 12)
KAPPA = Fraction(1, 10)
DT = Fraction(1, 256)
BASE = Fraction(2)
AMPLITUDE = Fraction(1, 2)
RHS_OFFSET = Fraction(3)
SOLVER_RTOL = 1e-13
SOLVER_ATOL = 1e-14


@dataclass(frozen=True)
class DiffusionFieldCase:
    case: object
    layout: object
    state: object
    field: object
    program: object
    initial: np.ndarray
    method: str
    refusal_limit: object
    retry: bool


def author_case(
    method="euler",
    *,
    layout_factory=None,
    field_solver=None,
    hierarchy_policy=None,
    rk2_composition="weighted_rates",
    labels=None,
    kappa=KAPPA,
    automatic_retry=False,
    populated_refusal=False,
):
    if method not in ("euler", "ssprk2"):
        raise ValueError(method)
    if rk2_composition not in ("weighted_rates", "weighted_predictor"):
        raise ValueError(rk2_composition)
    labels = {} if labels is None else dict(labels)
    label = lambda role, default: labels.get(role, default)
    # Physical model first. The diagnostic read is never part of ddt(U).
    frame = Rectangle("diffusion-field-domain", (0.0, 0.0), (1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model(label("model", "public_diffusion_field"), frame=frame)
    U = model.state("U", components=("u",))
    tau = model.auxiliary("stage_tau", frame=frame.canonical_id, unit="1")
    refusal_limit = model.param(RuntimeParam("late_refusal_norm_limit", default=0.0))
    limit_expression = model.value(refusal_limit) + 0.0 * U[0]
    if populated_refusal:
        limit_expression = limit_expression - 2560.0 * tau
    model.source(
        "late_refusal_limit_observation",
        on=U,
        value=(limit_expression,),
    )
    phi_read = model.aux("screened_phi")
    flux = model.diffusive_flux(
        "conduction", state=U, value=math.CoeffGradient(U[0], float(kappa))
    )
    diffusion = model.rate(
        label("rate", "diffusion"), equation=math.ddt(U) == math.div(flux)
    )
    phi = model.field("phi")
    operator = model.field_operator(
        "fields",
        unknown=phi,
        equation=-math.laplacian(phi) + math.Reaction(phi, 1)
        == (float(RHS_OFFSET) + tau) * U[0],
        outputs=(FieldOutput("screened_phi", phi),),
    )
    # A generic pointwise read converts the consumed Field output to a saved RHS
    # history. It contributes no increment to the physical state.
    model.source("stage_phi_observation", on=U, value=(phi_read,))

    case = pops.Case("diffusion_predictor_field_" + method)
    block = case.block("material", model)
    plan = DiscretizationPlan()
    plan.rates.add(diffusion, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    policy = {} if hierarchy_policy is None else dict(hierarchy_policy=hierarchy_policy)
    field = case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=field_solver
            or CartesianCG(
                tolerance=Relative(SOLVER_RTOL, floor=AbsoluteFloor(SOLVER_ATOL))
            ),
            **policy,
        ),
    )
    diagnostic = model.module.operator_handle("stage_phi_observation")
    limit_read = model.module.operator_handle("late_refusal_limit_observation")

    # Both methods are authored with the same public Program primitives.
    program = pops.Program(label("method", "authored_diffusion_field_" + method))
    state = program.state(block[U])
    stage0 = program.stage("diffusion_stage_0", c=0)
    stage1 = program.stage("field_predictor_stage", c=1)
    D = program.value(label("D", "D"), diffusion(state.n), at=stage0)
    Y = program.value(label("Y", "Y"), state.n + program.dt * D, at=stage1)
    fields = field(Y).consume(action=FailRun())
    observed_phi = program.value("phi_Y", diagnostic(Y, fields), at=stage1)
    program.store_history("predictor_Y", Y, depth=1)
    program.store_history("phi_stage", observed_phi, depth=1)
    if method == "ssprk2":
        D1 = program.value(label("D1", "D1"), diffusion(Y), at=stage1)
        if rk2_composition == "weighted_rates":
            final = (
                state.n
                + program.dt * Fraction(1, 2) * D
                + program.dt * Fraction(1, 2) * D1
            )
        else:
            final = (
                Fraction(1, 2) * state.n
                + Fraction(1, 2) * Y
                + program.dt * Fraction(1, 2) * D1
            )
        accepted = program.value(
            label("accepted", "accepted_RK2"), final, at=state.next.point
        )
    else:
        accepted = program.value(
            label("accepted", "accepted_Euler"), 1 * Y, at=state.next.point
        )
    # Field consumption, diagnostic RHS, histories and all physical slopes execute
    # before this actual collective predicate. It contributes no physical increment.
    if automatic_retry:
        delta = program.value(
            "late_refusal_predictor_increment", Y - state.n, at=stage1
        )
        condition = program.norm2(delta) <= 0.1
        action = RejectAttempt()
        role = GuardRole.ERROR_ESTIMATE
    else:
        limit_field = program.value(
            "late_refusal_limit_field", limit_read(Y), at=stage1
        )
        condition = program.norm2(Y) <= program.max(limit_field)
        action = FailRun()
        role = GuardRole.INVARIANT
    guarded = program.guard(
        "late_field_norm_refusal", accepted, condition, action=action, role=role
    )
    program.commit(state.next, guarded)
    if automatic_retry:
        program.step_strategy(
            ErrorControlledDt(
                dt_init=2 * float(DT),
                rtol=1e-3,
                atol=1e-8,
                dt_min=float(DT),
                dt_max=2 * float(DT),
                max_rejections=1,
                shrink=0.5,
                growth=1.0,
            )
        )
    else:
        program.step_strategy(FixedDt(float(DT)))
    model.module.aux_provider(
        AnalyticAux(
            model.module.aux_handle(model.module.aux()["stage_tau"]),
            analytic_time(program.clock),
            frame=frame,
            boundary=AuxiliaryBoundary(width=1, kind="foextrap"),
        )
    )
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=block[U], value=BindArray(), projection=ConservativeCellAverage()
        )
    )

    grid = CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes(frame.axes))
    layout = (
        Uniform(grid)
        if layout_factory is None
        else layout_factory(case, frame, grid, block[U], program)
    )
    nx, ny = CELLS
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    # Exact averages of sin(2*pi*x) over each cell, constant in y.
    average_mode = np.sinc(1 / nx) * np.sin(2 * np.pi * x)
    initial = np.ascontiguousarray(
        np.broadcast_to(float(BASE) + float(AMPLITUDE) * average_mode, (1, ny, nx))
    )
    return DiffusionFieldCase(
        case,
        layout,
        block[U],
        field,
        program,
        initial,
        method,
        block[refusal_limit],
        automatic_retry,
    )


def one_level_amr_layout(case, frame, grid, state, program):
    """Actual one-level AMR authority, same grid/PDE; no refinement qualification."""
    from pops.amr import (
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
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import ValueExpr
    from pops.params import RuntimeParam
    from pops.time import every

    transfer = AMRTransfer()
    transfer.state(state, StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=3.0))
    return AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=1, ratios=()),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(state)["u"] > case.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
