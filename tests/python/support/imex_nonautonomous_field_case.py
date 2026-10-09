"""Public IMEX Euler with a time-dependent, genuinely solved screened field.

The runtime clock is tau with tau0=0. Physical time is t=1+tau, so
I=-(2+tau)U and -Delta F+F=(3+tau)U realize the requested nonautonomous
scalar law on spatially constant data. No clock setter or custom provider.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pops
from pops.analytic import time as analytic_time
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.domain import Rectangle
from pops.fields import (AnalyticAux, AuxiliaryBoundary, CellCenteredSecondOrder,
                         CompositeHierarchySolve, FieldDiscretization, FieldOutput)
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.lib.time import IMEX, IMEX_EULER_TABLEAU
from pops.math import Reaction, ValueExpr, ddt, laplacian
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers.elliptic import GeometricMG
from pops.solvers.tolerances import AbsoluteFloor, Relative
from pops.time import FixedDt, every

U0 = Fraction(3)
PHYSICAL_T0 = Fraction(1)
H = Fraction(1, 4)
Y_EXACT = Fraction(48, 25)
FIELD_EXACT = Fraction(144, 25)
ACCEPTED_EXACT = Fraction(84, 25)
WRONG_FIELD_TIME_ACCEPTED = Fraction(87, 25)
CELLS = (8, 8)
SOLVER_RTOL = 1e-13
SOLVER_ATOL = 1e-14


def error_bounds(cells=CELLS):
    """Fixed budget from residual tolerance, beta=1 and binary64 roundoff.

    N*max(|rhs|) bounds either unscaled or normalized residual norms on this
    constant witness. The periodic screened operator has unit constant-mode
    eigenvalue. The accepted error propagates by h; 2048 eps covers the small
    scalar solve and arithmetic. No observed result determines these bounds.
    """
    n = int(np.prod(cells))
    rounding = 2048 * np.finfo(np.float64).eps
    field = n * 6 * SOLVER_RTOL + SOLVER_ATOL + 6 * rounding
    accepted = float(H) * field + 4 * rounding
    return field, accepted


@dataclass(frozen=True)
class NonautonomousFieldCase:
    case: object
    layout: object
    state: object
    field: object
    program: object
    initial: np.ndarray
    cells: tuple[int, int]


def build_case(*, cells=CELLS, layout=None):
    """Author using ordinary Model/Case/Field/IMEX primitives.

    The caller may supply its real layout. There is no layout-kind rejection
    or synthetic backend branch in this physical witness.
    """
    frame = Rectangle("nonautonomous_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("nonautonomous_relaxation", frame=frame)
    state = model.state("U", components=("u",))
    tau = model.auxiliary("stage_tau", frame=frame.canonical_id, unit="1")
    field_read = model.aux("E")
    explicit = model.rate("read_solved_field", equation=ddt(state) == model.source(
        "field_forcing", on=state, value=(field_read,)))
    implicit = model.operator("time_dependent_relaxation", inputs=(),
        returns=model.local_linear_operator("time_dependent_relaxation", on=state,
                                           matrix=((-(2 + tau),),)))
    unknown = model.field("F")
    physical_field = model.field_operator("fields", unknown=unknown,
        equation=-laplacian(unknown) + Reaction(unknown, 1) == (3 + tau) * state[0],
        outputs=(FieldOutput("E", unknown),))
    case = pops.Case("nonautonomous_imex_field")
    block = case.block("material", model)
    # E is a pointwise source rate: allocate real Program state storage, no flux.
    # I keeps the public IMEX LocalLinear/DenseLU realization selected by the preset.
    numerics = DiscretizationPlan()
    numerics.rates.add(explicit, StateStorage())
    case.numerics(numerics, block=block)
    tolerance = Relative(SOLVER_RTOL, floor=AbsoluteFloor(SOLVER_ATOL))
    # One level has no coarse/fine correction; do not request multilevel FAC.
    field_solver = GeometricMG(tolerance=tolerance, max_cycles=100)
    field = case.field(physical_field, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
        solver=field_solver, hierarchy_policy=CompositeHierarchySolve()))
    program = IMEX(block[state], explicit_operator=explicit, implicit_operator=implicit,
                   fields_operator=field, tableau=IMEX_EULER_TABLEAU)
    # Attach to the final Module and the exact Clock created by the public factory.
    module = model.module
    module.aux_provider(AnalyticAux(module.aux_handle(module.aux()["stage_tau"]),
        analytic_time(program.clock), frame=frame,
        boundary=AuxiliaryBoundary(width=1, kind="foextrap")))
    program.step_strategy(FixedDt(float(H)))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    if layout is None:
        transfer = AMRTransfer()
        transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam("refine_threshold", default=float(U0)))
        # A genuine one-level hierarchy serves the public screened MG realization.
        # Tagging/regrid/transfer authorities exist; this witness does not exercise refinement.
        layout = AMR(grid=CartesianGrid(frame=frame, cells=tuple(cells),
                                       periodic=PeriodicAxes(frame.axes)),
            hierarchy=AMRHierarchy(max_levels=1, ratios=()),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["u"] > case.value(threshold)), Buffer(cells=0)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
            transfer=transfer, execution=AMRExecution.synchronous())
    initial = np.full((1, cells[1], cells[0]), float(U0), dtype=np.float64)
    return NonautonomousFieldCase(case, layout, block[state], field, program, initial, tuple(cells))
