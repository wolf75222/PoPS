"""Final Case/field resolution shared by native integration seams.

The tests importing this module intentionally exercise the low-level native installation API.  They
still obtain every field-install plan from the public ``validate -> resolve`` lifecycle: no codegen
test is allowed to invent a solver route or bypass Case ownership.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pops
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
from pops.domain import Rectangle
from pops.fields import (
    CellCenteredSecondOrder,
    CompositeHierarchySolve,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.math import ValueExpr
from pops.layouts import AMR, Uniform
from pops.lib.amr import EllipticRecompute, StateTransfer
from pops.lib.initial import Constant
from pops.math import ddt, div, laplacian
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.solvers.elliptic import GeometricMG
from pops.time import every


ProgramFactory = Callable[[Any, Any, Any], Any]


def _frame(name: str) -> Any:
    return Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())


def passive_field_model(name: str, *, coefficient: float) -> Model:
    """One conservative scalar with a linear source and a periodic Poisson field."""
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("growth", on=state, value=(coefficient * rho,))
    model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + source)
    potential = model.field("potential")
    model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == rho),
        outputs=(
            FieldOutput("phi", potential),
            GradientOutput("grad", potential),
        ),
    )
    return model


def passive_source_model(name: str, *, coefficient: float) -> Model:
    """One conservative scalar with a local source and no field-solve requirement."""
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("growth", on=state, value=(coefficient * rho,))
    model.rate("source_rate", equation=ddt(state) == source)
    return model


def scalar_advection_field_model(name: str) -> Model:
    """Conservative scalar advection with an authenticated periodic Poisson provider."""
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (rho,),
            y_axis: (0.25 * rho,),
        },
        waves={
            x_axis: (1.0 + 0.0 * rho,),
            y_axis: (0.25 + 0.0 * rho,),
        },
    )
    model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    potential = model.field("potential")
    model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == rho),
        outputs=(
            FieldOutput("phi", potential),
            GradientOutput("grad", potential),
        ),
    )
    return model


def resolve_periodic_field_program(
    model: Model,
    factory: ProgramFactory,
    *,
    name: str,
    block_name: str,
    target: str,
    n: int,
    regrid_every: int = 2,
) -> Any:
    """Return the exact public resolved plan consumed by one native integration compile."""
    if target not in {"system", "amr_system"}:
        raise ValueError("target must be 'system' or 'amr_system'")
    state = next(iter(model.states.values()))
    rate = model.operators["explicit_rhs"]
    flux = model.fluxes["transport"]
    field_operator = model.field_operators["electrostatic"]

    case = pops.Case("%s-case" % name)
    block = case.block(block_name, model)
    state_instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    field_instance = case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(
                BoundaryCondition(AllPhysicalBoundaries(), Periodic()),
            ),
            solver=GeometricMG(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
            hierarchy_policy=(
                CompositeHierarchySolve() if target == "amr_system" else None
            ),
        ),
    )
    program = factory(state_instance, rate, field_instance)
    case.program(program)

    if target == "system":
        grid_frame = _frame("%s-uniform-grid" % name)
        layout = Uniform(CartesianGrid(
            frame=grid_frame,
            cells=(n, n),
            periodic=PeriodicAxes(grid_frame.axes),
        ))
    else:
        case.initials.add(
            InitialCondition(
                state=state_instance,
                value=Constant((1.0,) + (0.0,) * (len(state.components) - 1)),
                projection=ConservativeCellAverage(),
            )
        )
        threshold = case.param(
            RuntimeParam("%s_refine_threshold" % name, default=0.5)
        )
        transfer = AMRTransfer()
        transfer.state(state_instance, StateTransfer())
        transfer.field(field_instance, EllipticRecompute())
        tagging = AMRTagging(
            rules=(
                Tag(ValueExpr(state_instance) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        )
        layout = AMR(
            grid=CartesianGrid(frame=model.frame, cells=(n, n)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=tagging,
            regrid=AMRRegrid(
                schedule=every(max(1, regrid_every), clock=program.clock)
            ),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )
    return pops.resolve(pops.validate(case), layout=layout)


def compile_block_model(model: Model, *, target: str) -> Any:
    """Compile a final board model through its explicit compiler-provider protocol."""
    return compiler_model(model).compile(backend="production", target=target)


def compiler_model(model: Model) -> Any:
    """Return the authenticated formula emitter paired with the final model's Module."""
    lowering = model.__pops_compiler_lowering__()
    if lowering.source_module is not model.module or lowering.facade is not model:
        raise ValueError("final Model compiler lowering changed its authenticated authority")
    return lowering.emit_model


__all__ = [
    "compile_block_model",
    "compiler_model",
    "passive_field_model",
    "passive_source_model",
    "resolve_periodic_field_program",
    "scalar_advection_field_model",
]
