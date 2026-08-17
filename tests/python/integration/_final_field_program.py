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
    PatchLayout,
    Tag,
)
from pops.domain import Rectangle
from pops.codegen import Production
from pops.fields import (
    CellCenteredSecondOrder,
    CompositeHierarchySolve,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet, Periodic
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.math import ValueExpr
from pops.layouts import AMR, Uniform
from pops.lib.amr import EllipticRecompute, StateTransfer
from pops.lib.initial import Constant
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, laplacian
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Density, Model
from pops.projection import ConservativeCellAverage
from pops.solvers.elliptic import CartesianCG, GeometricMG
from pops.time import FixedDt, every, on_start


ProgramFactory = Callable[[Any, Any, Any], Any]
ConsumerFactory = Callable[[Any, Any, Any, Any], tuple[Any, ...]]


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
        equation=(-laplacian(potential) == rho - 1.0),
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
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("growth", on=state, value=(coefficient * rho,))
    model.rate("source_rate", equation=ddt(state) == -div(flux) + source)
    return model


def density_advection_model(
    name: str,
    *,
    speed: float = 1.0,
    stability_speed: float | None = None,
    stability_dt: float | None = None,
    source_frequency: float | None = None,
) -> Model:
    """One public density component for low-level runtime/CFL fixture coverage."""
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state(
        "U",
        components=("density",),
        roles={"density": Density()},
    )
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (speed * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (abs(speed) + 0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    rhs = -div(flux)
    if source_frequency is not None:
        source = model.source("null_source", on=state, value=(0.0 * rho,))
        rhs = rhs + source
        # Stability traits still live on the single-state compiler facade; the board retains that
        # exact lowering authority as ``_dsl`` until the traits become first-class board handles.
        model._dsl.source_frequency(source_frequency + 0.0 * rho)
    model.rate("explicit_rhs", equation=ddt(state) == rhs)
    if stability_speed is not None:
        model._dsl.stability_speed(stability_speed + 0.0 * rho)
    if stability_dt is not None:
        model._dsl.stability_dt(stability_dt + 0.0 * rho)
    return model


def forward_euler_program(state: Any, rate: Any, _field: Any) -> Any:
    """Public one-rate Forward-Euler Program factory for artifact-backed fixtures."""
    program = ForwardEuler(state, rate=rate)
    program.step_strategy(FixedDt(0.01))
    return program


def scalar_advection_model(name: str) -> Model:
    """Conservative scalar advection without an unrelated field solve."""
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
    return model


def scalar_burgers_model(name: str) -> Model:
    """Conservative nonlinear scalar transport without an unrelated field solve."""
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
            x_axis: (0.5 * rho * rho,),
            y_axis: (0.125 * rho * rho,),
        },
        waves={
            x_axis: (rho,),
            y_axis: (0.25 * rho,),
        },
    )
    model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    return model


def scalar_advection_field_model(name: str) -> Model:
    """Conservative scalar advection with an authenticated periodic Poisson provider."""
    model = scalar_advection_model(name)
    state = next(iter(model.states.values()))
    (rho,) = state
    potential = model.field("potential")
    model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == rho - 1.0),
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
    rate_name: str = "explicit_rhs",
    regrid_every: int = 2,
    field_solver: Any = None,
    initial_profile: Any = None,
    components: tuple[Any, ...] = (),
    cxx: str | None = None,
    include: str | None = None,
    strict_restart: bool = False,
    consumer_factory: ConsumerFactory | None = None,
    anchored_field: bool = False,
    patch_layout: PatchLayout | None = None,
    clustering: Any = None,
    refine_threshold: float = 0.5,
) -> Any:
    """Return the exact public resolved plan consumed by one native integration compile."""
    if target not in {"system", "amr_system"}:
        raise ValueError("target must be 'system' or 'amr_system'")
    state = next(iter(model.states.values()))
    rate = model.operators[rate_name]
    flux = model.fluxes["transport"]
    field_operator = model.field_operators.get("electrostatic")

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
    solver = field_solver
    if solver is None:
        solver = GeometricMG() if target == "amr_system" else CartesianCG()
    field_instance = None if field_operator is None else case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(
                BoundaryCondition(
                    AllPhysicalBoundaries(),
                    Dirichlet(0.0) if anchored_field else Periodic(),
                ),
            ),
            solver=solver,
            nullspace=None if anchored_field else ConstantNullspace(),
            gauge=None if anchored_field else MeanValueGauge(0.0),
            hierarchy_policy=(
                CompositeHierarchySolve() if target == "amr_system" else None
            ),
        ),
    )
    program = factory(state_instance, rate, field_instance)
    case.program(program)
    consumers: list[Any] = []
    if strict_restart:
        from pops.output import Checkpoint

        consumers.append(
            Checkpoint(
                schedule=on_start(clock=program.clock),
                target="checkpoints/strict",
                bit_identical=True,
            )
        )
    if consumer_factory is not None:
        extra_consumers = consumer_factory(case, block, state_instance, program)
        if type(extra_consumers) is not tuple:
            raise TypeError("consumer_factory must return an exact tuple")
        consumers.extend(extra_consumers)
    if consumers:
        from pops.output import ConsumerGraph

        case.consumers(ConsumerGraph.from_consumers(tuple(consumers)))

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
                value=(
                    Constant((1.0,) + (0.0,) * (len(state.components) - 1))
                    if initial_profile is None else initial_profile
                ),
                projection=ConservativeCellAverage(),
            )
        )
        threshold = case.param(
            RuntimeParam("%s_refine_threshold" % name, default=refine_threshold)
        )
        transfer = AMRTransfer()
        transfer.state(state_instance, StateTransfer())
        if field_instance is not None:
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
            grid=CartesianGrid(
                frame=model.frame,
                cells=(n, n),
                periodic=PeriodicAxes(model.frame.axes),
            ),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=tagging,
            regrid=AMRRegrid(
                schedule=every(max(1, regrid_every), clock=program.clock)
            ),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
            patch_layout=PatchLayout() if patch_layout is None else patch_layout,
            clustering=clustering,
        )
    native_options: dict[str, Any] = {}
    if cxx is not None or include is not None:
        if cxx is None or include is None:
            raise ValueError("native resolution requires both cxx and include")
        native_options = {
            "backend": Production(),
            "compile_options": {"cxx": cxx, "include": include},
        }
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        components=components,
        **native_options,
    )


def compile_block_model(model: Model, *, target: str) -> Any:
    """Compile a final board model through its explicit compiler-provider protocol."""
    return compiler_model(model).compile(backend="production", target=target)


def compiler_model(model: Model) -> Any:
    """Return the authenticated formula emitter paired with the final model's Module."""
    lowering = model.__pops_compiler_lowering__()
    if lowering.source_module is not model.module or lowering.facade is not model:
        raise ValueError("final Model compiler lowering changed its authenticated authority")
    from pops.codegen.component_provider_packs import resolve_component_provider_packs

    lowering.bind_component_provider_packs(
        resolve_component_provider_packs(lowering.source_module)
    )
    return lowering.emit_model


__all__ = [
    "compile_block_model",
    "compiler_model",
    "density_advection_model",
    "forward_euler_program",
    "passive_field_model",
    "passive_source_model",
    "resolve_periodic_field_program",
    "scalar_advection_model",
    "scalar_burgers_model",
    "scalar_advection_field_model",
]
