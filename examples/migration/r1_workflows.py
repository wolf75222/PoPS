"""Run the eight bounded equation-oriented R1 migration workflows.

These workflows demonstrate the public Model -> Case -> DiscretizationPlan ->
Program -> validate -> resolve -> compile -> bind -> run lifecycle. They are
small user examples, not a replacement for the full migration qualification
matrix.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pops
from pops.domain import Rectangle
from pops.fields import (
    FieldBoundary,
    FieldDiscretization,
    FieldProblem,
    SharedMeanGauge,
    bcs,
)
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import DivCoeffGrad, ddt, div, grad, laplacian, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (
    Diffusion,
    DiscretizationPlan,
    FiniteVolume,
    JointEvaluation,
    reconstruction,
    riemann,
    variables,
)
from pops.numerics.terms import Flux, SourceTerm
from pops.solvers import CG, Newton
from pops.time import (
    DerivativeStrategy,
    FailRun,
    FixedDt,
    ImplicitDiffusionStage,
    SolveUnknown,
)


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
DEFAULT_CELLS = 16


@dataclass(frozen=True, slots=True)
class Workflow:
    """One command-line workflow and its execution requirements."""

    run: Callable[[argparse.Namespace], dict[str, Any]]
    summary: str


def _frame(name: str) -> Any:
    return Rectangle(name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())


def _layout(frame: Any, cells: int, *, periodic: bool = True) -> Any:
    grid = CartesianGrid(
        frame=frame,
        cells=(cells, cells),
        periodic=PeriodicAxes(frame.axes) if periodic else None,
    )
    return Uniform(grid)


def _execution_resources(artifact: Any) -> dict[str, Any]:
    communicator = artifact.platform_manifest.communicator.require("migration example communicator")
    if communicator == "serial":
        return {}
    if communicator == "MPI_COMM_WORLD":
        return {"execution_context": pops.ExecutionContext.mpi_world(artifact)}
    raise RuntimeError("unsupported communicator %r" % communicator)


def _compile(case: Any, layout: Any, *, components: tuple[Any, ...] = ()) -> Any:
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout, components=components)
    return pops.compile(resolved)


def _bind(artifact: Any, initial_state: dict[str, np.ndarray]) -> Any:
    return pops.bind(
        artifact,
        initial_state=initial_state,
        resources=_execution_resources(artifact),
    )


def _run_case(
    case: Any,
    layout: Any,
    initial_state: dict[str, np.ndarray],
    *,
    dt: float,
    steps: int,
    components: tuple[Any, ...] = (),
) -> tuple[Any, Any, Any]:
    artifact = _compile(case, layout, components=components)
    runtime = _bind(artifact, initial_state)
    report = pops.run(
        runtime,
        t_end=steps * dt,
        max_steps=steps,
        console=False,
    )
    return runtime, report, artifact


def _cell_centers(cells: int) -> tuple[np.ndarray, np.ndarray]:
    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    return np.meshgrid(coordinate, coordinate, indexing="xy")


def _state_summary(runtime: Any, block: str) -> dict[str, Any]:
    values = np.asarray(runtime.state_global(block), dtype=np.float64)
    return {
        "block": block,
        "shape": list(values.shape),
        "component_means": np.mean(values, axis=tuple(range(1, values.ndim))).tolist(),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _scalar_amr(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the canonical scalar AMR example without duplicating its physics."""

    script = SOURCE_ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"
    command = [sys.executable, str(script), "--output-dir", str(args.work_dir / "scalar-amr")]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError("scalar AMR example failed with exit code %d" % completed.returncode)
    return {
        "workflow": "scalar-amr",
        "delegated_entrypoint": str(script.relative_to(SOURCE_ROOT)),
        "output_dir": str(args.work_dir / "scalar-amr"),
        "exit_code": completed.returncode,
    }


def _field_consumer_case(
    cells: int, *, transport: bool
) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = _frame("field_consumer_square")
    x_axis, y_axis = frame.axes
    model = pops.Model("field_consumer", frame=frame)
    names = ("rho", "driver") if transport else ("rho", "mx", "my")
    state = model.state("U", components=names)
    rho = state[0]
    potential = model.field("potential")
    gradient = model.vector(
        "gradient",
        frame=frame,
        components={x_axis: grad(potential).x, y_axis: grad(potential).y},
    )
    if transport:
        driver = state[1]
        flux = model.flux(
            "field_transport",
            frame=frame,
            state=state,
            components={
                x_axis: (0 * rho, 0 * driver),
                y_axis: (gradient.y * rho, 0 * driver),
            },
            waves={
                x_axis: (0 * rho, 0 * driver),
                y_axis: (gradient.y, 0 * driver),
            },
        )
        physical_rate = model.rate("transport", equation=ddt(state) == -div(flux))
        driver_model = None
        driver_state = None
    else:
        _rho, mx, my = state
        speed = sqrt(0.5)
        flux = model.flux(
            "Euler",
            frame=frame,
            state=state,
            components={
                x_axis: (mx, mx * mx / rho + 0.5 * rho, mx * my / rho),
                y_axis: (my, mx * my / rho, my * my / rho + 0.5 * rho),
            },
            waves={
                x_axis: (mx / rho - speed, mx / rho, mx / rho + speed),
                y_axis: (my / rho - speed, my / rho, my / rho + speed),
            },
        )
        electric = model.source(
            "electric", on=state, value=(0 * rho, -rho * gradient.x, -rho * gradient.y)
        )
        physical_rate = model.rate("Euler_Poisson", equation=ddt(state) == -div(flux) + electric)
        driver_model = pops.Model("Poisson_driver", frame=frame)
        driver_state = driver_model.state("U", components=("load",))
        stationary = driver_model.flux(
            "stationary",
            frame=frame,
            state=driver_state,
            components={axis: (0 * driver_state[0],) for axis in frame.axes},
            waves={axis: (0 * driver_state[0],) for axis in frame.axes},
        )
        driver_rate = driver_model.rate(
            "frozen_load", equation=ddt(driver_state) == -div(stationary)
        )
        driver = driver_state[0]

    problem = FieldProblem(
        "Poisson",
        unknowns=(potential,),
        equations=(-laplacian(potential) == driver - 1,),
        boundaries=(
            FieldBoundary(
                potential,
                bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()),
            ),
        ),
        gauge=SharedMeanGauge((potential,)),
    )
    case = pops.Case("field_consumer_case")
    fluid_block = case.block("fluid", model)
    fluid_numerics = DiscretizationPlan()
    fluid_numerics.rates.add(
        physical_rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(fluid_numerics, block=fluid_block)

    driver_block = None
    if not transport:
        driver_block = case.block("driver", driver_model)
        driver_numerics = DiscretizationPlan()
        driver_numerics.rates.add(
            driver_rate,
            FiniteVolume(
                flux=stationary,
                variables=variables.Conservative(driver_state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(driver_numerics, block=driver_block)

    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=CG(max_iter=4000, rel_tol=1.0e-11, abs_tol=1.0e-12),
        ),
    )
    program = pops.Program("field_consumer_step")
    current = program.state(fluid_block[state])
    point = program.stage("evaluate", c=0)
    driver_current = current if transport else program.state(driver_block[driver_state])
    solve_inputs = (
        {fluid_block[state]: current.n}
        if transport
        else {driver_block[driver_state]: driver_current.n}
    )
    observations = field.observe(
        program.solve(field, values=solve_inputs, at=point).consume(action=FailRun())
    )
    solved_gradient = observations.gradient(field[potential], dimension=2)
    module = model.module
    carrier = fluid_block[module.field_handle(module.field_spaces()["fields"])]
    context = observations.publish(
        {
            (carrier, "potential_grad_x"): (solved_gradient, 0),
            (carrier, "potential_grad_y"): (solved_gradient, 1),
        },
        states=None if transport else {fluid_block[state]: current.n},
    )
    terms = [Flux()] if transport else [SourceTerm(fluid_block[module.operator_handle("electric")])]
    rhs = program.rhs(state=current.n, fields=context, terms=terms)
    program.store_history("gradient", solved_gradient, depth=1)
    dt = 0.125
    program.commit(
        current.next,
        program.value("advanced", current.n + dt * rhs, at=current.next.point),
    )
    if not transport:
        program.commit(
            driver_current.next,
            program.value("frozen_driver", 1 * driver_current.n, at=driver_current.next.point),
        )
    program.step_strategy(FixedDt(dt))
    case.program(program)

    x, y = _cell_centers(cells)
    amplitude = 0.1
    if transport:
        rho_values = 1 + 0.2 * np.sin(2 * math.pi * y)
        driver_values = 1 + amplitude * np.cos(2 * math.pi * y)
        fluid = np.stack((rho_values, driver_values))
        initial = {"fluid": fluid}
    else:
        rho_values = 1 + amplitude * np.cos(2 * math.pi * x) * np.cos(2 * math.pi * y)
        fluid = np.stack((rho_values, np.zeros_like(rho_values), np.zeros_like(rho_values)))
        initial = {"fluid": fluid, "driver": rho_values[None]}
    return case, _layout(frame, cells), initial, dt


def _field_consumer(args: argparse.Namespace, *, transport: bool) -> dict[str, Any]:
    case, layout, initial, dt = _field_consumer_case(args.cells, transport=transport)
    before_means = np.mean(initial["fluid"], axis=(1, 2))
    runtime, report, artifact = _run_case(case, layout, initial, dt=dt, steps=args.steps)
    after = np.asarray(runtime.state_global("fluid")).reshape(initial["fluid"].shape)
    gradient_values = np.asarray(runtime.history_global("gradient", 0))
    return {
        "workflow": "field-transport" if transport else "euler-poisson",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "state": _state_summary(runtime, "fluid"),
        "component_mean_change": (np.mean(after, axis=(1, 2)) - before_means).tolist(),
        "gradient_l2": float(np.sqrt(np.mean(gradient_values**2))),
    }


def _heterogeneous_case(cells: int) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = _frame("heterogeneous_interaction_square")
    model = pops.Model("heterogeneous_exchange", frame=frame)
    left = model.species("left", state=("p", "E"))
    right = model.species("right", state=("p", "E", "m"))
    force = right[0] / right[2] - left[0]
    power = (left[0] + right[0] / right[2]) * force / 2
    exchange = model.interaction(
        "exchange",
        outputs={left: (force, power), right: (-force, -power, 0)},
    )
    left_rate = model.rate("left_balance", equation=ddt(left) == exchange[left])
    right_rate = model.rate("right_balance", equation=ddt(right) == exchange[right])
    case = pops.Case("heterogeneous_interaction_case")
    left_block = case.block("left", model, states=(left,))
    right_block = case.block("right", model, states=(right,))
    for block, state, rate in (
        (left_block, left, left_rate),
        (right_block, right, right_rate),
    ):
        plan = DiscretizationPlan()
        plan.rates.add(rate, JointEvaluation(state))
        case.numerics(plan, block=block)

    program = pops.Program("heterogeneous_exchange_step")
    left_time = program.state(left_block[left])
    right_time = program.state(right_block[right])
    left_rhs = left_rate(left_time.n, right_time.n)
    right_rhs = right_rate(right_time.n, left_time.n)
    dt = 1.0e-3
    program.commit_many(
        {
            left_time.next: program.value(
                "left_next", left_time.n + dt * left_rhs, at=left_time.next.point
            ),
            right_time.next: program.value(
                "right_next", right_time.n + dt * right_rhs, at=right_time.next.point
            ),
        }
    )
    program.step_strategy(FixedDt(dt))
    case.program(program)
    ones = np.ones((cells, cells), dtype=np.float64)
    initial = {
        "left": np.stack((2 * ones, 5 * ones)),
        "right": np.stack((ones, 3 * ones, 2 * ones)),
    }
    return case, _layout(frame, cells), initial, dt


def _heterogeneous_interaction(args: argparse.Namespace) -> dict[str, Any]:
    case, layout, initial, dt = _heterogeneous_case(args.cells)
    before = {
        component: float(np.mean(initial["left"][component] + initial["right"][component]))
        for component in (0, 1)
    }
    runtime, report, artifact = _run_case(case, layout, initial, dt=dt, steps=args.steps)
    left = np.asarray(runtime.state_global("left"))
    right = np.asarray(runtime.state_global("right"))
    return {
        "workflow": "heterogeneous-interaction",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "left": _state_summary(runtime, "left"),
        "right": _state_summary(runtime, "right"),
        "pair_mean_defect": {
            str(component): float(np.mean(left[component] + right[component])) - before[component]
            for component in (0, 1)
        },
    }


def _diffusion_case(cells: int, *, implicit: bool) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = _frame("diffusion_square")
    model = pops.Model("scalar_diffusion", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    flux = model.diffusive_flux("conduction", state=state, value=0.1 * grad(u))
    rate = model.rate("heat", equation=ddt(state) == div(flux))
    case = pops.Case("implicit_diffusion" if implicit else "explicit_diffusion")
    block = case.block("heat", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    dt = 1.0e-4 if implicit else 0.05 / (0.1 * cells * cells)
    if implicit:
        program = pops.Program("backward_euler_diffusion")
        temporal = program.state(block[state])
        coordinates = program.value("stage_coordinates", temporal.n, at=temporal.next.point)
        stage = ImplicitDiffusionStage(rate, temporal.n, program.dt)
        request = stage.request(
            unknown=SolveUnknown("state", coordinates),
            seed=temporal.n,
            derivative=DerivativeStrategy("finite_difference"),
        )
        solved = program.solve(
            request,
            solver=Newton(
                tolerance=1.0e-12,
                max_iterations=20,
                linear_tolerance=1.0e-8,
                linear_max_iterations=100,
                restart=30,
            ),
        ).consume(action=FailRun())[0]
        program.commit(
            temporal.next,
            program.value("updated", solved, at=temporal.next.point),
        )
    else:
        program = pops.Program("forward_euler_diffusion")
        temporal = program.state(block[state])
        candidate = program.value(
            "updated", temporal.n + program.dt * rate(temporal.n), at=temporal.next.point
        )
        program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    x, y = _cell_centers(cells)
    initial = 2 + 0.4 * np.sin(2 * math.pi * x) * np.cos(2 * math.pi * y)
    return case, _layout(frame, cells), {"heat": initial[None]}, dt


def _diffusion(args: argparse.Namespace, *, implicit: bool) -> dict[str, Any]:
    case, layout, initial, dt = _diffusion_case(args.cells, implicit=implicit)
    initial_mass = float(np.mean(initial["heat"]))
    runtime, report, artifact = _run_case(case, layout, initial, dt=dt, steps=args.steps)
    final = np.asarray(runtime.state_global("heat"))
    return {
        "workflow": "implicit-diffusion" if implicit else "explicit-diffusion",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "state": _state_summary(runtime, "heat"),
        "mean_change": float(np.mean(final)) - initial_mass,
    }


def _variable_field_case(cells: int) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = _frame("variable_coefficient_field_square")
    load_model = pops.Model("field_load", frame=frame)
    reference_model = pops.Model("field_reference", frame=frame)
    load_state = load_model.state("U", components=("load", "coefficient"))
    reference_state = reference_model.state("U", components=("load", "coefficient"))
    authored = []
    for model, state in (
        (load_model, load_state),
        (reference_model, reference_state),
    ):
        stationary = model.flux(
            "stationary",
            frame=frame,
            state=state,
            components={axis: tuple(0 * value for value in state) for axis in frame.axes},
            waves={axis: tuple(0 * value for value in state) for axis in frame.axes},
        )
        rate = model.rate("frozen", equation=ddt(state) == -div(stationary))
        plan = DiscretizationPlan()
        plan.rates.add(
            rate,
            FiniteVolume(
                flux=stationary,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        authored.append(plan)

    case = pops.Case("variable_coefficient_field_case")
    blocks = (
        case.block("load", load_model),
        case.block("reference", reference_model),
    )
    for block, plan in zip(blocks, authored, strict=True):
        case.numerics(plan, block=block)

    from pops.model import Handle, OwnerPath

    potential = Handle("potential", kind="field", owner=OwnerPath.model("variable_field"))
    problem = FieldProblem(
        "variable_Poisson",
        unknowns=(potential,),
        equations=(-DivCoeffGrad(potential, load_state[1]) == load_state[0] + reference_state[0],),
        boundaries=(
            FieldBoundary(
                potential,
                bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()),
            ),
        ),
        gauge=SharedMeanGauge((potential,)),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=CG(max_iter=4000, rel_tol=1.0e-11, abs_tol=1.0e-12),
        ),
    )
    program = pops.Program("variable_field_solve")
    times = (
        program.state(blocks[0][load_state]),
        program.state(blocks[1][reference_state]),
    )
    point = program.stage("solve", c=0)
    values = {
        blocks[0][load_state]: times[0].n,
        blocks[1][reference_state]: times[1].n,
    }
    observation = field.observe(
        program.solve(field, values=values, at=point).consume(action=FailRun())
    )
    solution = observation[field[potential]]
    program.store_history("potential", solution, depth=1)
    for temporal in times:
        program.commit(
            temporal.next,
            program.value("unchanged_" + temporal.n.name, 1 * temporal.n, at=temporal.next.point),
        )
    dt = 0.125
    program.step_strategy(FixedDt(dt))
    case.program(program)
    x, y = _cell_centers(cells)
    coefficient = 1.5 + 0.25 * np.cos(2 * math.pi * x)
    load = np.sin(2 * math.pi * x) * np.cos(2 * math.pi * y)
    zeros = np.zeros_like(load)
    initial = {
        "load": np.stack((load, coefficient)),
        "reference": np.stack((zeros, np.ones_like(load))),
    }
    return case, _layout(frame, cells), initial, dt


def _variable_coefficient_field(args: argparse.Namespace) -> dict[str, Any]:
    case, layout, initial, dt = _variable_field_case(args.cells)
    runtime, report, artifact = _run_case(case, layout, initial, dt=dt, steps=args.steps)
    potential = np.asarray(runtime.history_global("potential", 0))
    return {
        "workflow": "variable-coefficient-field",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "coefficient_range": [
            float(np.min(initial["load"][1])),
            float(np.max(initial["load"][1])),
        ],
        "potential_mean": float(np.mean(potential)),
        "potential_l2": float(np.sqrt(np.mean(potential**2))),
    }


def _component_source(manifest: Any) -> bytes:
    template = (HERE / "native_scalar_flux.cpp.in").read_text(encoding="ascii")
    replacements = {
        "@COMPONENT_ID@": json.dumps(manifest.component_id),
        "@SEMANTIC_DIGEST@": json.dumps(manifest.semantic_digest.token),
        "@MANIFEST_DIGEST@": json.dumps(manifest.manifest_digest.token),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    if any(marker in template for marker in replacements):
        raise RuntimeError("native component template contains an unresolved marker")
    return template.encode("ascii")


def _native_flux_source_component(work_dir: Path) -> Any:
    from pops import interfaces
    from pops.external import build_source_package_manifest, load
    from pops.model import ComponentManifest

    interface = interfaces.NumericalFlux
    manifest = ComponentManifest(
        uri="pops://examples.migration/scalar-upwind",
        component_type="numerical_flux",
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "state_components": 1,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={
            "variants": [
                {
                    "dimension": 2,
                    "scalar": "float64",
                    "device": "cpu",
                    "features": [],
                }
            ]
        },
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = _component_source(manifest)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_name = "native_scalar_flux.cpp"
    (work_dir / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={"scalar_upwind": manifest},
        payloads={source_name: ("source", source)},
    )
    package_path = work_dir / "native_scalar_flux.pops.json"
    package_path.write_text(json.dumps(package_data, indent=2), encoding="utf-8")
    return load(package_path).require("scalar_upwind", interface=interfaces.NumericalFlux)()


def _native_flux_component(work_dir: Path) -> Any:
    from pops.external import compile_component

    component = _native_flux_source_component(work_dir)
    return compile_component(
        component,
        include=os.environ.get("POPS_INCLUDE"),
        native_dimension=2,
    )


def _imported_primitive_case(
    cells: int, component: Any
) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Outflow
    from pops.mesh.boundaries import (
        BlockInterfaceSide,
        ConservativeInterface,
    )

    frame = _frame("imported_primitive_square")
    model = pops.Model("imported_primitive_advection", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    velocity = model.vector("velocity", frame=frame, components={axis: 1.0 for axis in frame.axes})
    flux = model.flux(
        "advection",
        frame=frame,
        state=state,
        components={axis: (u,) for axis in frame.axes},
        waves={axis: (1.0,) for axis in frame.axes},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("imported_native_primitive_case")
    left_block = case.block("left", model)
    right_block = case.block("right", model)
    left_state = left_block[state]
    right_state = right_block[state]
    finite_volume = FiniteVolume(
        flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    )
    boundaries = frame.boundaries

    def numerics(subject: Any) -> DiscretizationPlan:
        plan = DiscretizationPlan()
        plan.rates.add(rate, finite_volume)
        plan.boundaries.add(
            TransportBoundarySet({boundary: Outflow(state=subject) for boundary in boundaries.all})
        )
        return plan

    left_plan = numerics(left_state)
    right_plan = numerics(right_state)
    ConservativeInterface(
        "native_scalar_interface",
        left=BlockInterfaceSide(left_state, boundaries.x_max),
        right=BlockInterfaceSide(right_state, boundaries.x_min),
        numerical_flux=component,
        permutation=(0,),
        right_normal_translation=1.0,
    ).attach(left_plan, right_plan)
    case.numerics(left_plan, block=left_block)
    case.numerics(right_plan, block=right_block)

    program = pops.Program("imported_primitive_forward_euler")
    left_time = program.state(left_state)
    right_time = program.state(right_state)
    stage = program.stage("evaluate", c=0)
    left_rhs = program.value("left_rhs", rate(left_time.n), at=stage)
    right_rhs = program.value("right_rhs", rate(right_time.n), at=stage)
    dt = 1.0e-3
    program.commit(
        left_time.next,
        program.value("left_next", left_time.n + dt * left_rhs, at=left_time.next.point),
    )
    program.commit(
        right_time.next,
        program.value("right_next", right_time.n + dt * right_rhs, at=right_time.next.point),
    )
    program.step_strategy(FixedDt(dt))
    case.program(program)
    initial = {
        "left": np.ones((1, cells, cells), dtype=np.float64),
        "right": np.full((1, cells, cells), 3.0, dtype=np.float64),
    }
    return case, _layout(frame, cells, periodic=False), initial, dt


def _imported_native_primitive(args: argparse.Namespace) -> dict[str, Any]:
    component = _native_flux_component(args.work_dir / "native-component")
    case, layout, initial, dt = _imported_primitive_case(args.cells, component)
    initial_total = sum(float(np.mean(values)) for values in initial.values())
    runtime, report, artifact = _run_case(
        case,
        layout,
        initial,
        dt=dt,
        steps=args.steps,
        components=(component,),
    )
    final_total = sum(
        float(np.mean(np.asarray(runtime.state_global(block)))) for block in ("left", "right")
    )
    return {
        "workflow": "imported-native-primitive",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "component_id": component.component_id,
        "component_binary": component.binary_identity.token,
        "source_package": component.source_package.token,
        "left": _state_summary(runtime, "left"),
        "right": _state_summary(runtime, "right"),
        "paired_mean_change": final_total - initial_total,
    }


WORKFLOWS: dict[str, Workflow] = {
    "scalar-amr": Workflow(_scalar_amr, "canonical scalar advection with AMR and restart"),
    "field-transport": Workflow(
        lambda args: _field_consumer(args, transport=True),
        "transport driven by a consumed Poisson field",
    ),
    "euler-poisson": Workflow(
        lambda args: _field_consumer(args, transport=False),
        "Euler momentum source driven by a consumed Poisson field",
    ),
    "heterogeneous-interaction": Workflow(
        _heterogeneous_interaction,
        "one exchange over unequal (p,E) and (p,E,m) states",
    ),
    "explicit-diffusion": Workflow(
        lambda args: _diffusion(args, implicit=False),
        "periodic scalar heat equation with Forward Euler",
    ),
    "implicit-diffusion": Workflow(
        lambda args: _diffusion(args, implicit=True),
        "periodic scalar heat equation with an implicit diffusion stage",
    ),
    "variable-coefficient-field": Workflow(
        _variable_coefficient_field,
        "scalar field solve with a nonconstant state-provided coefficient",
    ),
    "imported-native-primitive": Workflow(
        _imported_native_primitive,
        "two-block advection using an authenticated imported native interface flux",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", nargs="?", choices=tuple(WORKFLOWS))
    parser.add_argument("--list", action="store_true", help="list workflow names and requirements")
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/migration-r1"),
        help="generated component and delegated-example output directory",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        for name, workflow in WORKFLOWS.items():
            print("%-29s %s" % (name, workflow.summary))
        return
    if args.workflow is None:
        raise SystemExit("choose a workflow or pass --list")
    if isinstance(args.cells, bool) or args.cells < 4:
        raise SystemExit("--cells must be an integer >= 4")
    if isinstance(args.steps, bool) or args.steps < 1:
        raise SystemExit("--steps must be an integer >= 1")
    args.work_dir = args.work_dir.resolve()
    result = WORKFLOWS[args.workflow].run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
