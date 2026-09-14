"""Transport driven by a consumed Poisson field."""

from __future__ import annotations

import argparse
import math
from typing import Any
import numpy as np
import pops
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import ddt, div, grad, laplacian
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from .runtime import _cell_centers, _state_summary, _execution_resources


def build_case(cells: int) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = Rectangle("field_consumer_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = pops.Model("field_consumer", frame=frame)
    names = ("rho", "driver")
    state = model.state("U", components=names)
    rho = state[0]
    potential = model.field("potential")
    gradient = model.vector(
        "gradient", frame=frame, components={x_axis: grad(potential).x, y_axis: grad(potential).y}
    )
    driver = state[1]
    flux = model.flux(
        "field_transport",
        frame=frame,
        state=state,
        components={x_axis: (0 * rho, 0 * driver), y_axis: (gradient.y * rho, 0 * driver)},
        waves={x_axis: (0 * rho, 0 * driver), y_axis: (gradient.y, 0 * driver)},
    )
    physical_rate = model.rate("transport", equation=ddt(state) == -div(flux))
    problem = FieldProblem(
        "Poisson",
        unknowns=(potential,),
        equations=(-laplacian(potential) == driver - 1,),
        boundaries=(
            FieldBoundary(
                potential, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())
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
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=CG(max_iter=4000, rel_tol=1e-11, abs_tol=1e-12),
        ),
    )
    program = pops.Program("field_consumer_step")
    current = program.state(fluid_block[state])
    point = program.stage("evaluate", c=0)
    solve_inputs = {fluid_block[state]: current.n}
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
        states=None,
    )
    rhs = physical_rate(current.n, context)
    program.store_history("gradient", solved_gradient, depth=1)
    dt = 0.125
    program.commit(
        current.next, program.value("advanced", current.n + dt * rhs, at=current.next.point)
    )
    program.step_strategy(FixedDt(dt))
    case.program(program)
    x, y = _cell_centers(cells)
    amplitude = 0.1
    rho_values = 1 + 0.2 * np.sin(2 * math.pi * y)
    driver_values = 1 + amplitude * np.cos(2 * math.pi * y)
    fluid = np.stack((rho_values, driver_values))
    initial = {"fluid": fluid}
    return (
        case,
        Uniform(
            CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes))
        ),
        initial,
        dt,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    case, layout, initial, dt = build_case(args.cells)
    before_means = np.mean(initial["fluid"], axis=(1, 2))
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, initial_state=initial, resources=_execution_resources(artifact))
    report = pops.run(runtime, t_end=args.steps * dt, max_steps=args.steps, console=False)
    after = np.asarray(runtime.state_global("fluid")).reshape(initial["fluid"].shape)
    gradient_values = np.asarray(runtime.history_global("gradient", 0))
    return {
        "workflow": "field-transport",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "state": _state_summary(runtime, "fluid"),
        "component_mean_change": (np.mean(after, axis=(1, 2)) - before_means).tolist(),
        "gradient_l2": float(np.sqrt(np.mean(gradient_values**2))),
    }
