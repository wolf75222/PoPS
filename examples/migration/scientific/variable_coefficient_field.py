"""Poisson solve with a nonconstant coefficient supplied by state."""

from __future__ import annotations

import argparse
import math
from typing import Any
import numpy as np
import pops
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import DivCoeffGrad, ddt, div
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from .runtime import _cell_centers, _execution_resources


def build_case(cells: int) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = Rectangle(
        "variable_coefficient_field_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    load_model = pops.Model("field_load", frame=frame)
    reference_model = pops.Model("field_reference", frame=frame)
    load_state = load_model.state("U", components=("load", "coefficient"))
    reference_state = reference_model.state("U", components=("load", "coefficient"))
    authored = []
    for model, state in ((load_model, load_state), (reference_model, reference_state)):
        stationary = model.flux(
            "stationary",
            frame=frame,
            state=state,
            components={axis: tuple((0 * value for value in state)) for axis in frame.axes},
            waves={axis: tuple((0 * value for value in state)) for axis in frame.axes},
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
    blocks = (case.block("load", load_model), case.block("reference", reference_model))
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
                potential, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())
            ),
        ),
        gauge=SharedMeanGauge((potential,)),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=CG(max_iter=4000, rel_tol=1e-11, abs_tol=1e-12),
        ),
    )
    program = pops.Program("variable_field_solve")
    times = (program.state(blocks[0][load_state]), program.state(blocks[1][reference_state]))
    point = program.stage("solve", c=0)
    values = {blocks[0][load_state]: times[0].n, blocks[1][reference_state]: times[1].n}
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
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, initial_state=initial, resources=_execution_resources(artifact))
    report = pops.run(runtime, t_end=args.steps * dt, max_steps=args.steps, console=False)
    potential = np.asarray(runtime.history_global("potential", 0))
    return {
        "workflow": "variable-coefficient-field",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "coefficient_range": [float(np.min(initial["load"][1])), float(np.max(initial["load"][1]))],
        "potential_mean": float(np.mean(potential)),
        "potential_l2": float(np.sqrt(np.mean(potential**2))),
    }
