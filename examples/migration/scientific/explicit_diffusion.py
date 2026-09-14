"""Heat equation with a public spatial discretization and explicit user time method."""

from __future__ import annotations

import argparse
import math
from typing import Any
import numpy as np
import pops
from pops.math import ddt, div, grad
from pops.numerics import Diffusion, DiscretizationPlan
from pops.time import FixedDt
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from .runtime import _cell_centers, _state_summary, _execution_resources


def build_case(cells: int) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = Rectangle("diffusion_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("scalar_diffusion", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    flux = model.diffusive_flux("conduction", state=state, value=0.1 * grad(u))
    rate = model.rate("heat", equation=ddt(state) == div(flux))
    case = pops.Case("explicit_diffusion")
    block = case.block("heat", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    dt = 0.05 / (0.1 * cells * cells)
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
    return (
        case,
        Uniform(
            CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes))
        ),
        {"heat": initial[None]},
        dt,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    case, layout, initial, dt = build_case(args.cells)
    initial_mass = float(np.mean(initial["heat"]))
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, initial_state=initial, resources=_execution_resources(artifact))
    report = pops.run(runtime, t_end=args.steps * dt, max_steps=args.steps, console=False)
    final = np.asarray(runtime.state_global("heat"))
    return {
        "workflow": "explicit-diffusion",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "state": _state_summary(runtime, "heat"),
        "mean_change": float(np.mean(final)) - initial_mass,
    }
