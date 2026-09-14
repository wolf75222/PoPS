"""Conservative momentum and energy exchange between unequal species."""

from __future__ import annotations

import argparse
from typing import Any
import numpy as np
import pops
from pops.math import ddt
from pops.numerics import DiscretizationPlan, JointEvaluation
from pops.time import FixedDt
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from .runtime import _state_summary, _execution_resources


def build_case(cells: int) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    frame = Rectangle("heterogeneous_interaction_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("heterogeneous_exchange", frame=frame)
    left = model.species("left", state=("p", "E"))
    right = model.species("right", state=("p", "E", "m"))
    force = right[0] / right[2] - left[0]
    power = (left[0] + right[0] / right[2]) * force / 2
    exchange = model.interaction(
        "exchange", outputs={left: (force, power), right: (-force, -power, 0)}
    )
    left_rate = model.rate("left_balance", equation=ddt(left) == exchange[left])
    right_rate = model.rate("right_balance", equation=ddt(right) == exchange[right])
    case = pops.Case("heterogeneous_interaction_case")
    left_block = case.block("left", model, states=(left,))
    right_block = case.block("right", model, states=(right,))
    for block, state, rate in ((left_block, left, left_rate), (right_block, right, right_rate)):
        plan = DiscretizationPlan()
        plan.rates.add(rate, JointEvaluation(state))
        case.numerics(plan, block=block)
    program = pops.Program("heterogeneous_exchange_step")
    left_time = program.state(left_block[left])
    right_time = program.state(right_block[right])
    left_rhs = left_rate(left_time.n, right_time.n)
    right_rhs = right_rate(right_time.n, left_time.n)
    dt = 0.001
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
    before = {
        component: float(np.mean(initial["left"][component] + initial["right"][component]))
        for component in (0, 1)
    }
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, initial_state=initial, resources=_execution_resources(artifact))
    report = pops.run(runtime, t_end=args.steps * dt, max_steps=args.steps, console=False)
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
