"""Advection authored in Python with an authenticated external multiplication.

Two periodic blocks use different profiles and the same physical law. The native
library knows only scalar multiplication; Python owns velocity, conservation law,
Rusanov discretization, and the explicit temporal graph.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import FieldSpace, Signature
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain
from pops.native_components import PreparedNativeComponent
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.time import FixedDt

from .runtime import _cell_centers, _execution_resources, _state_summary


def prepare_arithmetic(work_dir: Path) -> PreparedNativeComponent:
    work_dir.mkdir(parents=True, exist_ok=True)
    header = Path(__file__).with_name("arithmetic.hpp")
    (work_dir / header.name).write_bytes(header.read_bytes())
    return PreparedNativeComponent.header_only(
        "migration.arithmetic",
        include_root=work_dir,
        entry_headers=(header.name,),
    )


def build_case(cells: int, component: PreparedNativeComponent, *, imported: bool = True):
    domain = Rectangle("native_arithmetic_square", lower=(0, 0), upper=(1, 1))
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("python_advection", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    scalar = FieldSpace("arithmetic_scalar", components=("value",))
    multiply = NativeFunction(
        component,
        "migration_arithmetic::multiply",
        Signature((state.space, scalar), scalar),
        execution_domains=("host", "device"),
        reads=((0, 0), (1, 0)),
        domains=(NativeInputDomain(0, 0), NativeInputDomain(1, 0)),
        derivatives=(
            NativeDerivative("exact", "migration_arithmetic::jacobian"),
            NativeDerivative("approximate", "migration_arithmetic::approximate"),
        ),
    )

    # Physical law: d_t u + d_x(a_x u) + d_y(a_y u) = 0.
    # The pure-expression variant is the independent execution control.
    ax, ay = 1.0, -0.25
    flux_x = multiply((u,), (ax,)).value[0] if imported else ax * u
    flux_y = multiply((u,), (ay,)).value[0] if imported else ay * u
    flux = model.flux(
        "advection",
        frame=frame,
        state=state,
        components={x_axis: (flux_x,), y_axis: (flux_y,)},
        waves={x_axis: (ax,), y_axis: (ay,)},
    )
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    case = pops.Case("imported_native_primitive_case")
    left = case.block("left", model)
    right = case.block("right", model)
    for block in (left, right):
        numerical_method = FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        )
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, numerical_method)
        case.numerics(numerics, block=block)

    program = pops.Program("user_forward_euler")
    stage = program.stage("evaluate", c=0)
    pending = []
    for block in (left, right):
        q = program.state(block[state])
        rhs = program.value(block.local_id + "_rhs", rate(q.n), at=stage)
        pending.append((block, q, rhs))
    for block, q, rhs in pending:
        next_value = program.value(
            block.local_id + "_advanced",
            q.n + program.dt * rhs,
            at=q.next.point,
        )
        program.commit(q.next, next_value)
    dt = 0.001
    program.step_strategy(FixedDt(dt))
    case.program(program)
    x, y = _cell_centers(cells)
    initial = {
        "left": (1 + 0.2 * np.sin(2 * np.pi * x))[None],
        "right": (3 + 0.3 * np.cos(2 * np.pi * y))[None],
    }
    layout = Uniform(
        CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes))
    )
    return case, layout, initial, dt


def run(args: argparse.Namespace) -> dict[str, Any]:
    component = prepare_arithmetic(args.work_dir / "native-arithmetic")
    case, layout, initial, dt = build_case(args.cells, component)
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, initial_state=initial, resources=_execution_resources(artifact))
    report = pops.run(runtime, t_end=args.steps * dt, max_steps=args.steps, console=False)
    final_total = sum(float(np.mean(runtime.state_global(block))) for block in initial)
    initial_total = sum(float(np.mean(values)) for values in initial.values())
    binaries = {
        block.name: hashlib.sha256(Path(block.model.so_path).read_bytes()).hexdigest()
        for block in artifact.blocks
    }
    return {
        "workflow": "imported-native-primitive",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "component_id": component.component_id,
        "component_binary": binaries["left"],
        "component_binaries": binaries,
        "source_package": component.manifest_sha256,
        "left": _state_summary(runtime, "left"),
        "right": _state_summary(runtime, "right"),
        "paired_mean_change": final_total - initial_total,
    }
