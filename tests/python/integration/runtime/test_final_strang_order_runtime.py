"""Runtime order proof for final typed Lie and Strang Program factories."""

from __future__ import annotations

import math
import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import ExternalTimeGrid


CELLS = 4
FINAL_TIME = 1.0
COARSE_STEP = 0.2
FIRST_RATE = 1.0
SECOND_RATE = -2.0

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _authoring(factory):
    frame = Rectangle(
        "split_unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = pops.Model("noncommuting_shear_flows", frame=frame)
    state = model.state(
        "U",
        components=("first_amplitude", "second_amplitude"),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    first_amplitude, second_amplitude = state
    zero_flux = (0.0 * first_amplitude, 0.0 * second_amplitude)
    x_axis, y_axis = frame.axes
    flux = model.flux(
        "inert_transport",
        frame=frame,
        state=state,
        components={x_axis: zero_flux, y_axis: zero_flux},
        waves={x_axis: (0.0, 0.0), y_axis: (0.0, 0.0)},
    )
    inert_rate = model.rate("inert_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        inert_rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    first_operator = model.operator(
        "upper_shear",
        returns=model.local_linear_operator(
            "upper_shear",
            on=state,
            matrix=((0.0, FIRST_RATE), (0.0, 0.0)),
        ),
    )
    second_operator = model.operator(
        "lower_shear",
        returns=model.local_linear_operator(
            "lower_shear",
            on=state,
            matrix=((0.0, 0.0), (SECOND_RATE, 0.0)),
        ),
    )
    case = pops.Case("noncommuting_split_%s" % factory.__name__.lower())
    block = case.block("oscillator", model=model)
    case.numerics(numerics, block=block)
    block_state = block[state]

    def subflow(operator, label):
        def build(program, current, fraction, *, at):
            linear = program.value(
                "%s_map" % label,
                operator(program=program),
                at=current.point,
            )
            rate = program.apply(linear, current)
            return program.value(
                "%s_flow" % label,
                current + (fraction * program.dt) * rate,
                at=at,
            )

        return build

    program = factory(
        block_state,
        first=subflow(first_operator, "upper"),
        second=subflow(second_operator, "lower"),
    )
    program.step_strategy(ExternalTimeGrid("time_grid"))
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame,
        cells=(CELLS, CELLS),
        periodic=PeriodicAxes(frame.axes),
    ))
    return case, layout


def _compile(factory):
    case, layout = _authoring(factory)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return pops.compile(resolved)


def _run(artifact, step):
    count = int(round(FINAL_TIME / step))
    grid = tuple(index * step for index in range(count + 1))
    initial_vector = np.array((1.0, 0.3), dtype=np.float64)
    initial = np.broadcast_to(
        initial_vector[:, None, None], (2, CELLS, CELLS)).copy()
    runtime = pops.bind(artifact, initial_state={"oscillator": initial})
    report = pops.run(
        runtime,
        t_end=FINAL_TIME,
        max_steps=count,
        time_grid=grid,
    )
    assert report.accepted_steps == count
    state = np.asarray(
        runtime.state_global("oscillator"), dtype=np.float64).reshape(
            2, CELLS, CELLS)
    assert np.all(np.isfinite(state))
    np.testing.assert_allclose(
        state,
        np.broadcast_to(state[:, :1, :1], state.shape),
        rtol=0.0,
        atol=2.0e-14,
    )
    return state[:, 0, 0]


def _analytic_solution():
    initial = np.array((1.0, 0.3), dtype=np.float64)
    frequency = math.sqrt(-FIRST_RATE * SECOND_RATE)
    generator = np.array(
        ((0.0, FIRST_RATE), (SECOND_RATE, 0.0)), dtype=np.float64)
    return (
        math.cos(frequency * FINAL_TIME) * initial
        + math.sin(frequency * FINAL_TIME) / frequency * (generator @ initial)
    )


def _orders(errors):
    return tuple(
        math.log2(coarse / fine)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    )


def test_strang_is_second_order_while_lie_is_first_order_after_real_runs(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    artifacts = {
        "strang": _compile(libtime.Strang),
        "lie": _compile(libtime.Lie),
    }
    exact = _analytic_solution()
    steps = (COARSE_STEP, COARSE_STEP / 2.0, COARSE_STEP / 4.0)
    errors = {
        name: tuple(
            float(np.linalg.norm(_run(artifact, step) - exact, ord=2))
            for step in steps
        )
        for name, artifact in artifacts.items()
    }
    strang_orders = _orders(errors["strang"])
    lie_orders = _orders(errors["lie"])

    assert all(1.75 < order < 2.25 for order in strang_orders), (
        errors["strang"], strang_orders)
    assert all(0.80 < order < 1.20 for order in lie_orders), (
        errors["lie"], lie_orders)
    assert errors["strang"][-1] < errors["lie"][-1]
