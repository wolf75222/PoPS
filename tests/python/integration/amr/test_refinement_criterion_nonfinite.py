"""A refinement criterion that reads a non-finite sample rejects the level (N12).

The case advects two components with the method of the parity cases: ``u``, the Gaussian of
tutorial 09, and ``q = 1e200 x``, whose discrete gradient overflows. Two rules read that gradient
behind a left operand: ``(u > and_left) & (|grad q| > 1)`` and ``(u > or_left) | (|grad q| > 1)``.
The tag-graph VM, removed in lot 3c-2b, evaluated every leaf of a rule, so it rejected the level at
``bind`` whatever the left operands decided; written as functions of ``cell``, the same rules are
rejected with the message that it gave, both when the left operands leave the right ones to decide and when they decide alone
(``u > 10`` is false and ``u > 0`` true on every cell): a region evaluates every comparison before
``&`` and ``|`` join them, as IR4 evaluates the operands of an operation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pytest
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    Coarsen,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    PatchLayout,
    Tag,
)
from pops.analytic import exp, x, y
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer, gradient_norm
from pops.lib.initial import Analytic, BindArray
from pops.lib.time import SSPRK2
from pops.math import ValueExpr, ddt, div, sqrt, where
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.runtime_environment import runtime_environment_report
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every

pytestmark = [pytest.mark.kokkos, pytest.mark.regression]

N = 32
REFUSAL = "rejected a non-finite indicator sample"


def _case(*, region=None):
    edge = 1.0 if region is None else float(N)
    domain = Rectangle("unit_square", lower=(0.0, 0.0), upper=(edge, edge)).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("refinement_criterion_overflow", frame=frame)
    U = model.state("U", components=("u", "q"), representation=Conservative(),
                    space=CellState(frame=frame))
    flux = model.flux("advection_flux", frame=frame, state=U,
                      components={x_axis: tuple(1.0 * q for q in U),
                                  y_axis: tuple(0.25 * q for q in U)},
                      waves={x_axis: (1.0, 1.0), y_axis: (0.25, 0.25)})
    rate = model.rate("advection_rate", equation=ddt(U) == -div(flux))
    method = FiniteVolume(
        flux=flux, variables=variables.Conservative(U),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()), riemann=riemann.Rusanov())
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("refinement_criterion_nonfinite")
    tracer = case.block("tracer", model=model)
    state = tracer[U]
    boundaries = frame.boundaries
    numerics.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=state, value=(0.05, 0.05)),
        boundaries.x_max: Outflow(state=state),
        boundaries.y_min: Inflow(state=state, value=(0.05, 0.05)),
        boundaries.y_max: Outflow(state=state),
    }))
    case.numerics(numerics, block=tracer)
    program = SSPRK2(state, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))
    case.program(program)
    if region is None:
        bump = exp(-120.0 * ((x(frame) - 0.30) ** 2 + (y(frame) - 0.35) ** 2))
        case.initials.add(InitialCondition(
            state=state,
            value=Analytic(frame=frame, components=(0.05 + 0.95 * bump, 1.0e200 * x(frame))),
            projection=ConservativeCellAverage()))
    else:
        case.initials.add(InitialCondition(
            state=state, value=BindArray(), projection=ConservativeCellAverage()))
    params = {key: case.param(RuntimeParam("%s_threshold" % key, default=value))
              for key, value in {"and_left": 0.0, "or_left": 10.0, "right": 1.0}.items()}
    value = {key: case.value(handle) for key, handle in params.items()}
    u, q = ValueExpr(state)["u"], ValueExpr(state)["q"]
    stencil = method.amr_indicator_stencil(dimension=2)

    def steep(cell):
        return gradient_norm(cell, q, axes=frame.axes, stencil=stencil) > value["right"]

    rules = ((Tag(lambda cell: (cell.value(u) > value["and_left"]) & steep(cell)),
              Tag(lambda cell: (cell.value(u) > value["or_left"]) | steep(cell)))
             if region is None else region(u, q, x_axis, value["right"]))
    tagging = AMRTagging(rules=rules + (Buffer(cells=2 if region is None else 0),),
                         hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                         conflict_policy=ConflictPolicy.REFINE_WINS)
    transfer = AMRTransfer()
    transfer.state(state, StateTransfer())
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(N, N)),
                 hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)), tagging=tagging,
                 regrid=AMRRegrid(schedule=every(2, clock=program.clock)), transfer=transfer,
                 execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
                 **({} if region is None else {"patch_layout": PatchLayout(coarse_max_grid=8)}))
    return case, layout, params, state


@pytest.mark.compiler
@pytest.mark.native_loader
def test_a_non_finite_sample_rejects_the_level(isolated_native_cache) -> None:
    del isolated_native_cache
    case, layout, params, _state = _case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    # The left operands leave the gradient to decide, then decide alone without reading it.
    for and_left, or_left in ((0.0, 10.0), (10.0, 0.0)):
        with pytest.raises(RuntimeError, match=REFUSAL):
            pops.bind(artifact, params={params["and_left"]: and_left,
                                        params["or_left"]: or_left})


def _real():
    return {4: np.float32, 8: np.float64}[runtime_environment_report()["real_bytes"]]


def _precision_rules(operation, threshold):
    def rules(u, q, axis, bound):
        cutoff = bound if operation == "bound_gradient" else threshold

        def sample(cell):
            if operation == "value":
                return cell.value(q)
            difference = (
                cell.shift(axis, 1).value(q) - cell.shift(axis, -1).value(q)
            ) * 0.5 / cell.width(axis)
            return sqrt(difference * difference)

        return (
            Tag(lambda cell: ~(sample(cell) < cutoff)),
            Coarsen(lambda cell: sample(cell) < cutoff),
        )
    return rules


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize(("operation", "threshold"), (
    ("value", 1.2), ("value", 2.4), ("gradient", 2.4), ("bound_gradient", 2.4),
))
def test_criterion_keeps_the_native_threshold(operation, threshold, isolated_native_cache):
    real = _real()
    cutoff = real(threshold)
    triplet = np.array([np.nextafter(cutoff, real(0)), cutoff,
                        np.nextafter(cutoff, real(3))], dtype=real)
    centers = np.array([7, 15, 23])
    initial = np.zeros((2, N, N), dtype=np.float64)
    initial[0] = 3.0
    if operation == "value":
        initial[1][:, centers] = triplet
    else:
        # Each central stencil straddles a coarse patch boundary, with width one.
        initial[1][:, centers + 1] = real(2) * triplet
    case, layout, params, state = _case(region=_precision_rules(operation, threshold))
    plan = pops.resolve(pops.validate(case), layout=layout)
    artifact = pops.compile(plan)
    program_path = Path(artifact.program.so_path)
    before = program_path.stat()
    warm = pops.compile(plan)
    assert Path(warm.program.so_path) == program_path
    assert program_path.stat().st_mtime_ns == before.st_mtime_ns
    runtime = pops.bind(artifact, initial_values={state: initial},
                        params={params["right"]: threshold})
    coarse = np.asarray(runtime.block_level_state_global("tracer", 0)).reshape(initial.shape)
    np.testing.assert_array_equal(coarse, initial)
    q = coarse[1].astype(real)
    if operation == "value":
        measured = q[:, centers]
    else:
        derivative = real(0.5) * (q[:, centers + 1] - q[:, centers - 1])
        measured = np.sqrt(derivative * derivative)
    np.testing.assert_array_equal(measured, np.broadcast_to(triplet, measured.shape))
    requests = np.frombuffer(runtime._executor._tagging_requests(0), dtype=np.uint8).reshape(N, N)
    np.testing.assert_array_equal(requests[:, centers],
                                  np.broadcast_to([2, 0, 1], measured.shape))
    assert runtime.n_levels() == 2
    assert runtime.patch_boxes()
    pops.step(runtime, t_bound=0.01)
    assert runtime.macro_step() == 1


@pytest.mark.compiler
@pytest.mark.native_loader
def test_criterion_rejects_an_active_out_of_range_literal(isolated_native_cache):
    def rules(u, q, axis, bound):
        def active(cell):
            return where(cell.value(q) < 1.0e100, lambda: 2.0, lambda: 0.0)

        return (Tag(lambda cell: where(cell.value(q) > 0, lambda: active(cell), lambda: 0.0) > 1),)

    case, layout, _params, state = _case(region=rules)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.zeros((2, N, N), dtype=np.float64)
    initial[0] = 3
    initial[1] = -1
    assert pops.bind(artifact, initial_values={state: initial}).n_levels() == 1
    initial[1] = 1
    if _real() is np.float32:
        with pytest.raises(RuntimeError, match=REFUSAL):
            pops.bind(artifact, initial_values={state: initial})
    else:
        assert pops.bind(artifact, initial_values={state: initial}).n_levels() == 2
