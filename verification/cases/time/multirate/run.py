"""TM-06 in-memory multirate BE plus a public two-species Case.

Fast species takes r BE substeps of size Δt/r. Slow species takes one BE
step of size Δt. r = 1 is identical to single-rate BE. The public Case
authors the uncoupled pair as local linear operators and a Program with a
child clock plus subcycle. pops.lib.time has IMEX/BDF, not a Multirate
factory. Optional native compile/bind/run. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.solvers import DenseLU
from pops.spaces import CellState
from pops.time import (
    Clock,
    FailRun,
    FixedDt,
    LocalLinear,
    Program,
    SampleAndHold,
    TimePoint,
)
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import (
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = 1
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def backward_euler_step(u, dt, lam) -> float:
    """One implicit Euler step of u' = -λ u: u / (1 + λ Δt)."""
    return float(u) / (1.0 + float(lam) * float(dt))


def single_rate_step(
    y,
    z,
    dt,
    *,
    lambda_f=_exact.LAMBDA_F,
    lambda_s=_exact.LAMBDA_S,
):
    """One BE step of size dt on both components."""
    return (
        backward_euler_step(y, dt, lambda_f),
        backward_euler_step(z, dt, lambda_s),
    )


def multirate_step(
    y,
    z,
    dt,
    r,
    *,
    lambda_f=_exact.LAMBDA_F,
    lambda_s=_exact.LAMBDA_S,
):
    """r fast BE substeps of dt/r; one slow BE step of size dt."""
    ratio = int(r)
    if ratio < 1:
        raise ValueError("substep ratio r must be a positive integer")
    y_new = float(y)
    sub_dt = float(dt) / float(ratio)
    for _ in range(ratio):
        y_new = backward_euler_step(y_new, sub_dt, lambda_f)
    z_new = backward_euler_step(z, dt, lambda_s)
    return y_new, z_new


def fast_error(
    r,
    dt=_exact.DT,
    *,
    y0=_exact.Y0,
    z0=_exact.Z0,
    lambda_f=_exact.LAMBDA_F,
    lambda_s=_exact.LAMBDA_S,
) -> float:
    """Absolute error of the fast component vs the exact exponential."""
    y_num, _ = multirate_step(
        y0, z0, dt, r, lambda_f=lambda_f, lambda_s=lambda_s
    )
    return abs(y_num - _exact.exact_y(dt, y0, lambda_f=lambda_f))


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt", "r")

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        dt: float,
        r: int,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt
        self.r = r


def _frame():
    return CartesianDomain("tm06_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _backward_euler(builder, state, operator, name, point):
    """One local-linear BE step of u' = L u at ``point``."""
    predictor = builder.value(f"{name}_rhs", state, at=point)
    linear = builder.value(
        f"{name}_map",
        operator(program=builder),
        at=point,
    )
    solved = builder.solve(
        LocalLinear(
            operator=builder.I - builder.dt * linear,
            rhs=predictor,
        ),
        solver=DenseLU(),
        name=f"{name}_solve",
    ).consume(action=FailRun())
    return builder.value(f"{name}_step", solved, at=point)


def _author(dt, r=1, *, n_cells: int = N_CELLS) -> _Authoring:
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    ratio = int(r)
    if ratio < 1:
        raise ValueError("substep ratio r must be a positive integer")
    step = float(dt)
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm06_multirate", frame=frame)
    state = model.state(
        "U",
        components=("y", "z"),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    y, z = state
    zero_flux = (0.0 * y, 0.0 * z)
    flux = model.flux(
        "inert_flux",
        frame=frame,
        state=state,
        components={x_axis: zero_flux},
        waves={x_axis: (0.0, 0.0)},
    )
    inert_rate = model.rate("inert_rate", equation=ddt(state) == -div(flux))
    lambda_f = float(_exact.LAMBDA_F)
    lambda_s = float(_exact.LAMBDA_S)
    # Fast/slow maps are uncoupled. The zero rows hold the other species.
    operator_f = model.operator(
        "operator_fast",
        returns=model.local_linear_operator(
            "operator_fast",
            on=state,
            matrix=((-lambda_f, 0.0), (0.0, 0.0)),
        ),
    )
    operator_s = model.operator(
        "operator_slow",
        returns=model.local_linear_operator(
            "operator_slow",
            on=state,
            matrix=((0.0, 0.0), (0.0, -lambda_s)),
        ),
    )
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
    case = pops.Case("tm06_multirate")
    pair = case.block("pair", model=model, states=(state,))
    instance = pair[state]
    case.numerics(numerics, block=pair)
    program = Program("tm06_multirate_be")
    temporal = program.state(instance)
    fast = Clock("fast", owner=program.owner_path)
    child = program.state(instance, clock=fast)
    on_fast = program.synchronize(
        temporal.n,
        at=TimePoint(fast),
        relation=SampleAndHold(),
        name="to_fast",
    )

    def _fast_tick(builder, value):
        return _backward_euler(
            builder, value, operator_f, "fast", child.next.point
        )

    advanced = program.subcycle(
        on_fast,
        clock=fast,
        within=program.clock,
        count=ratio,
        body_fn=_fast_tick,
        name="fast_ticks",
    )
    on_macro = program.synchronize(
        advanced,
        at=temporal.next.point,
        relation=SampleAndHold(),
        name="to_macro",
    )
    program.commit(
        temporal.next,
        _backward_euler(program, on_macro, operator_s, "slow", temporal.next.point),
    )
    program.step_strategy(FixedDt(step))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(
        case=case,
        instance=instance,
        frame=frame,
        n_cells=count,
        dt=step,
        r=ratio,
    )


def build_case(dt, r=1, *, n_cells: int = N_CELLS) -> pops.Case:
    """Author the 1-d periodic two-species Case. Does not compile or run."""
    return _author(dt, r, n_cells=n_cells).case


def resolve_plan(dt, r=1, *, n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(dt, r, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(dt, r=1, t_end=_exact.DT, *, n_cells: int = N_CELLS):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(dt, r, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.broadcast_to(
        np.asarray((_exact.Y0, _exact.Z0), dtype=np.float64)[:, np.newaxis],
        (2, authored.n_cells),
    ).copy()
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("pair"), dtype=np.float64)
    return np.ravel(field)
