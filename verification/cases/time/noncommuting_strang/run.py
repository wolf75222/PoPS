"""TM-02 in-memory exact Lie/Strang plus official ``pops.lib.time.Strang``.

In-memory subflows stay the exact 0-d matrix exponentials (Lie order 1,
Strang order 2). The public Case is the official non-commuting split:
two ``local_linear_operator`` maps composed by ``pops.lib.time.Strang``
or ``Lie``, with inert (zero) flux so FiniteVolume is not inside the
split. That matches
``tests/python/integration/amr/test_amr_refined_strang_program.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pops
import pops.lib.time as libtime
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.native_evidence import NULL_COUPLING, apply_campaign_request, maybe_campaign_payload, require_bind_request
from verification.pops_verify.case_authoring import (
    attach_case_diagnostics,
    bind_public,
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = 16
MAX_STEPS = 100_000
SPLITTERS = ("lie", "strang")


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def lie_step(state, dt):
    """Godunov/Lie step: Φ_B(dt) ∘ Φ_A(dt)."""
    return _exact.flow_B(_exact.flow_A(state, dt), dt)


def strang_step(state, dt):
    """Strang step: Φ_B(dt/2) ∘ Φ_A(dt) ∘ Φ_B(dt/2)."""
    half = 0.5 * float(dt)
    return _exact.flow_B(_exact.flow_A(_exact.flow_B(state, half), dt), half)


def advance(stepper, state, dt, t_end):
    """Apply stepper n = t_end/dt times. dt must tile t_end."""
    current = np.asarray(state, dtype=np.float64)
    steps = int(round(float(t_end) / float(dt)))
    if not np.isclose(steps * float(dt), float(t_end)):
        raise ValueError("dt must tile t_end")
    for _ in range(steps):
        current = np.asarray(stepper(current, dt), dtype=np.float64)
    return current


def splitting_error(stepper, dt, *, t_end=_exact.T_END, u0=_exact.U0) -> float:
    """L∞ error of the split flow against exp((A+B) t_end) u0."""
    numerical = advance(stepper, u0, dt, t_end)
    exact = _exact.exact_state(t_end, u0)
    return float(np.max(np.abs(numerical - exact)))


def error_series(stepper, dts=None, *, t_end=_exact.T_END, u0=_exact.U0):
    """L∞ errors of stepper on the manufactured Δt series."""
    steps = _exact.DT_SERIES if dts is None else tuple(dts)
    return tuple(splitting_error(stepper, step, t_end=t_end, u0=u0) for step in steps)


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int, dt: float) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt


def _frame():
    return CartesianDomain("tm02_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _subflow(operator, label):
    """Official Strang/Lie subflow: Euler step of a local linear map."""

    def build(program, current, fraction, *, at):
        linear = program.value(
            "%s_map" % label,
            operator(program=program),
            at=current.point,
        )
        change = program.apply(linear, current)
        return program.value(
            "%s_flow" % label,
            current + (fraction * program.dt) * change,
            at=at,
        )

    return build


def _uniform_initial(n_cells: int) -> np.ndarray:
    count = int(n_cells)
    return np.ascontiguousarray(
        np.stack(
            (
                np.full(count, float(_exact.U0[0]), dtype=np.float64),
                np.full(count, float(_exact.U0[1]), dtype=np.float64),
            )
        ),
        dtype=np.float64,
    )


def _author(
    dt,
    *,
    n_cells: int = N_CELLS,
    method: str = "strang",
    reconstruction_kind: str = "first_order",
) -> _Authoring:
    del reconstruction_kind
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    step = float(dt)
    kind = str(method)
    if kind not in SPLITTERS:
        raise ValueError(f"unknown splitter {method!r}")
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm02_noncommuting_strang", frame=frame)
    state = model.state(
        "U",
        components=("q0", "q1"),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    q0, q1 = state
    zero_flux = (0.0 * q0, 0.0 * q1)
    flux = model.flux(
        "inert_split_flux",
        frame=frame,
        state=state,
        components={x_axis: zero_flux},
        waves={x_axis: (0.0, 0.0)},
    )
    rate = model.rate("inert_split_rate", equation=ddt(state) == -div(flux))
    operator_a = model.operator(
        "operator_A",
        returns=model.local_linear_operator(
            "operator_A",
            on=state,
            matrix=((float(_exact.A1), 0.0), (0.0, float(_exact.A2))),
        ),
    )
    operator_b = model.operator(
        "operator_B",
        returns=model.local_linear_operator(
            "operator_B",
            on=state,
            matrix=(
                (-float(_exact.NU), float(_exact.NU)),
                (float(_exact.NU), -float(_exact.NU)),
            ),
        ),
    )
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
    case = pops.Case("tm02_noncommuting_strang")
    oscillator = case.block("oscillator", model=model, states=(state,))
    instance = oscillator[state]
    case.numerics(numerics, block=oscillator)
    first = _subflow(operator_a, "A")
    second = _subflow(operator_b, "B")
    if kind == "strang":
        program = libtime.Strang(instance, first=first, second=second)
    else:
        program = libtime.Lie(instance, first=first, second=second)
    program.step_strategy(FixedDt(step))
    case.program(program)
    attach_case_diagnostics(case, oscillator, program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count, dt=step)


def build_case(
    dt,
    *,
    n_cells: int = N_CELLS,
    method: str = "strang",
    reconstruction_kind: str = "first_order",
) -> pops.Case:
    """Author the official Strang/Lie Case. Does not compile or run."""
    return _author(
        dt, n_cells=n_cells, method=method, reconstruction_kind=reconstruction_kind
    ).case


def resolve_plan(
    dt,
    *,
    n_cells: int = N_CELLS,
    method: str = "strang",
    reconstruction_kind: str = "first_order",
):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(
        dt, n_cells=n_cells, method=method, reconstruction_kind=reconstruction_kind
    )
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(
    dt=None,
    t_end=_exact.T_END,
    *,
    n_cells: int = N_CELLS,
    method: str = "strang",
    reconstruction_kind: str = "first_order",
    request=None,
):
    """Compile, bind, and run official Strang/Lie. Raises NativeUnavailable without a compiler."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='TM-02', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    if dt is None:
        dt = float(globals().get('DT', 0.1))
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(
        dt, n_cells=n_cells, method=method, reconstruction_kind=reconstruction_kind
    )
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    simulation = bind_public(artifact, initial_values={authored.instance: _uniform_initial(authored.n_cells)}, mpi_mode=require_bind_request(request, NativeUnavailable, 'TM-02'))
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("oscillator"), dtype=np.float64)
    field = np.reshape(field, (2, authored.n_cells))
    if request is not None:
        return maybe_campaign_payload(
            request,
            field,
            artifact=artifact,
            simulation=simulation,
            coupling=dict(NULL_COUPLING),
            n_cells=n_cells,
            t_end=t_end,
            time_program="Strang" if str(method) == "strang" else "Lie",
            cfl=0.0,
            dimension=1,
        )
    return field


def run_order_campaign(
    dts=None,
    *,
    t_end=_exact.T_END,
    n_cells: int = N_CELLS,
    method: str = "strang",
    reconstruction_kind: str = "first_order",
):
    """Native Δt series vs tiled ``exact_state``. Subflows are official Euler maps."""
    del reconstruction_kind
    steps = _exact.DT_SERIES if dts is None else tuple(float(dt) for dt in dts)
    volumes = np.full(int(n_cells), 1.0 / float(n_cells), dtype=np.float64)
    oracle = np.full(int(n_cells), float(_exact.exact_state(t_end)[0]), dtype=np.float64)
    errors = []
    for step in steps:
        field = run_native(step, t_end=t_end, n_cells=n_cells, method=method)
        err = reference_errors(field[0], oracle, volumes)
        errors.append(float(err.linf))
    return {
        "method": method,
        "n_cells": int(n_cells),
        "t_end": float(t_end),
        "dts": steps,
        "linf": tuple(errors),
    }
