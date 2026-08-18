"""Public 1-d periodic scalar Case for TM-05 AP limit.

Keeps the in-memory implicit backward-Euler and explicit Euler helpers. The
public Case authors the stiff relaxation as a field-independent local linear
operator L = -1/ε and uses pops.lib.time.IMEX (Euler tableau, DenseLU). An
inert zero flux satisfies FiniteVolume; the AP map is the implicit source, not
a private stiff stepper. Optional native compile/bind/run. ``pops.run`` is
used only inside ``run_native``. Does not call ROMEO.
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
from pops.lib.time import IMEX
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
from verification.pops_verify.native_evidence import apply_campaign_request, maybe_campaign_payload, require_bind_request
from verification.pops_verify.case_authoring import (
    bind_public,
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = 1
MAX_STEPS = 100_000
DEFAULT_EPS = float(_exact.EPS_SWEEP[-1])


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt", "eps")

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        dt: float,
        eps: float,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt
        self.eps = eps


def implicit_step(y, dt, eps, *, g=_exact.G, f=_exact.F):
    """One backward-Euler step of dy/dt = -(y-g)/ε + f."""
    stiffness = float(dt) / float(eps)
    return (float(y) + stiffness * float(g) + float(dt) * float(f)) / (1.0 + stiffness)


def explicit_step(y, dt, eps, *, g=_exact.G, f=_exact.F):
    """One forward-Euler step of dy/dt = -(y-g)/ε + f."""
    return float(y) + float(dt) * (-(float(y) - float(g)) / float(eps) + float(f))


def _frame():
    return CartesianDomain("tm05_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(dt, *, eps: float = DEFAULT_EPS, n_cells: int = N_CELLS) -> _Authoring:
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    step = float(dt)
    scale = float(eps)
    if step <= 0.0:
        raise ValueError("dt must be positive")
    if scale <= 0.0:
        raise ValueError("eps must be positive")
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm05_ap_limit", frame=frame)
    state = model.state(
        "U",
        components=("y",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (y,) = state
    flux = model.flux(
        "inert_flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * y,)},
        waves={x_axis: (0.0,)},
    )
    inert_rate = model.rate("inert_rate", equation=ddt(state) == -div(flux))
    relaxation = model.operator(
        "relaxation",
        returns=model.local_linear_operator(
            "relaxation",
            on=state,
            matrix=((-1.0 / scale,),),
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
    case = pops.Case("tm05_ap_limit")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    # IMEX Euler + L = -1/ε is the backward-Euler map y / (1 + Δt/ε).
    program = IMEX(
        instance,
        explicit_operator=inert_rate,
        implicit_operator=relaxation,
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
        eps=scale,
    )


def build_case(dt, *, eps: float = DEFAULT_EPS, n_cells: int = N_CELLS) -> pops.Case:
    """Author the 1-d periodic AP relaxation Case. Does not compile or run."""
    return _author(dt, eps=eps, n_cells=n_cells).case


def resolve_plan(dt, *, eps: float = DEFAULT_EPS, n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(dt, eps=eps, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(dt=None, t_end=None, *, eps: float = DEFAULT_EPS, n_cells: int = N_CELLS, request=None):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='TM-05', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    if dt is None:
        dt = float(globals().get('DT', 0.1))
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(dt, eps=eps, n_cells=n_cells)
    horizon = authored.dt if t_end is None else float(t_end)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.full((1, authored.n_cells), float(_exact.Y0), dtype=np.float64)
    simulation = bind_public(artifact, initial_values={authored.instance: initial}, mpi_mode=require_bind_request(request, NativeUnavailable, 'TM-05'))
    pops.run(simulation, t_end=horizon, max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    field = np.ravel(field)
    if request is not None:
        return maybe_campaign_payload(
            request,
            field,
            n_cells=n_cells,
            t_end=horizon,
            time_program='IMEX',
            cfl=0.0,
            dimension=1,
        )
    return field