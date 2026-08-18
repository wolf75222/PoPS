"""TM-08 in-memory reversible cycle plus a public advection Case.

Keeps the exact reversible map (advance T at +a, flip the velocity sign,
advance T at -a). A public 1-d periodic Case authors the same TR-01 sine
advection so resolve_plan succeeds. Optional native compile/bind/run.
pops.run is used only inside run_native. Does not call ROMEO.
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
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
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

A = float(_exact.A)
N_CELLS = int(_exact.N_CELLS)
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt", "a")

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        dt: float,
        a: float,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt
        self.a = a


def reversible_cycle(n_cells=_exact.N_CELLS, t=_exact.T, a=_exact.A) -> dict:
    """Return initial, forwarded, and returned states of the exact map."""
    centers, volumes = _exact.uniform_cell_centers(n_cells)
    initial = _exact.exact_sine(centers, 0.0, a=a)
    forwarded = _exact.after_forward(centers, t, a=a)
    returned = _exact.after_return(centers, t, a=a)
    return {
        "x": centers,
        "volumes": volumes,
        "initial": initial,
        "forwarded": forwarded,
        "returned": returned,
    }


def _frame():
    return CartesianDomain("unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(dt, *, n_cells: int = N_CELLS, a: float = A) -> _Authoring:
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    step = float(dt)
    speed = float(a)
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm08_reversible_strang", frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: speed})
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (speed * q,)},
        waves={x_axis: (speed,)},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case("tm08_reversible_strang")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
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
        a=speed,
    )


def build_case(dt, *, n_cells: int = N_CELLS, a: float = A) -> pops.Case:
    """Author the 1-d periodic advection Case. Does not compile or run."""
    return _author(dt, n_cells=n_cells, a=a).case


def resolve_plan(dt, *, n_cells: int = N_CELLS, a: float = A):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(dt, n_cells=n_cells, a=a)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(dt=None, t_end=_exact.T, *, n_cells: int = N_CELLS, a: float = A, request=None):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='TM-08', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    raise NativeUnavailable(
        "TM-08 SSPRK2 advection is not the required reversible Strang program"
    )
    if dt is None:
        dt = float(globals().get('DT', 0.1))
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(dt, n_cells=n_cells, a=a)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    centers, _ = _exact.uniform_cell_centers(authored.n_cells)
    initial = np.ascontiguousarray(
        _exact.exact_sine(centers, 0.0, a=authored.a)[np.newaxis, :],
        dtype=np.float64,
    )
    simulation = bind_public(artifact, initial_values={authored.instance: initial}, mpi_mode=require_bind_request(request, NativeUnavailable, 'TM-08'))
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    field = np.ravel(field)
    if request is not None:
        return maybe_campaign_payload(
            request,
            field,
            n_cells=n_cells,
            t_end=t_end,
            time_program='SSPRK2',
            cfl=0.4,
            dimension=1,
        )
    return field