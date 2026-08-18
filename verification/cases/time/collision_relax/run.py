"""Public 1-d periodic Cases for TM-03 collision relaxation.

Keeps the in-memory exact map ``relax`` for the scalar BGK-style contract
and an in-memory two-species map ``relax_two_species``. The scalar Case
authors a cell-local source toward the manufactured IC barycenter (a
constant). The two-species Case authors the coupled drag
ρ1 dq0/dt = K(q1-q0), ρ2 dq1/dt = K(q0-q1) as a public source. An inert
zero flux satisfies FiniteVolume; the collision is the source, not a
private operator. Optional native compile/bind/run. ``pops.run`` is used
only inside ``run_native`` and ``run_native_two_species``. Does not call
ROMEO.
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

NU = float(_exact.NU)
N_CELLS = int(_exact.N_CELLS)
MAX_STEPS = 100_000
# Defaults give λ = K(1/ρ1 + 1/ρ2) = 2, matching exact.NU.
K = 1.0
RHO1 = 1.0
RHO2 = 1.0
TWO_SPECIES_N_CELLS = 1
Q0_INIT = 1.0
Q1_INIT = 0.0


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt", "u_bar")

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        dt: float,
        u_bar: float,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt
        self.u_bar = u_bar


def _frame():
    return CartesianDomain("unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _ic_barycenter(n_cells: int) -> float:
    centers, volumes = _exact.uniform_cell_centers(n_cells)
    return _exact.barycenter(_exact.initial_field(centers), volumes)


def relax(u0, t, *, nu, volumes=None):
    """Return the exact relaxation of u0 at time t toward its barycenter."""
    u_bar = _exact.barycenter(u0, volumes)
    return _exact.exact_relax(u0, t, nu=nu, u_bar=u_bar)


def collision_lambda(k: float = K, rho1: float = RHO1, rho2: float = RHO2) -> float:
    """Return λ = K(1/ρ1 + 1/ρ2) for the two-species slip decay."""
    density1 = float(rho1)
    density2 = float(rho2)
    if density1 == 0.0 or density2 == 0.0:
        raise ValueError("rho1 and rho2 must be nonzero")
    return float(k) * (1.0 / density1 + 1.0 / density2)


def two_species_barycenter(q0, q1, *, rho1: float = RHO1, rho2: float = RHO2):
    """Return V = (ρ1 q0 + ρ2 q1) / (ρ1 + ρ2)."""
    density1 = float(rho1)
    density2 = float(rho2)
    mass = density1 + density2
    if mass == 0.0:
        raise ValueError("rho1 + rho2 must be nonzero")
    left = density1 * np.asarray(q0, dtype=np.float64)
    right = density2 * np.asarray(q1, dtype=np.float64)
    return (left + right) / mass


def relax_two_species(
    q0,
    q1,
    t,
    *,
    k: float = K,
    rho1: float = RHO1,
    rho2: float = RHO2,
):
    """Return the exact two-species pair at time t.

    V is constant. The slip w = q0 - q1 decays as w(0) exp(-λ t) with
    λ = K(1/ρ1 + 1/ρ2).
    """
    barycenter = two_species_barycenter(q0, q1, rho1=rho1, rho2=rho2)
    slip0 = np.asarray(q0, dtype=np.float64) - np.asarray(q1, dtype=np.float64)
    slip = slip0 * np.exp(-collision_lambda(k, rho1, rho2) * float(t))
    mass = float(rho1) + float(rho2)
    q0_t = barycenter + (float(rho2) / mass) * slip
    q1_t = barycenter - (float(rho1) / mass) * slip
    return q0_t, q1_t


def _author(dt, *, n_cells: int = N_CELLS) -> _Authoring:
    count = int(n_cells)
    step = float(dt)
    u_bar = _ic_barycenter(count)
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm03_collision_relax", frame=frame)
    state = model.state(
        "U",
        components=("u",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (u,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: 0.0})
    flux = model.flux(
        "inert",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * u,)},
        waves={x_axis: (0.0,)},
    )
    source = model.source("collision", on=state, value=(-NU * (u - u_bar),))
    rate = model.rate("collision_rate", equation=ddt(state) == -div(flux) + source)
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
    case = pops.Case("tm03_collision_relax")
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
        u_bar=u_bar,
    )


def build_case(dt, *, n_cells: int = N_CELLS) -> pops.Case:
    """Author the 1-d periodic relaxation Case. Does not compile or run."""
    return _author(dt, n_cells=n_cells).case


def resolve_plan(dt, *, n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(dt, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(dt=None, t_end=1.0, *, n_cells: int = N_CELLS, request=None):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='TM-03', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    if dt is None:
        dt = float(globals().get('DT', 0.1))
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(dt, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    centers, _ = _exact.uniform_cell_centers(authored.n_cells)
    initial = np.ascontiguousarray(
        _exact.initial_field(centers)[np.newaxis, :],
        dtype=np.float64,
    )
    simulation = bind_public(artifact, initial_values={authored.instance: initial}, mpi_mode=require_bind_request(request, NativeUnavailable, 'TM-03'))
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    field = np.ravel(field)
    if request is not None:
        return maybe_campaign_payload(
            request,
            field,
            n_cells=n_cells,
            t_end=t_end,
            time_program='FixedDt',
            cfl=0.0,
            dimension=1,
        )
    return field

class _TwoSpeciesAuthoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt", "k", "rho1", "rho2")

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        dt: float,
        k: float,
        rho1: float,
        rho2: float,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt
        self.k = k
        self.rho1 = rho1
        self.rho2 = rho2


def _two_species_frame():
    return CartesianDomain("tm03_two_species_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _author_two_species(
    dt,
    *,
    n_cells: int = TWO_SPECIES_N_CELLS,
    k: float = K,
    rho1: float = RHO1,
    rho2: float = RHO2,
) -> _TwoSpeciesAuthoring:
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    stiffness = float(k)
    density1 = float(rho1)
    density2 = float(rho2)
    if density1 == 0.0 or density2 == 0.0:
        raise ValueError("rho1 and rho2 must be nonzero")
    step = float(dt)
    frame = _two_species_frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm03_two_species", frame=frame)
    state = model.state(
        "U",
        components=("q0", "q1"),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    q0, q1 = state
    zero_flux = (0.0 * q0, 0.0 * q1)
    flux = model.flux(
        "inert",
        frame=frame,
        state=state,
        components={x_axis: zero_flux},
        waves={x_axis: (0.0, 0.0)},
    )
    source = model.source(
        "collision",
        on=state,
        value=(
            (stiffness / density1) * (q1 - q0),
            (stiffness / density2) * (q0 - q1),
        ),
    )
    rate = model.rate("collision_rate", equation=ddt(state) == -div(flux) + source)
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
    case = pops.Case("tm03_two_species")
    pair = case.block("pair", model=model, states=(state,))
    instance = pair[state]
    case.numerics(numerics, block=pair)
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
    return _TwoSpeciesAuthoring(
        case=case,
        instance=instance,
        frame=frame,
        n_cells=count,
        dt=step,
        k=stiffness,
        rho1=density1,
        rho2=density2,
    )


def build_two_species(
    dt,
    *,
    n_cells: int = TWO_SPECIES_N_CELLS,
    k: float = K,
    rho1: float = RHO1,
    rho2: float = RHO2,
) -> pops.Case:
    """Author the 1-d periodic two-species Case. Does not compile or run."""
    return _author_two_species(dt, n_cells=n_cells, k=k, rho1=rho1, rho2=rho2).case


def two_species_resolve(
    dt,
    *,
    n_cells: int = TWO_SPECIES_N_CELLS,
    k: float = K,
    rho1: float = RHO1,
    rho2: float = RHO2,
):
    """Validate and resolve the two-species Case. Does not compile or call pops.run."""
    authored = _author_two_species(dt, n_cells=n_cells, k=k, rho1=rho1, rho2=rho2)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def run_native_two_species(
    dt,
    t_end=1.0,
    *,
    n_cells: int = TWO_SPECIES_N_CELLS,
    k: float = K,
    rho1: float = RHO1,
    rho2: float = RHO2,
    q0: float = Q0_INIT,
    q1: float = Q1_INIT,
):
    """Compile, bind, and run the two-species Case. Raises NativeUnavailable without a compiler."""
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author_two_species(dt, n_cells=n_cells, k=k, rho1=rho1, rho2=rho2)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.broadcast_to(
        np.asarray((q0, q1), dtype=np.float64)[:, np.newaxis],
        (2, authored.n_cells),
    ).copy()
    simulation = bind_public(
        artifact, initial_values={authored.instance: initial}, mpi_mode="off"
    )
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("pair"), dtype=np.float64)
    return np.ravel(field)
