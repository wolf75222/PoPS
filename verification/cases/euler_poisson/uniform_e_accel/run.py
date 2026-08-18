"""Public 1-d periodic Euler–Poisson uniform-E acceleration (CP-08).

A periodic Poisson solve of a uniform neutralizing plasma yields E_self = 0.
The prescribed uniform field E0 is added in the Lorentz source. SSPRK2 is
wired with ``fields=`` so Poisson is still solved at each stage. ``run_native``
compiles, binds, and advances the Case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

GAMMA = float(_EXACT.GAMMA)
N_CELLS = 32
Q = float(_EXACT.Q)
MASS = float(_EXACT.MASS)
E0 = float(_EXACT.E0)
N0 = float(_EXACT.N0)
EPS0 = 1.0
CFL = 0.4
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class AuthoringPending(RuntimeError):
    """Kept for compatibility. Resolve now succeeds with SSPRK2(fields=...)."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / count
    return (np.arange(count, dtype=np.float64) + 0.5) * width, width


def initial_primitives(
    n_cells: int = N_CELLS,
    *,
    q=_EXACT.Q,
    mass=_EXACT.MASS,
    e0=_EXACT.E0,
    u0=_EXACT.U0,
    n0=_EXACT.N0,
    p0=_EXACT.P0,
):
    """Primitive IC W(x,0). Shape (3, n)."""
    centers, _ = cell_centers(n_cells)
    return _EXACT.primitives(centers, 0.0, q=q, mass=mass, e0=e0, u0=u0, n0=n0, p0=p0)


def primitives_to_conserved(primitives) -> np.ndarray:
    """Convert primitive (n, u, p) to conserved (n, n u, E) with mass density n*m=n."""
    number, speed, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * number * speed * speed
    return np.stack((number, number * speed, energy))


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from ``primitives(..., t=0)``."""
    return primitives_to_conserved(initial_primitives(n_cells))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("cp08-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


def _author(n_cells: int = N_CELLS) -> _Authoring:
    import pops
    from pops.fields import (
        CellCenteredSecondOrder,
        ConstantNullspace,
        FieldDiscretization,
        FieldOutput,
        GradientOutput,
        MeanValueGauge,
    )
    from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.lib.time import SSPRK2
    from pops.math import ddt, div, laplacian, sqrt
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Energy, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.solvers.elliptic import FFT
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("cp08-uniform-e-accel", frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "E"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "E": Energy(),
        },
    )
    rho, momentum, energy = state
    velocity = momentum / rho
    pressure = (GAMMA - 1.0) * (energy - 0.5 * rho * velocity * velocity)
    sound = sqrt(GAMMA * pressure / rho)
    flux = model.flux(
        "euler",
        frame=frame,
        state=state,
        components={
            x_axis: (
                momentum,
                momentum * velocity + pressure,
                velocity * (energy + pressure),
            ),
        },
        waves={x_axis: (velocity - sound, velocity, velocity + sound)},
    )
    potential = model.field("phi")
    phi_aux = model.aux("potential")
    electric = model.aux("phi_grad_x")
    electric_total = electric + E0
    lorentz = (Q / MASS) * rho * electric_total
    charge = model.source(
        "electric",
        on=state,
        value=(0.0 * rho + 0.0 * phi_aux, lorentz, velocity * lorentz),
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + charge)
    operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == (Q / EPS0) * (rho - N0)),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("phi_grad", potential, sign=-1),
        ),
    )
    case = pops.Case("cp08-uniform-e-accel")
    block = case.block("gas", model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiter=limiters.VanLeer()),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    field = case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=FFT(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )
    program = SSPRK2(instance, rate=rate, fields=field)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = N_CELLS):
    """Author a 1-d periodic Euler–Poisson Case with prescribed E0. Does not run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or execute a run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05):
    """Compile, bind, and run the uniform-E case."""
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(initial_conserved(authored.n_cells), dtype=np.float64)
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return np.reshape(field, (3, authored.n_cells))
