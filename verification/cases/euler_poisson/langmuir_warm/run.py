"""Public 1-d periodic warm-electron Euler–Poisson authoring and native run.

Isothermal closure ``p = c_e² n`` matches the oracle ``ω² = ω_pe² + c_e² k²``.
SSPRK2 is wired with ``fields=`` so Poisson is solved at each stage (TM-07).
``run_native`` compiles, binds, and advances the Case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.native_evidence import NULL_COUPLING, apply_campaign_request, maybe_campaign_payload, require_bind_request
from verification.pops_verify.case_authoring import bind_public, load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = _EXACT.N_CELLS if hasattr(_EXACT, "N_CELLS") else 32
E_CHARGE = _EXACT.E_CHARGE
Q_E = -E_CHARGE
N0 = _EXACT.N0
N_I = _EXACT.N0
EPS0 = _EXACT.EPSILON_0
M_E = 1.0
C_E = _EXACT.C_E
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
    return _EXACT.uniform_cell_centers(n_cells)


def initial_fields(n_cells: int = N_CELLS, *, cycles: int = 1, t: float = 0.0):
    """Warm-Langmuir fields at one time for one canonical wavenumber."""
    centers, volumes = cell_centers(n_cells)
    fields = _EXACT.exact_fields(centers, float(t), k=_EXACT.wavenumber(cycles))
    fields["x"] = centers
    fields["volumes"] = volumes
    return fields


def initial_conserved(n_cells: int = N_CELLS, *, cycles: int = 1):
    """Conserved IC (n, n u) from the closed oracle at t=0. Shape (2, n)."""
    sample = initial_fields(n_cells, cycles=cycles, t=0.0)
    density = np.asarray(sample["n_e"], dtype=np.float64)
    velocity = np.asarray(sample["u_e"], dtype=np.float64)
    return np.stack((density, density * velocity))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("cp03-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


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
    from pops.math import ddt, div, laplacian
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.solvers.elliptic import FFT
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("cp03-langmuir-warm", frame=frame)
    state = model.state(
        "U",
        components=("n", "n_u"),
        roles={
            "n": Density(),
            "n_u": Momentum(axis=x_axis),
        },
    )
    density, momentum = state
    velocity = momentum / density
    pressure = C_E * C_E * density
    flux = model.flux(
        "warm_electron",
        frame=frame,
        state=state,
        components={x_axis: (momentum, momentum * velocity + pressure)},
        waves={x_axis: (velocity - C_E, velocity + C_E)},
    )
    potential = model.field("phi")
    phi_aux = model.aux("potential")
    electric = model.aux("phi_grad_x")
    charge = model.source(
        "electric",
        on=state,
        value=(0.0 * density + 0.0 * phi_aux, (Q_E / M_E) * density * electric),
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + charge)
    operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == (E_CHARGE / EPS0) * (N_I - density)),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("phi_grad", potential, sign=-1),
        ),
    )
    case = pops.Case("cp03-langmuir-warm")
    block = case.block("electrons", model, states=(state,))
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
    """Author a 1-d periodic warm Euler–Poisson Case. Does not compile or run."""
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


def _bind_native(n_cells: int = N_CELLS, *, cycles: int = 1, request=None):
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
    initial = np.ascontiguousarray(
        initial_conserved(authored.n_cells, cycles=cycles), dtype=np.float64
    )
    simulation = bind_public(
        artifact,
        initial_values={authored.instance: initial},
        mpi_mode=require_bind_request(request, NativeUnavailable, "CP-03"),
    )
    return authored, simulation


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, cycles: int = 1, request=None):
    """Compile, bind, and run the warm Langmuir case."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='CP-03', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    import pops

    authored, simulation = _bind_native(n_cells, cycles=cycles, request=request)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("electrons"), dtype=np.float64)
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
            time_program='SSPRK2',
            cfl=0.4,
            dimension=1,
        )
    return field

def run_native_series(times, n_cells: int = N_CELLS, *, cycles: int = 1):
    """Advance one bound run through increasing ``times`` and return (2, n) snapshots.

    Compiles once. Each entry of ``times`` must be strictly increasing. The
    returned array has shape ``(len(times), 2, n_cells)``.
    """
    import pops

    stamps = np.asarray(times, dtype=np.float64)
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("times must be a non-empty 1-d array")
    if np.any(np.diff(stamps) <= 0.0):
        raise ValueError("times must be strictly increasing")
    authored, simulation = _bind_native(n_cells, cycles=cycles)
    snapshots = np.empty((stamps.size, 2, authored.n_cells), dtype=np.float64)
    for index, stamp in enumerate(stamps):
        pops.run(simulation, t_end=float(stamp), max_steps=MAX_STEPS)
        field = np.asarray(simulation.state_global("electrons"), dtype=np.float64)
        snapshots[index] = np.reshape(field, (2, authored.n_cells))
    return snapshots
