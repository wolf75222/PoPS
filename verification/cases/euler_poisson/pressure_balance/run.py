"""Public 1-d periodic Euler–Poisson pressure–field equilibrium (CP-07).

Isothermal electrons ``p = T n`` plus Poisson. The fixed background charge is
the remainder that makes the Boltzmann potential satisfy Poisson at the
cosine IC. SSPRK2 is wired with ``fields=`` so the field is recomputed at
each stage. ``run_native`` compiles, binds, and advances the Case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.native_evidence import apply_campaign_request, maybe_campaign_payload, require_bind_request
from verification.pops_verify.case_authoring import bind_public, load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

GAMMA = 1.4
N_CELLS = 64
EPS0 = 1.0
MASS = 1.0
CFL = 0.4
MAX_STEPS = 100_000
TWO_PI = 2.0 * np.pi


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


def build_oracle(n_cells: int = N_CELLS, profile: str = "cosine"):
    """In-memory exact n, u, p, φ, E, ∇p on a uniform 1-d grid."""
    centers, volumes = cell_centers(n_cells)
    fields = _EXACT.exact_fields(centers, profile=profile)
    fields["x"] = centers
    fields["volumes"] = volumes
    return fields


def initial_primitives(n_cells: int = N_CELLS, profile: str = "cosine") -> np.ndarray:
    """Primitive IC W=(n, u, p). Shape (3, n)."""
    fields = build_oracle(n_cells, profile=profile)
    return np.stack((fields["n"], fields["u"], fields["p"]))


def _spectral_laplacian(field, spacing: float) -> np.ndarray:
    samples = np.asarray(field, dtype=np.float64)
    wave = TWO_PI * np.fft.fftfreq(samples.size, d=float(spacing))
    return np.fft.ifft(-(wave * wave) * np.fft.fft(samples)).real


def background_charge(n_cells: int = N_CELLS, profile: str = "cosine") -> np.ndarray:
    """Fixed ρ_bg such that -Δφ_Boltzmann = (q n + ρ_bg) / ε0 at the IC."""
    fields = build_oracle(n_cells, profile=profile)
    spacing = 1.0 / float(n_cells)
    lap_phi = _spectral_laplacian(fields["phi"], spacing)
    charge = float(fields["q"])
    density = np.asarray(fields["n"], dtype=np.float64)
    return -EPS0 * lap_phi - charge * density


def initial_conserved(n_cells: int = N_CELLS, profile: str = "cosine") -> np.ndarray:
    """Conserved IC (n, n u, ρ_bg). Shape (3, n)."""
    fields = build_oracle(n_cells, profile=profile)
    density = np.asarray(fields["n"], dtype=np.float64)
    velocity = np.asarray(fields["u"], dtype=np.float64)
    return np.stack((density, density * velocity, background_charge(n_cells, profile)))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("cp07-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


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
    temperature = float(_EXACT.T)
    charge = float(_EXACT.Q)
    sound = float(np.sqrt(temperature))
    model = pops.Model("cp07-pressure-balance", frame=frame)
    state = model.state(
        "U",
        components=("n", "n_u", "rho_bg"),
        roles={
            "n": Density(),
            "n_u": Momentum(axis=x_axis),
        },
    )
    density, momentum, rho_bg = state
    velocity = momentum / density
    pressure = temperature * density
    flux = model.flux(
        "isothermal",
        frame=frame,
        state=state,
        components={x_axis: (momentum, momentum * velocity + pressure, 0.0 * rho_bg)},
        waves={x_axis: (velocity - sound, velocity, velocity + sound)},
    )
    potential = model.field("phi")
    phi_aux = model.aux("potential")
    electric = model.aux("phi_grad_x")
    charge_src = model.source(
        "electric",
        on=state,
        value=(
            0.0 * density + 0.0 * phi_aux,
            (charge / MASS) * density * electric,
            0.0 * rho_bg,
        ),
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + charge_src)
    operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == (charge * density + rho_bg) / EPS0),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("phi_grad", potential, sign=-1),
        ),
    )
    case = pops.Case("cp07-pressure-balance")
    block = case.block("plasma", model, states=(state,))
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
    """Author a 1-d periodic isothermal Euler–Poisson Case. Does not compile or run."""
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


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, profile: str = "cosine", request=None):
    """Compile, bind, and run the pressure-balance case."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='CP-07', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    if profile != "cosine":
        raise NativeUnavailable("native CP-07 supports the periodic cosine profile only")
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(
        initial_conserved(authored.n_cells, profile=profile), dtype=np.float64
    )
    simulation = bind_public(artifact, initial_values={authored.instance: initial}, mpi_mode=require_bind_request(request, NativeUnavailable, 'CP-07'))
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("plasma"), dtype=np.float64)
    field = np.reshape(field, (3, authored.n_cells))
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