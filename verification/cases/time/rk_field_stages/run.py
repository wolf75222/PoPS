"""TM-07 field-solve counts plus a public CP-02 Langmuir Case.

In-memory helpers document one field solve at each required RK stage, with
frozen-field as the negative control. The public Case is the cold Langmuir
Euler–Poisson program with ``SSPRK2(..., fields=)`` so Poisson is solved at
each stage. ``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.native_evidence import NULL_COUPLING, apply_campaign_request, maybe_campaign_payload, require_bind_request
from verification.pops_verify.case_authoring import bind_public, load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

# CP-02 cold Langmuir units: e = m_e = ε0 = n0 = 1 ⇒ ω_pe = 1.
E_CHARGE = 1.0
M_E = 1.0
EPS0 = 1.0
N0 = 1.0
N_I = 1.0
A = 1.0e-4
K = 2.0 * np.pi
Q_E = -E_CHARGE
N_CELLS = 64
CFL = 0.4
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def field_solves_per_step(integrator: str, *, frozen_field: bool = False) -> int:
    """Return documented field solves per step for the named integrator."""
    if frozen_field:
        return int(_exact.FROZEN_FIELD_SOLVES_PER_STEP)
    return _exact.required_field_solves(_exact.stage_count(integrator))


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC (n, n u) from the t=0 Langmuir standing wave. Shape (2, n)."""
    count = int(n_cells)
    width = 1.0 / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    density = N0 + A * np.cos(K * centers)
    velocity = np.zeros(count, dtype=np.float64)
    return np.stack((density, density * velocity))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("tm07-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


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
    model = pops.Model("tm07-langmuir-cold", frame=frame)
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
    flux = model.flux(
        "cold_electron",
        frame=frame,
        state=state,
        components={x_axis: (momentum, momentum * velocity)},
        waves={x_axis: (velocity, velocity)},
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
    case = pops.Case("tm07-rk-field-stages")
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
    """Author a 1-d periodic cold Euler–Poisson Case. Does not compile or run."""
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


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, request=None):
    """Compile, bind, and run the Langmuir case. Raises NativeUnavailable without Kokkos."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='TM-07', allowed_dims=(1,), unavailable=NativeUnavailable
    )
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
    simulation = bind_public(artifact, initial_values={authored.instance: initial}, mpi_mode=require_bind_request(request, NativeUnavailable, 'TM-07'))
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