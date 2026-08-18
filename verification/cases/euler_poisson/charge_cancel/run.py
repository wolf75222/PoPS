"""Public 1-d periodic Poisson authoring for CP-12 charge cancellation.

The hyperbolic rate is stationary (zero flux) so the Program exists only to
host one field solve per step. The bound RHS is the cancelled two-species
charge. ``run_native`` compiles, binds, advances one step, and returns φ.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.native_evidence import NULL_COUPLING, apply_campaign_request, maybe_campaign_payload, require_bind_request
from verification.pops_verify.case_authoring import bind_public, load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
N_CELLS = 32
MAX_STEPS = 4


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class AuthoringPending(RuntimeError):
    """Kept for compatibility. Resolve now succeeds with a stationary Program."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _exact_module():
    return load_sibling_module(_CASE_DIR / "exact.py")


def build_oracle(n_cells: int = N_CELLS, *, delta: float = 0.0):
    """Return in-memory cancelled charge, φ, and E on a uniform 1-d grid."""
    exact = _exact_module()
    centers, volumes = exact.uniform_cell_grid(n_cells)
    return {
        "x": centers,
        "volumes": volumes,
        "density": exact.density(centers, delta=delta),
        "net_charge": exact.net_charge(centers, delta=delta),
        "rhs": exact.poisson_rhs(centers, delta=delta),
        "phi": exact.phi_exact(centers),
        "e": exact.e_exact(centers),
    }


def _author(n_cells: int = N_CELLS) -> _Authoring:
    import pops
    from pops.domain import CartesianDomain
    from pops.fields import (
        CellCenteredSecondOrder,
        ConstantNullspace,
        FieldDiscretization,
        FieldOutput,
        GradientOutput,
        MeanValueGauge,
    )
    from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
    from pops.frames import Cartesian1D
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.lib.time import ForwardEuler
    from pops.math import ddt, div, laplacian
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Model
    from pops.problem import Case
    from pops.projection import ConservativeCellAverage
    from pops.solvers.elliptic import FFT
    from pops.time import FixedDt

    count = int(n_cells)
    frame = CartesianDomain("cp12-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    (x_axis,) = frame.axes
    model = Model("cp12_charge_cancel", frame=frame)
    state = model.state("U", components=["rhs"])
    (rhs,) = state
    flux = model.flux(
        "stationary",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rhs,)},
        waves={x_axis: (0.0 * rhs,)},
    )
    rate = model.rate("hold", equation=ddt(state) == -div(flux))
    potential = model.field("potential")
    operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case("cp12-charge-cancel")
    block = case.block("electrostatic", model, states=(state,))
    instance = block[state]
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
    program = ForwardEuler(instance, rate=rate, fields=field)
    program.step_strategy(FixedDt(1.0))
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
    """Author a 1-d periodic cancelled-charge Poisson Case with a stationary Program."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or execute a run."""
    from verification.pops_verify.case_authoring import resolve_case, uniform_periodic_layout

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


def run_native(n_cells: int = N_CELLS, t_end: float = 1.0, *, delta: float = 0.0, request=None):
    """Compile, bind the cancelled charge, and return the solved potential."""
    n_cells = apply_campaign_request(
        n_cells, request, case_id='CP-12', allowed_dims=(1,), unavailable=NativeUnavailable
    )
    import pops

    from verification.pops_verify.case_authoring import resolve_case, uniform_periodic_layout

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    sample = build_oracle(authored.n_cells, delta=delta)
    initial = np.ascontiguousarray(sample["rhs"][np.newaxis, :], dtype=np.float64)
    simulation = bind_public(artifact, initial_values={authored.instance: initial}, mpi_mode=require_bind_request(request, NativeUnavailable, 'CP-12'))
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    slots = tuple(simulation.field_provider_slots())
    if not slots:
        raise NativeUnavailable("native runtime exposed no field-provider slot")
    phi = np.asarray(simulation.field_potential_global(slots[0]), dtype=np.float64)
    field = np.ravel(phi)[: authored.n_cells]
    if request is not None:
        return maybe_campaign_payload(
            request,
            field,
            artifact=artifact,
            simulation=simulation,
            coupling=dict(NULL_COUPLING),
            n_cells=n_cells,
            t_end=t_end,
            time_program='ForwardEuler',
            cfl=0.0,
            dimension=1,
        )
    return field