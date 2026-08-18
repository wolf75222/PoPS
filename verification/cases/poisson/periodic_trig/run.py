"""Public 1-d periodic Poisson authoring and native FFT solve for PO-01.

The hyperbolic rate is stationary (zero flux) so the Program exists only to
host one field solve per step. ``run_native`` compiles, binds the manufactured
RHS, advances one step, and returns the solved potential.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
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
from verification.pops_verify.native_toolchain import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import resolve_case, uniform_periodic_layout

_EXACT_PATH = Path(__file__).with_name("exact.py")
MAX_STEPS = 4


class AuthoringPending(RuntimeError):
    """Kept for compatibility. Resolve now succeeds with a stationary Program."""


class NativeUnavailable(RuntimeError):
    """Raised when the native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _exact_module():
    spec = importlib.util.spec_from_file_location("po01_periodic_trig_exact", _EXACT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_EXACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_rhs_and_oracle(n_cells: int):
    """Return cell-average RHS and pointwise oracle φ, E on a uniform 1-d grid."""
    from verification.pops_verify.cell_averages import analytic_cell_averages

    exact = _exact_module()
    centers, volumes = exact.uniform_cell_grid(n_cells)
    width = float(volumes[0])
    lo = centers - 0.5 * width
    hi = centers + 0.5 * width
    return {
        "x": centers,
        "volumes": volumes,
        "phi": exact.phi_exact(centers),
        "rhs": analytic_cell_averages(exact.rhs_exact, lo, hi),
        "e": exact.e_exact(centers),
    }


def _author(n_cells: int) -> _Authoring:
    count = int(n_cells)
    frame = CartesianDomain("po01-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    (x_axis,) = frame.axes
    model = Model("po01_periodic_trig", frame=frame)
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
        "poisson",
        unknown=potential,
        equation=(-laplacian(potential) == rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case("po01-periodic-trig")
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


def build_case(n_cells: int) -> Case:
    """Author a 1-d periodic Poisson Case with a stationary Program."""
    return _author(n_cells).case


def resolve_plan(n_cells: int):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int = 16, t_end: float = 1.0, *, request=None):
    """Compile, bind the manufactured RHS, and return the solved potential."""
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"PO-01 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    sample = build_rhs_and_oracle(authored.n_cells)
    initial = np.ascontiguousarray(sample["rhs"][np.newaxis, :], dtype=np.float64)
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    slots = tuple(simulation.field_provider_slots())
    if not slots:
        raise NativeUnavailable("native runtime exposed no field-provider slot")
    phi = np.ravel(
        np.asarray(simulation.field_potential_global(slots[0]), dtype=np.float64)
    )[: authored.n_cells]
    return maybe_campaign_payload(
        request,
        phi,
        artifact=artifact,
        simulation=simulation,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="ForwardEuler",
        cfl=1.0,
        dimension=1,
    )
