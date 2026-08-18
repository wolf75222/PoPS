"""Public 1-d periodic AMR Poisson authoring for PO-06 CF gradient placement.

The hyperbolic rate is stationary (zero flux) so the Program exists only to
host one composite field solve per step. Interface-band helpers stay the
in-memory CF-placement oracle. ``run_native`` compiles the public AMR Case,
binds the manufactured RHS, advances one step, and returns the solved
potential. FFT is refused on AMR, so the field solver is GeometricMG with
CompositeHierarchySolve. The frozen two-level coarse-fine interface is at
x = 0.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredSecondOrder,
    CompositeHierarchySolve,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import EllipticRecompute, StateTransfer
from pops.lib.initial import BindArray
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div, laplacian
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.problem import Case
from pops.projection import ConservativeCellAverage
from pops.solvers.elliptic import GeometricMG
from pops.time import FixedDt
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import load_sibling_module, resolve_case
from verification.pops_verify.interface_error import (
    band_max_abs_error,
    interface_band_mask,
)
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
MANUFACTURED_ERROR_SCALE = 0.04
BAND_WEIGHT = 0.5
INTERFACE = 0.5
N_LEVELS = 2
REFINEMENT_RATIO = 2
MAX_STEPS = 4


class AuthoringPending(RuntimeError):
    """Kept for compatibility. Resolve now succeeds with a stationary AMR Program."""


class NativeUnavailable(RuntimeError):
    """Raised when the native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "layout")

    def __init__(
        self, case: Any, instance: Any, frame: Any, n_cells: int, layout: Any
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.layout = layout


def _exact_module():
    return load_sibling_module(_CASE_DIR / "exact.py")


def build_rhs_and_oracle(n_cells: int):
    """Return in-memory cell-center RHS and exact φ, E on a uniform 1-d grid."""
    exact = _exact_module()
    centers, volumes = exact.uniform_cell_grid(n_cells)
    return {
        "x": centers,
        "volumes": volumes,
        "phi": exact.phi_exact(centers),
        "rhs": exact.rhs_exact(centers),
        "e": exact.e_exact(centers),
    }


def placement_sample(n_cells: int, placement: str):
    """Return the PO-01 oracle plus periodic distance to a named CF interface."""
    exact = _exact_module()
    sample = build_rhs_and_oracle(n_cells)
    x_interface = exact.interface_location(placement)
    sample["placement"] = placement
    sample["x_interface"] = x_interface
    sample["distance"] = exact.periodic_distance(sample["x"], x_interface)
    return sample


def manufactured_gradient(n_cells: int, placement: str):
    """Return manufactured second-order E error at one CF placement."""
    sample = placement_sample(n_cells, placement)
    spacing = 1.0 / float(n_cells)
    floor = MANUFACTURED_ERROR_SCALE * spacing**2
    band = interface_band_mask(sample["distance"], h_fine=spacing)
    manufactured = sample["e"] + floor * (1.0 + BAND_WEIGHT * band.astype(np.float64))
    field = reference_errors(manufactured, sample["e"], sample["volumes"])
    complement = np.logical_not(band)
    sample["manufactured_e"] = manufactured
    sample["band"] = band
    sample["field_error"] = float(field.linf)
    sample["e_cf"] = band_max_abs_error(manufactured, sample["e"], band)
    sample["e_bulk"] = band_max_abs_error(manufactured, sample["e"], complement)
    return sample


def _author(n_cells: int) -> _Authoring:
    count = int(n_cells)
    frame = CartesianDomain("po06-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    (x_axis,) = frame.axes
    model = Model("po06_cf_gradient", frame=frame)
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
        # ρ = (2π)² sin(2πx) is negative on [INTERFACE, 1]. Bind U = -ρ and
        # solve -∇²φ = -U so Tag(U > 0) covers that prescribed fine patch.
        equation=(-laplacian(potential) == -rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case("po06-cf-gradient")
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
            solver=GeometricMG(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
            hierarchy_policy=CompositeHierarchySolve(),
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
    threshold = case.param(RuntimeParam("po06-refine", default=0.0))
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer())
    transfer.field(field, EllipticRecompute())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(count,),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(
            max_levels=int(N_LEVELS),
            ratios=(int(REFINEMENT_RATIO),),
        ),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(instance) > ValueExpr(threshold)),
                Buffer(cells=0),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    return _Authoring(
        case=case, instance=instance, frame=frame, n_cells=count, layout=layout
    )


def build_case(n_cells: int) -> Case:
    """Author a 1-d periodic AMR Poisson Case with a stationary Program."""
    return _author(n_cells).case


def resolve_plan(n_cells: int):
    """Validate and resolve the AMR Case. Does not compile or call pops.run."""
    authored = _author(n_cells)
    return resolve_case(authored.case, layout=authored.layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int, t_end: float = 1.0):
    """Compile, bind the manufactured RHS, and return the solved potential."""
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    plan = resolve_case(authored.case, layout=authored.layout)
    artifact = pops.compile(plan)
    sample = build_rhs_and_oracle(authored.n_cells)
    initial = np.ascontiguousarray(-sample["rhs"][np.newaxis, :], dtype=np.float64)
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    slots = tuple(simulation.field_provider_slots())
    if not slots:
        raise NativeUnavailable("native runtime exposed no field-provider slot")
    phi = np.asarray(simulation.field_potential_global(slots[0]), dtype=np.float64)
    return np.ravel(phi)
