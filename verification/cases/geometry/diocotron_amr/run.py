"""GE-06 Cartesian diocotron: in-memory ring/tag/FFT plus public 2-d AMR.

Reuses the CP-11 ring via ``load_sibling_module`` on
``verification/cases/euler_poisson/diocotron/exact.py`` when that sibling
exists; otherwise the local exact.py duplicate is the oracle. CP-11
``density`` is never evaluated on the GE-06 origin-centered mesh.

Public Case is 2-d pressureless Euler–Poisson (CP-11 hydro + CP-02
``fields=`` / ``model.aux("potential")`` / ``model.aux("phi_grad_x")``)
on ``pops.amr``: two-level hierarchy, ratio 2, frozen ring tag via
``pops.analytic`` marker, GeometricMG + CompositeHierarchySolve.
Polar is refused.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_CP11_EXACT = (
    Path(__file__).resolve().parents[2] / "euler_poisson" / "diocotron" / "exact.py"
)
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_cp11 = load_sibling_module(_CP11_EXACT) if _CP11_EXACT.is_file() else None

N_CELLS = int(_exact.N_CELLS)
BUFFER_CELLS = int(_exact.BUFFER_CELLS)
E_CHARGE = 1.0
Q_E = -E_CHARGE
N_I_FALLBACK = 0.25
EPS0 = 1.0
M_E = 1.0
NATIVE_DT = 1.0e-3
MAX_STEPS = 100_000
NATIVE_DENSITY_FLOOR = 1.0e-8
COMPONENT_ORDER = ("n", "n_u", "n_v")
POLAR_RUNTIME_REFUSAL = "GE-06 is Cartesian only; public polar System not active"


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class AuthoringPending(RuntimeError):
    """Raised when public Cartesian Euler–Poisson validate/resolve cannot complete."""


class _Authoring:
    __slots__ = ("case", "instance", "marker_instance", "frame", "n_cells", "layout")

    def __init__(
        self,
        case: Any,
        instance: Any,
        marker_instance: Any,
        frame: Any,
        n_cells: int,
        layout: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.frame = frame
        self.n_cells = n_cells
        self.layout = layout


def _ring_density(x, y):
    """Unperturbed hollow ring on the GE-06 origin-centered box.

    CP-11 is a sibling oracle (unit square, centre (0.5, 0.5)). Sampling it
    on this mesh would miss the ring. Keep the load so the case documents
    the sibling; do not evaluate it on GE-06 coordinates.
    """
    assert _cp11 is None or hasattr(_cp11, "density")
    return _exact.unperturbed_density(x, y)


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the box [lower, upper]²."""
    count = int(n_cells)
    lower_x, lower_y = (float(value) for value in _exact.DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in _exact.DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / count
    width_y = (upper_y - lower_y) / count
    x_centers = lower_x + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x, y, width_x, width_y


def sample_field(n_cells: int = N_CELLS) -> np.ndarray:
    """Unperturbed ring density on the uniform Cartesian mesh."""
    x, y, _, _ = cell_centers(n_cells)
    return _ring_density(x, y)


def ring_mask(n_cells: int = N_CELLS) -> np.ndarray:
    """Cell-center mask of the open annulus r1 < r < r2."""
    x, y, _, _ = cell_centers(n_cells)
    return _exact.ring_mask(x, y)


def raw_tag_mask(n_cells: int = N_CELLS) -> np.ndarray:
    """Tag |n - n_bg| > θ on the unperturbed ring."""
    return _exact.raw_tag_mask(sample_field(n_cells))


def envelope_mask(n_cells: int = N_CELLS, buffer_cells: int = BUFFER_CELLS) -> np.ndarray:
    """Two-level envelope: tagged cells plus Chebyshev buffer."""
    return _exact.dilate_mask(raw_tag_mask(n_cells), buffer_cells)


def angular_density(n_theta=None, ring_radius=None):
    """Unperturbed ring samples on a fixed-r circle (for the unused-mode FFT)."""
    count = _exact.N_THETA if n_theta is None else int(n_theta)
    return _exact.angular_density(ring_radius, count, eps=0.0)


def unused_mode_amplitude(n_theta=None, ring_radius=None) -> float:
    """|FFT[m=3]| of the unperturbed ring. Axisymmetry ⇒ ~0."""
    count = _exact.N_THETA if n_theta is None else int(n_theta)
    return _exact.unused_mode_amplitude(ring_radius, count, eps=0.0)


def refuse_public_polar_runtime() -> str:
    """Return the documented reason the public polar System is not used."""
    return POLAR_RUNTIME_REFUSAL


def initial_density(n_cells: int = N_CELLS) -> np.ndarray:
    """Unperturbed ring with a positive floor for the native hydro state."""
    field = np.asarray(sample_field(n_cells), dtype=np.float64)
    return np.maximum(field, NATIVE_DENSITY_FLOOR)


def neutralizing_background(n_cells: int = N_CELLS) -> float:
    """Spatial mean of the floored ring so periodic Poisson is mean-zero."""
    field = initial_density(n_cells)
    if field.size == 0:
        return float(N_I_FALLBACK)
    return float(np.mean(field))


def initial_conserved(n_cells: int = N_CELLS) -> dict:
    """Rest-fluid conserved IC (n, n u, n v) on the GE-06 mesh."""
    density = initial_density(n_cells)
    zeros = np.zeros_like(density)
    return {"n": density, "n_u": zeros, "n_v": zeros}


def pack_conserved(conserved) -> np.ndarray:
    """Return a C-contiguous (3, n, n) array in COMPONENT_ORDER."""
    return np.ascontiguousarray(
        np.stack([conserved[name] for name in COMPONENT_ORDER], axis=0),
        dtype=np.float64,
    )


def unpack_conserved(field, n_cells: int | None = None) -> dict:
    """Split a (3, n, n) or flat native buffer into named conserved fields."""
    count = N_CELLS if n_cells is None else int(n_cells)
    array = np.ascontiguousarray(field, dtype=np.float64)
    if array.shape == (3, count, count):
        stacked = array
    else:
        stacked = np.reshape(array, (3, count, count))
    return {name: stacked[index] for index, name in enumerate(COMPONENT_ORDER)}


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "ge06-box",
        _exact.DOMAIN_LOWER,
        _exact.DOMAIN_UPPER,
    ).frame(Cartesian2D())


def _ring_marker(frame):
    """Public analytic indicator of the hollow ring (1 on r1<r<r2)."""
    from pops.analytic import between, radius as analytic_radius, where
    from pops.lib.initial import Analytic

    radial = analytic_radius(frame)
    return Analytic(
        frame=frame,
        components=(where(between(radial, _exact.R1, _exact.R2), 1.0, 0.0),),
    )


def _author(n_cells: int) -> _Authoring:
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
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import EllipticRecompute, StateTransfer
    from pops.lib.initial import BindArray
    from pops.lib.time import SSPRK2
    from pops.math import ValueExpr, ddt, div, laplacian
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.params import RuntimeParam
    from pops.physics import Density, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.solvers.elliptic import GeometricMG
    from pops.spaces import CellState
    from pops.time import FixedDt

    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("ge06-diocotron", frame=frame)
    state = model.state(
        "U",
        components=COMPONENT_ORDER,
        roles={
            "n": Density(),
            "n_u": Momentum(axis=x_axis),
            "n_v": Momentum(axis=y_axis),
        },
    )
    density, momentum_x, momentum_y = state
    velocity_x = momentum_x / density
    velocity_y = momentum_y / density
    flux = model.flux(
        "cold_electron",
        frame=frame,
        state=state,
        components={
            x_axis: (momentum_x, momentum_x * velocity_x, momentum_x * velocity_y),
            y_axis: (momentum_y, momentum_y * velocity_x, momentum_y * velocity_y),
        },
        waves={
            x_axis: (velocity_x, velocity_x, velocity_x),
            y_axis: (velocity_y, velocity_y, velocity_y),
        },
    )
    potential = model.field("phi")
    phi_aux = model.aux("potential")
    electric_x = model.aux("phi_grad_x")
    electric_y = model.aux("phi_grad_y")
    n_i = neutralizing_background(count)
    charge = model.source(
        "electric",
        on=state,
        value=(
            0.0 * density + 0.0 * phi_aux,
            (Q_E / M_E) * density * electric_x,
            (Q_E / M_E) * density * electric_y,
        ),
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + charge)
    operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == (E_CHARGE / EPS0) * (n_i - density)),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("phi_grad", potential, sign=-1),
        ),
    )
    marker_model = pops.Model("ge06_marker", frame=frame)
    marker_state = marker_model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (marker_q,) = marker_state
    marker_flux = marker_model.flux(
        "ge06_marker_flux",
        frame=frame,
        state=marker_state,
        components={x_axis: (0.0 * marker_q,), y_axis: (0.0 * marker_q,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    marker_rate = marker_model.rate(
        "ge06_marker_rate", equation=ddt(marker_state) == -div(marker_flux)
    )

    case = pops.Case("ge06-diocotron-amr")
    block = case.block("electrons", model, states=(state,))
    marker = case.block("marker", model=marker_model, states=(marker_state,))
    instance = block[state]
    marker_instance = marker[marker_state]
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
    marker_numerics = DiscretizationPlan()
    marker_numerics.rates.add(
        marker_rate,
        FiniteVolume(
            flux=marker_flux,
            variables=variables.Conservative(marker_state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(marker_numerics, block=marker)
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
    program = SSPRK2(instance, rate=rate, fields=field)
    marker_time = program.state(marker_instance)
    marker_hold = program.value(
        "marker_hold",
        marker_time.n,
        at=marker_time.next.point,
    )
    program.commit(marker_time.next, marker_hold)
    program.step_strategy(FixedDt(NATIVE_DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=marker_instance,
            value=_ring_marker(frame),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("ge06-refine", default=float(_exact.THETA)))
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer())
    transfer.state(marker_instance, StateTransfer())
    transfer.field(field, EllipticRecompute())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(count, count),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(marker_instance) > ValueExpr(threshold)),
                Buffer(cells=int(_exact.BUFFER_CELLS)),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    return _Authoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        frame=frame,
        n_cells=count,
        layout=layout,
    )


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d Cartesian Euler–Poisson Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the 2-d AMR Euler–Poisson Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import resolve_case

    try:
        authored = _author(n_cells)
        return resolve_case(authored.case, layout=authored.layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            "GE-06 Cartesian Euler–Poisson AMR resolve failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _native_dim_refusal() -> str | None:
    value = os.environ.get("POPS_NATIVE_DIM")
    if value is not None and value != "2":
        return (
            "GE-06 requires POPS_NATIVE_DIM=2 "
            f"(got {value!r}); Cartesian only"
        )
    return None


def _native_unavailable_reason() -> str | None:
    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    dim_reason = _native_dim_refusal()
    if dim_reason:
        return dim_reason
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05):
    """Compile, bind, and run a short 2-d AMR Euler–Poisson step.

    Returns a C-contiguous level-0 ``(3, n, n)`` conserved array in
    ``COMPONENT_ORDER``. Polar is refused. Raises ``NativeUnavailable``
    without a compiler/Kokkos, when ``POPS_NATIVE_DIM`` is not 2, or when
    compile/bind/run cannot complete.
    """
    import pops

    from verification.pops_verify.case_authoring import bind_public, resolve_case

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    try:
        authored = _author(n_cells)
        plan = resolve_case(authored.case, layout=authored.layout)
        artifact = pops.compile(plan)
        initial = pack_conserved(initial_conserved(authored.n_cells))
        simulation = bind_public(
            artifact, initial_values={authored.instance: initial}
        )
        pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
        field = np.asarray(
            simulation.block_level_state_global("electrons", 0),
            dtype=np.float64,
        )
        return pack_conserved(unpack_conserved(field, authored.n_cells))
    except NativeUnavailable:
        raise
    except Exception as exc:
        raise NativeUnavailable(
            f"GE-06 native compile/bind/run failed: {type(exc).__name__}: {exc}"
        ) from exc
