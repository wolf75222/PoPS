"""Public 1-d periodic Euler–Poisson AMR authoring for AM-11.

In-memory leaf-only charge helpers stay. The public Case takes hydro +
Poisson from CP-02 (SSPRK2(..., fields=), operator name ``fields``,
``model.aux("potential")`` / ``model.aux("phi_grad_x")``) and the AMR
layout from AM-10 (Cartesian1D, two levels, ratio 2, frozen right-half
patch, GeometricMG + CompositeHierarchySolve, EllipticRecompute).
``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

CFL = 0.4
MAX_STEPS = 100_000
DEFAULT_N_CELLS = 16
INTERFACE = 0.5
N_LEVELS = 2
REFINEMENT_RATIO = 2
REFINE_THRESHOLD = 0.5
E_CHARGE = 1.0
Q_E = -E_CHARGE
N_I = 1.0
EPS0 = 1.0
M_E = 1.0


class AuthoringPending(RuntimeError):
    """Raised when public Euler–Poisson AMR validate/resolve cannot complete."""


class NativeUnavailable(RuntimeError):
    """Raised when the native compile/run path cannot run."""


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


def compose_charge(rho, volumes, leaf_mask) -> float:
    """Return the leaf-only net charge of an in-memory Euler–Poisson hierarchy."""
    return _exact.leaf_net_charge(rho, volumes, leaf_mask)


def initial_conserved(n_cells: int):
    """Level-0 conserved IC (n, n u) from the AM-11 density, u=0. Shape (2, n)."""
    count = int(n_cells)
    width = 1.0 / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    density = np.asarray(_exact.density(centers), dtype=np.float64)
    momentum = np.zeros_like(density)
    return np.stack((density, momentum))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("am11-ep-sync", (0.0,), (1.0,)).frame(Cartesian1D())


def _right_half_marker(frame):
    """Spatial indicator: 1 on x>0.5, 0 on the coarse left half."""
    from pops.analytic import where, x as analytic_x
    from pops.lib.initial import Analytic

    return Analytic(
        frame=frame,
        components=(where(analytic_x(frame) > INTERFACE, 1.0, 0.0),),
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
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("am11_ep_sync", frame=frame)
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

    marker_model = pops.Model("am11_marker", frame=frame)
    marker_state = marker_model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (marker_q,) = marker_state
    marker_flux = marker_model.flux(
        "am11_marker_flux",
        frame=frame,
        state=marker_state,
        components={x_axis: (0.0 * marker_q,)},
        waves={x_axis: (0.0,)},
    )
    marker_rate = marker_model.rate(
        "am11_marker_rate", equation=ddt(marker_state) == -div(marker_flux)
    )

    case = pops.Case("am11-ep-sync")
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
    program.step_strategy(AdaptiveCFL(cfl=CFL))
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
            value=_right_half_marker(frame),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("am11-refine", default=REFINE_THRESHOLD))
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer())
    transfer.state(marker_instance, StateTransfer())
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
                Tag(ValueExpr(marker_instance) > ValueExpr(threshold)),
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
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        frame=frame,
        n_cells=count,
        layout=layout,
    )


def build_case(n_cells: int = DEFAULT_N_CELLS) -> Any:
    """Author a 1-d periodic Euler–Poisson AMR Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = DEFAULT_N_CELLS):
    """Validate and resolve the AMR Case. Does not compile or call pops.run."""
    import pops

    from verification.pops_verify.case_authoring import resolve_case

    authored = _author(n_cells)
    try:
        validated = pops.validate(authored.case)
    except Exception as exc:
        raise AuthoringPending(
            "AM-11 public Euler–Poisson AMR validate failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        return resolve_case(validated, layout=authored.layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            "AM-11 Euler–Poisson AMR resolve failed after public validate "
            f"({type(exc).__name__}: {exc}); FFT is refused on AMR, "
            "this Case uses GeometricMG + CompositeHierarchySolve"
        ) from exc


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


def _level0_electrons(simulation, n_cells: int):
    """Return conserved electrons on level 0 as a documented (2, n) array."""
    getter = getattr(simulation, "block_level_state_global", None)
    if callable(getter):
        field = np.asarray(getter("electrons", 0), dtype=np.float64)
    else:
        field = np.asarray(simulation.state_global("electrons"), dtype=np.float64)
    return np.reshape(field, (2, int(n_cells)))


def run_native(n_cells: int = DEFAULT_N_CELLS, t_end: float = 0.05):
    """Compile, bind, and run the Euler–Poisson AMR Case.

    Returns the level-0 electron conserved state with shape ``(2, n_cells)``.
    Fine-level leaf data is not packed into this array. Raises
    ``NativeUnavailable`` without a compiler/Kokkos, or when compile/bind/run
    cannot complete.
    """
    import pops

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    try:
        authored = _author(n_cells)
        from verification.pops_verify.case_authoring import resolve_case

        plan = resolve_case(authored.case, layout=authored.layout)
    except AuthoringPending as exc:
        raise NativeUnavailable(str(exc)) from exc
    except Exception as exc:
        raise NativeUnavailable(
            f"AM-11 Euler–Poisson AMR resolve failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        artifact = pops.compile(plan)
        initial = np.ascontiguousarray(
            initial_conserved(authored.n_cells), dtype=np.float64
        )
        simulation = pops.bind(
            artifact, initial_values={authored.instance: initial}
        )
        pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
        return _level0_electrons(simulation, authored.n_cells)
    except NativeUnavailable:
        raise
    except Exception as exc:
        raise NativeUnavailable(
            f"AM-11 native compile/bind/run failed: {type(exc).__name__}: {exc}"
        ) from exc
