"""Public 1-d periodic AMR Case for AM-05 regrid-frequency sweep.

In-memory helpers stay. The tagged window follows a marker advected at the
TR-02 exact speed ``a`` from ``x0``. Regrid cadence is
``AMRRegrid(schedule=every(k, clock=program.clock))`` for
``k in {1, 2, 4, 8, 16}``. Optional native compile/bind/run when Kokkos and
a compiler exist. ``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_TR02_DIR = Path(__file__).resolve().parents[2] / "transport" / "gaussian_pulse"
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_tr02 = load_sibling_module(_TR02_DIR / "exact.py")

N_CELLS = _exact.N_CELLS
K_VALUES = _exact.K_VALUES
A = _tr02.A
X0 = _tr02.X0
CFL = 0.45
MAX_STEPS = 100_000
REFINE_THRESHOLD = 0.5
REGRID_EVERY = 2


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class AuthoringPending(RuntimeError):
    """Raised when public 1-d periodic AMR validate/resolve cannot complete."""


class _Authoring:
    __slots__ = (
        "case",
        "instance",
        "marker_instance",
        "frame",
        "n_cells",
        "program",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        marker_instance: Any,
        frame: Any,
        n_cells: int,
        program: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.frame = frame
        self.n_cells = n_cells
        self.program = program


def uniform_cells(n_cells: int = N_CELLS):
    """Return uniform cell centers and volumes on the periodic unit interval."""
    width = 1.0 / float(n_cells)
    centers = (np.arange(int(n_cells), dtype=np.float64) + 0.5) * width
    volumes = np.full(int(n_cells), width, dtype=np.float64)
    return centers, volumes


def sweep_regrid_frequency(k, *, n_steps=None, n_cells: int = N_CELLS):
    """Advance N steps, rebuild the tagged window every k steps, return counts."""
    steps = _exact.N_STEPS if n_steps is None else int(n_steps)
    requested = int(k)
    centers, volumes = uniform_cells(n_cells)
    rebuilds = 0
    window = None
    for step in range(steps):
        time = float(step) * float(_exact.DT)
        if _exact.should_rebuild(step, requested):
            window = _exact.tagged_window_mask(centers, _exact.patch_center(time))
            rebuilds += 1
    if rebuilds == 0:
        raise ValueError(f"regrid interval k={requested} produced no rebuilds")
    t_end = float(steps) * float(_exact.DT)
    field = _exact.exact_field(centers, t_end)
    k_regrid = float(steps) / float(rebuilds)
    return {
        "rebuilds": int(rebuilds),
        "k_requested": requested,
        "k_regrid": k_regrid,
        "leftover": _exact.interval_leftover(k_regrid, requested),
        "field": field,
        "window": window,
        "centers": centers,
        "volumes": volumes,
    }


def count_rebuilds(k, *, n_steps=None, n_cells: int = N_CELLS) -> int:
    """Return how many times the tagged window is rebuilt over N steps."""
    return sweep_regrid_frequency(k, n_steps=n_steps, n_cells=n_cells)["rebuilds"]


def observed_regrid_interval(k, *, n_steps=None, n_cells: int = N_CELLS) -> float:
    """Return k_regrid = N / rebuilds for the requested cadence."""
    sample = sweep_regrid_frequency(k, n_steps=n_steps, n_cells=n_cells)
    return float(sample["k_regrid"])


def interval_leftover(k, *, n_steps=None, n_cells: int = N_CELLS) -> float:
    """Return |k_regrid - k_requested| for one requested cadence."""
    sample = sweep_regrid_frequency(k, n_steps=n_steps, n_cells=n_cells)
    return float(sample["leftover"])


def exact_field_at_regrid_frequency(k, *, n_steps=None, n_cells: int = N_CELLS):
    """Return the exact pulse at t = N Δt after a sweep at this k."""
    return sweep_regrid_frequency(k, n_steps=n_steps, n_cells=n_cells)["field"]


def field_leftover_vs_inv_k(*, k_values=None, n_steps=None, n_cells: int = N_CELLS):
    """Return (1/k, L∞ leftover, slope) of the exact field against k-free oracle."""
    frequencies = _exact.K_VALUES if k_values is None else tuple(k_values)
    steps = _exact.N_STEPS if n_steps is None else int(n_steps)
    centers, volumes = uniform_cells(n_cells)
    reference = _exact.exact_field(centers, float(steps) * float(_exact.DT))
    inv_k = []
    leftovers = []
    for k in frequencies:
        field = exact_field_at_regrid_frequency(k, n_steps=steps, n_cells=n_cells)
        leftovers.append(reference_errors(field, reference, volumes).linf)
        inv_k.append(1.0 / float(k))
    slope = float(
        np.polyfit(
            np.asarray(inv_k, dtype=np.float64),
            np.asarray(leftovers, dtype=np.float64),
            1,
        )[0]
    )
    return inv_k, leftovers, slope


def _normalize_regrid_every(regrid_every) -> int:
    count = int(regrid_every)
    if count < 1:
        raise ValueError(f"regrid_every must be >= 1, got {regrid_every!r}")
    return count


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("am05_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _tr02_gaussian_initial(frame):
    """Analytic Gaussian matching the TR-02 pulse at t=0."""
    from pops.analytic import exp, x as analytic_x
    from pops.lib.initial import Analytic

    displacement = analytic_x(frame) - X0
    profile = _tr02.Q0 + _tr02.AMP * exp(
        -(displacement * displacement) / (2.0 * _tr02.SIGMA**2)
    )
    return Analytic(frame=frame, components=(profile,))


def _marker_bump_initial(frame):
    """Localized bump around x0; advected at speed a so the patch follows the pulse."""
    from pops.analytic import exp, x as analytic_x
    from pops.lib.initial import Analytic

    displacement = analytic_x(frame) - X0
    profile = exp(-(displacement * displacement) / (2.0 * _tr02.SIGMA**2))
    return Analytic(frame=frame, components=(profile,))


def _scalar_transport(name, frame, *, speed: float):
    import pops
    from pops.math import ddt, div
    from pops.representations import Conservative
    from pops.spaces import CellState

    (x_axis,) = frame.axes
    model = pops.Model(name, frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: speed})
    flux = model.flux(
        f"{name}_flux",
        frame=frame,
        state=state,
        components={x_axis: (speed * q,)},
        waves={x_axis: (speed,)},
    )
    rate = model.rate(f"{name}_rate", equation=ddt(state) == -div(flux))
    return model, state, flux, rate, velocity


def _author(n_cells: int) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import RungeKutta, RungeKuttaRoute, SSPRK2_TABLEAU
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    frame = _frame()
    tracer_model, tracer_state, tracer_flux, tracer_rate, velocity = _scalar_transport(
        "am05_tracer", frame, speed=A
    )
    marker_model, marker_state, marker_flux, marker_rate, marker_velocity = _scalar_transport(
        "am05_marker", frame, speed=A
    )

    case = pops.Case("am05_regrid_frequency")
    tracer = case.block("tracer", model=tracer_model, states=(tracer_state,))
    marker = case.block("marker", model=marker_model, states=(marker_state,))
    instance = tracer[tracer_state]
    marker_instance = marker[marker_state]

    def _add_numerics(block, state, flux, rate, riemann_solver):
        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
                riemann=riemann_solver,
            ),
        )
        case.numerics(numerics, block=block)

    _add_numerics(
        tracer,
        tracer_state,
        tracer_flux,
        tracer_rate,
        riemann.ScalarUpwind(velocity=velocity),
    )
    _add_numerics(
        marker,
        marker_state,
        marker_flux,
        marker_rate,
        riemann.ScalarUpwind(velocity=marker_velocity),
    )

    program = RungeKutta(
        routes=(
            RungeKuttaRoute(instance, tracer_rate),
            RungeKuttaRoute(marker_instance, marker_rate),
        ),
        tableau=SSPRK2_TABLEAU,
    )
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=instance,
            value=_tr02_gaussian_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=marker_instance,
            value=_marker_bump_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        frame=frame,
        n_cells=count,
        program=program,
    )


def amr_periodic_layout(authored: _Authoring, *, regrid_every: int = REGRID_EVERY):
    """1-d periodic AMR layout: fine patch follows the exact-speed marker bump."""
    from pops.amr import (
        AMRClockRelation,
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
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.math import ValueExpr
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.params import RuntimeParam
    from pops.time import every

    cadence = _normalize_regrid_every(regrid_every)
    case = authored.case
    threshold = case.param(RuntimeParam("am05_refine_marker", default=REFINE_THRESHOLD))
    transfer = AMRTransfer()
    transfer.state(authored.instance, StateTransfer())
    transfer.state(authored.marker_instance, StateTransfer())
    return AMR(
        grid=CartesianGrid(
            frame=authored.frame,
            cells=(authored.n_cells,),
            periodic=PeriodicAxes(authored.frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(authored.marker_instance) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(cadence, clock=authored.program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )


def build_case(n_cells=None, regrid_every=REGRID_EVERY):
    """Author the 1-d periodic AMR Case. Does not compile or run."""
    _normalize_regrid_every(regrid_every)
    count = N_CELLS if n_cells is None else int(n_cells)
    return _author(count).case


def resolve_plan(n_cells=None, regrid_every=REGRID_EVERY):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import resolve_case

    count = N_CELLS if n_cells is None else int(n_cells)
    cadence = _normalize_regrid_every(regrid_every)
    try:
        authored = _author(count)
        layout = amr_periodic_layout(authored, regrid_every=cadence)
        return resolve_case(authored.case, layout=layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            f"AM-05 1-d periodic AMR resolve failed: {type(exc).__name__}: {exc}"
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


def run_native(n_cells=None, t_end=1.0, regrid_every=REGRID_EVERY):
    """Compile, bind, and run the Case.

    Returns the level-0 tracer state as a 1-d array of shape ``(n_cells,)``.
    Fine-level leaf data is not packed into this array. Raises
    ``NativeUnavailable`` without a compiler/Kokkos, or when compile/bind/run
    cannot complete.
    """
    import pops

    from verification.pops_verify.case_authoring import resolve_case

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    count = N_CELLS if n_cells is None else int(n_cells)
    cadence = _normalize_regrid_every(regrid_every)
    try:
        authored = _author(count)
        layout = amr_periodic_layout(authored, regrid_every=cadence)
        plan = resolve_case(authored.case, layout=layout)
        artifact = pops.compile(plan)
        simulation = pops.bind(artifact)
        pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
        field = np.asarray(
            simulation.block_level_state_global("tracer", 0),
            dtype=np.float64,
        )
        return np.reshape(field, (authored.n_cells,))
    except NativeUnavailable:
        raise
    except Exception as exc:
        raise NativeUnavailable(
            f"AM-05 native compile/bind/run failed: {type(exc).__name__}: {exc}"
        ) from exc
