"""Public 1-d periodic AMR Case for AM-03 live tracer tagging.

In-memory helpers stay (sample_field, raw_tag_mask, buffered_tag_mask,
pulse_core_mask, hysteresis_update). The live Case tags the numerical
tracer, not a prescribed marker. Optional native compile/bind/run when
Kokkos and a compiler exist. ``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_TR02_DIR = Path(__file__).resolve().parents[2] / "transport" / "gaussian_pulse"
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_tr02 = load_sibling_module(_TR02_DIR / "exact.py")

N_CELLS = _exact.N_CELLS
A = _tr02.A
X0 = _tr02.X0
CFL = 0.45
MAX_STEPS = 100_000
REFINE_THRESHOLD = 0.5
COARSEN_THRESHOLD = 0.25
REGRID_EVERY = 2


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class AuthoringPending(RuntimeError):
    """Raised when public 1-d periodic AMR validate/resolve cannot complete."""


class _Authoring:
    __slots__ = (
        "case",
        "instance",
        "frame",
        "n_cells",
        "program",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        frame: Any,
        n_cells: int,
        program: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.program = program


def sample_field(n_cells=None, t=0.0):
    """Return (centers, TR-02 exact Gaussian) on a uniform periodic mesh."""
    count = N_CELLS if n_cells is None else int(n_cells)
    centers = _exact.uniform_centers(count)
    field = _tr02.exact_gaussian(centers, t, x0=_tr02.X0, a=_tr02.A)
    return centers, field


def raw_tag_mask(n_cells=None, t=0.0, *, theta=None, theta2=None) -> np.ndarray:
    """Tag |Δq| > θ or |second difference| > θ2 on the TR-02 exact field."""
    _, field = sample_field(n_cells, t)
    return _exact.raw_tag_mask(
        field,
        theta=_exact.THETA if theta is None else theta,
        theta2=_exact.THETA2 if theta2 is None else theta2,
    )


def buffered_tag_mask(buffer_cells, n_cells=None, t=0.0, *, theta=None, theta2=None):
    """Dilate the raw tag mask by `buffer_cells` on the periodic mesh."""
    return _exact.dilate_mask(raw_tag_mask(n_cells, t, theta=theta, theta2=theta2), buffer_cells)


def pulse_core_mask(n_cells=None, t=0.0) -> np.ndarray:
    """Documented pulse core: cells within CORE_RADIUS of the exact peak."""
    centers, _ = sample_field(n_cells, t)
    peak = (_tr02.X0 + _tr02.A * float(t)) % _tr02.PERIOD
    return _exact.pulse_core_mask(centers, x0=peak, radius=_exact.CORE_RADIUS)


def hysteresis_update(previous, n_cells=None, t=0.0, *, theta=None, theta2=None):
    """One refine/coarsen hysteresis step on the static TR-02 field."""
    _, field = sample_field(n_cells, t)
    return _exact.hysteresis_update(
        previous,
        _exact.first_difference(field),
        _exact.second_difference(field),
        theta=_exact.THETA if theta is None else theta,
        theta2=_exact.THETA2 if theta2 is None else theta2,
    )


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("am03_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _tr02_gaussian_initial(frame):
    """Analytic Gaussian matching the TR-02 pulse at t=0."""
    from pops.analytic import exp, x as analytic_x
    from pops.lib.initial import Analytic

    displacement = analytic_x(frame) - X0
    profile = _tr02.Q0 + _tr02.AMP * exp(
        -(displacement * displacement) / (2.0 * _tr02.SIGMA**2)
    )
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
        "am03_tracer", frame, speed=A
    )

    case = pops.Case("am03_tagging")
    tracer = case.block("tracer", model=tracer_model, states=(tracer_state,))
    instance = tracer[tracer_state]

    numerics = DiscretizationPlan()
    numerics.rates.add(
        tracer_rate,
        FiniteVolume(
            flux=tracer_flux,
            variables=variables.Conservative(tracer_state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case.numerics(numerics, block=tracer)

    program = RungeKutta(
        routes=(RungeKuttaRoute(instance, tracer_rate),),
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
    return _Authoring(
        case=case,
        instance=instance,
        frame=frame,
        n_cells=count,
        program=program,
    )


def amr_periodic_layout(authored: _Authoring):
    """1-d periodic AMR layout: live tagging of the advected tracer pulse."""
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Coarsen,
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

    case = authored.case
    refine = case.param(RuntimeParam("am03_refine_tracer", default=REFINE_THRESHOLD))
    coarsen = case.param(RuntimeParam("am03_coarsen_tracer", default=COARSEN_THRESHOLD))
    transfer = AMRTransfer()
    transfer.state(authored.instance, StateTransfer())
    return AMR(
        grid=CartesianGrid(
            frame=authored.frame,
            cells=(authored.n_cells,),
            periodic=PeriodicAxes(authored.frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(authored.instance) > case.value(refine)),
                Coarsen(ValueExpr(authored.instance) < case.value(coarsen)),
                Buffer(cells=2),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(REGRID_EVERY, clock=authored.program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )


def build_case(n_cells=None):
    """Author the 1-d periodic AMR Case. Does not compile or run."""
    count = N_CELLS if n_cells is None else int(n_cells)
    return _author(count).case


def resolve_plan(n_cells=None):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import resolve_case

    count = N_CELLS if n_cells is None else int(n_cells)
    try:
        authored = _author(count)
        layout = amr_periodic_layout(authored)
        return resolve_case(authored.case, layout=layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            f"AM-03 1-d periodic AMR resolve failed: {type(exc).__name__}: {exc}"
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


def run_native(n_cells=None, t_end=1.0):
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
    try:
        authored = _author(count)
        layout = amr_periodic_layout(authored)
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
            f"AM-03 native compile/bind/run failed: {type(exc).__name__}: {exc}"
        ) from exc
