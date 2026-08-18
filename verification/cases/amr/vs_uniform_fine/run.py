"""Public 1-d periodic Cases for AM-07 three-series comparison.

In-memory manufactured E∝h² helpers stay. Optional native compile/bind/run
when Kokkos and a compiler exist. ``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

SERIES_UNIFORM_H = "uniform_h"
SERIES_AMR_H_FINE_H2 = "amr_h_fine_h2"
SERIES_UNIFORM_H2 = "uniform_h2"
SERIES = (SERIES_UNIFORM_H, SERIES_AMR_H_FINE_H2, SERIES_UNIFORM_H2)

A = 1.0
CFL = 0.45
MAX_STEPS = 100_000
REFINE_THRESHOLD = 0.5
INTERFACE_X = 0.5
DEFAULT_N_CELLS = 16


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class AuthoringPending(RuntimeError):
    """Raised when public 1-d periodic validate/resolve cannot complete."""


class _UniformAuthoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


class _AmrAuthoring:
    __slots__ = (
        "case",
        "instance",
        "marker_instance",
        "frame",
        "n_coarse",
        "program",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        marker_instance: Any,
        frame: Any,
        n_coarse: int,
        program: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.frame = frame
        self.n_coarse = n_coarse
        self.program = program


def local_spacing(series: str, h) -> float:
    """Return the local cell spacing used by one of the three series."""
    if series == SERIES_UNIFORM_H:
        return float(h)
    if series in (SERIES_AMR_H_FINE_H2, SERIES_UNIFORM_H2):
        return _exact.fine_spacing(h)
    raise ValueError(f"unknown series {series!r}")


def series_error(series: str, h) -> float:
    """Manufactured E∝h² evaluated at the series local spacing."""
    return _exact.manufactured_error(local_spacing(series, h))


def three_series(h) -> dict[str, float]:
    """Return the three AM-07 error scalars at base spacing h."""
    return {name: series_error(name, h) for name in SERIES}


def _normalize_n_cells(n_cells) -> int:
    count = DEFAULT_N_CELLS if n_cells is None else int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    return count


def _normalize_series(series) -> str:
    name = SERIES_AMR_H_FINE_H2 if series is None else str(series)
    if name not in SERIES:
        raise ValueError(f"unknown series {series!r}")
    return name


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("am07_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _sine_initial(frame):
    """Public analytic IC matching the TR-01 sine at t=0."""
    import math

    from pops.analytic import sin, x as analytic_x
    from pops.lib.initial import Analytic

    q0 = 1.0
    eps = 1.0e-2
    wave = 2.0 * math.pi
    return Analytic(frame=frame, components=(q0 + eps * sin(wave * analytic_x(frame)),))


def _right_half_marker(frame):
    """Spatial indicator: 1 on x>0.5, 0 on the coarse left half."""
    from pops.analytic import where, x as analytic_x
    from pops.lib.initial import Analytic

    return Analytic(
        frame=frame,
        components=(where(analytic_x(frame) > INTERFACE_X, 1.0, 0.0),),
    )


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


def _author_uniform(n_cells: int) -> _UniformAuthoring:
    """TR-01 Uniform periodic scalar advection on ``n_cells``."""
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import SSPRK2
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    frame = _frame()
    model, state, flux, rate, velocity = _scalar_transport(
        "am07_advection", frame, speed=A
    )
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case("am07_uniform")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_sine_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    return _UniformAuthoring(case=case, instance=instance, frame=frame, n_cells=count)


def _author_amr(n_coarse: int) -> _AmrAuthoring:
    """AM-01 style 2-level AMR: fine patch on the right half x>0.5."""
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import SSPRK2
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_coarse)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    frame = _frame()
    tracer_model, tracer_state, tracer_flux, tracer_rate, velocity = _scalar_transport(
        "am07_tracer", frame, speed=A
    )
    marker_model, marker_state, marker_flux, marker_rate, _ = _scalar_transport(
        "am07_marker", frame, speed=0.0
    )

    case = pops.Case("am07_amr_h_fine_h2")
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
        riemann.Rusanov(),
    )

    program = SSPRK2(instance, rate=tracer_rate)
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
            value=_sine_initial(frame),
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
    return _AmrAuthoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        frame=frame,
        n_coarse=count,
        program=program,
    )


def amr_periodic_layout(authored: _AmrAuthoring):
    """1-d periodic AMR layout: fine patch on the right half x>0.5."""
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

    case = authored.case
    threshold = case.param(RuntimeParam("am07_refine_x", default=REFINE_THRESHOLD))
    transfer = AMRTransfer()
    transfer.state(authored.instance, StateTransfer())
    transfer.state(authored.marker_instance, StateTransfer())
    return AMR(
        grid=CartesianGrid(
            frame=authored.frame,
            cells=(authored.n_coarse,),
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
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )


def _prepare(n_cells=None, series=None):
    """Author the series Case and its layout. Does not compile or run."""
    from verification.pops_verify.case_authoring import uniform_periodic_layout

    count = _normalize_n_cells(n_cells)
    name = _normalize_series(series)
    if name == SERIES_AMR_H_FINE_H2:
        authored = _author_amr(count)
        return authored, amr_periodic_layout(authored), count
    n_uniform = count if name == SERIES_UNIFORM_H else 2 * count
    authored = _author_uniform(n_uniform)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return authored, layout, n_uniform


def build_case(n_cells=None, series=SERIES_AMR_H_FINE_H2):
    """Author the public Case for one series. Does not compile or run."""
    authored, _, _ = _prepare(n_cells, series)
    return authored.case


def resolve_plan(n_cells=None, series=SERIES_AMR_H_FINE_H2):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import resolve_case

    try:
        authored, layout, _ = _prepare(n_cells, series)
        return resolve_case(authored.case, layout=layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        name = _normalize_series(series)
        raise AuthoringPending(
            f"AM-07 series {name!r} resolve failed: {type(exc).__name__}: {exc}"
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


def run_native(n_cells=None, t_end=1.0, series=SERIES_AMR_H_FINE_H2):
    """Compile, bind, and run one series.

    Uniform series return the full tracer as a 1-d array of shape ``(n,)``
    or ``(2 n,)``. The AMR series returns the level-0 tracer as shape
    ``(n_cells,)``. Raises ``NativeUnavailable`` without a compiler/Kokkos,
    or when compile/bind/run cannot complete.
    """
    import numpy as np
    import pops

    from verification.pops_verify.case_authoring import resolve_case

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    name = _normalize_series(series)
    authored, layout, n_out = _prepare(n_cells, name)
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    simulation = pops.bind(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    if name == SERIES_AMR_H_FINE_H2:
        field = np.asarray(
            simulation.block_level_state_global("tracer", 0),
            dtype=np.float64,
        )
        return np.reshape(field, (n_out,))
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    return np.ravel(field)
