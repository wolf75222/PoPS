"""Public 1-d periodic AMR Case for AM-06 level-count authoring.

The in-memory ``request_level_count`` gate stays. Request 1, 2, or 3 is
accepted. Request 4 is refused before any compile artefact, with
``requested_level_count=`` and ``supported_level_count=`` in the message.
Optional native compile/bind/run when Kokkos and a compiler exist.
``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

A = 1.0
CFL = 0.45
MAX_STEPS = 100_000
REFINE_THRESHOLD = 0.5
INTERFACE_X = 0.5
DEFAULT_N_COARSE = 8
DEFAULT_LEVELS = 2


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


def request_level_count(requested_level_count: int) -> int:
    """Accept a requested level count if it does not exceed the documented support."""
    supported = _exact.supported_level_count()
    requested = int(requested_level_count)
    if requested > supported:
        raise ValueError(
            f"requested_level_count={requested} exceeds supported_level_count={supported}"
        )
    return requested


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("am06_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


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


def _author(n_coarse: int) -> _Authoring:
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
        raise ValueError("n_coarse must be positive")
    frame = _frame()
    tracer_model, tracer_state, tracer_flux, tracer_rate, velocity = _scalar_transport(
        "am06_tracer", frame, speed=A
    )
    marker_model, marker_state, marker_flux, marker_rate, _ = _scalar_transport(
        "am06_marker", frame, speed=0.0
    )

    case = pops.Case("am06_level_count")
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
    return _Authoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        frame=frame,
        n_coarse=count,
        program=program,
    )


def _hierarchy_for_levels(levels: int):
    from pops.amr import AMRHierarchy

    if levels == 1:
        return AMRHierarchy(max_levels=1, ratios=())
    if levels == 2:
        return AMRHierarchy(max_levels=2, ratios=(2,))
    return AMRHierarchy(max_levels=3, ratios=(2, 2))


def _execution_for_levels(levels: int):
    from pops.amr import AMRClockRelation, AMRExecution

    if levels == 1:
        return AMRExecution.synchronous()
    return AMRExecution.subcycled(
        tuple(AMRClockRelation(level, level + 1, 2) for level in range(levels - 1))
    )


def amr_periodic_layout(authored: _Authoring, *, levels: int = DEFAULT_LEVELS):
    """1-d periodic layout with the requested materialized level count.

    Level 1 is Uniform (single-level). Levels 2 and 3 are AMR hierarchies.
    """
    from pops.amr import (
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
    from verification.pops_verify.case_authoring import uniform_periodic_layout

    count = request_level_count(levels)
    if count == 1:
        return uniform_periodic_layout(authored.frame, (authored.n_coarse,))
    case = authored.case
    threshold = case.param(RuntimeParam("am06_refine_x", default=REFINE_THRESHOLD))
    transfer = AMRTransfer()
    transfer.state(authored.instance, StateTransfer())
    transfer.state(authored.marker_instance, StateTransfer())
    return AMR(
        grid=CartesianGrid(
            frame=authored.frame,
            cells=(authored.n_coarse,),
            periodic=PeriodicAxes(authored.frame.axes),
        ),
        hierarchy=_hierarchy_for_levels(count),
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
        execution=_execution_for_levels(count),
    )


def build_case(n_coarse=None, levels=DEFAULT_LEVELS):
    """Author the 1-d periodic AMR Case. Does not compile or run."""
    request_level_count(levels)
    count = DEFAULT_N_COARSE if n_coarse is None else int(n_coarse)
    return _author(count).case


def resolve_plan(n_coarse=None, levels=DEFAULT_LEVELS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import resolve_case

    requested = request_level_count(levels)
    count = DEFAULT_N_COARSE if n_coarse is None else int(n_coarse)
    try:
        authored = _author(count)
        layout = amr_periodic_layout(authored, levels=requested)
        return resolve_case(authored.case, layout=layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            f"AM-06 1-d periodic AMR resolve failed: {type(exc).__name__}: {exc}"
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


def run_native(n_coarse=None, t_end=1.0, levels=DEFAULT_LEVELS):
    """Compile, bind, and run the Case.

    Returns the level-0 tracer state as a 1-d array of shape ``(n_coarse,)``.
    Fine-level leaf data is not packed into this array. Raises
    ``NativeUnavailable`` without a compiler/Kokkos, or when compile/bind/run
    cannot complete. Over-request (levels > 3) raises ``ValueError`` from
    ``request_level_count`` before any compile artefact.
    """
    import pops

    from verification.pops_verify.case_authoring import resolve_case

    requested = request_level_count(levels)
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    count = DEFAULT_N_COARSE if n_coarse is None else int(n_coarse)
    authored = _author(count)
    layout = amr_periodic_layout(authored, levels=requested)
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    simulation = pops.bind(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(
        simulation.block_level_state_global("tracer", 0),
        dtype=np.float64,
    )
    return np.reshape(field, (authored.n_coarse,))
