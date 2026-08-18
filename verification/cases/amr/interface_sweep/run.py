"""Public 1-d periodic AMR Case for AM-08 interface-placement sweep.

In-memory manufactured E∝h² helpers stay. Optional native compile/bind/run
when Kokkos and a compiler exist. ``pops.run`` is used only inside
``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.interface_error import (
    band_max_abs_error,
    interface_band_mask,
)

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

A = 1.0
CFL = 0.45
MAX_STEPS = 100_000
REFINE_THRESHOLD = 0.5
DEFAULT_N_CELLS = 32
DEFAULT_INTERFACE = 0.5
FINE_FRACTION = 0.5


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
        "interface",
        "program",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        marker_instance: Any,
        frame: Any,
        n_cells: int,
        interface: float,
        program: Any,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.frame = frame
        self.n_cells = n_cells
        self.interface = interface
        self.program = program


def interface_error_at(x0, n_cells: int) -> float:
    """Return manufactured E_cf at one interface placement."""
    centers, _ = _exact.uniform_cell_centers(n_cells)
    spacing = 1.0 / float(n_cells)
    field = _exact.manufactured_field(centers, spacing, x0)
    oracle = _exact.oracle(centers)
    mask = interface_band_mask(
        _exact.periodic_distance(centers, x0),
        h_fine=spacing,
        band_cells=_exact.BAND_CELLS,
    )
    return band_max_abs_error(field, oracle, mask)


def bulk_error_at(x0, n_cells: int) -> float:
    """Return manufactured E_bulk (complement of the interface band)."""
    centers, _ = _exact.uniform_cell_centers(n_cells)
    spacing = 1.0 / float(n_cells)
    field = _exact.manufactured_field(centers, spacing, x0)
    oracle = _exact.oracle(centers)
    interface = interface_band_mask(
        _exact.periodic_distance(centers, x0),
        h_fine=spacing,
        band_cells=_exact.BAND_CELLS,
    )
    return band_max_abs_error(field, oracle, np.logical_not(interface))


def sweep_interface_errors(n_cells: int):
    """Return (placements, E_cf) for the default x0 sweep at one resolution."""
    placements = _exact.interface_placements()
    errors = np.array(
        [interface_error_at(x0, n_cells) for x0 in placements],
        dtype=np.float64,
    )
    return placements, errors


def error_envelope(n_cells: int) -> tuple[float, float]:
    """Return (min, max) of manufactured E_cf over the x0 sweep."""
    _, errors = sweep_interface_errors(n_cells)
    return float(np.min(errors)), float(np.max(errors))


def worst_placement(n_cells: int) -> float:
    """Return the x0 that maximises manufactured E_cf at this resolution."""
    placements, errors = sweep_interface_errors(n_cells)
    return float(placements[int(np.argmax(errors))])


def worst_error_series(resolutions=None):
    """Return (errors, spacings, worst_x0) for the manufactured E ∝ h² series."""
    series = tuple(resolutions if resolutions is not None else _exact.RESOLUTIONS)
    worst_x0 = worst_placement(series[0])
    errors = np.array(
        [interface_error_at(worst_x0, n_cells) for n_cells in series],
        dtype=np.float64,
    )
    spacings = np.array([1.0 / float(n_cells) for n_cells in series], dtype=np.float64)
    return errors, spacings, worst_x0


def _normalize_n_cells(n_cells) -> int:
    count = DEFAULT_N_CELLS if n_cells is None else int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    return count


def _normalize_interface(interface) -> float:
    x0 = DEFAULT_INTERFACE if interface is None else float(interface)
    x0 = x0 % 1.0
    if x0 < 0.0:
        x0 += 1.0
    if x0 >= 1.0:
        x0 = 0.0
    return x0


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("am08_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _sine_initial(frame):
    """Public analytic IC matching the TR-01 sine at t=0."""
    import math

    from pops.analytic import sin, x as analytic_x
    from pops.lib.initial import Analytic

    q0 = 1.0
    eps = 1.0e-2
    wave = 2.0 * math.pi
    return Analytic(frame=frame, components=(q0 + eps * sin(wave * analytic_x(frame)),))


def _interface_marker(frame, x0: float):
    """Spatial indicator: 1 on the periodic half-interval starting at x0.

    At ``x0=0.5`` this is ``where(x > 0.5, 1, 0)``. Other placements wrap
    across the periodic seam so the fine patch stays length 1/2.
    """
    from pops.analytic import where, x as analytic_x
    from pops.lib.initial import Analytic

    x_coord = analytic_x(frame)
    delta = x_coord - float(x0)
    wrapped = where(delta >= 0.0, delta, delta + 1.0)
    return Analytic(
        frame=frame,
        components=(where(wrapped < FINE_FRACTION, 1.0, 0.0),),
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


def _author(n_cells: int, interface: float) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import SSPRK2
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = _normalize_n_cells(n_cells)
    x0 = _normalize_interface(interface)
    frame = _frame()
    tracer_model, tracer_state, tracer_flux, tracer_rate, velocity = _scalar_transport(
        "am08_tracer", frame, speed=A
    )
    marker_model, marker_state, marker_flux, marker_rate, _ = _scalar_transport(
        "am08_marker", frame, speed=0.0
    )

    case = pops.Case("am08_interface_sweep")
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
            value=_interface_marker(frame, x0),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(
        case=case,
        instance=instance,
        marker_instance=marker_instance,
        frame=frame,
        n_cells=count,
        interface=x0,
        program=program,
    )


def amr_periodic_layout(authored: _Authoring):
    """1-d periodic AMR layout: frozen fine patch on the half-interval at x0."""
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
    threshold = case.param(RuntimeParam("am08_refine_x", default=REFINE_THRESHOLD))
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
        regrid=AMRRegrid.frozen(),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )


def build_case(n_cells=None, interface=DEFAULT_INTERFACE):
    """Author the 1-d periodic AMR Case. Does not compile or run."""
    return _author(_normalize_n_cells(n_cells), _normalize_interface(interface)).case


def resolve_plan(n_cells=None, interface=DEFAULT_INTERFACE):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    from verification.pops_verify.case_authoring import resolve_case

    count = _normalize_n_cells(n_cells)
    x0 = _normalize_interface(interface)
    try:
        authored = _author(count, x0)
        layout = amr_periodic_layout(authored)
        return resolve_case(authored.case, layout=layout)
    except AuthoringPending:
        raise
    except Exception as exc:
        raise AuthoringPending(
            f"AM-08 1-d periodic AMR resolve failed: {type(exc).__name__}: {exc}"
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


def run_native(n_cells=None, t_end=1.0, interface=DEFAULT_INTERFACE):
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
    count = _normalize_n_cells(n_cells)
    x0 = _normalize_interface(interface)
    try:
        authored = _author(count, x0)
        layout = amr_periodic_layout(authored)
        plan = resolve_case(authored.case, layout=layout)
    except AuthoringPending as exc:
        raise NativeUnavailable(str(exc)) from exc
    except Exception as exc:
        raise NativeUnavailable(
            f"AM-08 1-d periodic AMR resolve failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
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
            f"AM-08 native compile/bind/run failed: {type(exc).__name__}: {exc}"
        ) from exc
