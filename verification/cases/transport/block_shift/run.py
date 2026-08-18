"""TR-05 translated block faces plus a public 1-d AMR coarse-fine join.

In-memory helpers sample the exact Gaussian on three cell-face joins. The
live Case is AM-01-style static AMR whose fine patch starts at a chosen
interface, so translating the join is a real multi-block crossing.
``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_TR02 = load_sibling_module(_CASE_DIR.parent / "gaussian_pulse" / "exact.py")

ADVECTION_SPEED = float(_TR02.A)
CFL = 0.4
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def exact_fields_for_interfaces(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0):
    """Return exact fields keyed by the three canonical interface positions."""
    return {
        float(interface): _exact.exact_on_decomposition(n_cells, interface, t)
        for interface in _exact.INTERFACE_X
    }


def max_decomposition_difference(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return the max pairwise L∞ between the three exact placements."""
    fields = exact_fields_for_interfaces(n_cells, t)
    volumes = _exact.cell_volumes(n_cells)
    linf = 0.0
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        linf = max(linf, errors.linf)
    return float(linf)


class _Authoring:
    __slots__ = (
        "case",
        "instance",
        "marker_instance",
        "frame",
        "n_cells",
        "program",
        "interface",
    )

    def __init__(
        self,
        case: Any,
        instance: Any,
        marker_instance: Any,
        frame: Any,
        n_cells: int,
        program: Any,
        interface: float,
    ) -> None:
        self.case = case
        self.instance = instance
        self.marker_instance = marker_instance
        self.frame = frame
        self.n_cells = n_cells
        self.program = program
        self.interface = interface


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("tr05-line", (0.0,), (1.0,)).frame(Cartesian1D())


def _gaussian_initial(frame):
    """Public analytic IC matching the TR-02 pulse at t=0."""
    from pops.analytic import exp, x as analytic_x
    from pops.lib.initial import Analytic

    displacement = analytic_x(frame) - float(_TR02.X0)
    profile = float(_TR02.Q0) + float(_TR02.AMP) * exp(
        -(displacement * displacement) / (2.0 * float(_TR02.SIGMA) ** 2)
    )
    return Analytic(frame=frame, components=(profile,))


def _interface_marker(frame, interface: float):
    """Spatial indicator: 1 on x>interface (translated two-block join)."""
    from pops.analytic import where, x as analytic_x
    from pops.lib.initial import Analytic

    return Analytic(
        frame=frame,
        components=(where(analytic_x(frame) > float(interface), 1.0, 0.0),),
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


def _author(n_cells: int, interface: float = 0.25) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import SSPRK2
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    cut = float(interface)
    frame = _frame()
    tracer_model, tracer_state, tracer_flux, tracer_rate, velocity = _scalar_transport(
        "tr05_tracer", frame, speed=ADVECTION_SPEED
    )
    marker_model, marker_state, marker_flux, marker_rate, _ = _scalar_transport(
        "tr05_marker", frame, speed=0.0
    )
    case = pops.Case("tr05-block-shift")
    tracer = case.block("tracer", model=tracer_model, states=(tracer_state,))
    marker = case.block("marker", model=marker_model, states=(marker_state,))
    instance = tracer[tracer_state]
    marker_instance = marker[marker_state]

    def _add_numerics(block, state, flux, rate, solver):
        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
                riemann=solver,
            ),
        )
        case.numerics(numerics, block=block)

    _add_numerics(
        tracer, tracer_state, tracer_flux, tracer_rate, riemann.ScalarUpwind(velocity=velocity)
    )
    _add_numerics(marker, marker_state, marker_flux, marker_rate, riemann.Rusanov())
    program = SSPRK2(instance, rate=tracer_rate)
    marker_time = program.state(marker_instance)
    program.commit(
        marker_time.next,
        program.value("marker_hold", marker_time.n, at=marker_time.next.point),
    )
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_gaussian_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=marker_instance,
            value=_interface_marker(frame, cut),
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
        interface=cut,
    )


def amr_shift_layout(authored: _Authoring):
    """1-d periodic AMR layout: fine patch on x>interface."""
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
    threshold = case.param(RuntimeParam("tr05_refine_x", default=0.5))
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


def build_case(n_cells: int = _exact.DEFAULT_N_CELLS, interface: float = 0.25):
    """Author the AMR Case whose fine patch starts at ``interface``."""
    return _author(n_cells, interface=interface).case


def resolve_plan(n_cells: int = _exact.DEFAULT_N_CELLS, interface: float = 0.25):
    """Validate and resolve the AMR Case. Does not compile or execute a run."""
    from verification.pops_verify.case_authoring import resolve_case

    authored = _author(n_cells, interface=interface)
    return resolve_case(authored.case, layout=amr_shift_layout(authored))


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


def run_native(n_cells: int = 32, t_end: float = 0.25, interface: float = 0.25):
    """Compile, bind, and run the pulse across a translated AMR join."""
    import pops

    from verification.pops_verify.case_authoring import bind_public, resolve_case

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells, interface=interface)
    plan = resolve_case(authored.case, layout=amr_shift_layout(authored))
    artifact = pops.compile(plan)
    simulation = bind_public(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(
        simulation.block_level_state_global("tracer", 0), dtype=np.float64
    )
    return np.ravel(field)[: authored.n_cells]
