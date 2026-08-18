"""TR-04 two-block placements plus a public 1-d AMR coarse-fine crossing.

In-memory helpers sample the exact Gaussian on the two-block join at the
three crossing times. The live Case is AM-01-style static AMR: fine patch
on x>0.5 so the TR-02 pulse crosses a real coarse-fine interface.
``pops.run`` is used only inside ``run_native``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

ADVECTION_SPEED = float(_exact.A)
CFL = 0.4
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def two_block_join():
    """Return the 1-d two-block edges (face at 0.5)."""
    return _exact.BLOCK_EDGES


def sample_placement(name, n_cells=None):
    """Sample the exact Gaussian on the two-block join at a named placement."""
    count = _exact.N_CELLS if n_cells is None else int(n_cells)
    centers, volumes = _exact.two_block_cell_centers(count)
    field = _exact.exact_on_placement(name, centers)
    return centers, field, volumes


def placement_fields(n_cells=None):
    """Return {placement: (x, q, volumes)} for face, edge, and corner."""
    count = _exact.N_CELLS if n_cells is None else int(n_cells)
    return {name: sample_placement(name, count) for name in _exact.PLACEMENTS}


class _Authoring:
    __slots__ = ("case", "instance", "marker_instance", "frame", "n_cells", "program")

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


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("tr04-line", (0.0,), (1.0,)).frame(Cartesian1D())


def _gaussian_initial(frame):
    """Public analytic IC matching the TR-02 pulse at t=0."""
    from pops.analytic import exp, x as analytic_x
    from pops.lib.initial import Analytic

    displacement = analytic_x(frame) - float(_exact.X0)
    profile = float(_exact.Q0) + float(_exact.AMP) * exp(
        -(displacement * displacement) / (2.0 * float(_exact.SIGMA) ** 2)
    )
    return Analytic(frame=frame, components=(profile,))


def _right_half_marker(frame):
    """Spatial indicator: 1 on x>0.5, the two-block join."""
    from pops.analytic import where, x as analytic_x
    from pops.lib.initial import Analytic

    return Analytic(
        frame=frame,
        components=(where(analytic_x(frame) > float(_exact.FACE), 1.0, 0.0),),
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


def _author(n_cells: int) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
    from pops.lib.time import SSPRK2
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    frame = _frame()
    tracer_model, tracer_state, tracer_flux, tracer_rate, velocity = _scalar_transport(
        "tr04_tracer", frame, speed=ADVECTION_SPEED
    )
    marker_model, marker_state, marker_flux, marker_rate, _ = _scalar_transport(
        "tr04_marker", frame, speed=0.0
    )
    case = pops.Case("tr04-block-crossings")
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
            value=_right_half_marker(frame),
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


def amr_crossing_layout(authored: _Authoring):
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
    threshold = case.param(RuntimeParam("tr04_refine_x", default=0.5))
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


def build_case(n_cells: int = _exact.N_CELLS):
    """Author a 1-d periodic conservative scalar advection Case."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = _exact.N_CELLS):
    """Validate and resolve the AMR crossing Case. Does not compile or execute a run."""
    from verification.pops_verify.case_authoring import resolve_case

    authored = _author(n_cells)
    return resolve_case(authored.case, layout=amr_crossing_layout(authored))


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


def run_native(n_cells: int = _exact.N_CELLS, t_end: float | None = None):
    """Compile, bind, and run the pulse across the AMR coarse-fine join."""
    import pops

    from verification.pops_verify.case_authoring import bind_public, resolve_case

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    plan = resolve_case(authored.case, layout=amr_crossing_layout(authored))
    artifact = pops.compile(plan)
    simulation = bind_public(artifact)
    horizon = _exact.placement_time("face") if t_end is None else float(t_end)
    pops.run(simulation, t_end=horizon, max_steps=MAX_STEPS)
    field = np.asarray(
        simulation.block_level_state_global("tracer", 0), dtype=np.float64
    )
    return np.ravel(field)[: authored.n_cells]
