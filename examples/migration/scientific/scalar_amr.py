"""Scalar advection with AMR and the same user SSPRK2 method inline or imported.

The physical flux, reconstruction, Riemann solver, initial condition, and AMR
policy are authored here. The run function exposes the complete public lifecycle.
"""

from __future__ import annotations
import argparse
from typing import Any
from fractions import Fraction
import pops


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
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Gaussian
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt, every

from .my_time_schemes import ssprk2
from .runtime import _execution_resources, _state_summary
from .restart_checks import (
    _snapshot,
    _require_same_snapshot,
    _require_refined_hierarchy,
    _require_regrid_progress,
)

AX, AY = 1.0, 0.25
FAR_FIELD = 0.05


def inline_ssprk2(state, rate, *, dt):
    """Build the explicit two-stage Shu--Osher method with ordinary Program values."""
    program = pops.Program("user_ssprk2")
    q = program.state(state)
    stage_0 = program.stage("stage_0", c=0)
    k0 = program.value("k0", rate(q.n), at=stage_0)
    stage_1 = program.stage("stage_1", c=1)
    q1 = program.value("q1", q.n + program.dt * k0, at=stage_1)
    k1 = program.value("k1", rate(q1), at=stage_1)
    half = Fraction(1, 2)
    advanced = program.value(
        "advanced",
        q.n + program.dt * half * k0 + program.dt * half * k1,
        at=q.next.point,
    )
    program.commit(q.next, advanced)
    program.step_strategy(FixedDt(dt))
    return program


def build_case(cells: int, *, scheme: str = "inline"):
    if scheme not in ("inline", "imported"):
        raise ValueError("scheme must be inline or imported")
    NX = NY = cells
    dt = min(0.01, 0.2 / cells)
    # 1. Domaine, repere et grille grossiere de la hierarchie adaptative.
    domain = Rectangle(
        "unit_square",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).tag("fluid")

    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    grid = CartesianGrid(frame=frame, cells=(NX, NY))

    # 2. Physique : d_t U + div(a U) = 0.
    model = pops.Model("scalar_advection_amr", frame=frame)

    U = model.state(
        "U",
        components=("u",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (u,) = U

    velocity = model.vector(
        "a",
        frame=frame,
        components={x_axis: AX, y_axis: AY},
    )

    physical_flux = model.flux(
        "advection_flux",
        frame=frame,
        state=U,
        components={x_axis: (AX * u,), y_axis: (AY * u,)},
        waves={x_axis: (AX,), y_axis: (AY,)},
    )

    advection_rate = model.rate(
        "advection_rate",
        equation=ddt(U) == -div(physical_flux),
    )

    # 3. Methode spatiale identique a la variante preset.
    finite_volume = FiniteVolume(
        flux=physical_flux,
        variables=variables.Conservative(U),
        reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    )

    numerics = DiscretizationPlan()
    numerics.rates.add(advection_rate, finite_volume)

    # 4. Bloc qualifie et conditions aux limites de transport.
    case = pops.Case("tutorial_scalar_advection_amr")
    tracer = case.block("tracer", model=model)
    tracer_U = tracer[U]

    boundaries = frame.boundaries
    transport_boundaries = TransportBoundarySet(
        {
            boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
            boundaries.x_max: Outflow(state=tracer_U),
            boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
            boundaries.y_max: Outflow(state=tracer_U),
        }
    )

    numerics.boundaries.add(transport_boundaries)
    case.numerics(numerics, block=tracer)

    # 5. La meme methode peut etre ecrite ici ou importee comme du Python ordinaire.
    method = inline_ssprk2 if scheme == "inline" else ssprk2
    program = method(tracer_U, advection_rate, dt=dt)
    case.program(program)

    # 6. Condition initiale analytique, projetee conservativement sur chaque niveau AMR.
    case.initials.add(
        InitialCondition(
            state=tracer_U,
            value=Gaussian(
                frame=frame,
                center={x_axis: 0.30, y_axis: 0.35},
                background=FAR_FIELD,
                amplitude=0.95,
                inverse_width=120.0,
            ),
            projection=ConservativeCellAverage(),
        )
    )

    # 7. Meme hierarchie, meme tagging et meme subcycling que la variante preset.
    refine_threshold = case.param(RuntimeParam("refine_u", default=0.30))
    coarsen_threshold = case.param(RuntimeParam("coarsen_u", default=0.20))

    tagging = AMRTagging(
        rules=(
            Tag(ValueExpr(tracer_U) > case.value(refine_threshold)),
            Coarsen(ValueExpr(tracer_U) < case.value(coarsen_threshold)),
            Buffer(cells=2),
        ),
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )

    transfer = AMRTransfer()
    transfer.state(tracer_U, StateTransfer())

    layout = AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=tagging,
        regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )

    return case, layout, dt


def run(args: argparse.Namespace) -> dict[str, Any]:
    case, layout, dt = build_case(args.cells, scheme=getattr(args, "scheme", "inline"))
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, resources=_execution_resources(artifact))
    steps = getattr(args, "steps", None)
    steps = round(0.2 / dt) if steps is None else steps
    report = pops.run(runtime, t_end=steps * dt, max_steps=steps, console=False)
    accepted = _snapshot(runtime)
    _require_refined_hierarchy(accepted, where="accepted scalar run")
    output_dir = args.work_dir / "scalar-amr"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = runtime.checkpoint(output_dir / "accepted_restart")
    resumed = pops.bind(artifact, resources=_execution_resources(artifact))
    resumed.restart(checkpoint)
    restored = _snapshot(resumed)
    _require_same_snapshot(accepted, restored, where="strict restart")
    final_time = 2 * steps * dt
    pops.run(runtime, t_end=final_time, max_steps=2 * steps, console=False)
    pops.run(resumed, t_end=final_time, max_steps=2 * steps, console=False)
    continuous, restarted = _snapshot(runtime), _snapshot(resumed)
    _require_same_snapshot(continuous, restarted, where="continued restart")
    _require_regrid_progress(accepted, continuous, where="continuous trajectory")
    _require_regrid_progress(restored, restarted, where="restarted trajectory")
    patches = runtime.amr.patch_table()
    regrid = runtime.amr.explain_regrid()
    return {
        "workflow": "scalar-amr",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "scheme": getattr(args, "scheme", "inline"),
        "restart_bit_identical": True,
        "checkpoint": str(checkpoint),
        "levels": runtime.n_levels(),
        "fine_patches": patches.n_patches,
        "regrids": regrid.regrid_count,
        "state": _state_summary(runtime, "tracer"),
        "exit_code": 0,
        "output_dir": str(args.work_dir / "scalar-amr"),
    }
