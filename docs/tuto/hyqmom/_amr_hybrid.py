"""Execution AMR hybride commune aux tutoriels HyQMOM.

Ce module ne contient aucune physique HyQMOM. Il porte seulement le maillage
adaptatif, le marqueur de raffinement, MPI/OpenMP et la sortie ParaView.
"""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pops
import pops.output as output

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
    PatchLayout,
    Tag,
)
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray, Gaussian
from pops.math import ValueExpr, ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.output import ParallelMode, ScientificOutput
from pops.params import RuntimeParam
from pops.physics import Density
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import every, every_dt


BASE_CELLS = 16
FINE_CELLS = 32


def add_static_refinement_marker(
    case: Any,
    frame: Any,
    program: Any,
) -> tuple[Any, Any]:
    """Ajoute un bloc scalaire immobile qui definit une zone raffinee locale."""
    x_axis, y_axis = frame.axes
    marker_model = pops.Model("hyqmom_amr_marker", frame=frame)
    marker_state = marker_model.state(
        "U",
        components=("marker",),
        representation=Conservative(),
        space=CellState(frame=frame),
        roles={"marker": Density()},
    )
    (marker_value,) = marker_state
    marker_flux = marker_model.flux(
        "marker_flux",
        frame=frame,
        state=marker_state,
        components={
            x_axis: (0.0 * marker_value,),
            y_axis: (0.0 * marker_value,),
        },
        waves={
            x_axis: (0.0 * marker_value,),
            y_axis: (0.0 * marker_value,),
        },
    )
    marker_rate = marker_model.rate(
        "marker_rate",
        equation=ddt(marker_state) == -div(marker_flux),
    )

    marker_block = case.block(
        "amr_marker",
        model=marker_model,
        states=(marker_state,),
    )
    marker = marker_block[marker_state]
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
    case.numerics(marker_numerics, block=marker_block)

    marker_time = program.state(marker)
    program.commit(
        marker_time.next,
        program.value("accepted_amr_marker", marker_time.n, at=marker_time.next.point),
    )
    case.initials.add(InitialCondition(
        state=marker,
        value=Gaussian(
            frame=frame,
            center={x_axis: 0.0, y_axis: 0.0},
            background=0.0,
            amplitude=1.0,
            inverse_width=80.0,
        ),
        projection=ConservativeCellAverage(),
    ))
    return marker_block, marker


def build_layout(
    case: Any,
    grid: Any,
    program: Any,
    plasma_state: Any,
    marker_state: Any,
) -> AMR:
    """Construit deux niveaux 16x16 -> 32x32 avec patches distribues."""
    case.initials.add(InitialCondition(
        state=plasma_state,
        value=BindArray(),
        projection=ConservativeCellAverage(),
    ))
    threshold = case.param(RuntimeParam(
        "hyqmom_amr_marker_refine",
        default=0.15,
    ))
    tagging = AMRTagging(
        rules=(
            Tag(ValueExpr(marker_state) > case.value(threshold)),
            Buffer(cells=1),
        ),
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )
    transfer = AMRTransfer()
    transfer.state(plasma_state, StateTransfer())
    transfer.state(marker_state, StateTransfer())
    return AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=tagging,
        regrid=AMRRegrid(schedule=every(5, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
        patch_layout=PatchLayout(
            # CompositeFAC conserve le niveau zero comme ancre repliquee ; les
            # patches raffines restent distribues entre les rangs MPI.
            distribute_coarse=False,
            coarse_max_grid=BASE_CELLS,
        ),
    )


def paraview_output(program: Any, plasma_state: Any, t_end: float) -> ScientificOutput:
    """Sortie temporelle AMR lisible directement par ParaView."""
    return ScientificOutput(
        format=output.ParaView(
            mode=ParallelMode.PER_RANK,
            preset=output.ParaViewPreset(
                color_by="U",
                color_map="Viridis",
                representation="Surface With Edges",
            ),
        ),
        schedule=every_dt(t_end / 4.0, clock=program.clock),
        fields=(plasma_state,),
        target="solution/plasma",
    )


def bind_hybrid(
    artifact: Any,
    plasma_state: Any,
    initial_state: np.ndarray,
    *,
    params: Any = None,
) -> tuple[Any, Any, int]:
    """Lie un artefact MPI au monde natif et conserve OpenMP dans chaque rang."""
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    simulation = pops.bind(
        artifact,
        initial_values={plasma_state: initial_state},
        params={} if params is None else params,
        resources={"execution_context": context},
    )
    return simulation, world, int(world.rank)


def prepare_output_root(root: Path, world: Any, rank: int) -> None:
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    world.barrier()


def coarse_state(simulation: Any, block: str, shape: tuple[int, ...]) -> np.ndarray:
    """Lit la copie locale complete du niveau grossier, repliquee par ``PatchLayout``."""
    return np.asarray(
        simulation.block_level_state(block, 0),
        dtype=np.float64,
    ).reshape(shape)


def require_pvd(root: Path) -> Path:
    paths = tuple(sorted(root.rglob("*.pvd")))
    if not paths:
        raise RuntimeError("the AMR ParaView output did not publish a PVD collection")
    return paths[-1]


__all__ = [
    "BASE_CELLS",
    "FINE_CELLS",
    "add_static_refinement_marker",
    "bind_hybrid",
    "build_layout",
    "coarse_state",
    "paraview_output",
    "prepare_output_root",
    "require_pvd",
]
