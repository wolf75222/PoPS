"""Shared public-API authoring for verification cases.

Layouts, bind, and boundary helpers wrap only public PoPS types. Does not
compile or run except ``bind_public``, which calls ``pops.bind``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pops
from pops.codegen import Production
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sibling_module(path: Path):
    """Load a case-local module by file path (never via a shared ``exact`` name)."""
    spec = importlib.util.spec_from_file_location(
        f"pops_verify_case_{path.stem}_{path.parent.name}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def uniform_periodic_layout(frame, cells: tuple[int, ...]):
    """Return a Uniform layout on a fully periodic Cartesian grid."""
    return Uniform(
        CartesianGrid(
            frame=frame,
            cells=cells,
            periodic=PeriodicAxes(frame.axes),
        )
    )


def uniform_open_layout(frame, cells: tuple[int, ...]):
    """Return a Uniform layout with no periodic axes (physical faces)."""
    return Uniform(CartesianGrid(frame=frame, cells=cells))


def physical_boundaries(frame):
    """Return the physical faces of ``frame`` in canonical min/max axis order."""
    faces = frame.boundaries
    names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    return tuple(getattr(faces, name) for name in names if hasattr(faces, name))


def transmissive_boundary_set(frame, state):
    """Public outflow on every physical face."""
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Outflow

    return TransportBoundarySet(
        {face: Outflow(state=state) for face in physical_boundaries(frame)}
    )


def slip_wall_boundary_set(frame, state):
    """Public slip wall on every physical face."""
    from pops.boundary import SlipWall, TransportBoundarySet

    return TransportBoundarySet(
        {face: SlipWall(state=state) for face in physical_boundaries(frame)}
    )


def wall_and_outflow_boundary_set(frame, state, *, wall_faces):
    """Slip walls on named faces, outflow on the remaining physical faces."""
    from pops.boundary import SlipWall, TransportBoundarySet
    from pops.boundary.transport import Outflow

    walls = set(wall_faces)
    mapping = {}
    for face in physical_boundaries(frame):
        mapping[face] = SlipWall(state=state) if face in walls else Outflow(state=state)
    return TransportBoundarySet(mapping)


def attach_case_diagnostics(case, block, program, *, every_n: int = 1):
    """Attach public ``pops.diagnostics`` on a ConsumerGraph."""
    from pops.diagnostics import ConservationCheck, Integral
    from pops.time import every

    from verification.pops_verify.native_diagnostics import attach_state_diagnostics

    cadence = every(int(every_n), clock=program.clock)
    case.consumers(
        attach_state_diagnostics(
            block=block,
            cadence=cadence,
            conservation_check=ConservationCheck(
                Integral(block=block, cadence=cadence)
            ),
        )
    )
    return case


def bind_public(artifact, **kwargs: Any):
    """Bind through the public pipeline.

    When the compiled artifact proves ``MPI_COMM_WORLD``, the official
    ``ExecutionContext.mpi_world`` communicator is used. Otherwise bind is
    serial. Python never launches ranks.
    """
    try:
        context = pops.ExecutionContext.mpi_world(artifact)
    except Exception:
        context = None
    if context is None:
        return pops.bind(artifact, **kwargs)
    resources = dict(kwargs.get("resources") or {})
    resources["execution_context"] = context
    bind_kwargs = dict(kwargs)
    bind_kwargs["resources"] = resources
    return pops.bind(artifact, **bind_kwargs)


def resolve_case(case, *, layout, include: Path | None = None):
    """Validate and resolve a Case. Does not compile or call pops.run."""
    options = {"include": str(include or (REPO_ROOT / "include"))}
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options=options,
    )
