"""Strict LayoutPlan contracts for tests that manually assemble resolved phase records."""
from __future__ import annotations

import hashlib
from typing import Any

from pops.codegen._layout_resolution import layout_lowering_coverage
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
from pops._ir.expr import Const
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, PeriodicAxes, normalize_layout_plan
from pops.layouts import AMR, Uniform
from pops.model import Handle, OwnerPath
from pops.time import Clock, always


def cartesian_grid(
    n: int = 64,
    L: float = 1.0,
    periodic: bool = True,
    *,
    name: str = "test-square",
) -> CartesianGrid:
    """Build the canonical public square-grid authoring value used by tests."""
    frame = Rectangle(name, (0.0, 0.0), (L, L)).frame(Cartesian2D())
    topology = PeriodicAxes(frame.axes) if periodic else None
    return CartesianGrid(frame=frame, cells=(n, n), periodic=topology)


def final_amr_layout(
    grid: Any,
    *,
    max_levels: int = 2,
    ratio: int = 2,
) -> AMR:
    """Small complete final AMR authority for tests needing only layout structure."""
    if type(grid) is not CartesianGrid:
        raise TypeError("final_amr_layout requires an exact public CartesianGrid")
    return AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=max_levels, ratios=(ratio,) * (max_levels - 1)),
        tagging=AMRTagging(
            rules=(Tag(Const(1.0) > Const(0.0)), Buffer(1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(always(clock=Clock("macro"))),
        transfer=AMRTransfer(),
        execution=AMRExecution.subcycled(tuple(
            AMRClockRelation(level, level + 1, ratio)
            for level in range(max_levels - 1)
        )),
    )


def resolved_layout_contract(
    layout: Any, *, target: str, block_names: Any,
) -> tuple[Any, Any]:
    """Return an exact plan/coverage pair for a synthetic already-resolved test record."""
    names = tuple(block_names)
    token = hashlib.sha256((target + "\0" + "\0".join(names)).encode()).hexdigest()[:12]
    owner = OwnerPath.case("test-resolved-layout-%s" % token)
    blocks = tuple(Handle(name, kind="block", owner=owner) for name in names)
    descriptor = layout
    required = ("validate", "options", "requirements", "capabilities")
    if not all(callable(getattr(descriptor, name, None)) for name in required):
        grid = cartesian_grid(n=8)
        descriptor = final_amr_layout(grid, max_levels=2, ratio=2) \
            if target == "amr_system" else Uniform(grid)
    plan = normalize_layout_plan(descriptor, owner=owner, blocks=blocks)
    return plan, layout_lowering_coverage(plan)


__all__ = ["cartesian_grid", "final_amr_layout", "resolved_layout_contract"]
