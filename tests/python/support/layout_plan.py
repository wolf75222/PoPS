"""Strict LayoutPlan contracts for tests that manually assemble resolved phase records."""
from __future__ import annotations

import hashlib
from typing import Any

from pops.codegen._layout_resolution import layout_lowering_coverage
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.ir.expr import Const
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, CartesianMesh, normalize_layout_plan
from pops.layouts import AMR, Uniform
from pops.mesh.amr.transfer import AMRTransfer
from pops.model import Handle, OwnerPath
from pops.time import Clock, always


def final_amr_layout(
    grid: Any,
    *,
    max_levels: int = 2,
    ratio: int = 2,
) -> AMR:
    """Small complete final AMR authority for tests needing only layout structure."""
    if type(grid) is CartesianMesh:
        grid = CartesianGrid(
            frame=Rectangle(
                "test-amr-grid", (0.0, 0.0), (grid.L, grid.L)
            ).frame(Cartesian2D()),
            cells=(grid.n, grid.n),
        )
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
        execution=AMRExecution.subcycled(),
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
        mesh = CartesianMesh(n=8)
        descriptor = final_amr_layout(mesh, max_levels=2, ratio=2) \
            if target == "amr_system" else Uniform(mesh)
    plan = normalize_layout_plan(descriptor, owner=owner, blocks=blocks)
    return plan, layout_lowering_coverage(plan)


__all__ = ["final_amr_layout", "resolved_layout_contract"]
