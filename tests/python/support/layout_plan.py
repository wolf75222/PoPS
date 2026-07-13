"""Strict LayoutPlan contracts for tests that manually assemble resolved phase records."""
from __future__ import annotations

import hashlib
from typing import Any

from pops.codegen._layout_resolution import layout_lowering_coverage
from pops.mesh import CartesianMesh, normalize_layout_plan
from pops.mesh.layouts import AMR, Uniform
from pops.model import Handle, OwnerPath


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
        descriptor = AMR(mesh, max_levels=2, ratio=2) \
            if target == "amr_system" else Uniform(mesh)
    plan = normalize_layout_plan(descriptor, owner=owner, blocks=blocks)
    return plan, layout_lowering_coverage(plan)


__all__ = ["resolved_layout_contract"]
