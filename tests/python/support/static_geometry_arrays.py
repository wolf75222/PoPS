"""The static geometry arrays of the layout of a surface case, generated natively without compiling.

N17 L1b: the elaborated layout of a case (``pops.Graded`` gives the static case of the
geometry-evolution contract) is normalized, its native spatial layout is resolved, and the arrays
are generated and certified in a bare native System of the requested rank, as ``pops.bind`` does
before the first block. Nothing is compiled, and the case itself need not resolve: probe 3 does
not yet, its value relations under a diffusion waiting on G04.
"""
from __future__ import annotations

from typing import Any


def layout_static_arrays(case: Any, *, dimension: int) -> dict:
    """The snapshot of the static geometry arrays of the layout of ``case``."""
    from pops._native_selector import select_native_dimension
    from pops.codegen._layout_resolution import resolve_native_spatial_layouts
    from pops.mesh import normalize_layout_plan
    from pops.runtime._runtime_mesh_lowering import (
        install_static_geometry, system_config_from_layout)

    select_native_dimension(dimension)
    from pops.runtime._system import System

    elaborated = case.elaborated
    plan = normalize_layout_plan(elaborated.layout, owner=elaborated.case.owner_path.canonical())
    resolve_native_spatial_layouts(plan)
    (layout,) = plan.layouts
    engine = System(system_config_from_layout(layout.native_spatial_layout))
    install_static_geometry(engine, layout)
    return engine._s._static_geometry_snapshot()
