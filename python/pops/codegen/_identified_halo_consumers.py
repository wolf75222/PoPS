"""Qualification of consumers of the oriented Cartesian halo (G09 I1)."""
from __future__ import annotations

from typing import Any

from pops.codegen._static_geometry_consumers import STATIC_GEOMETRY_PROGRAM_OPS
from pops.codegen.program_emit_kernels import _named_fluxes
from pops.codegen.program_lowerability import all_ops
from pops.time._step.strategy import FixedDt


def validate_identified_halo_consumers(plan: Any) -> None:
    layouts = plan.layout_plan.layouts
    mapped = tuple(layout for layout in layouts if
                   layout.native_spatial_layout is not None and
                   layout.native_spatial_layout.identifications)
    if not mapped:
        return
    if len(layouts) != 1 or mapped[0].adaptive or mapped[0].capabilities.get("embedded_boundary"):
        raise NotImplementedError("G09 I1 identified halos require one non-embedded Uniform layout")
    if plan.target != "system" or plan.backend == "cuda":
        raise NotImplementedError("G09 I1 identified halos require the host Uniform execution route")
    if plan.field_plans or plan.program_field_plans or plan.component_inputs:
        raise NotImplementedError(
            "G09 I1 identified halos have no qualified field, solver or external component consumer")
    if type(plan.time._step_strategy) is not FixedDt:
        raise NotImplementedError("G09 I1 identified halos serve the fixed-time-step flux route")
    from pops.numerics.spatial import FiniteVolume

    for block in plan.blocks:
        numerics = block.numerics
        if (numerics.boundaries or numerics.interfaces or numerics.fields or
                numerics.principal_groups or numerics.sources or block.model.module.aux()):
            raise NotImplementedError(
                "G09 I1 identified halos have no qualified physical, auxiliary or interface provider")
        for rate in numerics.rates:
            if type(rate.method) is not FiniteVolume:
                raise NotImplementedError(
                    "G09 I1 identified halos have no qualified %s rate consumer" %
                    type(rate.method).__name__)
    for value in all_ops(plan.time):
        access = value.attrs.get("stencil_access")
        spatial_inputs = access is not None and access.required_ghost_depth > 0
        if value.op not in STATIC_GEOMETRY_PROGRAM_OPS or (value.op != "rhs" and spatial_inputs):
            raise NotImplementedError(
                "G09 I1 identified halos have no qualified %s Program consumer" % value.op)
        if value.op == "rhs" and (_named_fluxes(value) is not None or value.attrs.get("flux_handles")
                                  or value.attrs.get("sources")):
            raise NotImplementedError("G09 I1 identified halos require one default transport route")
