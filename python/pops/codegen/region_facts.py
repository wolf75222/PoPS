"""Region facts of a resolved plan (spec 0.3 IR14, IR17, IR20, IR21).

The face kernels of a plan are regions, one per axis: the finite-volume method of each rate of a
block, and each stored face expression that the program evaluates, at the top level or in a
nested block. Their builders give the reads, derived from a body or declared by a native
catalogue (``Region.basis``), and the class they expect. ``check_expected_classes`` classifies
every region from the normal forms of its maps and raises ``ClassNotReached`` when a region does
not reach the class its builder expects (IR21), before any native work of ``compile``; nothing
falls back silently (I15).

The rates are the only source of the finite-volume methods: the spatial method of a block is
``primary_spatial()`` of its numerics, the method of one of its rates or a state storage, so it
adds no kernel. Every region has the rank of the plan, ``resolved_dimension``: the native layouts
of one plan never mix dimensions. The program values are walked as the codegen walks them
(``walk_program_nodes``), so the check sees every value that is emitted. Each value and each
region is read once: the cost follows the size of the plan, never an extent or a trip count (B4).
"""
from __future__ import annotations

from pops.time._graph.value_traversal import walk_program_nodes


def plan_regions(plan):
    """The regions of the face kernels of ``plan``, in a deterministic order."""
    from pops.numerics.spatial import FiniteVolume
    from pops.numerics.stencil_flux import StencilFlux

    regions = []
    for block in plan.blocks:
        for rate in getattr(block.numerics, "rates", ()):
            if isinstance(rate.method, FiniteVolume):
                regions.extend(rate.method.regions(
                    "block %s/rate %s" % (block.name, rate.rate.local_id),
                    plan.resolved_dimension))
    for value in walk_program_nodes(plan.time._values):
        declaration = value.attrs.get("stencil_flux")
        if isinstance(declaration, StencilFlux):
            block = getattr(value.block, "local_id", value.block)
            regions.extend(declaration.regions("block %s/program %s" % (block, value.name)))
    return tuple(regions)


def check_expected_classes(plan):
    """``(name, expected, reached, basis)`` for every region; ``ClassNotReached`` on a miss."""
    return tuple((region.name, region.expected, region.check_expected_class(), region.basis)
                 for region in plan_regions(plan))


__all__ = ["check_expected_classes", "plan_regions"]
