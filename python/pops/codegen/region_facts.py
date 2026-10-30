"""Region facts of a resolved plan (spec 0.3 IR14, IR17, IR20, IR21).

The face kernels of a plan are regions, one per axis: the finite-volume method of each rate of a
block, and each stored face expression that the program evaluates. Their builders give the reads,
derived from a body or declared by a native catalogue (``Region.basis``), and the class they
expect. ``check_expected_classes`` classifies every region from the normal forms of its maps and
raises ``ClassNotReached`` when a region does not reach the class its builder expects (IR21),
before any native work of ``compile``; nothing falls back silently (I15).

The walk reads the program values and their nested blocks once, and each region once: its cost
follows the size of the plan, never an extent or a trip count (B4).
"""
from __future__ import annotations


def _program_values(program):
    pending = [tuple(program._values)]
    while pending:
        for value in pending.pop():
            yield value
            for attribute in value.attrs.values():
                if (isinstance(attribute, tuple) and attribute
                        and all(hasattr(item, "op") and hasattr(item, "attrs")
                                for item in attribute)):
                    pending.append(attribute)


def _rank(block, plan):
    frame = getattr(block.model, "frame", None)
    coordinates = getattr(frame, "coordinates", None)
    dimension = getattr(coordinates, "dimension", None)
    return dimension if type(dimension) is int else plan.resolved_dimension


def plan_regions(plan):
    """The regions of the face kernels of ``plan``, in a deterministic order."""
    from pops.numerics.spatial import FiniteVolume
    from pops.numerics.stencil_flux import StencilFlux

    regions = []
    for block in plan.blocks:
        methods = []
        for rate in getattr(block.numerics, "rates", ()):
            if isinstance(rate.method, FiniteVolume):
                methods.append(("block %s/rate %s" % (block.name, rate.rate.local_id),
                                rate.method))
        # The spatial method of a block is the method of its rate, resolved once more.
        if isinstance(block.spatial, FiniteVolume) and all(
                block.spatial.to_data() != method.to_data() for _, method in methods):
            methods.append(("block %s/spatial" % block.name, block.spatial))
        for name, method in methods:
            regions.extend(method.regions(name, _rank(block, plan)))
    for value in _program_values(plan.time):
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
