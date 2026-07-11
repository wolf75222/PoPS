"""Pure resolve-time layout selection and reference authentication."""
from __future__ import annotations

from typing import Any


def validate_layout(problem: Any, layout: Any) -> None:
    from pops.mesh.layouts import Uniform

    criteria = problem._constraints.refinement
    if criteria.get("refine") is not None and isinstance(layout, Uniform) \
            and layout.ignore_amr is None:
        raise ValueError(
            "pops.resolve: Uniform layout cannot consume active AMR refinement criteria")
    context = {"layout": layout}
    layout.validate(context)
    problem._field_registry.validate(context).raise_if_error()


def resolve_layout(problem: Any, layout: Any) -> Any:
    """Select one layout authority, merge Problem AMR policies, and resolve every Handle leaf."""
    from pops.mesh.layouts import AMR, Uniform

    authored = problem.layout
    if layout is None:
        if authored is None:
            raise ValueError("pops.resolve requires layout=Uniform(...) or layout=AMR(...)")
        selected = authored
    else:
        if authored is not None and layout is not authored and layout != authored:
            raise ValueError("pops.resolve received two competing layout authorities")
        selected = layout

    def resolved(value: Any) -> Any:
        if value is None:
            return None
        protocol = getattr(value, "resolve_references", None)
        return protocol(problem.resolve) if callable(protocol) else value

    if isinstance(selected, Uniform):
        return Uniform(
            mesh=selected.mesh, embedded_boundary=selected.embedded_boundary,
            refine=resolved(selected.refine), ignore_amr=selected.ignore_amr)
    if not isinstance(selected, AMR):
        raise TypeError("pops.resolve layout must be a typed Uniform or AMR descriptor")

    criteria = problem._constraints.refinement
    policies = {}
    for slot in ("refine", "regrid", "nesting", "patches"):
        layout_value = getattr(selected, slot)
        problem_value = criteria.get(slot)
        if layout_value is not None and problem_value is not None:
            raise ValueError("pops.resolve: AMR %s has two competing authorities" % slot)
        policies[slot] = problem_value if problem_value is not None else layout_value
    refine = policies["refine"]
    if refine is not None and criteria.get("refine") is None:
        refine = resolved(refine)
    if refine is not None and not getattr(refine, "references_authenticated", False):
        raise ValueError("pops.resolve: AMR refinement references are not authenticated")
    return AMR(
        base=selected.base, max_levels=selected.max_levels, ratio=selected.ratio,
        regrid=policies["regrid"], patches=policies["patches"], refine=refine,
        nesting=policies["nesting"], checkpoint=selected.checkpoint,
        output=resolved(selected.output), clustering=selected.clustering)


__all__ = ["resolve_layout", "validate_layout"]
