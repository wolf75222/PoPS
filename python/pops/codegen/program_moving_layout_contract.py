"""Prepare-time pairing of layout motion and coupled physical SSA authority."""
from ._compiled_parameter import _thaw_json


def validate_moving_layout_program(program,layout_plan):
    from .program_lowerability import all_ops
    values=tuple(all_ops(program))
    bindings=tuple(value for value in values if value.op=="geometry_state")
    assignments={row.subject.local_id:row.layout for row in layout_plan.assignments
                 if row.subject_kind=="block"}
    by_layout={row.handle:[] for row in layout_plan.layouts}
    for binding in bindings:
        layout=assignments.get(binding.block.local_id)
        if layout is None: raise ValueError("moving physical state has no exact layout assignment")
        by_layout[layout].append(binding)
    for layout in layout_plan.layouts:
        evolution=layout.requirements.get("geometry_evolution")
        selected=by_layout[layout.handle]
        if evolution is None and selected:
            raise ValueError("coupled geometry_state requires its explicit MovingControlVolumes layout")
        if evolution is None: continue
        if len(selected)!=1:
            raise ValueError("moving layout requires exactly one coupled physical geometry_state authority")
        if _thaw_json(selected[0].attrs["evolution"])!=_thaw_json(evolution):
            raise ValueError("Program coordinate law differs from its exact moving layout evolution")
        if selected[0].space.frame!=layout.geometry.frame_id:
            raise ValueError("moving state and reference layout have different physical frames")
    if bindings:
        from .program_emit_moving import deferred_moving_rates,moving_updates
        allowed=deferred_moving_rates(program)|{value.id for value in bindings}|{
            value.id for value in moving_updates(program)}|{value.inputs[0].id for value in bindings}
        for value in values:
            if value.id not in allowed and any(item.id in allowed for item in value.inputs):
                raise NotImplementedError("this moving-state consumer needs an interval-qualified "
                                          "geometry/provider projection: %s"%value.op)


def validate_moving_metric_consumers(graph,layout_plan):
    """Do not multiply a static-dx diagnostic by a moving physical field."""
    if graph is None: return
    moving={row.handle.qualified_id for row in layout_plan.layouts
            if row.requirements.get("geometry_evolution") is not None}
    for node in graph.nodes:
        for quantity in node.diagnostic_quantities:
            if quantity.layout_id not in moving: continue
            if any(operation.get("metric_weighted") for operation in quantity.execution.get("operations",())):
                raise NotImplementedError("moving metric-weighted diagnostics need an accepted-measure "
                                          "reduction provider; scientific snapshots expose actual volumes")
