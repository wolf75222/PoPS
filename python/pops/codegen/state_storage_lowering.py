"""Exact physical frame admission for source-only Program state storage."""
from pops._cartesian_axes import canonical_axis_mapping


def prepare_source_storage_carrier(emitter, module, *, state_space=None):
    impl = getattr(emitter, "_m", emitter)
    if impl._flux:
        return
    states = module.state_spaces()
    if state_space is None:
        if len(states) != 1:
            return
        state = next(iter(states.values()))
    else:
        state = state_space
    operators = tuple(module.operator_registry())
    if any(op.kind == "grid_operator" and state in op.signature.inputs for op in operators):
        return
    frames = set()
    for operator in operators:
        view = operator.lowering.get("physical_balance")
        if (view is None or not view.accumulation.is_identity or not view.occurrences
                or any(row.kind != "source" for row in view.occurrences)
                or operator.signature.inputs[0] != state):
            continue
        axes = tuple(operator.capabilities.get("storage_axes", ()))
        frame = operator.capabilities.get("storage_frame")
        if not axes or frame != state.frame:
            raise ValueError("source-only state storage requires its exact authored Cartesian frame")
        axes = tuple(canonical_axis_mapping(dict.fromkeys(axes), where="source-only storage frame"))
        frames.add((axes, frame))
    if not frames:
        return
    if len(frames) != 1:
        raise ValueError("source-only state storage has conflicting physical frames")
    axes, _ = next(iter(frames))
    previous = getattr(impl, "_program_only_storage_axes", axes)
    if previous != axes:
        raise ValueError("source-only state storage differs from the selected physical frame")
    object.__setattr__(impl, "_program_only_storage_axes", axes)
