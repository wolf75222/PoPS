"""Preparation gate for the versioned Program computed-frontier interval rule."""


def _frontier_values(program):
    from pops.codegen.program_lowerability import all_ops_with_ancestry
    from pops.time.value_support import _ProgramValueBase, _AffineExpressionBase

    # Region ownership is authenticated before following captures/results. A capture
    # can revisit an earlier node; it must not introduce an unrecorded operation.
    values = tuple(value for value, _ in all_ops_with_ancestry(program))
    issued = {id(value) for value in values}
    seen = set()
    pending = list(values)
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        if id(value) not in issued:
            raise ValueError("computed frontier references an unrecorded Program operation")
        seen.add(id(value))
        pending.extend(value.inputs)
        for key in ("cond", "body", "true_result", "false_result", "residual", "iterate",
                    "guess", "apply_result", "apply_in", "apply_out"):
            ref = value.attrs.get(key)
            if isinstance(ref, _ProgramValueBase):
                pending.append(ref)
            elif isinstance(ref, _AffineExpressionBase):
                pending.extend(term for term, _ in ref.terms)
    return values


def require_computed_frontier_support(program, *, target):
    if target != "system":
        raise NotImplementedError("ComputedDt version 1 requires a uniform Program; AMR interval "
                                  "exchanges cannot yet be remapped to a computed endpoint")
    values = _frontier_values(program)
    for value in values:
        if value.op in {"diffusive_rhs", "principal_rate", "coupled_rate", "subcycle",
                        "synchronize", "layout_map_export", "layout_map_import", "post_synchronization"} \
                or (value.op == "rhs" and value.attrs.get("flux", True)) \
                or value.attrs.get("schedule") is not None:
            raise NotImplementedError("ComputedDt version 1 refuses spatial interval exchanges, "
                                      "nested clocks and held scheduler caches")
    if getattr(program, "_integral_transfers", ()):
        raise NotImplementedError("ComputedDt version 1 refuses persistent interval transfers")
    declared = False
    for value in values:
        if value.op == "reached_duration":
            declared = True
        elif declared and value.op == "store_history":
            raise NotImplementedError("computed outgoing history intervals must be stored before "
                                      "the Program returns its duration")
