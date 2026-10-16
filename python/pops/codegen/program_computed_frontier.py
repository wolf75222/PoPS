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
                pending.extend(term for term, _ in getattr(ref, "terms"))
    return values


def require_computed_frontier_support(program, *, target):
    values = _frontier_values(program)
    affine = [value for value in values if value.op == "accept_increment"]
    if affine:
        from pops.codegen.program_lowerability import all_ops_with_ancestry
        if target != "amr_system" or len(affine) != 1:
            raise ValueError("accept_increment requires one AMR parent finalization")
        by_id = {value.id: value for value in values}
        ancestry = next(path for value, path in all_ops_with_ancestry(program) if value is affine[0])
        if len(ancestry) != 1 or by_id[ancestry[0][0]].op != "post_synchronization":
            raise ValueError("accept_increment must belong directly to after_synchronization")
        if getattr(program, "_post_sync_commits", {}):
            raise ValueError("accept_increment owns all affine commits; manual post-sync commits are forbidden")
        from pops.time._program.affine_history import validate_history_contract
        validate_history_contract(program, affine[0].attrs.get("histories", {}))
        forbidden = {"reached_duration", "history_snapshot",
                     "interval_begin", "interval_check", "history_interval_begin",
                     "subcycle", "synchronize", "layout_map_export", "layout_map_import"}
        if any(value.op in forbidden or value.attrs.get("schedule") is not None for value in values):
            raise NotImplementedError("affine macro publication requires an explicit remap law for "
                                      "retained histories, nested Program clocks and held caches")
        if getattr(program, "_integral_transfers", ()):
            raise NotImplementedError("affine macro publication requires a persistent integral remap law")
        return
    if target != "system":
        raise NotImplementedError("ComputedDt version 1 requires a uniform Program; AMR interval "
                                  "exchanges cannot yet be remapped to a computed endpoint")
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
