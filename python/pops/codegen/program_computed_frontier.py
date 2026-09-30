"""Preparation gate for the versioned Program computed-frontier interval rule."""


def require_computed_frontier_support(program, *, target):
    if target != "system":
        raise NotImplementedError("ComputedDt version 1 requires a uniform Program; AMR interval "
                                  "exchanges cannot yet be remapped to a computed endpoint")
    for value in program._values:
        if value.op in {"diffusive_rhs", "principal_rate", "coupled_rate", "subcycle",
                        "synchronize", "layout_map_export", "layout_map_import", "post_synchronization"} \
                or (value.op == "rhs" and value.attrs.get("flux", True)) \
                or value.attrs.get("schedule") is not None:
            raise NotImplementedError("ComputedDt version 1 refuses spatial interval exchanges, "
                                      "nested clocks and held scheduler caches")
    if getattr(program, "_integral_transfers", ()):
        raise NotImplementedError("ComputedDt version 1 refuses persistent interval transfers")
    declared = False
    for value in program._values:
        if value.op == "reached_duration":
            declared = True
        elif declared and value.op == "store_history":
            raise NotImplementedError("computed outgoing history intervals must be stored before "
                                      "the Program returns its duration")
