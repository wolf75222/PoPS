"""Memory-estimate helpers for compiled artifact inspection."""


def amr_patch_budget(layout, state_field, cell_field, n_elliptic):
    """A conservative AMR patch budget from an ``AMR`` layout descriptor.

    Returns ``(layout_kind, amr_patch_bytes, notes)``. Imports mesh descriptors
    lazily so codegen stays free of mesh imports at module load time.
    """
    from pops.mesh.layouts import AMR, Uniform

    if isinstance(layout, Uniform):
        return "uniform", None, []
    if not isinstance(layout, AMR):
        raise TypeError("estimate_memory(layout=): expected a pops.mesh.layouts.AMR / Uniform; "
                        "got %r" % type(layout).__name__)
    max_levels = int(getattr(layout, "max_levels", 1) or 1)
    ratio = int(getattr(layout, "ratio", 2) or 2)
    if max_levels <= 1:
        return "amr", 0, ["AMR layout with a single level: no extra patch budget"]
    refine_factor = sum(ratio ** (2 * k) for k in range(1, max_levels))
    per_cell_levels = state_field + n_elliptic * cell_field
    amr_bytes = refine_factor * per_cell_levels
    notes = [
        "AMR estimate is CONSERVATIVE: assumes EVERY level (1..%d) fully refines the whole domain "
        "at ratio %d (worst case); a real regrid tags a fraction of cells, so the true footprint is "
        "smaller. A tight AMR figure needs a bind (the regrid pattern is data-dependent)."
        % (max_levels - 1, ratio),
        "AMR refine factor (sum of r^(2k), k=1..%d) = %d base-grid equivalents"
        % (max_levels - 1, refine_factor),
    ]
    return "amr", amr_bytes, notes


__all__ = ["amr_patch_budget"]
