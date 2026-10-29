"""Explicit resource ceilings, distinct from scientific admissibility conditions."""


def validate_flux_expression_budget(value):
    if value is None:
        return
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or not 1 <= item <= (1 << 63) - 1 for item in value)
    ):
        raise ValueError(
            "flux_expression_budget requires (max_bases, max_terms) positive int64 ceilings"
        )


def validate_flux_expression_abi(budgets):
    """All generated resource accessors have exact uint64 return values."""
    maximum = (1 << 64) - 1
    for bases, terms in budgets:
        if (
            type(bases) is not int
            or type(terms) is not int
            or not 0 <= bases <= maximum
            or not 0 <= terms <= maximum
            or (bases == 0) != (terms == 0)
        ):
            raise ValueError("flux expression budgets require paired uint64 native bounds")
    if sum(bases for bases, _ in budgets) > maximum:
        raise ValueError("aggregate flux expression interface budget exceeds native uint64 ABI")


def validate_terminal_failure_capacities(per_rank, aggregate):
    """Explicit storage limits for complete typed failures, independent of equations."""
    for name, value in (("terminal_failure_byte_capacity", per_rank),
                        ("terminal_failure_aggregate_byte_capacity", aggregate)):
        if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
            raise ValueError(f"{name} must be an integer in [1, INT64_MAX]")


def validate_history_field_metadata_budget(value):
    """Retained descriptions and transient codec workspace are separate ceilings."""
    if value is None:
        return
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) is not int or not 1 <= item <= (1 << 63) - 1 for item in value)):
        raise ValueError(
            "history_field_metadata_budget requires (retained_bytes, workspace_bytes) positive int64 capacities"
        )


# Plan default of the History Field metadata resource: a case whose History metadata fits it
# authors no memory budget to retain Field support and representation (spec 0.3, B3). It bounds
# the metadata only, never the exchange or FLX image nested in a checkpoint envelope. Most API 0.2
# examples and fixtures declare this capacity explicitly; two declare (8 MiB, 8 MiB).
DEFAULT_HISTORY_FIELD_METADATA_BUDGET = (8 * 1024 * 1024, 16 * 1024 * 1024)


def resolved_history_field_metadata_budget(program):
    """Installed (retained_bytes, workspace_bytes), or None for a Program without History rings.

    An authored budget takes precedence over the plan default. Every native consumer (the
    prelude, the AMR checkpoint exports and the exchange checkpoint bound) reads this one value.
    """
    budget = getattr(program, "history_field_metadata_budget", None)
    validate_history_field_metadata_budget(budget)
    if budget is not None:
        return budget
    return DEFAULT_HISTORY_FIELD_METADATA_BUDGET if program._histories else None


def validate_representation_workspace_capacity(value):
    """Workspace for exact closed-read certificates, independent of failure/flux storage."""
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        raise ValueError("representation_workspace_byte_capacity must be an integer in [1, INT64_MAX]")
