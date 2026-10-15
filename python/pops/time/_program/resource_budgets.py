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
