"""Test-only installation helpers for exact prepared AMRTagging programs."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI


def install_prepared_threshold_union(
    system: Any,
    criteria: Iterable[tuple[str, str, float]],
    *,
    provider_identity: str = "test::prepared-named-threshold-union@1",
) -> None:
    """Install an OR of exact block-qualified ``value > threshold`` leaves."""
    rows = tuple(criteria)
    if not rows:
        raise ValueError("test prepared threshold union requires at least one leaf")
    if any(
        not isinstance(block, str) or not block
        or not isinstance(variable, str) or not variable
        for block, variable, _threshold in rows
    ):
        raise ValueError("test prepared threshold leaves require block and variable names")

    above = int(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"]["above"])
    any_of = int(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"]["any_of"])
    refine_ops = [above] * len(rows)
    refine_args = list(range(len(rows)))
    if len(rows) > 1:
        refine_ops.append(any_of)
        refine_args.append(len(rows))
    native = getattr(system, "_s", system)
    blocks = [block for block, _variable, _threshold in rows]
    native._set_bootstrap_tagging(
        ["state"] * len(rows),
        [
            "pops://runtime/amr-direct-state/%d:%s" % (len(block), block)
            for block in blocks
        ],
        blocks,
        [variable for _block, variable, _threshold in rows],
        [-1] * len(rows),
        [above] * len(rows),
        [float(threshold) for _block, _variable, threshold in rows],
        [-1] * len(rows),
        [None] * len(rows),
        [],
        refine_ops,
        refine_args,
        [],
        [],
        0,
        "hold",
        "error",
        "test::prepared-tagging-clock",
        provider_identity,
    )


__all__ = ["install_prepared_threshold_union"]
