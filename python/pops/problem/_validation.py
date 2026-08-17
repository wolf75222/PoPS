"""Strict validation helpers for behavior-bearing Case fields."""
from __future__ import annotations

from typing import Any


def account_block_plan_fields(report: Any, block_registry: Any) -> Any:
    """Reject block fields for which the resolved/install plan has no exact consumer."""
    for block_name, spec in block_registry.items():
        time = spec.get("time")
        if time is not None and hasattr(time, "ir_nodes"):
            report = report.error(
                "block", "unlowered_block_time",
                "block %r declares a per-block time Program, but the resolved/install plans do "
                "not carry it; refusing the historical runtime-default substitution" % block_name,
                context={"block": block_name},
                alternatives=("declare the whole-system Program with case.program(...)",))
        elif time is not None and (
            not hasattr(time, "kind") or not hasattr(time, "stride")
        ):
            report = report.error(
                "block", "unlowered_block_time",
                "block %r declares a per-block time value that is neither a cadence "
                "descriptor nor a Program" % block_name,
                context={"block": block_name},
                alternatives=("pass time=Explicit(stride=M) or IMEX(stride=M)",),
            )
        if spec.get("diagnostics"):
            report = report.error(
                "block", "unlowered_block_diagnostics",
                "block %r declares per-block diagnostics, but no resolved-plan consumer exists; "
                "refusing to drop them" % block_name,
                context={"block": block_name, "count": len(spec["diagnostics"])},
                alternatives=("attach diagnostics with problem.runtime(...)",))
    return report


__all__ = ["account_block_plan_fields"]
