"""Strict validation helpers for behavior-bearing Problem fields."""
from __future__ import annotations

from typing import Any

from pops.mesh.layouts import Uniform


def refuse_uniform_with_amr_criteria(report: Any, layout: Any) -> Any:
    """Reject active AMR criteria on a single-level layout unless explicitly ignored."""
    criterion = getattr(layout, "refine", None) if isinstance(layout, Uniform) else None
    if criterion is None or getattr(layout, "ignore_amr", None) is not None:
        return report
    sub_criteria = getattr(criterion, "criteria", None)
    names = [item.name for item in sub_criteria] if sub_criteria is not None else [criterion.name]
    return report.error(
        "amr", "uniform_with_amr_criteria",
        "layout=Uniform(...) carries active AMR criteria (%s) but a single-level layout has no "
        "level to refine onto; a criterion is never silently ignored. Use layout=AMR(...) to "
        "actually refine, or pass Uniform(mesh, refine=..., "
        "ignore_amr=pops.mesh.amr.IgnoreAMRCriteria()) to keep the criterion attached but "
        "explicitly unused." % ", ".join(names), context={"criteria": names})


def account_block_plan_fields(report: Any, block_registry: Any) -> Any:
    """Reject block fields for which the resolved/install plan has no exact consumer."""
    for block_name, spec in block_registry.items():
        if spec.get("time") is not None:
            report = report.error(
                "block", "unlowered_block_time",
                "block %r declares a per-block time Program, but the resolved/install plans do "
                "not carry it; refusing the historical runtime-default substitution" % block_name,
                context={"block": block_name},
                alternatives=("declare the whole-system Program with problem.time(...)",))
        if spec.get("diagnostics"):
            report = report.error(
                "block", "unlowered_block_diagnostics",
                "block %r declares per-block diagnostics, but no resolved-plan consumer exists; "
                "refusing to drop them" % block_name,
                context={"block": block_name, "count": len(spec["diagnostics"])},
                alternatives=("attach diagnostics with problem.runtime(...)",))
    return report


__all__ = ["account_block_plan_fields", "refuse_uniform_with_amr_criteria"]
