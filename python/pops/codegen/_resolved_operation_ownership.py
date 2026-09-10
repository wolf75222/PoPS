"""Authenticate a resolved numerical plan at its Case-block execution boundary."""
from __future__ import annotations

from typing import Any, NoReturn, cast


def require_block_plan_owner(plan: Any, owner_qid: Any, *, where: str,
                             required: bool = False) -> None:
    from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringRejection
    from pops.codegen.resolved_operations import ResolvedOperationPlan

    is_plan = type(plan) is ResolvedOperationPlan
    case_plan = is_plan and any("block_instance" in operation.guarantees
                                for operation in plan.operations)

    def reject(message: str) -> NoReturn:
        raise LoweringRejection(
            "%s %s" % (where, message),
            coverage_report=plan.coverage if is_plan else LoweringCoverageReport(()),
            source="block:" + (owner_qid if type(owner_qid) is str and owner_qid else "<missing>"),
            gate="resolved_operation_block_owner_mismatch")

    if type(owner_qid) is not str or not owner_qid.strip():
        absent = owner_qid is None or (type(owner_qid) is str and owner_qid == "")
        if required or case_plan or not absent:
            reject("requires its exact non-empty Case-block instance owner")
        # Standalone Module plans and explicit None legacy adapters have no Case
        # instance authority. Their existing source/realization guards still apply.
        if plan is None or is_plan:
            return
    if plan is None and not required:
        return
    if type(plan) is not ResolvedOperationPlan:
        reject("requires its exact Case-block resolved operation plan")
    mismatches = tuple(operation.identity for operation in plan.operations
                       if operation.guarantees.get("block_instance") != owner_qid)
    if mismatches:
        raise LoweringRejection(
            "%s received a resolved operation plan for a different block instance: %s"
            % (where, ", ".join(mismatches)), coverage_report=plan.coverage,
            source="block:" + cast(str, owner_qid), gate="resolved_operation_block_owner_mismatch")
