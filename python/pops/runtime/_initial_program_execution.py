"""Execute the authenticated uniform bind-time program through native kernels."""
from __future__ import annotations

from typing import Any


def initial_solve_report_data(report: Any, *, provider_slot: str, ordinal: int) -> dict[str, Any]:
    """Copy the actual native SolveReport into array-free diagnostic facts."""
    evidence = report.residual_evaluation
    return {
        "ordinal": ordinal, "provider_slot": provider_slot,
        "iters": report.iters, "evaluations": report.evaluations,
        "residual_norm": report.residual_norm,
        "reference_residual_norm": report.reference_residual_norm,
        "rel_residual": report.rel_residual,
        "status": report.status, "action": report.action, "reason": report.reason,
        "residual_evaluation": None if evidence is None else {
            name: getattr(evidence, name) for name in (
                "initial_residual_norm", "rhs_norm", "initial_evaluation_bound",
                "evaluation_bound", "requested_threshold", "applied_threshold",
                "roundoff_limited", "relative_to_rhs_upper")},
    }


def initial_program_reports_snapshot(engine: Any) -> dict[str, Any]:
    """Return an independent metadata snapshot; no mutable executor authority escapes."""
    from copy import deepcopy
    return deepcopy(getattr(engine, "_initial_program_reports", {}))


def resolved_initial_field_slot(name: str, field_plans: Any) -> str:
    """Select one exact resolved Case field for this initialized block; no default route."""
    from collections.abc import Mapping
    from pops.codegen.field_install_plan import ResolvedFieldInstallPlan
    from pops.fields._identity import field_identity

    if not isinstance(field_plans, Mapping) or not field_plans:
        raise ValueError("native initial FieldSolve requires its resolved Case field plans")
    selected = []
    for plan in field_plans.values():
        if type(plan) is not ResolvedFieldInstallPlan or plan.target != "system":
            raise TypeError("native initial FieldSolve requires exact Uniform field install plans")
        if plan.identity != field_identity("resolved-field-install", plan.to_data(include_identity=False)):
            raise ValueError("native initial FieldSolve field plan identity changed")
        publication = plan.output_publication_data()
        route = plan.native_options["output_route"]
        if route["owner_block"] != name:
            continue
        slot = plan.native_options.get("provider_slot")
        if type(slot) is not str or not slot or slot == "pops.system.default-field" \
                or publication["block"] != name:
            raise ValueError("native initial FieldSolve requires its exact named field slot")
        selected.append(slot)
    if len(selected) != 1:
        raise ValueError("native initial FieldSolve requires exactly one qualified field for block %r; "
                         "found %d" % (name, len(selected)))
    return selected[0]


def execute_uniform_initial_program(native: Any, name: str, source: Any, *, params: Any,
                                    field_plans: Any) -> tuple[dict[str, Any], ...]:
    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    def mapped(components: Any, inputs: Any, consumer_qid: str) -> None:
        lowered = lower_analytic_components(components, frame_id=source["frame_id"], bindings=params)
        bindings = [{key: value for key, value in row.items() if key != "value_id"}
                    for row in sorted(inputs, key=lambda row: row["value_id"])]
        native._set_analytic_mapped_state(name,
            [list(opcodes) for opcodes, _ in lowered],
            [list(literals) for _, literals in lowered], bindings, consumer_qid)

    # Authenticate before any seed publication. The historical solve_fields() binding
    # selects a default solver and is deliberately never used by this program.
    field_slot = resolved_initial_field_slot(name, field_plans)
    # A no-input map is the existing native cell-centre evaluation path. It deliberately
    # does not invoke the separate quadrature-based Analytic cell-average materializer.
    mapped(source["seed_components"], (), source["consumer_qid"])
    reports = []
    for stage in source["stages"]:
        if stage["operation"] == "solve_fields":
            report = native._solve_initial_field(field_slot)
            reports.append(initial_solve_report_data(report, provider_slot=field_slot,
                                                      ordinal=len(reports)+1))
        else:
            mapped(stage["components"], stage["inputs"], stage["consumer_qid"])
    return tuple(reports)


__all__ = ["execute_uniform_initial_program", "resolved_initial_field_slot",
           "initial_solve_report_data", "initial_program_reports_snapshot"]
