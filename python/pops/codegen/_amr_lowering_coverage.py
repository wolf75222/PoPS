"""Exact AMR authoring-to-runtime rows for the global lowering coverage report."""
from __future__ import annotations

from typing import Any

from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringCoverageRow
from pops.identity import make_identity


def amr_lowering_coverage(
    *,
    resolved_hierarchy: Any,
    transfer: Any,
    bootstrap: Any,
    execution: Any,
) -> LoweringCoverageReport:
    """Project the resolved AMR authorities onto their executable runtime routes."""

    from pops.amr.authoring import AMRExecution
    from pops.mesh._amr._bootstrap_contracts import BootstrapPlan
    from pops.mesh._amr._transfer_contracts import ResolvedAMRTransfer
    from pops.mesh._amr.hierarchy_resolution import ResolvedHierarchy

    if type(resolved_hierarchy) is not ResolvedHierarchy:
        raise TypeError("AMR lowering coverage requires an exact ResolvedHierarchy")
    if type(transfer) is not ResolvedAMRTransfer:
        raise TypeError("AMR lowering coverage requires an exact ResolvedAMRTransfer")
    if type(bootstrap) is not BootstrapPlan:
        raise TypeError("AMR lowering coverage requires an exact BootstrapPlan")
    if type(execution) is not AMRExecution:
        raise TypeError("AMR lowering coverage requires an exact AMRExecution")

    hierarchy_identity = resolved_hierarchy.identity.token
    transfer_identity = transfer.identity.token
    bootstrap_identity = bootstrap.identity.token
    execution_identity = make_identity("amr-execution", execution.to_data()).token
    tagging = bootstrap.tagging
    tagging_target = "amr-runtime-tagging:%s" % tagging.qualified_id

    rows = [
        LoweringCoverageRow(
            source="amr-hierarchy:%s" % hierarchy_identity,
            disposition="lowered",
            targets=("amr-runtime-hierarchy:%s" % hierarchy_identity,),
        ),
        LoweringCoverageRow(
            source="amr-regrid:%s" % resolved_hierarchy.plan.regrid.identity.token,
            disposition="lowered",
            targets=("amr-runtime-regrid:%s" % hierarchy_identity,),
        ),
        LoweringCoverageRow(
            source="amr-tagging-graph:%s" % tagging.qualified_id,
            disposition="lowered",
            targets=(tagging_target,),
        ),
        LoweringCoverageRow(
            source="amr-tagging-hysteresis:%s" % tagging.qualified_id,
            disposition="lowered",
            targets=("%s:hysteresis" % tagging_target,),
        ),
        LoweringCoverageRow(
            source="amr-tagging-conflict-policy:%s" % tagging.qualified_id,
            disposition="lowered",
            targets=("%s:conflict-policy" % tagging_target,),
        ),
        LoweringCoverageRow(
            source="amr-transfer-plan:%s" % transfer_identity,
            disposition="lowered",
            targets=("amr-runtime-transfer:%s" % transfer_identity,),
        ),
        LoweringCoverageRow(
            source="amr-execution:%s" % execution_identity,
            disposition="lowered",
            targets=("amr-runtime-execution:%s" % execution.mode,),
        ),
        LoweringCoverageRow(
            source="amr-bootstrap:%s" % bootstrap_identity,
            disposition="lowered",
            targets=("amr-runtime-bootstrap:%s" % bootstrap_identity,),
        ),
    ]
    registrations = {
        registration.node_type: registration
        for registration in tagging.registrations
    }

    def append_predicate(node: Any, path: str) -> None:
        registration = registrations[node.node_type]
        rows.append(LoweringCoverageRow(
            source="amr-tagging-predicate:%s:%s:%s"
            % (tagging.qualified_id, path, node.node_type),
            disposition="lowered",
            targets=(registration.lowering.qualified_id,),
        ))
        for index, child in enumerate(node.operands()):
            append_predicate(child, "%s/%d" % (path, index))

    append_predicate(tagging.graph.refine, "refine")
    if tagging.graph.coarsen is not None:
        append_predicate(tagging.graph.coarsen, "coarsen")
    rows.extend(
        LoweringCoverageRow(
            source="amr-transfer-entry:%s" % entry.identity.token,
            disposition="lowered",
            targets=(
                "amr-runtime-transfer-operation:%s:%s"
                % (
                    entry.native_materialization.to_data()["action"],
                    entry.key.operation.name,
                ),
            ),
        )
        for entry in transfer.entries
    )
    rows.extend(
        LoweringCoverageRow(
            source="amr-subcycling:%s:%d-%d"
            % (execution_identity, relation.parent_level, relation.child_level),
            disposition="lowered",
            targets=(
                "amr-runtime-clock-relation:%d-%d:%d/%d"
                % (
                    relation.parent_level,
                    relation.child_level,
                    relation.temporal_ratio.numerator,
                    relation.temporal_ratio.denominator,
                ),
            ),
        )
        for relation in execution.relations
    )
    return LoweringCoverageReport(rows)


__all__ = ["amr_lowering_coverage"]
