"""Continuation barriers for invocation-owned, synchronized hierarchy field resources."""
from __future__ import annotations

import json
from typing import Any


def hierarchy_region_solves(program: Any) -> tuple[Any, ...]:
    """Select the resolved per-invocation provider contract, independent of the field physics."""
    from pops.codegen.program_lowerability import all_ops
    from pops.solvers.providers import prepared_hierarchy_solver_provider_from_attrs

    nodes = tuple(all_ops(program))
    scoped = tuple(value for value in nodes
                   if value.op == "solve_linear" and value.attrs.get("scope") == "hierarchy")
    selected = tuple(value for value in scoped if "hierarchy_field_identity" in value.attrs)
    if not selected:
        return ()
    top_level = {id(value) for value in program._values}
    barriers = selected + tuple(value for value in nodes if value.op == "field_publication")
    if any(id(value) not in top_level for value in barriers):
        raise NotImplementedError("hierarchy field continuations require top-level qualified barriers")
    if len(selected) != len(scoped) or any(value.op == "solve_spatial_nonlinear"
                                         for value in nodes):
        raise NotImplementedError(
            "invocation-owned hierarchy field barriers cannot share a region with a legacy "
            "singleton or spatial nonlinear hierarchy provider")
    for value in selected:
        prepared_hierarchy_solver_provider_from_attrs(value.attrs).validate_node(
            value, target="amr_system")
    return selected


def open_hierarchy_continuation(program: Any, value: Any, values: Any, var: Any,
                                lines: list[str], once: list[str], *, kind: str) -> None:
    from pops.identity import canonical_bytes
    from pops.time._evaluation_point import evaluation_stage_fraction
    from pops.codegen.program_emit_mapping_regions import continuation_capture

    if kind not in ("linear_solve", "field_publication"):
        raise ValueError("unknown typed hierarchy barrier kind")
    point = evaluation_stage_fraction(value)
    contract = "pops.program-hierarchy-dependencies.cbor.v1:" + canonical_bytes({
        "schema_version": 1,
        "barrier": program._serialize_node(value, include_provenance=False),
        "inputs": [program._serialize_node(source, include_provenance=False)
                   for source in value.inputs],
    }).hex()
    capture = continuation_capture(values, var)
    lines.append("ctx.set_stage_time(%d, %d);" % (point.numerator, point.denominator))
    lines.append("ctx.suspend_hierarchy_barrier(")
    lines.append("  std::remove_reference_t<decltype(ctx)>::HierarchyBarrierKind::%s, %d," %
                 (kind, value.id))
    lines.append("  %s, %s, %d, %d," % (json.dumps(program._ir_hash()), json.dumps(contract),
                                        point.numerator, point.denominator))
    lines.append("  [%s]() {" % capture)
    lines.append("auto& ctx = *ctx_owner;")
    lines.extend(once)
    lines.append("}, [%s]() {" % capture)
    lines.append("auto& ctx = *ctx_owner;")
