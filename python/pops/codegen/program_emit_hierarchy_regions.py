"""Continuation barriers for invocation-owned, synchronized hierarchy field resources."""
from __future__ import annotations

import json
from typing import Any


def hierarchy_path_rhs(program: Any) -> tuple[Any, ...]:
    from pops.codegen.program_lowerability import all_ops
    nodes = tuple(all_ops(program))
    selected = tuple(value for value in nodes
                     if value.op == "rhs" and value.attrs.get("path_conservative", False))
    if not selected:
        return ()
    top_level = {id(value) for value in program._values}
    if any(id(value) not in top_level or value.attrs.get("schedule") is not None for value in selected):
        raise NotImplementedError("path-conservative hierarchy barriers require unscheduled top-level RHS")
    if any(value.op == "solve_spatial_nonlinear" for value in nodes):
        raise NotImplementedError("path RHS cannot share a hierarchy with spatial nonlinear solves")
    return selected


def has_hierarchy_continuations(program: Any) -> bool:
    return bool(hierarchy_region_solves(program) or hierarchy_path_rhs(program))


def hierarchy_region_solves(program: Any) -> tuple[Any, ...]:
    """Select the resolved per-invocation provider contract, independent of the field physics."""
    from pops.codegen.program_lowerability import all_ops
    from pops.solvers.providers import prepared_hierarchy_solver_provider_from_attrs

    nodes = tuple(all_ops(program))
    scoped = tuple(value for value in nodes
                   if value.op == "solve_linear" and value.attrs.get("scope") == "hierarchy")
    selected = tuple(value for value in scoped if "hierarchy_field_identity" in value.attrs)
    paths = hierarchy_path_rhs(program)
    if paths and scoped and not selected:
        # One existing tensor provider may own the source prefix. Its singleton
        # storage is consumed once before any path RHS or field publication;
        # the synchronized gather and continuation preserve that exact lifetime.
        top = list(program._values)
        ordinals = {id(value): index for index, value in enumerate(top)}
        path_ids = {id(value) for value in paths}
        first_barrier = min(index for index, value in enumerate(top)
                            if value.op in ("layout_map_export", "layout_map_import", "field_publication")
                            or id(value) in path_ids)
        if (len(scoped) != 1 or id(scoped[0]) not in ordinals
                or ordinals[id(scoped[0])] >= first_barrier
                or any(value.op in ("while", "range", "branch") for value in top)):
            raise NotImplementedError(
                "a singleton hierarchy source must be the unique top-level prefix before path barriers")
        selected = scoped
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

    if kind not in ("linear_solve", "field_publication", "spatial_rhs"):
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
