"""Continuation barriers for invocation-owned, synchronized hierarchy field resources."""
from __future__ import annotations

import json
from typing import Any


def _structured_region_nodes(program: Any) -> set[int]:
    allowed: set[int] = set()
    pending = list(program._values)
    while pending:
        value = pending.pop()
        allowed.add(id(value))
        if value.op in ("branch", "structured_fold", "capture_result", "structured_iterate"):
            for key in ("true_block", "false_block", "body_block", "cond_block"):
                pending.extend(value.attrs.get(key, ()))
    return allowed


def hierarchy_path_rhs(program: Any) -> tuple[Any, ...]:
    from pops.codegen.program_lowerability import all_ops
    from .program_emit_hierarchy_operations import is_path_rhs, validate_operation_barrier
    nodes = tuple(all_ops(program))
    selected = tuple(value for value in nodes if is_path_rhs(value))
    if not selected:
        return ()
    allowed = _structured_region_nodes(program)
    if any(id(value) not in allowed for value in selected):
        raise NotImplementedError("path RHS suspension needs an explicit region continuation")
    for value in selected:
        validate_operation_barrier(program, value)
    return selected


def hierarchy_spatial_solves(program: Any) -> tuple[Any, ...]:
    from pops.codegen.program_lowerability import all_ops
    from .program_emit_hierarchy_operations import is_spatial_solve, validate_operation_barrier
    selected = tuple(value for value in all_ops(program) if is_spatial_solve(value))
    allowed = _structured_region_nodes(program)
    if any(id(value) not in allowed for value in selected):
        raise NotImplementedError("composite spatial solve needs an explicit region continuation")
    for value in selected:
        validate_operation_barrier(program, value)
    return selected


def has_hierarchy_continuations(program: Any) -> bool:
    return bool(hierarchy_region_solves(program) or hierarchy_path_rhs(program)
                or hierarchy_field_publications(program) or hierarchy_named_field_barriers(program)
                or hierarchy_original_field_solves(program) or hierarchy_spatial_solves(program))


def hierarchy_original_field_solves(program: Any) -> tuple[Any, ...]:
    """Each authored nonlinear field solve owns its synchronized provider and barrier."""
    from pops.codegen.program_lowerability import all_ops
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    selected = tuple(value for value in all_ops(program) if value.op == "solve_spatial_field")
    allowed = set()
    pending = list(program._values)
    while pending:
        value = pending.pop()
        allowed.add(id(value))
        if value.op in ("branch", "structured_fold", "capture_result", "structured_iterate"):
            for key in ("true_block", "false_block", "body_block", "cond_block"):
                pending.extend(value.attrs.get(key, ()))
    if any(id(value) not in allowed or value.attrs.get("schedule") is not None
           for value in selected):
        raise NotImplementedError("original AMR field suspension needs an explicit region continuation")
    for value in selected:
        validate_nonlinear_field_request(program, value)
    return selected


def hierarchy_named_field_barriers(program: Any) -> tuple[Any, ...]:
    """Named composite solves and iterate publication consume one common-time hierarchy."""
    from pops.codegen.program_lowerability import all_ops
    selected = tuple(value for value in all_ops(program)
                     if value.op in ("solve_fields", "solve_fields_from_blocks", "field_warm_start"))
    top_level = {id(value) for value in program._values}
    if any(id(value) not in top_level or value.attrs.get("schedule") is not None
           for value in selected):
        raise NotImplementedError("named AMR field barriers require unscheduled top-level operations")
    return selected


def hierarchy_field_publications(program: Any) -> tuple[Any, ...]:
    """Select publication barriers on layouts without their own hierarchy solve.

    A hierarchy solve's validated lowering already owns its publication barrier,
    either through invocation continuations or the gather/solve/publish regions.
    Only a consumer without such a solve needs a publication-only continuation.
    """
    from pops.codegen.program_lowerability import all_ops

    nodes = tuple(all_ops(program))
    if any(value.op in ("solve_spatial_field", "solve_spatial_nonlinear")
           or value.op == "solve_linear" and value.attrs.get("scope") == "hierarchy"
           for value in nodes):
        return ()
    selected = tuple(value for value in nodes if value.op == "field_publication")
    top_level = {id(value) for value in program._values}
    if any(id(value) not in top_level for value in selected):
        raise NotImplementedError("hierarchy field continuations require top-level qualified barriers")
    return selected


def hierarchy_region_solves(program: Any) -> tuple[Any, ...]:
    """Select the resolved per-invocation provider contract, independent of the field physics."""
    from pops.codegen.program_lowerability import all_ops
    from pops.solvers.providers import prepared_hierarchy_solver_provider_from_attrs

    from pops.time._program.temporal_field import is_temporal_field_solve
    nodes = tuple(all_ops(program))
    scoped = tuple(value for value in nodes
                   if value.op == "solve_linear" and value.attrs.get("scope") == "hierarchy"
                   and not is_temporal_field_solve(value))
    # Temporal evaluations own all-level input reconstruction at the active
    # invocation point. Only ordinary common-time solves suspend the level body;
    # their distinct solve resources preserve observations across that suspension.
    selected = tuple(value for value in scoped if "hierarchy_field_identity" in value.attrs)
    paths = hierarchy_path_rhs(program)
    named = hierarchy_named_field_barriers(program)
    if named and scoped and not selected:
        top_level = {id(value) for value in program._values}
        if len(scoped) != 1 or id(scoped[0]) not in top_level:
            raise NotImplementedError("named field barriers support one top-level singleton hierarchy solve")
        selected = scoped
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
        if "hierarchy_field_identity" in value.attrs:
            prepared_hierarchy_solver_provider_from_attrs(value.attrs).validate_node(
                value, target="amr_system")
    return selected


def open_hierarchy_continuation(program: Any, value: Any, values: Any, var: Any,
                                lines: list[str], once: list[str], *, kind: str) -> None:
    from pops.identity import canonical_bytes
    from pops.time.canonical_data import strict_data
    from pops.time._evaluation_point import evaluation_stage_fraction
    from pops.codegen.program_emit_mapping_regions import continuation_capture

    if kind not in ("linear_solve", "field_publication", "spatial_rhs", "named_field_solve", "field_warm_start", "nonlinear_solve", "control"):
        raise ValueError("unknown typed hierarchy barrier kind")
    point = evaluation_stage_fraction(value)
    contract = "pops.program-hierarchy-dependencies.cbor.v2:" + canonical_bytes(strict_data({
        "schema_version": 2,
        "barrier": program._serialize_node(value, include_provenance=False),
        "inputs": [program._serialize_node(source, include_provenance=False)
                   for source in value.inputs],
    }, where="hierarchy dependency contract")).hex()
    capture = continuation_capture(values, var)
    from .program_amr_finite_scope import continuation_scope_capture, resume_scope
    capture = continuation_scope_capture(capture, var)
    lines.append("ctx.set_stage_time(%d, %d);" % (point.numerator, point.denominator))
    lines.append("ctx.suspend_hierarchy_barrier(")
    lines.append("  std::remove_reference_t<decltype(ctx)>::HierarchyBarrierKind::%s, %d," %
                 (kind, value.id))
    lines.append("  %s, %s, %d, %d," % (json.dumps(program._ir_hash()), json.dumps(contract),
                                        point.numerator, point.denominator))
    lines.append("  [%s]() {" % capture)
    lines.append("auto& ctx = *ctx_owner;")
    resume_scope(lines, var)
    lines.extend(once)
    lines.append("}, [%s]() {" % capture)
    lines.append("auto& ctx = *ctx_owner;")
    resume_scope(lines, var)
