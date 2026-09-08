"""Exact same-invocation reuse for pure physical field tuple solves.

The cache is an emission-local alias proof. Runtime solved buffers never survive as
reuse inputs across a new step, rejected attempt, restart, or layout generation.
"""
from __future__ import annotations

import json
from typing import Any

from pops.time._program.serialization import _json_ready
from pops.identity import canonical_bytes

# Every omitted operation is an effect barrier. In particular in-place projection,
# field publication, callback evaluation and control flow cannot preserve this proof.
_SAFE = frozenset({"state", "scalar_field", "linear_combine", "matrix_free_operator",
    "field_problem_load", "field_problem_coefficients", "field_component", "field_gradient", "solve_linear",
    "solve_outcome", "solve_outcome_component", "reduce", "record_scalar", "scalar_op", "compare"})


def observe_operation(value: Any, var: Any, *, model: Any = None) -> None:
    if value.op == "local_transform" and value.region == 0:
        # This existing operation writes fresh state storage and consumes its domain
        # guard before the next operation. Only direct copies of exact native input
        # coordinates preserve component versions; opaque arithmetic never does.
        from pops.codegen.program_emit_kernels import _model_impl
        from pops._ir.expr import Var
        impl = _model_impl(model) if model is not None else None
        declaration = getattr(impl, "_local_transforms", {}).get(value.attrs["transform"])
        if declaration is not None and len(value.inputs) == 1:
            expressions = declaration["expressions"]
            names = tuple(impl.cons_names)
            if len(expressions) == len(names):
                versions = var.setdefault(("field_component_versions",), {})
                source = value.inputs[0]
                for component, expression in enumerate(expressions):
                    if type(expression) is Var and expression.kind == "cons" \
                            and expression.name == names[component]:
                        versions[(value.id, component)] = versions.get(
                            (source.id, component), (source.id, component))
                return
    if value.op not in _SAFE or value.region != 0:
        var[("field_reuse_cache",)] = {}


def _key(value: Any, var: Any, *, target: str) -> tuple[Any, ...] | None:
    if target != "system" or value.region != 0 or value.op != "solve_linear":
        return None
    rhs = value.inputs[1]
    request = value.attrs.get("solve_request")
    if rhs.op != "field_problem_load" or request is None:
        return None
    from pops.time._program.solve_request import validate_solve_request_node
    validate_solve_request_node(value.prog, value)
    from pops.fields._program_problem import validate_field_apply
    apply = value.inputs[0].attrs["apply_result"]
    validate_field_apply(apply)
    coefficient = apply.inputs[2]
    versions = var.get(("field_component_versions",), {})

    def frozen_expression(expression: Any, inputs: Any) -> Any:
        if expression[0] == "input":
            _, ordinal, component, identity = expression
            source = inputs[ordinal]
            return ("input", _json_ready(identity), component,
                    versions.get((source.id, component), (source.id, component)))
        if expression[0] == "literal":
            return _json_ready(expression)
        return (expression[0], *(frozen_expression(child, inputs) for child in expression[1:]))

    def expressions(node: Any) -> bytes:
        return canonical_bytes(tuple(frozen_expression(row, node.inputs)
                                     for row in node.attrs["expressions"]))

    field = _json_ready(rhs.attrs["field_handle"])
    return (field["qualified_id"], rhs.attrs["field_problem_identity"], value.point,
            expressions(rhs), expressions(coefficient),
            canonical_bytes(_json_ready(apply.attrs["reaction"])),
            apply.attrs["physical_boundary"], request["solver_identity"],
            request["initialization_identity"])


def _counters(value: Any, var: Any, lines: Any) -> tuple[str, str, str]:
    field = _json_ready(value.inputs[1].attrs["field_handle"])["qualified_id"]
    counters = var.setdefault(("field_counters",), {})
    if field not in counters:
        index = len(counters)
        solve, reuse = "field_solve_count_%d" % index, "field_reuse_count_%d" % index
        lines.append("pops::Real %s = pops::Real(0), %s = pops::Real(0);" % (solve, reuse))
        counters[field] = (solve, reuse)
    solve, reuse = counters[field]
    return field, solve, reuse


def _record(field: str, solve: str, reuse: str, lines: Any) -> None:
    lines.append("ctx.record_scalar(%s, %s);" % (json.dumps("field.solves/" + field), solve))
    lines.append("ctx.record_scalar(%s, %s);" % (json.dumps("field.reuses/" + field), reuse))


def reuse_field_solve(value: Any, var: Any, lines: Any, *, target: str) -> bool:
    key = _key(value, var, target=target)
    if key is None:
        return False
    cache = var.setdefault(("field_reuse_cache",), {})
    prior = cache.get(key)
    if prior is None:
        return False
    field, solves, reuses = _counters(value, var, lines)
    # The earlier emitted solve guard dominates this alias, so only an explicitly
    # accepted solution can reach it. Failed candidates are never readable here.
    var[value.id] = prior
    lines.append("++%s;" % reuses)
    _record(field, solves, reuses, lines)
    return True


def publish_field_solve(value: Any, var: Any, lines: Any, *, target: str) -> None:
    key = _key(value, var, target=target)
    if key is None:
        return
    field, solves, reuses = _counters(value, var, lines)
    lines.append("++%s;" % solves)
    _record(field, solves, reuses, lines)
    var.setdefault(("field_reuse_cache",), {})[key] = var[value.id]
