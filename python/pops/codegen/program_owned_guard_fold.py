"""Plan a closed owned-cell reduction for an exclusively consumed guard."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pops.codegen.program_emit_expressions import checked_pointwise_rows
from pops.codegen.program_source_map import operation_group
from pops.model.spaces import StateSpace
from pops.time._program.affine_expressions import affine_field_terms
from pops.time.expressions import component_names


@dataclass(frozen=True, slots=True)
class OwnedGuardFold:
    prefix: tuple[Any, ...]
    guard: Any
    field: Any
    component: int
    expression: Any


def _guard_shape(guard: Any) -> tuple[dict[int, int], Any, int] | None:
    """Return the exact use edges of the finite/min or checked-expression/min cut."""
    field, condition = guard.inputs
    if field.vtype != "state" or type(field.space) is not StateSpace or field.space.centering != "cell":
        return None
    edges = {condition.id: guard.id}
    positive = condition
    if condition.op == "bool_op":
        if condition.attrs != {"operation": "and"} or len(condition.inputs) != 2:
            return None
        finite, positive = condition.inputs
        if finite.op != "scalar_isfinite" or finite.attrs or len(finite.inputs) != 1:
            return None
        norm, = finite.inputs
        if (norm.op != "reduce" or norm.attrs != {
                "kind": "norm_inf_all", "component_contract": "pops.program.norm-inf-all@1",
        } or len(norm.inputs) != 1 or norm.inputs[0] is not field):
            return None
        edges.update({positive.id: condition.id, finite.id: condition.id, norm.id: finite.id})
    if (positive.op != "compare" or len(positive.inputs) != 1
            or set(positive.attrs) != {"cmp", "rhs"} or positive.attrs["cmp"] != ">"
            or positive.attrs["rhs"].to_python() != 0):
        return None
    minimum, = positive.inputs
    if (minimum.op != "reduce" or len(minimum.inputs) != 1
            or set(minimum.attrs) != {"kind", "comp"} or minimum.attrs["kind"] != "min"):
        return None
    edges[minimum.id] = positive.id
    subject, = minimum.inputs
    component = minimum.attrs["comp"]
    if type(component) is not int or not 0 <= component < len(component_names(subject)):
        return None
    if condition is not positive:
        return (edges, None, component) if subject is field else None
    if (subject.op != "pointwise_expression" or len(subject.inputs) != 1
            or subject.inputs[0] is not field or subject.state_ref != field.state_ref
            or subject.space != field.space or set(subject.attrs) != {"expressions", "expression_nodes"}
            or affine_field_terms(subject) is not None):
        return None
    edges[subject.id] = minimum.id
    return edges, subject, component


def owned_guard_folds(program: Any, values: list[Any], variables: Any) -> dict[int, OwnedGuardFold]:
    """Keep contiguous, same-point cuts whose removed storage has no other reader."""
    from pops.codegen.program_balance_due import balance_value_due_expression
    from pops.codegen.program_source_map import emission_program_hash
    from pops.time._program.guard_failure_sites import guard_failure_site
    from pops.time.solve_outcome import FailRun, RejectAttempt

    readers: dict[int, set[int]] = defaultdict(set)
    for value in values:
        for operand in (*value.inputs, *program._subblock_value_refs(value)):
            readers[operand.id].add(value.id)
    authority = variables[("program_value_authority",)]
    roots = {value.id for value in (*program._commits.values(), *program._post_sync_commits.values())}
    roots.update(row.source for row in authority.fields)
    roots.update(row.ssa for row in authority.producers)
    roots.update(row[1] for row in getattr(program, "_integral_transfers", ()))
    if program._dt_bound is not None:
        body, result = program._dt_bound
        roots.add(result.id)
        for value in body:
            roots.update(item.id for item in (*value.inputs, *program._subblock_value_refs(value)))
    if program._advance_outputs:
        return {}  # Observation regions can retain genuinely live values off the main body.

    folds = {}
    program_hash = emission_program_hash(program)
    for end, guard in enumerate(values):
        if guard.op != "acceptance_guard" or len(guard.inputs) != 2:
            continue
        action = guard.attrs["action"]
        terminal = type(action) is FailRun and set(guard.attrs) == {"action", "guard"}
        trial = (type(action) is RejectAttempt and set(guard.attrs) == {"action", "guard", "trial_site"}
                 and guard_failure_site(program, guard, program_hash) is not None)
        if not (terminal or trial):
            continue
        shape = _guard_shape(guard)
        if shape is None:
            continue
        edges, expression, component = shape
        field = guard.inputs[0]
        first = end - len(edges)
        prefix = tuple(values[first:end])
        if (first < 0 or set(edges) != {value.id for value in prefix}
                or field.block is None or guard.block != field.block or guard.point != field.point
                or roots.intersection(edges)
                or any(readers[value.id] != {edges[value.id]}
                       or (value.op != "bool_op" and value.block != field.block)
                       or value.point != field.point or value.region != guard.region
                       or balance_value_due_expression(variables, value.id) is not None
                       for value in prefix)):
            continue
        folds[first] = OwnedGuardFold(prefix, guard, field, component, expression)
    return folds


def emit_owned_guard_fold(fold: OwnedGuardFold, program: Any, variables: Any, model: Any,
                          lines: Any, block_indices: Any) -> None:
    """Emit the fast reduction and the exact original lowering on the other branch."""
    from .program_emit_ops import _emit_op

    field, guard = fold.field, fold.guard
    names = dict(variables)
    ncomp = len(component_names(field))
    condition = "owned_guard_condition_%d" % guard.id
    active = "owned_guard_active_%d" % guard.id
    status = "owned_guard_status_%d" % guard.id
    mapped = fold.expression is not None
    columns = ["field_view(index, %d)" % component for component in range(ncomp)]
    lines += [
        "const bool %s = owned_guard_host_lane && ctx.owned_guard_unmasked();" % active,
        "bool %s;" % condition,
        "if (%s) {" % active,
    ]
    variables[guard.inputs[1].id] = condition
    variables[("owned_guard_result", guard.id)] = active
    with operation_group(fold.prefix, lines, environment=variables, channel="body"):
        lines += [
            "const auto %s = ctx.owned_guard_fold<%s>(%d, %s, %d," % (
                status, "true" if mapped else "false", block_indices[field.block],
                variables[field.id], ncomp),
            "  KOKKOS_LAMBDA(const auto& field_view, const auto& index) -> pops::Real {",
        ]
        if mapped:
            body, results, invalid = checked_pointwise_rows(fold.expression, [columns])
            lines += ["    " + line for line in body]
            lines.append("    if (%s) return pops::Real(2);" % invalid)
            positive = "%s > pops::Real(0)" % results[fold.component]
        else:
            # On this unmasked nonempty owner domain, the norm is finite iff all
            # stored values are finite and a strict positive min rejects both zeros.
            positive = " && ".join(
                ["Kokkos::isfinite(%s)" % column for column in columns]
                + ["%s > pops::Real(0)" % columns[fold.component]])
        lines += ["    return (%s) ? pops::Real(0) : pops::Real(1);" % positive, "  });"]
        if mapped:
            lines += [
                "if (%s == pops::Real(2))" % status,
                "  throw pops::runtime::program::StepAttemptRejected("
                "pops::SolveStatus::kInvalidEvaluation, \"pointwise_expression\", "
                "\"non-finite scientific expression input or intermediate\");",
            ]
        lines += ["%s = %s == pops::Real(0);" % (condition, status)]
    lines.append("} else {")
    for value in fold.prefix:
        _emit_op(program, value, None, frozenset(), names, model, lines, None, block_indices)
    lines += ["%s = %s;" % (condition, names[guard.inputs[1].id]), "}"]
