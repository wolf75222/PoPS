"""Native direct field load/coefficient kernels and consumed component observations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.identity import Identity, canonical_bytes
from pops.fields._program_expression import field_expression_cpp
from pops.time._program.serialization import _json_ready


def hierarchy_field_solve(value: Any) -> Any:
    """Find the unique authenticated solve consuming this exact load/coefficient value."""
    if value.op == "field_component":
        from pops.fields._observation_contract import validate_field_observation
        return validate_field_observation(value)[2]
    matches = []
    for solve in value.prog._values:
        if solve.op != "solve_linear" or solve.attrs.get("hierarchy_field_identity") != value.attrs.get("field_problem_identity"):
            continue
        coefficient = None
        if value.op == "field_problem_coefficients":
            from pops.fields._program_problem import validate_field_apply
            apply = solve.inputs[0].attrs.get("apply_result")
            validate_field_apply(apply)
            coefficient = apply.inputs[2]
        if (value.op == "field_problem_load" and solve.inputs[1] is value) or coefficient is value:
            matches.append(solve)
    if len(matches) != 1:
        raise ValueError("field hierarchy storage requires one exact consuming solve identity")
    return matches[0]


def emit_field_problem_value(value: Any, var: Any, lines: list[str], prelude: Any,
                             *, target: str) -> None:
    if target not in ("system", "amr_system") or prelude is None:
        raise NotImplementedError("field expression storage requires a supported native Program body")
    hierarchy = target == "amr_system"
    if hierarchy and value.op != "field_component" and value.attrs.get("scope") != "hierarchy":
        raise ValueError("AMR general fields require a synchronized hierarchy numerical solver")
    identity = Identity.from_token(value.attrs.get("field_problem_identity"))
    if identity.domain != "field-problem":
        raise ValueError("field expression lost its physical field-problem identity")
    ncomp = value.attrs.get("ncomp")
    if type(ncomp) is not int or ncomp < 1:
        raise ValueError("field expression requires a positive exact component count")
    component = value.op == "field_component"
    sources = tuple(value.inputs)
    expressions = []
    if component:
        from pops.fields._observation_contract import validate_field_observation

        width, selected, _solve = validate_field_observation(value)
        expressions = ["input0(index, %d)" % selected]
    else:
        if value.op not in ("field_problem_load", "field_problem_coefficients"):
            raise ValueError("unsupported field expression operation")
        encoded = value.attrs.get("expressions")
        if not isinstance(encoded, (tuple, list)) or len(encoded) != ncomp:
            raise ValueError("field expression component count differs from its encoded expressions")
        reads = {}
        for expression in encoded:
            code, dependencies = field_expression_cpp(
                expression, sources, views=tuple("input%d" % i for i in range(len(sources))))
            expressions.append(code)
            for handle in dependencies:
                reads[handle.qualified_id] = handle.canonical_identity()
        declared = value.attrs.get("field_dependencies", ())
        if any(not isinstance(item, Mapping) for item in declared) or {
                canonical_bytes(_json_ready(item)) for item in declared} != {
                canonical_bytes(item) for item in reads.values()}:
            raise ValueError("field expression dependency metadata differs from its actual input reads")

    token = "program_field_%d" % value.id
    if hierarchy:
        solve = hierarchy_field_solve(value)
        if component:
            lines.append("auto& %s = ctx.scalar_scratch(%d, 0, %s, 1, 1);" % (token, value.id, var[sources[0].id]))
        else:
            slot = "coefficients" if value.op == "field_problem_coefficients" else "rhs"
            lines.append('auto& %s = ctx.hierarchy_field_assembly(%d, "pops.general-field.%s");' % (token, solve.id, slot))
        lines.append("auto* %s_status = &ctx.scalar_scratch(%d, 1, %s, 1, 0);" % (token, value.id, token))
        pointer = "(&%s)" % token
    else:
        prelude.append("auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
                       "ctx.alloc_scalar_field(%d, 1));" % (token, ncomp))
        prelude.append("auto %s_status = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
                       "ctx.alloc_scalar_field(1, 0));" % token)
        pointer = token
    var[value.id] = "(*%s)" % pointer
    var[("field_pointer", value.id)] = pointer
    if hierarchy:
        # This token is a reference to context-owned storage, even though the ordinary
        # expression spelling uses a pointer wrapper. Continuations retain the object by reference.
        var[("continuation_reference", value.id)] = token
    if component:
        var[("field_observation", value.id)] = identity.token
    destination = "(*%s)" % pointer
    if hierarchy and not component and var.get(("hierarchy_field_phase",)) in ("solve", "observe", "publish"):
        return
    lines.extend(["{", "  long field_layout_invalid = 0;"])
    for i, source in enumerate(sources):
        source_cpp = var[source.id]
        source_width = width if component else len(source.space.components)
        lines.append("  const auto& field_source_%d = %s;" % (i, source_cpp))
        lines.append("  field_layout_invalid |= field_source_%d.ncomp() != %d ? 1L : 0L;"
                     % (i, source_width))
        lines.append("  field_layout_invalid |= (field_source_%d.layout() != %s.layout() || "
                     "field_source_%d.distribution() != %s.distribution() || "
                     "field_source_%d.local_rank() != %s.local_rank() || "
                     "field_source_%d.local_size() != %s.local_size() || "
                     "field_source_%d.shares_storage_with(%s)) ? 1L : 0L;"
                     % (i, destination, i, destination, i, destination, i, destination, i, destination))
    lines.extend([
        "  if (pops::all_reduce_max(field_layout_invalid, ctx.prepared_execution_lane()) != 0)",
        '    throw std::invalid_argument("field expression inputs require exact layout/distribution identity");',
        "  for (std::size_t li = 0; li < %s.local_size(); ++li) {" % destination,
        "    auto output = %s.fab(li).view();" % destination,
        "    auto status = %s_status->fab(li).view();" % token,
    ])
    for i, _source in enumerate(sources):
        lines.append("    const auto input%d = field_source_%d.fab(li).view();" % (i, i))
    lines.append("    pops::for_each_cell(%s.box(li), [=] POPS_HD("
                 "const pops::CellIndex<pops::kNativeDimension>& index) {" % destination)
    lines.append("      bool finite = true;")
    for i, expression in enumerate(expressions):
        lines.extend(["      const pops::Real value_%d = %s;" % (i, expression),
                      "      finite = finite && std::isfinite(value_%d);" % i,
                      "      output(index, %d) = value_%d;" % (i, i)])
    lines.extend([
        "      status(index, 0) = finite ? pops::Real(0) : pops::Real(1);",
        "    });",
        "  }",
        "  if (pops::all_reduce_max(pops::reduce_max_local(*%s_status), "
        "ctx.prepared_execution_lane()) > 0)" % token,
        "    throw pops::runtime::program::StepAttemptRejected("
        "pops::SolveStatus::kInvalidEvaluation, "
        "pops::runtime::program::StepAttemptDisposition::kReject, 0, "
        '"field_evaluation", "nonfinite field expression");',
        "}",
    ])


__all__ = ["emit_field_problem_value"]


def field_observation_reduction(value: Any, var: Any, *, target: str) -> str:
    """Reduce an already authenticated Cartesian field observation without a species owner."""
    if target not in ("system", "amr_system") or len(value.inputs) != 1:
        raise ValueError("field observation reduction requires one authenticated native field value")
    source = value.inputs[0]
    if source.op != "field_component" or var.get(("field_observation", source.id)) != source.attrs.get("field_problem_identity"):
        raise ValueError("unowned reduction requires an authenticated consumed field observation")
    kind = value.attrs.get("kind")
    methods = {"sum": "sum_component", "abs_sum": "abs_sum_component",
               "max": "max_component", "min": "min_component"}
    if kind not in methods:
        raise NotImplementedError("this independent field observation reduction has no native realization")
    component = value.attrs.get("comp", 0)
    if type(component) is not int or component != 0:
        raise ValueError("scalar field observation has exactly one component")
    if target == "amr_system":
        solve = hierarchy_field_solve(source)
        return 'ctx.reduce_hierarchy_field_component(%d, %d, "%s")' % (
            solve.id, source.attrs["component"], kind)
    return "ctx.%s(%s, 0)" % (methods[kind], var[source.id])
