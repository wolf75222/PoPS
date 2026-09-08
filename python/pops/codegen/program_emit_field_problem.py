"""Native direct field load/coefficient kernels and consumed component observations."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.identity import Identity, canonical_bytes
from pops.fields._program_expression import field_expression_cpp
from pops.time._program.serialization import _json_ready


def emit_field_problem_value(value: Any, var: Any, lines: list[str], prelude: Any,
                             *, target: str) -> None:
    if target != "system" or prelude is None:
        raise NotImplementedError("field expression storage requires a Uniform Program body")
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
        if ncomp != 1 or len(sources) != 1:
            raise ValueError("field observation requires one consumed packed source")
        if sources[0].op != "solve_outcome_component":
            raise ValueError("field observation requires an explicitly consumed solve result")
        selected = value.attrs.get("component")
        width = sources[0].attrs.get("ncomp")
        if type(selected) is not int or type(width) is not int or not 0 <= selected < width:
            raise ValueError("field observation component is outside its solved unknown tuple")
        from pops.model import Handle

        unknown = Handle.from_canonical_identity(_json_ready(value.attrs.get("field_unknown")))
        if unknown.kind != "field" or unknown.block_ref is not None:
            raise ValueError("field observation lost its independent solved-field identity")
        from pops.time._graph.base import strict_data

        consumed = sources[0]
        try:
            solve = consumed.inputs[0].inputs[0]
            physical = solve.attrs["solve_request"]["physical_problem"]
            expected_unknown = physical["unknown_components"][selected]
        except (IndexError, KeyError, AttributeError, TypeError) as exc:
            raise ValueError("field observation lost its consumed physical solve authority") from exc
        if canonical_bytes(strict_data(unknown.canonical_identity(), where="field unknown")) != \
                canonical_bytes(strict_data(expected_unknown, where="solved field unknown")):
            raise ValueError("field observation does not select its declared solved unknown")
        from pops.codegen.program_field_plan import _nodes, _reachable

        native_sources = tuple(node for node in _reachable(solve, _nodes(value.prog))
                               if node.op in ("field_problem_load", "field_problem_apply"))
        if not native_sources or any(node.attrs.get("field_problem_identity") != identity.token
                                     for node in native_sources):
            raise ValueError("field observation belongs to a different native field problem")
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
    prelude.append("auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
                   "ctx.alloc_scalar_field(%d, 1));" % (token, ncomp))
    prelude.append("auto %s_status = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
                   "ctx.alloc_scalar_field(1, 0));" % token)
    var[value.id] = "(*%s)" % token
    var[("field_pointer", value.id)] = token
    if component:
        var[("field_observation", value.id)] = identity.token
    destination = "(*%s)" % token
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
    if target != "system" or len(value.inputs) != 1:
        raise ValueError("field observation reduction requires one Uniform field value")
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
    return "ctx.%s(%s, 0)" % (methods[kind], var[source.id])
