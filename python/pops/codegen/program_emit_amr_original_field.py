"""Original FieldProblem body on one synchronized native composite hierarchy."""
from __future__ import annotations

import json
from typing import Any

from pops.identity.scalar import scalar_cpp


def original_amr_native_component() -> Any:
    from pops.native_components import PreparedNativeComponent
    return PreparedNativeComponent.pops_builtin("pops.hierarchy.original-field-residual", entry_headers=(
        "pops/runtime/program/prepared_amr_field_residual.hpp",
        "pops/numerics/elliptic/nd/prepared_composite_general_field.hpp"))


def original_field_consumer(value: Any) -> Any:
    """Authenticate an exact prototype/coefficient consumer, without a layout owner fallback."""
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    matches = [node for node in value.prog._values if node.op == "solve_spatial_field"
               and (any(source is value for source in node.inputs[:2]) or
                    (value.op == "scalar_field" and node.attrs.get("seed_index") is not None
                     and node.inputs[-1] is value))]
    if not matches:
        return None
    if len(matches) != 1:
        raise NotImplementedError("original AMR field storage requires one consuming barrier")
    validate_nonlinear_field_request(value.prog, matches[0])
    return matches[0]


def emit_amr_original_field(program: Any, value: Any, variables: Any, lines: list,
                            prelude: list, block_indices: Any) -> None:
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    from pops.fields._program_expression import field_expression_cpp
    from pops.model import Handle
    from pops.time._program.serialization import _json_ready
    from pops.time._program.equation_identity import _equation_value
    from pops.identity import make_identity
    from pops.time._program.spatial_solve import spatial_newton_options, spatial_scalar
    from pops.codegen.program_emit_solve import _append_solve_report_guard, _consumed_solve_action, _solve_stage_fraction

    validate_nonlinear_field_request(program, value)
    if prelude is None:
        raise NotImplementedError("original AMR fields require a synchronized top-level barrier")
    width, stem = value.attrs["ncomp"], "amr_original_field_%d" % value.id
    captures = value.inputs[2:2 + value.attrs["capture_count"]]
    options = spatial_newton_options(value.attrs["newton_controls"])
    policy_argument = (", pops::runtime::program::AmrFieldRightPreconditioner::kSpatialBasisJacobi"
                       if value.attrs.get("right_preconditioner") is not None else "")
    per_candidate = value.attrs.get("coefficient_evaluation") is not None
    if per_candidate:
        policy_argument = ", pops::runtime::program::AmrFieldRightPreconditioner::kIdentity, pops::runtime::program::AmrFieldCoefficientEvaluation::kPerCandidate"
    if value.attrs.get("right_preconditioner") == "pops.amr.full-residual-basis-lu@1":
        from pops.time._program.spatial_solve import validate_dense_resource_contract
        budget = validate_dense_resource_contract(value.attrs["right_preconditioner_resources"])
        coefficient_mode = "kPerCandidate" if per_candidate else "kFrozen"
        policy_argument = (", pops::runtime::program::AmrFieldRightPreconditioner::kFullResidualBasisLU, "
                           "pops::runtime::program::AmrFieldCoefficientEvaluation::%s, std::uint64_t{%dULL}" %
                           (coefficient_mode, budget))
    controls = "pops::FieldNewtonOptions{" + ", ".join(".%s = %s" %
        (key, str(options[key]) if type(options[key]) is int else scalar_cpp(options[key]))
        for key in ("tolerance", "max_iterations", "linear_tolerance", "linear_max_iterations",
                    "restart", "armijo", "minimum_step")) + "}"
    equation = _json_ready(value.attrs["solve_request"])["equation_identity"]
    graph = program._ir_hash()
    identities = [make_identity("original-field-capture", _equation_value(program, source)).token for source in captures]
    widths = [len(source.space.components) for source in captures]
    owners = [block_indices.get(source.block) for source in captures]
    if any(owner is None for owner in owners):
        raise ValueError("original AMR field capture requires its exact physical State owner")
    stage = _solve_stage_fraction(value)
    authority_args = "%d, %s_invocation->equation, %s_invocation->graph, %s_invocation->clock, pops::amr::Rational(%d, %d), %s_invocation->identities, %s_invocation->widths, %s_invocation->owners" % (
        value.id, stem, stem, stem, stage.numerator, stage.denominator, stem, stem, stem)
    entries = [("coefficient_components", "std::uint64_t{%d}" % (width * width)),
               ("restart", "std::uint64_t{%d}" % options["restart"]),
               ("modes.count", "std::uint64_t{0}")]
    entries += [("reaction.%d.%d" % (i, j), "double{0}")
                for i in range(width) for j in range(width)]
    physical = "periodic" if value.attrs["physical_boundary"] == "periodic" else "neumann"
    native_options = ('[&]() { pops::PreparedProviderOptions options{"pops.hierarchy.general-field@1", {%s}}; '
        'for (int axis=0; axis<pops::kNativeDimension; ++axis) for (int side=0; side<2; ++side) '
        'options.values.emplace("physical." + std::to_string(axis) + "." + std::to_string(side), '
        'std::string(%s)); return options; }()' % (
            ", ".join("{%s, %s}" % (json.dumps(key), cpp) for key, cpp in entries), json.dumps(physical)))
    prelude += [
        "ctx.register_hierarchy_tensor_solver_provider(std::make_shared<pops::elliptic::nd::CompositeGeneralFieldProvider<pops::kNativeDimension>>());",
        'ctx.configure_hierarchy_field_solver(%d, %d, %s, "pops.hierarchy.general-field.gmres@1", %s, %s, '
        '{"pops.general-field.coefficients", "pops.general-field.rhs"}, "pops.general-field.solution", %s);'
        % (value.id, width, json.dumps(value.attrs["field_problem_identity"]),
           json.dumps(value.attrs["solver_identity"]), json.dumps(equation), native_options)]
    phase = variables.get(("original_field_hierarchy_phase",))
    if phase == "gather":
        lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
        for index, source in enumerate(captures):
            lines += ["auto& %s_capture_%d = ctx.hierarchy_field_scratch(%d, %d, %d, %d, 0);" %
                      (stem, index, value.id, value.id, 2 + index, widths[index]),
                      "ctx.prepare_spatial_collectively([&] { pops::PureFieldAlgebra::copy(%s_capture_%d, %s); Kokkos::fence(); });" %
                      (stem, index, variables[source.id])]
        if value.attrs["seed_index"] is not None:
            lines += ["auto& %s_seed = ctx.hierarchy_field_scratch(%d, %d, 1, %d, 0);" %
                      (stem, value.id, value.id, width),
                      "ctx.prepare_spatial_collectively([&] { pops::PureFieldAlgebra::copy(%s_seed, %s); Kokkos::fence(); });" %
                      (stem, variables[value.inputs[-1].id])]
        variables[value.id] = "ctx.hierarchy_field_solution(%d)" % value.id
        return
    if phase not in ("solve",):
        variables[value.id] = "ctx.hierarchy_field_solution(%d)" % value.id
        lines.append("auto& %s = ctx.hierarchy_field_solution(%d);" % (stem, value.id))
        variables[value.id] = stem
        return

    lines += ["using %s_Core = pops::runtime::program::PreparedAmrFieldResidual<pops::kNativeDimension>;" % stem,
              "auto %s_provider = ctx.retain_hierarchy_field_solver(%d);" % (stem, value.id),
              "struct %s_Invocation { std::string equation, graph, clock; std::vector<std::string> identities; std::vector<int> widths, owners; };" % stem,
              "std::shared_ptr<const %s_Invocation> %s_invocation;" % (stem, stem),
              "ctx.prepare_spatial_collectively([&] {",
              "  %s_invocation = std::make_shared<%s_Invocation>(%s_Invocation{%s, %s, %s, {%s}, {%s}, {%s}});" % (stem, stem, stem, json.dumps(equation), json.dumps(graph), json.dumps(program.clock.qualified_id), ", ".join(json.dumps(identity) for identity in identities), ", ".join(map(str, widths)), ", ".join(map(str, owners))),
              "});",
              "auto %s_authority = ctx.original_hierarchy_field_authority(%s);" % (stem, authority_args),
              "std::vector<%s_Core::hierarchy_type> %s_captures;" % (stem, stem),
              "%s_Core::hierarchy_type %s_seed;" % (stem, stem),
              "ctx.prepare_spatial_collectively([&] { %s_captures.resize(%d); });" % (stem, len(captures)),
              "for (int level = 0; level < ctx.nlev(); ++level) ctx.with_program_attempt_level(level, [&] {"]
    for index in range(len(captures)):
        lines += ["  auto& input_%d = ctx.hierarchy_field_scratch(%d, %d, %d, %d, 0, false);" %
                  (index, value.id, value.id, 2 + index, widths[index]),
                  "  ctx.prepare_spatial_collectively([&] { %s_captures[%d].emplace_back(input_%d); });" % (stem, index, index)]
    if value.attrs["seed_index"] is not None:
        lines += ["  auto& seed = ctx.hierarchy_field_scratch(%d, %d, 1, %d, 0, false);" % (value.id, value.id, width),
                  "  ctx.prepare_spatial_collectively([&] { %s_seed.emplace_back(seed); });" % stem]
    lines += ["});", "std::vector<const %s_Core::hierarchy_type*> %s_capture_views;" % (stem, stem),
              "ctx.prepare_spatial_collectively([&] { for (const auto& tower : %s_captures) %s_capture_views.push_back(&tower); });" % (stem, stem),
              "auto %s_core = %s_Core::prepare(%s_provider, %s_authority, %s_capture_views, %s, %s, ctx.prepared_execution_lane()%s);" %
              (stem, stem, stem, stem, stem, controls, scalar_cpp(spatial_scalar(value.attrs["finite_difference_step"])), policy_argument)]
    from pops.fields._evolved_stage_contract import emit_issued_duration
    duration = emit_issued_duration(value.attrs["source_contract"].get("temporal_tau"), program, value.point, stem + "_issued_dt", lines, operation_id=value.id)
    unknowns = tuple(Handle.from_canonical_identity(_json_ready(item))
                     for item in value.attrs["source_contract"]["unknown_components"])
    callback = stem + "_body"
    lines += ["auto %s = [&](const auto& q, const auto& captured, auto& result, int evaluation) {" % callback,
              "  (void)evaluation;", "  pops::Real invalid = 0;",
              "  for (std::size_t level = 0; level < result.size(); ++level)",
              "    for (std::size_t patch = 0; patch < result[level].local_size(); ++patch) {",
              "      const auto candidate = q[level].fab(patch).view();",
              "      const auto output = result[level].fab(patch).view();"]
    for index in range(len(captures)):
        lines.append("      const auto capture%d = captured[%d][level].fab(patch).view();" % (index, index))
    lines += ["      invalid = std::max(invalid, pops::for_each_cell_reduce_max(result[level].box(patch),",
              "        [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& index) {",
              "          bool finite = true;"]
    if duration is not None:
        start = next(index for index in range(len(lines)-1, -1, -1) if lines[index].startswith("auto %s =" % callback)) + 1
        lines.insert(start, '  if (ctx.boundary_evaluation_point(%d).dt != %s) throw std::logic_error("original stage issued frame duration changed");' % (value.id, duration))
    for component, expression in enumerate(value.attrs["local_expressions"]):
        code, _ = field_expression_cpp(expression, captures,
            views=tuple("capture%d" % i for i in range(len(captures))), unknowns=unknowns, duration_name=duration)
        lines += ["          const pops::Real component_%d = output(index, %d) + %s;" % (component, component, code),
                  "          finite = finite && std::isfinite(candidate(index, %d)) && std::isfinite(component_%d);" % (component, component),
                  "          output(index, %d) = component_%d;" % (component, component)]
    lines += ["          return finite ? pops::Real(0) : pops::Real(1);", "        }));", "    }",
              "  if (invalid != 0) throw std::invalid_argument(\"nonfinite_original_field_residual\");", "};"]
    coefficient_callback = stem + "_coefficients"
    if per_candidate:
        lines += ["auto %s = [&](const auto& q, const auto& captured, auto& coefficients, int evaluation) {" % coefficient_callback,
                  "  (void)evaluation;", "  pops::Real invalid = 0;",
                  "  for (std::size_t level = 0; level < coefficients.size(); ++level)",
                  "    for (std::size_t patch = 0; patch < coefficients[level]->local_size(); ++patch) {",
                  "      const auto candidate = q[level].fab(patch).view();",
                  "      const auto output = coefficients[level]->fab(patch).view();"]
        if duration is not None:
            start = next(index for index in range(len(lines)-1, -1, -1) if lines[index].startswith("auto %s =" % coefficient_callback)) + 1
            lines.insert(start, '  if (ctx.boundary_evaluation_point(%d).dt != %s) throw std::logic_error("original stage issued frame duration changed");' % (value.id, duration))
        for index in range(len(captures)):
            lines.append("      const auto capture%d = captured[%d][level].fab(patch).view();" % (index, index))
        lines += ["      invalid = std::max(invalid, pops::for_each_cell_reduce_max(coefficients[level]->box(patch),",
                  "        [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& index) {", "          bool finite = true;"]
        for component, expression in enumerate(value.attrs["source_contract"]["diffusion"]):
            code, _ = field_expression_cpp(expression, captures,
                views=tuple("capture%d" % i for i in range(len(captures))), unknowns=unknowns, duration_name=duration)
            lines += ["          const pops::Real coefficient_%d = %s;" % (component, code),
                      "          finite = finite && std::isfinite(coefficient_%d);" % component,
                      "          output(index, %d) = coefficient_%d;" % (component, component)]
        lines += ["          return finite ? pops::Real(0) : pops::Real(1);", "        }));", "    }",
                  '  if (invalid != 0) throw std::invalid_argument("nonfinite_candidate_diffusion");', "};"]
    report, outcome = stem + "_report", stem + "_outcome"
    action_kind, _ = _consumed_solve_action(program, value)
    action = "pops::SolveAction::kRejectAttempt" if action_kind == "reject_attempt" else "pops::SolveAction::kFailRun"
    seed = "nullptr" if value.attrs["seed_index"] is None else "&%s_seed" % stem
    solve_call = ("  %s = %s_core->solve_candidate(%s_authority, %s, %s, %s, ctx.prepared_execution_lane());" %
                  (report, stem, stem, seed, callback, coefficient_callback)) if per_candidate else (
                  "  %s = %s_core->solve(%s_authority, %s, %s, ctx.prepared_execution_lane());" % (report, stem, stem, seed, callback))
    lines += ["pops::SolveReport %s;" % report, "try {", solve_call,
              "} catch (const std::exception& failure) {",
              "  %s.mark_failed(pops::SolveStatus::kInvalidEvaluation, %s, failure.what());" % (report, action), "}",
              "const %s_Core::hierarchy_type* %s_candidate = nullptr;" % (stem, stem),
              "if (%s.solved_value_available()) %s_candidate = &%s_core->candidate(%s_authority, ctx.prepared_execution_lane());" % (report, stem, stem, stem),
              "auto& %s_operator = pops::runtime::program::require_original_field_operator(*%s_provider, ctx.prepared_execution_lane());" % (stem, stem),
              "std::function<void()> %s_validate;" % stem,
              "std::function<void(%s_Core::hierarchy_type&)> %s_synchronize;" % (stem, stem),
              "ctx.prepare_spatial_collectively([&] {",
              "  %s_validate = [&, core = %s_core, %s_invocation = %s_invocation] { core->require_authority(ctx.original_hierarchy_field_authority(%s), ctx.prepared_execution_lane()); };" % (stem, stem, stem, stem, authority_args),
              "  %s_synchronize = [&](auto& candidate) { %s_operator.synchronize_original_field_candidate(candidate); };" % (stem, stem),
              "});",
              "auto %s = %s_provider->stage_original_field_candidate_collectively(std::move(%s), %s_candidate, %d, ctx.prepared_execution_lane(), %s_provider, std::move(%s_validate), std::move(%s_synchronize));" %
              (outcome, stem, report, stem, options["max_iterations"], stem, stem, stem)]
    _append_solve_report_guard(program, value, outcome, lines, label="original_amr_field_residual", phase="solve")
    closed_sources = [node for node in program._values if node.op == "spatial_interaction"
                      and node.attrs.get("contract") == "pops.spatial-interaction@3"]
    closed_sources = [node for node in closed_sources if node.inputs[0].inputs[0].inputs[0].inputs[0] is value]
    if closed_sources:
        snapshot_budget = min(int(node.attrs["max_workspace_bytes"]["uint64_hex"], 16) for node in closed_sources)
        from pops.codegen.program_emit_spatial_interaction import closed_interaction_identity
        closed_bindings = ", ".join("{%d, %d, %d, %d, %s}" % (node.id, node.inputs[0].id,
            node.inputs[0].attrs["component"], block_indices[node.attrs["closed_field_source"]["owner_block"]],
            json.dumps(closed_interaction_identity(node))) for node in closed_sources)
        lines += ["std::vector<std::tuple<std::int64_t, std::int64_t, int, int, std::string>> %s_closed_bindings;" % stem,
                  "ctx.prepare_spatial_collectively([&] { %s_closed_bindings = {%s}; });" % (stem, closed_bindings)]
        lines += ["std::function<pops::runtime::program::AmrFieldResidualAuthority()> %s_closed_authority;" % stem,
                  "ctx.prepare_spatial_collectively([&] {",
                  "  %s_closed_authority = [&, %s_invocation = %s_invocation] {" % (stem, stem, stem),
                  "    return ctx.original_hierarchy_field_authority(%s);" % authority_args,
                  "  };", "});",
                  "ctx.seal_original_field_source(%d, %s_provider, %s_core, %s_closed_authority, %s_closed_bindings, %s, %dULL);" %
                  (value.id, stem, stem, stem, stem, json.dumps(value.attrs["field_problem_identity"]), snapshot_budget)]
    for member in ("residual_norm", "reference_residual_norm", "rel_residual"):
        lines.append("ctx.record_scalar(%s, %s.report().%s);" % (json.dumps(stem + "." + member), outcome, member))
    lines += ["ctx.record_scalar(%s, static_cast<pops::Real>(%s_core->residual_evaluations()));" %
              (json.dumps(stem + ".full_residual_evaluations"), stem),
              "ctx.record_scalar(%s, static_cast<pops::Real>(%s_core->derivative_evaluations()));" %
              (json.dumps(stem + ".finite_difference_jvps"), stem),
              "auto& %s = ctx.hierarchy_field_solution(%d);" % (stem, value.id)]
    variables[value.id] = stem
