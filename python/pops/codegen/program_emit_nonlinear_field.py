"""Uniform original mixed residual, using native Kokkos stencils and Newton/GMRES."""
from __future__ import annotations

import json
from typing import Any

from pops.identity.scalar import scalar_cpp


def emit_nonlinear_field(program: Any, value: Any, variables: Any, lines: list,
                         prelude: list, *, target: str, block_indices: Any) -> None:
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    from pops.fields._program_expression import field_expression_cpp
    from pops.time._program.spatial_solve import spatial_newton_options, spatial_scalar
    from pops.codegen.program_emit_solve import _append_solve_report_guard, _consumed_solve_action

    validate_nonlinear_field_request(program, value)
    if target == "amr_system":
        from pops.codegen.program_emit_amr_original_field import emit_amr_original_field
        emit_amr_original_field(program, value, variables, lines, prelude, block_indices)
        return
    if value.attrs.get("right_preconditioner") is not None:
        raise NotImplementedError("SpatialBasisJacobi@1 is an original composite AMR realization; Uniform is unsupported")
    if target != "system" or prelude is None:
        raise NotImplementedError("pops.spatial-field-residual@1 requires a Uniform Cartesian native body")
    width, stem = value.attrs["ncomp"], "field_residual_%d" % value.id
    prototype, coefficients = (variables[part.id] for part in value.inputs[:2])
    coefficient_pointer = variables[("field_pointer", value.inputs[1].id)]
    captures = value.inputs[2:2 + value.attrs["capture_count"]]
    per_candidate = value.attrs.get("coefficient_evaluation") is not None
    allocation_start = len(prelude)
    if per_candidate:
        coefficient_pointer = stem + "_candidate_coefficients"
        coefficients = "(*%s)" % coefficient_pointer
        prelude.append("auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.alloc_scalar_field(%d, 1));" % (coefficient_pointer, width * width))
    from pops.model import Handle
    from pops.time._program.serialization import _json_ready
    unknowns = tuple(Handle.from_canonical_identity(_json_ready(item))
                     for item in value.attrs["source_contract"]["unknown_components"])
    options = spatial_newton_options(value.attrs["newton_controls"])
    controls = "pops::FieldNewtonOptions{" + ", ".join(".%s = %s" %
        (key, str(options[key]) if type(options[key]) is int else scalar_cpp(options[key]))
        for key in ("tolerance", "max_iterations", "linear_tolerance", "linear_max_iterations",
                    "restart", "armijo", "minimum_step")) + "}"
    workspace, trial, status, output, boundary, coefficient_boundary = (stem + suffix for suffix in
        ("_workspace", "_trial", "_status", "_output", "_boundary", "_coefficient_boundary"))
    prelude += [
        "auto %s = std::make_shared<pops::runtime::program::PreparedSpatialResidual<pops::kNativeDimension>>(%s, %s, %s%s);"
        % (workspace, prototype, controls, scalar_cpp(spatial_scalar(value.attrs["finite_difference_step"])), ", true, &ctx.prepared_execution_lane()" if per_candidate else ""),
        "auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.alloc_scalar_field(%d, 1));" % (trial, width),
        "auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.alloc_scalar_field(1, 0));" % status,
        "auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.alloc_scalar_field(%d, 0));" % (output, width),
        "auto %s = ctx.prepare_mesh_boundary_session(*%s, ctx.prepared_execution_lane());" % (boundary, trial),
        "auto %s = ctx.prepare_mesh_boundary_session(*%s, ctx.prepared_execution_lane());" % (coefficient_boundary, coefficient_pointer),
    ]
    frozen = []
    lines.append("long %s_layout_invalid = 0;" % stem)
    for index, capture in enumerate(captures):
        owner = block_indices.get(capture.block)
        if owner is None:
            raise ValueError("field residual capture has no authenticated physical layout witness")
        lines.append("ctx.require_cartesian_generated_operator(%d, \"original_field_residual\");" % owner)
        name = stem + "_capture_%d" % index
        frozen.append(name)
        source = variables[capture.id]
        ncomp = len(capture.space.components)
        prelude.append("auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.alloc_scalar_field(%d, 0));" % (name, ncomp))
        lines.append("%s_layout_invalid |= ((%s).layout() != %s->layout() || (%s).distribution() != %s->distribution() || "
                     "(%s).local_rank() != %s->local_rank() || (%s).local_size() != %s->local_size() || (%s).ncomp() != %d) ? 1L : 0L;"
                     % (stem, source, name, source, name, source, name, source, name, source, ncomp))
    if per_candidate:
        from pops.codegen.program_emit_field_problem import guard_candidate_allocations
        prelude[allocation_start:] = guard_candidate_allocations(prelude[allocation_start:])
    seed = "nullptr" if value.attrs["seed_index"] is None else "&(%s)" % variables[value.inputs[-1].id]
    if seed != "nullptr":
        source = variables[value.inputs[-1].id]
        lines.append("%s_layout_invalid |= ((%s).layout() != (%s).layout() || (%s).distribution() != (%s).distribution() || "
                     "(%s).local_rank() != (%s).local_rank() || (%s).local_size() != (%s).local_size() || (%s).ncomp() != %d) ? 1L : 0L;"
                     % (stem, source, prototype, source, prototype, source, prototype, source, prototype, source, width))
    lines += ["if (pops::all_reduce_max(%s_layout_invalid, ctx.prepared_execution_lane()) != 0)" % stem,
              '  throw std::invalid_argument("field residual inputs require exact co-located layout/distribution/width");']
    if per_candidate:
        lines += ["std::exception_ptr %s_capture_error;" % stem, "try {"]
    for capture, name in zip(captures, frozen, strict=True):
        lines.append("pops::PureFieldAlgebra::copy(*%s, %s);" % (name, variables[capture.id]))
    if per_candidate:
        lines += ["} catch (...) { %s_capture_error = std::current_exception(); }" % stem,
                  "try { Kokkos::fence(); } catch (...) { if (!%s_capture_error) %s_capture_error = std::current_exception(); }" % (stem, stem),
                  'pops::collectively_rethrow_exception(%s_capture_error, ctx.prepared_execution_lane(), "candidate coefficient frozen captures");' % stem]
    if not per_candidate:
        lines.append("pops::elliptic::nd::prepare_general_field_coefficients<pops::kNativeDimension, %d, %d, false>(*%s, *%s);"
                     % (width, width * width, coefficient_pointer, coefficient_boundary))
    if per_candidate:
        lines += ["const auto* %s_lane = &ctx.prepared_execution_lane();" % stem,
                  "pops::runtime::program::PreparedResourceAttempt %s_attempt;" % stem,
                  "pops::runtime::multiblock::BoundaryEvaluationPoint %s_point;" % stem,
                  "std::exception_ptr %s_authority_error;" % stem,
                  "try { %s_attempt = ctx.resource_attempt(); %s_point = ctx.boundary_evaluation_point(%d); }" % (stem, stem, value.id),
                  "catch (...) { %s_authority_error = std::current_exception(); }" % stem,
                  'pops::collectively_rethrow_exception(%s_authority_error, *%s_lane, "candidate coefficient invocation authority");' % (stem, stem)]
    arithmetic_argument = ", true" if value.attrs.get("coefficient_face_policy") is not None else ""
    if per_candidate:
        arithmetic_argument += ", true"
    callback = stem + "_evaluate"
    lines += ["auto %s = [&](const pops::MultiFab<pops::kNativeDimension>& q, pops::MultiFab<pops::kNativeDimension>& result, int evaluation) {" % callback,
              "  (void)evaluation;"]
    if not per_candidate:
        lines.append("  pops::PureFieldAlgebra::copy(*%s, q);" % trial)
    if per_candidate:
        lines += ["  std::exception_ptr coefficient_error;", "  try {",
                  "    if (&ctx.prepared_execution_lane() != %s_lane || !%s_attempt.visible() || !%s_attempt.same_attempt(ctx.resource_attempt()) || %s_point != ctx.boundary_evaluation_point(%d))" % (stem, stem, stem, stem, value.id),
                  '      throw std::logic_error("candidate coefficient point/attempt/lane authority changed");',
                  "    pops::PureFieldAlgebra::copy(*%s, q);" % trial,
                  "    pops::Real invalid = 0;",
                  "    for (std::size_t patch = 0; patch < %s->local_size(); ++patch) {" % coefficient_pointer,
                  "      const auto candidate = %s->fab(patch).view();" % trial,
                  "      const auto coefficient = %s->fab(patch).view();" % coefficient_pointer]
        for index, name in enumerate(frozen):
            lines.append("      const auto capture%d = std::as_const(*%s).fab(patch).view();" % (index, name))
        lines += ["      invalid = std::max(invalid, pops::for_each_cell_reduce_max(%s->box(patch)," % coefficient_pointer,
                  "        [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& index) {", "          bool finite = true;"]
        for component, expression in enumerate(value.attrs["source_contract"]["diffusion"]):
            code, _ = field_expression_cpp(expression, captures,
                views=tuple("capture%d" % i for i in range(len(frozen))), unknowns=unknowns)
            lines += ["          const pops::Real value_%d = %s;" % (component, code),
                      "          finite = finite && std::isfinite(value_%d);" % component,
                      "          coefficient(index, %d) = value_%d;" % (component, component)]
        lines += ["          return finite ? pops::Real(0) : pops::Real(1);", "        }));", "    }",
                  '    if (invalid != 0) throw std::invalid_argument("nonfinite_candidate_diffusion");',
                  "  } catch (...) { coefficient_error = std::current_exception(); }",
                  "  try { Kokkos::fence(); } catch (...) { if (!coefficient_error) coefficient_error = std::current_exception(); }",
                  '  pops::collectively_rethrow_exception(coefficient_error, *%s_lane, "candidate diffusion local evaluation");' % stem,
                  "  pops::elliptic::nd::prepare_general_field_coefficients<pops::kNativeDimension, %d, %d, false, true>(*%s, *%s);" %
                  (width, width * width, coefficient_pointer, coefficient_boundary)]
    lines += ["  pops::elliptic::nd::apply_general_field<pops::kNativeDimension, %d, %d%s>(result, *%s, %s, *%s, std::array<pops::Real, %d>{},"
              % (width, width * width, arithmetic_argument, trial, coefficients, boundary, width * width),
              "      [] { std::array<pops::elliptic::nd::PhysicalFieldBoundary, 2 * pops::kNativeDimension> laws{}; "
              "laws.fill(pops::elliptic::nd::PhysicalFieldBoundary::%s); return laws; }());" % value.attrs["physical_boundary"],
              "  for (std::size_t patch = 0; patch < result.local_size(); ++patch) {",
              "    auto output = result.fab(patch).view();", "    const auto candidate = %s->fab(patch).view();" % trial,
              "    auto status = %s->fab(patch).view();" % status]
    if per_candidate:
        # The apply primitive has already voted; protect the following local body too.
        offset = next(index for index in range(len(lines)-1, -1, -1)
                      if lines[index] == "  for (std::size_t patch = 0; patch < result.local_size(); ++patch) {")
        lines[offset:offset] = ["  std::exception_ptr body_error;", "  pops::Real body_invalid = 0;", "  try {"]
    for index, name in enumerate(frozen):
        lines.append("    const auto capture%d = %s->fab(patch).view();" % (index, name))
    lines += ["    pops::for_each_cell(result.box(patch), [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& index) {",
              "      bool finite = true;"]
    for component, expression in enumerate(value.attrs["local_expressions"]):
        code, _ = field_expression_cpp(expression, captures, views=tuple("capture%d" % i for i in range(len(frozen))), unknowns=unknowns)
        lines += ["      const pops::Real component_%d = output(index, %d) + %s;" % (component, component, code),
                  "      finite = finite && std::isfinite(candidate(index, %d)) && std::isfinite(component_%d);" % (component, component),
                  "      output(index, %d) = component_%d;" % (component, component)]
    lines += ["      status(index, 0) = finite ? pops::Real(0) : pops::Real(1);", "    });", "  }",
              "  if (pops::all_reduce_max(pops::reduce_sum_local(*%s), ctx.prepared_execution_lane()) > 0)" % status,
              '    throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::kInvalidEvaluation, "original_field_residual", "nonfinite_original_field_residual");',
              "};"]
    if per_candidate:
        offset = next(index for index in range(len(lines)-1, -1, -1)
                      if lines[index].startswith("  if (pops::all_reduce_max(pops::reduce_sum_local"))
        # The reduction belongs to the local phase as well; its value is then voted.
        lines[offset:offset] = ["    body_invalid = pops::reduce_sum_local(*%s);" % status, "  } catch (...) { body_error = std::current_exception(); }",
                               "  try { Kokkos::fence(); } catch (...) { if (!body_error) body_error = std::current_exception(); }",
                               '  pops::collectively_rethrow_exception(body_error, *%s_lane, "candidate original local body");' % stem]
        lines[offset + 4] = "  if (pops::all_reduce_max(body_invalid, *%s_lane) > 0)" % stem
    report, outcome = stem + "_report", stem + "_outcome"
    action_kind, _ = _consumed_solve_action(program, value)
    action = "pops::SolveAction::kRejectAttempt" if action_kind == "reject_attempt" else "pops::SolveAction::kFailRun"
    lines += ["pops::SolveReport %s;" % report, "int %s_original_rechecks = 0;" % stem, "try {",
              "  %s = %s->solve(%s, %s, ctx.prepared_execution_lane());" % (report, workspace, seed, callback),
              "  if (%s.solved_value_available()) {" % report,
              "    ++%s_original_rechecks;" % stem,
              "    %s(%s->candidate(), *%s, 0);" % (callback, workspace, output),
              "    const pops::Real original_norm = std::sqrt(pops::all_reduce_sum(pops::dot_all_local(*%s, *%s), ctx.prepared_execution_lane()));" % (output, output),
              "    if (!std::isfinite(original_norm) || original_norm > %s * std::max(pops::Real(1), %s.reference_residual_norm))"
              % (scalar_cpp(options["tolerance"]), report),
              '      %s.mark_failed(pops::SolveStatus::kInvalidEvaluation, %s, "original_field_residual_recheck_failed");' % (report, action),
              "    %s.residual_norm = original_norm;" % report,
              "    %s.rel_residual = original_norm / (%s.reference_residual_norm > pops::Real(0) ? %s.reference_residual_norm : pops::Real(1));" % (report, report, report), "  }",
              "} catch (const pops::runtime::program::StepAttemptRejected& failure) {",
              "  %s.mark_failed(failure.status(), %s, failure.what());" % (report, action), "}",
              "%s.evaluations = %s->residual_evaluations() + %s_original_rechecks;" % (report, workspace, stem),
              "pops::SolveOutcome %s = pops::SolveOutcome::collective_lane(std::move(%s), ctx.prepared_execution_lane());" % (outcome, report)]
    if per_candidate:
        solve_offset = next(index for index in range(len(lines)-1, -1, -1)
                            if "%s->solve(" % workspace in lines[index])
        lines[solve_offset] = lines[solve_offset].replace("ctx.prepared_execution_lane()", "*%s_lane" % stem)
        offset = next(index for index in range(len(lines)-1, -1, -1)
                      if "const pops::Real original_norm =" in lines[index])
        lines[offset:offset+1] = ["    pops::Real original_squared = 0;", "    std::exception_ptr recheck_error;",
                                "    try { original_squared = pops::dot_all_local(*%s, *%s); } catch (...) { recheck_error = std::current_exception(); }" % (output, output),
                                "    try { Kokkos::fence(); } catch (...) { if (!recheck_error) recheck_error = std::current_exception(); }",
                                '    pops::collectively_rethrow_exception(recheck_error, *%s_lane, "candidate original residual norm");' % stem,
                                "    const pops::Real original_norm = std::sqrt(pops::all_reduce_sum(original_squared, *%s_lane));" % stem]
    _append_solve_report_guard(program, value, outcome, lines, label="original_field_residual", phase="solve")
    for member in ("residual_norm", "reference_residual_norm", "rel_residual"):
        lines.append("ctx.record_scalar(%s, %s.report().%s);" % (json.dumps(stem + "." + member), outcome, member))
    lines += ["pops::PureFieldAlgebra::copy(*%s, %s->candidate());" % (output, workspace),
              "ctx.record_scalar(%s, static_cast<pops::Real>(%s->residual_evaluations() + %s_original_rechecks));" % (json.dumps(stem + ".full_residual_evaluations"), workspace, stem),
              "ctx.record_scalar(%s, static_cast<pops::Real>(%s->derivative_evaluations()));" % (json.dumps(stem + ".finite_difference_jvps"), workspace)]
    variables[value.id] = "(*%s)" % output
