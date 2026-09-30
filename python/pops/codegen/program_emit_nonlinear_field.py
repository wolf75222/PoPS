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
    if target != "system" or prelude is None:
        raise NotImplementedError("pops.spatial-field-residual@1 requires a Uniform Cartesian native body")
    width, stem = value.attrs["ncomp"], "field_residual_%d" % value.id
    prototype, coefficients = (variables[part.id] for part in value.inputs[:2])
    coefficient_pointer = variables[("field_pointer", value.inputs[1].id)]
    captures = value.inputs[2:2 + value.attrs["capture_count"]]
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
        "auto %s = std::make_shared<pops::runtime::program::PreparedSpatialResidual<pops::kNativeDimension>>(%s, %s, %s);"
        % (workspace, prototype, controls, scalar_cpp(spatial_scalar(value.attrs["finite_difference_step"]))),
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
    seed = "nullptr" if value.attrs["seed_index"] is None else "&(%s)" % variables[value.inputs[-1].id]
    if seed != "nullptr":
        source = variables[value.inputs[-1].id]
        lines.append("%s_layout_invalid |= ((%s).layout() != (%s).layout() || (%s).distribution() != (%s).distribution() || "
                     "(%s).local_rank() != (%s).local_rank() || (%s).local_size() != (%s).local_size() || (%s).ncomp() != %d) ? 1L : 0L;"
                     % (stem, source, prototype, source, prototype, source, prototype, source, prototype, source, width))
    lines += ["if (pops::all_reduce_max(%s_layout_invalid, ctx.prepared_execution_lane()) != 0)" % stem,
              '  throw std::invalid_argument("field residual inputs require exact co-located layout/distribution/width");']
    for capture, name in zip(captures, frozen, strict=True):
        lines.append("pops::PureFieldAlgebra::copy(*%s, %s);" % (name, variables[capture.id]))
    lines.append("pops::elliptic::nd::prepare_general_field_coefficients<pops::kNativeDimension, %d, %d, false>(*%s, *%s);"
                 % (width, width * width, coefficient_pointer, coefficient_boundary))
    callback = stem + "_evaluate"
    lines += ["auto %s = [&](const pops::MultiFab<pops::kNativeDimension>& q, pops::MultiFab<pops::kNativeDimension>& result, int evaluation) {" % callback,
              "  (void)evaluation;", "  pops::PureFieldAlgebra::copy(*%s, q);" % trial,
              "  pops::elliptic::nd::apply_general_field<pops::kNativeDimension, %d, %d>(result, *%s, %s, *%s, std::array<pops::Real, %d>{},"
              % (width, width * width, trial, coefficients, boundary, width * width),
              "      [] { std::array<pops::elliptic::nd::PhysicalFieldBoundary, 2 * pops::kNativeDimension> laws{}; "
              "laws.fill(pops::elliptic::nd::PhysicalFieldBoundary::%s); return laws; }());" % value.attrs["physical_boundary"],
              "  for (std::size_t patch = 0; patch < result.local_size(); ++patch) {",
              "    auto output = result.fab(patch).view();", "    const auto candidate = %s->fab(patch).view();" % trial,
              "    auto status = %s->fab(patch).view();" % status]
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
    _append_solve_report_guard(program, value, outcome, lines, label="original_field_residual", phase="solve")
    for member in ("residual_norm", "reference_residual_norm", "rel_residual"):
        lines.append("ctx.record_scalar(%s, %s.report().%s);" % (json.dumps(stem + "." + member), outcome, member))
    lines += ["pops::PureFieldAlgebra::copy(*%s, %s->candidate());" % (output, workspace),
              "ctx.record_scalar(%s, static_cast<pops::Real>(%s->residual_evaluations() + %s_original_rechecks));" % (json.dumps(stem + ".full_residual_evaluations"), workspace, stem),
              "ctx.record_scalar(%s, static_cast<pops::Real>(%s->derivative_evaluations()));" % (json.dumps(stem + ".finite_difference_jvps"), workspace)]
    variables[value.id] = "(*%s)" % output
