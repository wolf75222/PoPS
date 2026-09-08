"""Execute an authenticated scalar spatial residual through native Newton/GMRES."""
from __future__ import annotations

import json
from typing import Any

from pops.identity.scalar import scalar_cpp


def emit_spatial_solve(program: Any, value: Any, base: Any, variables: Any, model: Any,
                       lines: Any, prelude: Any, block_indices: Any, field_plans: Any,
                       target: str, emit_node: Any) -> None:
    from pops.time._program.spatial_solve import (
        validate_spatial_request, validate_spatial_commit, spatial_newton_options,
        spatial_scalar, spatial_rate_weight,
    )
    from pops.codegen.program_models import model_for_node
    from pops.codegen.program_emit_diffusion import (
        _emit_diffusive_preparation, _emit_diffusive_rhs, _emit_diffusive_accepted,
    )
    from pops.codegen.program_emit_solve import (
        _append_solve_report_guard, _consumed_solve_action, _solve_stage_fraction,
    )
    from pops.codegen.program_emit_kernels import _coeff_cpp

    validate_spatial_commit(program, value)
    validate_spatial_request(program, value)
    if target != "system" or prelude is None:
        raise NotImplementedError("spatial residual currently requires a Uniform native install scope")
    owner = block_indices[value.block]
    stem = "spatial_%d" % value.id
    workspace, trial = stem + "_workspace", stem + "_trial"
    options = spatial_newton_options(value.attrs["newton_controls"])
    option_order = ("tolerance", "max_iterations", "linear_tolerance", "linear_max_iterations",
                    "restart", "armijo", "minimum_step")
    controls = "pops::FieldNewtonOptions{" + ", ".join(
        ".%s = %s" % (key, str(options[key]) if type(options[key]) is int else scalar_cpp(options[key]))
        for key in option_order) + "}"
    prelude += [
        "auto %s = std::make_shared<pops::runtime::program::PreparedSpatialResidual<"
        "pops::kNativeDimension>>(ctx.state(%d), %s, %s);"
        % (workspace, owner, controls, scalar_cpp(spatial_scalar(value.attrs["finite_difference_step"]))),
        "auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
        "ctx.scratch_state_like(ctx.state(%d)));" % (trial, owner),
    ]
    lines.append("ctx.require_cartesian_generated_operator(%d, \"spatial_implicit_diffusion\");" % owner)
    stage = _solve_stage_fraction(value)
    lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
    subvars = dict(variables)
    subvars[value.attrs["iterate"].id] = stem + "_iterate"
    preparations = {}
    for node in value.attrs["residual_block"]:
        if node.op == "diffusive_rhs":
            prepared = "spatial_diffusion_%d" % node.id
            preparations[node.id] = prepared
            lines += _emit_diffusive_preparation(
                node, "(*%s)" % trial, prepared, model_for_node(model, node),
                variables.get(("program_provider_plans",)), owner, target)
            lines.append("(void)ctx.rhs_scratch(%d, 0, *%s);" % (node.id, trial))
            source_count = sum(row.kind == "source" for row in node.attrs["physical_balance"].occurrences)
            for slot in range(1, source_count + 1):
                lines.append("(void)ctx.rhs_scratch(%d, %d, *%s);" % (node.id, slot, trial))
        elif node.op in ("linear_combine", "source"):
            # Prewarm the exact node/slot/shape before Newton. Repeated scratch access
            # inside the residual only resets values; it does not allocate a new field.
            allocator = "scratch_state" if node.op == "linear_combine" else "rhs_scratch"
            lines.append("(void)ctx.%s(%d, 0, *%s);" % (allocator, node.id, trial))
    body = ["pops::PureFieldAlgebra::copy(*%s, q);" % trial,
            "auto& %s_iterate = *%s;" % (stem, trial)]
    for node in value.attrs["residual_block"]:
        if node is value.attrs["iterate"]:
            continue
        if node.op == "diffusive_rhs":
            _emit_diffusive_rhs(
                node, subvars, body, model_for_node(model, node),
                variables.get(("program_provider_plans",)), owner, target,
                prepared_var=preparations[node.id])
        else:
            emit_node(node, value.attrs["iterate"], subvars, body, prelude)
    body.append("pops::PureFieldAlgebra::copy(result, %s);" % subvars[value.attrs["residual"].id])
    callback = stem + "_residual"
    lines.append("auto %s = [&](const pops::MultiFab<pops::kNativeDimension>& q, "
                 "pops::MultiFab<pops::kNativeDimension>& result, int evaluation) {" % callback)
    lines.append("  (void)evaluation;")
    lines += ["  " + line for line in body]
    lines.append("};")
    report, outcome = stem + "_report", stem + "_outcome"
    seed = "nullptr" if len(value.inputs) == 1 else "&(%s)" % variables[value.inputs[1].id]
    action_kind, _ = _consumed_solve_action(program, value)
    action = "pops::SolveAction::kRejectAttempt" if action_kind == "reject_attempt" \
        else "pops::SolveAction::kFailRun"
    lines += [
        "pops::SolveReport %s;" % report,
        "try {",
        "  %s = %s->solve(%s, %s, ctx.prepared_execution_lane());"
        % (report, workspace, seed, callback),
        "} catch (const pops::runtime::program::StepAttemptRejected& failure) {",
        "  %s.mark_failed(failure.status(), %s, failure.what());" % (report, action),
        "  %s.evaluations = %s->residual_evaluations();" % (report, workspace),
        "} catch (const pops::runtime::program::DiffusiveEvaluationError& failure) {",
        "  %s.mark_failed(pops::SolveStatus::kInvalidEvaluation, "
        "failure.status() == 3 ? pops::SolveAction::kFailRun : %s, "
        "std::string(failure.what()) + \" native_status=\" + std::to_string(failure.status()) + "
        "\" native_reason=\" + std::to_string(failure.reason()));" % (report, action),
        "  %s.evaluations = %s->residual_evaluations();" % (report, workspace),
        "}",
        "pops::SolveOutcome %s = pops::SolveOutcome::collective_lane("
        "std::move(%s), ctx.prepared_execution_lane());" % (outcome, report),
    ]
    _append_solve_report_guard(program, value, outcome, lines, label="spatial_implicit", phase="solve")
    # A solved Newton report follows the residual of its accepted iterate. Its retained faces
    # therefore belong to that iterate; rejected line-search/FD evaluations never reach this hook.
    for node in value.attrs["residual_block"]:
        if node.op == "diffusive_rhs":
            _emit_diffusive_accepted(
                node, preparations[node.id], lines, _coeff_cpp(spatial_rate_weight(value, node)),
                "implicit-stage:" + str(value.point) + "/solve:" + str(value.id))
    result = stem + "_result"
    lines += [
        "auto& %s = ctx.scratch_state(%d, 0, ctx.state(%d));" % (result, value.id, owner),
        "pops::PureFieldAlgebra::copy(%s, %s->candidate());" % (result, workspace),
        "ctx.record_scalar(%s, static_cast<pops::Real>(%s->residual_evaluations()));"
        % (json.dumps(stem + ".full_residual_evaluations"), workspace),
        "ctx.record_scalar(%s, static_cast<pops::Real>(%s->derivative_evaluations()));"
        % (json.dumps(stem + ".finite_difference_jvps"), workspace),
    ]
    variables[value.id] = result
