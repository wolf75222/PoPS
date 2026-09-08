"""One gather/solve/publish barrier for an authored composite temporal stage."""

from __future__ import annotations
import json
from pops.identity.scalar import scalar_cpp


def emit_amr_spatial_solve(
    program, value, base, variables, model, lines, prelude, block_indices, field_plans, emit_node
):
    from pops.time._program.spatial_solve import (
        spatial_newton_options,
        spatial_scalar,
        spatial_rate_weight,
    )
    from pops.codegen.program_models import model_for_node
    from pops.codegen.program_emit_diffusion import (
        _selected,
        _boundary_cpp,
        _emit_diffusive_rhs,
        _emit_diffusive_accepted,
    )
    from pops.codegen.program_emit_solve import (
        _solve_stage_fraction,
        _consumed_solve_action,
        _append_solve_report_guard,
    )
    from pops.codegen.program_emit_kernels import _coeff_cpp

    rates = [node for node in value.attrs["residual_block"] if node.op == "diffusive_rhs"]
    if len(rates) != 1:
        raise NotImplementedError(
            "composite spatial stage requires exactly one scalar diffusive residual"
        )
    rate = rates[0]
    if any(row.kind in {"flux", "drift"} for row in rate.attrs["physical_balance"].occurrences):
        raise NotImplementedError(
            "composite implicit residual cannot yet include transport or fitted drift; retain transport in its explicit IMEX stage"
        )
    _, selected, occurrences = _selected(rate, model_for_node(model, rate))
    owner = block_indices[value.block]
    stem = "spatial_%d" % value.id
    workspace, trial, slot = stem + "_workspace", stem + "_trial", stem + "_diffusion"
    phase = variables.get(("spatial_hierarchy_phase",), "resources")
    options = spatial_newton_options(value.attrs["newton_controls"])
    order = (
        "tolerance",
        "max_iterations",
        "linear_tolerance",
        "linear_max_iterations",
        "restart",
        "armijo",
        "minimum_step",
    )
    controls = (
        "pops::FieldNewtonOptions{"
        + ", ".join(
            ".%s = %s"
            % (key, str(options[key]) if type(options[key]) is int else scalar_cpp(options[key]))
            for key in order
        )
        + "}"
    )
    result = stem + "_result"
    variables[value.id] = result
    if phase == "resources":
        prelude += [
            "auto %s = ctx.prepare_spatial_hierarchy(%d,%d,%s,%s);"
            % (
                workspace,
                value.id,
                owner,
                controls,
                scalar_cpp(spatial_scalar(value.attrs["finite_difference_step"])),
            ),
            "std::shared_ptr<pops::MultiFab<pops::kNativeDimension>> %s;" % trial,
            "std::shared_ptr<std::optional<pops::runtime::program::PreparedDiffusion<pops::kNativeDimension>>> %s;"
            % slot,
            "ctx.prepare_spatial_collectively([&] {",
            "  %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.scratch_state_like(ctx.state(%d)));"
            % (trial, owner),
            "  %s = std::make_shared<std::optional<pops::runtime::program::PreparedDiffusion<pops::kNativeDimension>>>();"
            % slot,
            "});",
        ]
        lines += [
            'throw std::logic_error("composite spatial stage requires the unique synchronized hierarchy barrier");',
            "auto& %s = ctx.scratch_state(%d,0,ctx.state(%d));" % (result, value.id, owner),
        ]
        # Residual local-transform emitters own persistent resource declarations too.
        # Discover them in the ordinary resource pass without executing a level solve.
        resource_vars = dict(variables)
        resource_vars[("spatial_hierarchy_phase",)] = "gather"
        emit_amr_spatial_solve(
            program,
            value,
            base,
            resource_vars,
            model,
            [],
            prelude,
            block_indices,
            field_plans,
            emit_node,
        )
        return
    stage = _solve_stage_fraction(value)
    lines.append("ctx.set_stage_time(%d,%d);" % (stage.numerator, stage.denominator))
    if phase == "gather":
        seed = "nullptr" if len(value.inputs) == 1 else "&(%s)" % variables[value.inputs[1].id]
        lines += [
            "ctx.stage_spatial_hierarchy_previous(%d,%d,%s,%s);"
            % (value.id, owner, variables[value.inputs[0].id], seed),
            "%s->emplace(ctx,*%s,%s,true);" % (slot, trial, _boundary_cpp(selected["physical"])),
        ]
        # Every scratch allocation happens before peers enter residual/JVP collectives.
        lines.append("ctx.prepare_spatial_collectively([&] {")
        for node in value.attrs["residual_block"]:
            if node.op in ("diffusive_rhs", "linear_combine", "source", "local_transform"):
                allocator = (
                    "rhs_scratch" if node.op in ("diffusive_rhs", "source") else "scratch_state"
                )
                lines.append("(void)ctx.%s(%d,0,*%s);" % (allocator, node.id, trial))
                if node.op == "diffusive_rhs":
                    for index, _ in enumerate(
                        (
                            row
                            for row in node.attrs["physical_balance"].occurrences
                            if row.kind == "source"
                        ),
                        1,
                    ):
                        lines.append("(void)ctx.rhs_scratch(%d,%d,*%s);" % (node.id, index, trial))
        lines.append("});")
        subvars = dict(variables)
        subvars[value.inputs[0].id] = "frozen_previous"
        subvars[value.attrs["iterate"].id] = stem + "_iterate"
        mapping_body = [
            "auto& ctx = *spatial_context;",
            "pops::PureFieldAlgebra::copy(*%s,q);" % trial,
            "auto& %s_iterate = *%s;" % (stem, trial),
        ]
        body = ["auto& ctx = *spatial_context;", "auto& %s_iterate = *%s;" % (stem, trial)]
        rate_seen = False
        for node in value.attrs["residual_block"]:
            if node is value.attrs["iterate"]:
                continue
            if node.op == "diffusive_rhs":
                mapping_body.append("return &(%s);" % subvars[node.inputs[0].id])
                subvars[node.inputs[0].id] = "current_conserved"
                rate_seen = True
                _emit_diffusive_rhs(
                    node,
                    subvars,
                    body,
                    model_for_node(model, node),
                    variables.get(("program_provider_plans",)),
                    owner,
                    "amr_system",
                    prepared_var=slot + "->value()",
                )
            else:
                emit_node(
                    node,
                    value.attrs["iterate"],
                    subvars,
                    body if rate_seen else mapping_body,
                    prelude,
                )
        body += [
            "pops::PureFieldAlgebra::copy(result,%s);" % subvars[value.attrs["residual"].id],
            "return &current_conserved;",
        ]
        lines += [
            "auto* spatial_context = &ctx;",
            "ctx.prepare_spatial_collectively([&] {",
            "  %s->mappings.at(ctx.level()) = [=](const pops::MultiFab<pops::kNativeDimension>& q) {"
            % workspace,
        ]
        lines += ["    " + line for line in mapping_body]
        lines += [
            "  };",
            "  %s->evaluations.at(ctx.level()) = [=](pops::MultiFab<pops::kNativeDimension>& current_conserved, const pops::MultiFab<pops::kNativeDimension>& frozen_previous, pops::MultiFab<pops::kNativeDimension>& result, const pops::MultiFab<pops::kNativeDimension>* parent) {"
            % workspace,
            "    return spatial_context->with_spatial_parent(%d,parent,[&]() {" % owner,
        ]
        lines += ["      " + line for line in body]
        lines += [
            "    });",
            "  };",
            "  %s->faces.at(ctx.level()) = &%s->value().faces();" % (workspace, slot),
        ]
        accepted = []
        _emit_diffusive_accepted(
            rate,
            slot + "->value()",
            accepted,
            _coeff_cpp(spatial_rate_weight(value, rate)),
            "implicit-stage:" + str(value.point) + "/solve:" + str(value.id),
            owner,
        )
        accepted = [
            line[:-2] + ",true);" if ".stage_accepted_exchanges(" in line else line
            for line in accepted
        ]
        lines += [
            "  %s->accepted.at(ctx.level()) = [=]() { auto& ctx = *spatial_context;" % workspace
        ]
        lines += ["    " + line for line in accepted]
        lines += ["  };", "});"]
        coefficient = sum(row.coefficient for row in occurrences)
        lines.append(
            "%s->residual_face_weight = -(%s)*(%s);"
            % (workspace, _coeff_cpp(spatial_rate_weight(value, rate)), scalar_cpp(coefficient))
        )
        return
    if phase == "solve":
        report, outcome = stem + "_report", stem + "_outcome"
        action_kind, _ = _consumed_solve_action(program, value)
        action = (
            "pops::SolveAction::kRejectAttempt"
            if action_kind == "reject_attempt"
            else "pops::SolveAction::kFailRun"
        )
        lines += [
            "pops::SolveReport %s;" % report,
            "try {",
            "  ctx.reconcile_spatial_hierarchy_previous(%d);" % value.id,
            "  %s = ctx.solve_spatial_hierarchy(*%s);" % (report, workspace),
            "} catch (const pops::runtime::program::StepAttemptRejected& failure) {",
            "  %s.mark_failed(failure.status(),%s,failure.what());" % (report, action),
            "} catch (const pops::runtime::program::DiffusiveEvaluationError& failure) {",
            "  %s.mark_failed(pops::SolveStatus::kInvalidEvaluation,failure.status()==3 ? pops::SolveAction::kFailRun : %s,failure.what());"
            % (report, action),
            "}",
            "pops::SolveOutcome %s = pops::SolveOutcome::collective_lane(std::move(%s),ctx.prepared_execution_lane());"
            % (outcome, report),
        ]
        _append_solve_report_guard(
            program, value, outcome, lines, label="spatial_implicit", phase="solve"
        )
        return
    if phase != "publish":
        raise ValueError("unknown composite spatial lowering phase")
    lines += [
        "%s->accepted.at(ctx.level())();" % workspace,
        "auto& %s = ctx.scratch_state(%d,0,ctx.state(%d));" % (result, value.id, owner),
        "pops::PureFieldAlgebra::copy(%s,%s->candidate(ctx.level()));" % (result, workspace),
        "ctx.record_scalar(%s,static_cast<pops::Real>(%s->residual_evaluations()));"
        % (json.dumps(stem + ".full_residual_evaluations"), workspace),
        "ctx.record_scalar(%s,static_cast<pops::Real>(%s->derivative_evaluations()));"
        % (json.dumps(stem + ".finite_difference_jvps"), workspace),
    ]
