"""Native pointwise cell representation of a sealed original accumulation."""

from __future__ import annotations


def emit_evolved_state(value, variables, lines, prelude, *, target, block_indices):
    from pops.fields._evolved_stage_contract import validate_evolved_state
    from pops.fields._program_expression import field_expression_cpp
    from pops.codegen.program_emit_field_problem import guard_candidate_allocations
    from pops.model import Handle
    from pops.time._program.serialization import _json_ready

    solve = validate_evolved_state(value)
    if target not in ("system", "amr_system") or prelude is None:
        raise NotImplementedError(
            "original accumulation needs a declared native spatial realization"
        )
    owner = block_indices.get(value.block)
    if owner is None:
        raise ValueError("original accumulation has no target physical layout owner")
    stem = "evolved_field_%d" % value.id
    packed = variables[value.inputs[0].id]
    captures = value.inputs[1:]
    previous = variables[captures[value.attrs["previous_capture_index"]].id]
    if target == "system":
        rows = [
            "auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.scratch_state_like(ctx.state(%d)));"
            % (stem, owner)
        ]
        prelude.extend(guard_candidate_allocations(rows))
        output = "(*%s)" % stem
    else:
        lines += [
            "pops::MultiFab<pops::kNativeDimension>* %s = nullptr;" % stem,
            "ctx.prepare_spatial_collectively([&] { %s = &ctx.scratch_state(%d, 0, ctx.state(%d)); });"
            % (stem, value.id, owner),
        ]
        output = "(*%s)" % stem
    variables[value.id] = output
    publication_start = len(lines)
    lines += [
        "{",
        "const auto& evolved_candidate = %s;" % packed,
        "const auto& evolved_previous = %s;" % previous,
        "auto& evolved_output = %s;" % output,
        "long evolved_layout_invalid = (evolved_candidate.layout() != evolved_output.layout() || evolved_candidate.distribution() != evolved_output.distribution() || evolved_candidate.local_rank() != evolved_output.local_rank() || evolved_candidate.local_size() != evolved_output.local_size() || evolved_candidate.ncomp() != %d || evolved_previous.ncomp() != %d) ? 1L : 0L;"
        % (solve.attrs["ncomp"], value.attrs["ncomp"]),
    ]
    for capture in captures:
        reference = variables[capture.id]
        lines.append(
            "evolved_layout_invalid |= (%s.layout() != evolved_output.layout() || %s.distribution() != evolved_output.distribution() || %s.local_rank() != evolved_output.local_rank() || %s.local_size() != evolved_output.local_size()) ? 1L : 0L;"
            % (reference, reference, reference, reference)
        )
    lines += [
        "if (pops::all_reduce_max(evolved_layout_invalid, ctx.prepared_execution_lane()) != 0)",
        '  throw std::invalid_argument("original accumulation layout/width changed");',
        "const auto* evolved_active = ctx.pointwise_active_mask(%d, evolved_output);" % owner,
        "pops::Real evolved_invalid = 0;",
        "std::exception_ptr evolved_error;",
        "try {",
        "pops::PureFieldAlgebra::copy(evolved_output, evolved_previous);",
        "for (std::size_t patch = 0; patch < evolved_output.local_size(); ++patch) {",
        "  const auto candidate = evolved_candidate.fab(patch).view();",
        "  const auto output = evolved_output.fab(patch).view();",
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> active = evolved_active ? std::as_const(*evolved_active).fab(patch).view() : pops::FieldView<const pops::Real, pops::kNativeDimension>{};",
        "  const bool masked = evolved_active != nullptr;",
        "  evolved_invalid = std::max(evolved_invalid, pops::for_each_cell_reduce_max(evolved_output.box(patch),",
        "    [=] POPS_HD(const pops::CellIndex<pops::kNativeDimension>& index) {",
        "      if (masked && active(index, 0) <= 0) return pops::Real(0);",
        "      bool finite = true;",
    ]
    unknowns = tuple(
        Handle.from_canonical_identity(_json_ready(row))
        for row in solve.attrs["source_contract"]["unknown_components"]
    )
    # Add co-located frozen capture views to the same pointwise publication kernel.
    offset = lines.index("  const auto output = evolved_output.fab(patch).view();", publication_start) + 1
    for index, capture in enumerate(captures):
        lines.insert(
            offset + index,
            "  const auto capture%d = %s.fab(patch).view();" % (index, variables[capture.id]),
        )
    for component, expression in enumerate(value.attrs["expressions"]):
        code, _ = field_expression_cpp(
            expression,
            captures,
            views=tuple("capture%d" % i for i in range(len(captures))),
            unknowns=unknowns,
        )
        lines += [
            "      const pops::Real q%d = %s;" % (component, code),
            "      finite = finite && std::isfinite(q%d);" % component,
            "      output(index, %d) = q%d;" % (component, component),
        ]
    lines += [
        "      return finite ? pops::Real(0) : pops::Real(1);",
        "    }));",
        "}",
        "} catch (...) { evolved_error = std::current_exception(); }",
        "try { Kokkos::fence(); } catch (...) { if (!evolved_error) evolved_error = std::current_exception(); }",
        'pops::collectively_rethrow_exception(evolved_error, ctx.prepared_execution_lane(), "original accumulation local evaluation");',
        "if (pops::all_reduce_max(evolved_invalid, ctx.prepared_execution_lane()) != 0)",
        '  throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::kInvalidEvaluation, "original_accumulation", "nonfinite_original_accumulation");',
        "}",
    ]
