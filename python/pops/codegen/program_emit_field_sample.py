"""Copy one authenticated cell provider/State component to explicitly bound scalar storage."""
from pops.codegen.program_emit_kernels import _prepare_provider_values, program_provider_consumer_qid
from pops.codegen.program_field_observations import emitted_sample_contract


def emit_field_sample(value, var, lines, prelude, model, provider_plans, block_index, target,
                      *, model_graph, field_plans=None):
    if target not in ("system", "amr_system") or prelude is None:
        raise NotImplementedError("typed FieldSample requires its exact prepared native Program route")
    template = value.inputs[0]
    selected_pack = emitted_sample_contract(value, model_graph, field_plans=field_plans, target=target)
    pointer = "sample%d" % value.id
    if target == "system":
        prelude.append("auto %s = ctx.alloc_shared_scalar_field_for_block(%d, 1, 1);"
                       % (pointer, block_index))
    else:
        lines.append("pops::MultiFab<pops::kNativeDimension>* %s = nullptr;" % pointer)
        lines.append("ctx.prepare_spatial_collectively([&] { %s = &ctx.scalar_scratch(%d, 0, %s, 1, 1); });" % (
            pointer, value.id, var[template.id]))
    var[value.id] = "(*%s)" % pointer
    var[("field_pointer", value.id)] = pointer
    binding = None
    if value.attrs["sample_kind"] in ("aux", "field"):
        if provider_plans is None:
            raise ValueError("FieldSample requires its exact provider plan")
        binding = provider_plans.bind_pack(selected_pack,
                    program_provider_consumer_qid(model, value.id, value.block))
    from pops.time._evaluation_point import evaluation_stage_fraction
    stage = evaluation_stage_fraction(value)
    lines.extend(["{", "auto sample_stage_%d = ctx.stage_evaluation_scope(%d, %d);"
                  % (value.id, stage.numerator, stage.denominator)])
    lines.append("ctx.require_scalar_sampling_layout(%d, %s, *%s, %d);" % (
        block_index, var[template.id], pointer,
        value.attrs.get("component", 0) if value.attrs["sample_kind"] == "state" else 0))
    if target == "amr_system" and value.attrs["sample_kind"] == "field":
        from pops.codegen.program_emit_field_routes import resolved_field_route
        from pops.codegen.cpp_strings import cpp_string_expression
        slot, plan = resolved_field_route(value.inputs[1].field_context.field, field_plans)
        if value.attrs["component"] != plan.native_options["output_route"]["components"][0]:
            raise NotImplementedError("AMR constrained field samples require the solved scalar potential")
        lines.append("ctx.copy_solved_field_scalar_sample(%s, %d, %s, *%s, %d);" % (
            cpp_string_expression(slot), block_index, var[template.id], pointer, value.id))
        lines.append("}")
        return
    lines.extend(_prepare_provider_values(binding, block_index, var[template.id]))
    lines.extend(["for (std::size_t li = 0; li < %s->local_size(); ++li) {" % pointer,
                  "  const auto output = %s->fab(li).view();" % pointer])
    if binding is not None:
        from pops.codegen.cpp_strings import cpp_string_expression
        lines.append("  const auto providers = ctx.template provider_values_view<1>(%s, %d, li);"
                     % (cpp_string_expression(binding["qid"]), block_index))
        rhs = "providers(index, 0)"
    else:
        lines.append("  const auto input = std::as_const(%s).fab(li).view();" % var[template.id])
        rhs = "input(index, %d)" % value.attrs["component"]
    lines.extend(["  pops::for_each_cell(%s->box(li), [=] POPS_HD(" % pointer,
                  "      const pops::Index<pops::kNativeDimension>& index) {",
                  "    output(index, 0) = %s;" % rhs, "  });", "}", "}"])


__all__ = ["emit_field_sample"]
