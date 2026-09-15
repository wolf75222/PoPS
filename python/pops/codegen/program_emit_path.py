"""Stage a complete path RHS before the synchronized hierarchy continuation."""
import json


def emit_path_rhs(value, var, lines, model, provider_plans, block, target):
    if target != "amr_system":
        raise NotImplementedError("path-conservative transport requires synchronous AMR execution")
    from pops.codegen.program_emit_kernels import _model_impl
    from pops.codegen.program_emit_ops import _rhs_flux_temporal_family
    from pops.codegen.program_emit_kernels import prepare_default_rhs_providers
    from pops.codegen.program_rhs_input_trace import emit_rhs_input_trace
    from pops.time._evaluation_point import evaluation_stage_fraction

    if getattr(_model_impl(model), "_path_conservative", None) is None:
        raise ValueError("path RHS has no authenticated complete native Fan–Li15 model")
    if (not value.attrs.get("flux") or value.attrs.get("fluxes")
            or tuple(value.attrs.get("sources", ())) or value.attrs.get("schedule") is not None):
        raise ValueError("path RHS must retain one complete unscheduled flux/product balance")
    state = value.inputs[0]
    var[value.id] = "r%d" % value.id
    lines.append("auto& %s = ctx.rhs_scratch(%d, 0, %s);" %
                 (var[value.id], value.id, var[state.id]))
    stage = evaluation_stage_fraction(value, ark_partition="explicit")
    lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
    lines += prepare_default_rhs_providers(model, value, block, var[state.id], provider_plans,
                                          target=target, flux=True, source=False)
    trace = emit_rhs_input_trace(value, block, var[state.id], lines, target, force=True)
    lines.append("ctx.stage_path_rhs(%d, %s, %s, %d, %s, %s, ctx.path_rhs_courant());" %
                 (block, var[state.id], var[value.id], value.id,
                  json.dumps(_rhs_flux_temporal_family(value)), trace))
