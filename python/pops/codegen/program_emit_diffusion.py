"""Program dispatch for the prepared physical diffusive face operator."""
from __future__ import annotations
import json
from pops.identity.scalar import scalar_cpp
from pops.identity import make_identity
from pops.codegen.program_emit_kernels import (
    _cell_locals, _has_runtime_param, _model_impl, _prepare_provider_values,
    program_provider_consumer_qid,
)
from pops.codegen.program_emit_model_kernels import _provider_binding, _emit_source_kernel


def _selected(v, node_model):
    view=v.attrs["physical_balance"]
    rows=tuple(row for row in view.occurrences if row.kind=="diffusion")
    if not rows or any(row.payload != rows[0].payload or row.coefficient<=0 for row in rows):
        raise ValueError("diffusive Program evaluation requires positive occurrences of one exact flux")
    impl=_model_impl(node_model)
    try:
        selected=impl._diffusive_laws[rows[0].payload.reg_name]
    except (AttributeError,KeyError):
        raise ValueError("diffusive Program has no authenticated native constitutive law") from None
    if v.attrs.get("fitted",False):
        drift=tuple(row for row in view.occurrences if row.kind=="drift")
        if len(drift)!=1 or drift[0].coefficient!=-1 or len(rows)!=1 or rows[0].coefficient!=1:
            raise ValueError("fitted flux must cover exactly one signed drift/diffusion occurrence pair")
        try:
            selected={**selected,"fitted":impl._drift_laws[drift[0].payload.reg_name]}
        except (AttributeError,KeyError):
            raise ValueError("fitted Program has no exact physical drift authority") from None
    return impl,selected,rows


def diffusive_flux_basis_count(v):
    """Return the exact number of native face bases one diffusive RHS publishes.

    Constitutive diffusion is materialized once even when its resolved balance contains
    repeated occurrences of the same law.  A selected default transport divergence owns one
    additional prepared-face basis; sources own none.  This mirrors the two attachment calls
    emitted by :func:`_emit_diffusive_rhs` and is the sole count used by AMR budget proofs.
    """
    if v.op != "diffusive_rhs":
        return 0
    view = v.attrs.get("physical_balance")
    occurrences = tuple(getattr(view, "occurrences", ()))
    diffusion = tuple(row for row in occurrences if row.kind == "diffusion")
    if not diffusion:
        raise ValueError("diffusive Program evaluation has no constitutive face occurrence")
    transport = tuple(row for row in occurrences if row.kind == "flux")
    if len(transport) > 1:
        raise ValueError("diffusive Program evaluation has several transport face providers")
    return 1 + len(transport)


def _law_expressions(selected):
    if "fitted" in selected:
        from pops._ir.expr import Const
        return selected["fitted"]["potential"],*selected["diagonal"],Const(0)
    return selected["variable"],*selected["diagonal"],selected["derivative"]


def _boundary_cpp(law):
    rows=[]
    for row in law.boundaries:
        slope=row.slope or (0,)*law.dimension
        rows.append("{pops::runtime::program::DiffusiveBoundaryKind::%s, %s, {%s}}" % (
            row.kind,scalar_cpp(row.value),", ".join(scalar_cpp(x) for x in slope)))
    return "std::array<pops::runtime::program::DiffusiveBoundary<pops::kNativeDimension>, 2*pops::kNativeDimension>{{%s}}" % ", ".join(rows)


def _emit_diffusive_preparation(v, state_var, prepared_var, node_model,
                                provider_plans=None, bidx=0, target="system"):
    if target not in {"system", "amr_system"}:
        raise ValueError("diffusion execution requires a Uniform or AMR native install scope")
    _,selected,_=_selected(v,node_model)
    lines = ["ctx.require_cartesian_generated_operator(%d, \"diffusive_face_evaluation\");" % bidx,
        "pops::runtime::program::PreparedDiffusion<pops::kNativeDimension> %s(ctx, %s, %s, %s);" % (
        prepared_var,state_var,_boundary_cpp(selected["physical"]),
        "true" if target == "amr_system" else "false")]
    if any(row.kind == "flux" for row in v.attrs["physical_balance"].occurrences):
        lines.append("std::vector<pops::nd::FaceField<pops::kNativeDimension>> %s_transport_faces;" % prepared_var)
    return lines


def _emit_diffusive_rhs(v, var, lines, node_model, provider_plans, bidx, target,
                        prepared_var=None):
    if target not in {"system", "amr_system"}:
        raise ValueError("diffusion execution requires a Uniform or AMR native install scope")
    if "evaluation_partition" in v.attrs:
        from pops.time._evaluation_point import evaluation_stage_fraction
        stage = evaluation_stage_fraction(v)
        lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
    impl,selected,rows=_selected(v,node_model)
    state_var=var[v.inputs[0].id]
    out="diffusive_rhs_%d" % v.id
    var[v.id]=out
    explicit = prepared_var is None
    if prepared_var is None:
        prepared_var="diffusion_prepared_%d" % v.id
        lines.extend(_emit_diffusive_preparation(v,state_var,prepared_var,node_model,
                                                 provider_plans,bidx,target))
    lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%d, 0, %s);" % (
        out,v.id,state_var))
    exprs=_law_expressions(selected)
    sources=tuple(row for row in v.attrs["physical_balance"].occurrences if row.kind=="source")
    roots=list(exprs)
    for row in sources:
        roots.extend(impl._source_terms[row.payload.reg_name])
    qid=program_provider_consumer_qid(node_model,v.id,v.block)
    binding=_provider_binding(impl,roots,provider_plans,qid)
    impl.assign_runtime_indices()
    if target == "amr_system":
        lines.append("ctx.prepare_generated_state(%d,%s,%d);" % (bidx,state_var,v.id))
    lines.extend(_prepare_provider_values(binding,bidx,state_var))
    if explicit:
        lines.append("try {")
    method="apply_fitted" if "fitted" in selected else "apply"
    lines.append("%s.%s(%s, %s, [&](std::size_t li) {" % (prepared_var,method,state_var,out))
    lines.append("  const auto %sA=std::as_const(%s).fab(li).view();" % (state_var,state_var))
    if binding["count"]:
        lines.append("  const auto providers=ctx.template provider_values_view<%d>(%s,%d,li);" % (
            binding["count"],json.dumps(binding["qid"]),bidx))
    if _has_runtime_param(roots):
        lines.append("  const auto params=ctx.program_params(%d);" % bidx)
    lines.append("  return [=] POPS_HD(const pops::Index<pops::kNativeDimension>& index) {")
    lines.extend("    "+line for line in _cell_locals(impl,exprs,state_var,with_cons=True,
                 with_prim=True,provider_binding=binding))
    from pops.codegen.native_constitutive import emit_native_constitutive
    endpoint = emit_native_constitutive(exprs)
    captured = tuple(v.attrs.get("native_functions", ()))
    if any(function not in captured for function in endpoint.functions):
        raise ValueError("diffusive endpoint native functions differ from the captured physical law")
    lines.extend(endpoint.lines)
    lines.append("    pops::runtime::program::DiffusiveLawResult<pops::kNativeDimension> result;")
    lines.append("    result.evaluation_status = %s; result.reason_code = %s;" % (
        endpoint.status, endpoint.reason))
    lines.append("    if (result.evaluation_status == 0) result.values = {%s};" % ", ".join(endpoint.values))
    lines.append("    return result;")
    lines.append("  };")
    if "fitted" in selected:
        ratio="(%s)/(%s)" % (selected["fitted"]["mobility"].to_cpp(),selected["diagonal"][0].to_cpp())
        lines.append("}, %s, %s);" % (ratio,_boundary_cpp(selected["fitted"]["physical"])))
    else:
        lines.append("});")
    if explicit:
        lines.extend(("} catch (const pops::runtime::program::DiffusiveEvaluationError& error) {",
            '  ctx.consume_pointwise_evaluation_status(%d,%d,error.status(),"diffusive_face_evaluation",error.reason());' % (bidx,v.id),
            "}"))
    coefficient=sum(row.coefficient for row in rows)
    if coefficient != 1:
        lines.append("pops::scale(%s,%s);" % (out,scalar_cpp(coefficient)))
    if target == "amr_system" and explicit:
        lines.append("ctx.attach_diffusive_flux_basis(%d,%s,%d,%s.faces(),%s);" % (
            bidx,out,v.id,prepared_var,scalar_cpp(coefficient)))
    transport = tuple(row for row in v.attrs["physical_balance"].occurrences if row.kind=="flux")
    if diffusive_flux_basis_count(v) != 1 + len(transport):
        raise AssertionError("diffusive face-basis count differs from its emitted operators")
    frequency = "%s*%s.explicit_frequency()" % (scalar_cpp(coefficient),prepared_var)
    if transport:
        if len(transport)!=1 or transport[0].coefficient!=-1 or not transport[0].payload.is_default:
            raise ValueError("diffusive lowering requires one exact default -div transport occurrence")
        temporary="diffusive_transport_%d" % v.id
        lines.append("auto& %s=ctx.rhs_scratch(%d,%d,%s);" % (temporary,v.id,len(sources)+1,state_var))
        # A native flux has its own exact read union, including derived providers that
        # a preceding field publication deliberately leaves dirty until consumption.
        # Do not inherit unrelated model inputs into the constitutive cell kernel.
        transport_pack = impl._component_operator_provider_packs[transport[0].payload.reg_name]
        transport_binding = provider_plans.bind_pack(transport_pack, qid+"/transport")
        lines.extend(_prepare_provider_values(transport_binding, bidx, state_var))
        lines.append("ctx.neg_div_flux_default_with_faces_into(%d,%s,%s,%d,%s_transport_faces);" % (
            bidx,state_var,temporary,v.id,prepared_var))
        # This combines spatial rates. Its unit coefficient has dt power zero; the
        # authored time update supplies the sole dt factor, including in AMR subcycles.
        lines.append("ctx.axpy(%s,1,%s,dt,{{0, 1, 1}});" % (out,temporary))
        inverse_spacing=" + ".join("1/ctx.geometry().spacing(%d)" % axis for axis in range(selected["physical"].dimension))
        frequency+=" + ctx.max_wave_speed(%d,%s)*(%s)" % (bidx,state_var,inverse_spacing)
    if explicit:
        lines.append("const pops::Real diffusion_frequency_%d=%s;" % (v.id,frequency))
        lines.append("if (!(std::isfinite(dt) && dt>=0 && dt*diffusion_frequency_%d<=1+32*std::numeric_limits<pops::Real>::epsilon()))" % v.id)
        lines.append('  ctx.consume_pointwise_evaluation_status(%d,%d,2,"combined_transport_diffusion_stability",502);' % (bidx,v.id))
    for ordinal,row in enumerate(sources,1):
        temporary="diffusive_source_%d_%d" % (v.id,ordinal)
        lines.append("auto& %s=ctx.rhs_scratch(%d,%d,%s);" % (temporary,v.id,ordinal,state_var))
        lines.extend(_emit_source_kernel(node_model,row.payload.reg_name,state_var,temporary,bidx,
            provider_plans=provider_plans,consumer_qid=qid,plan_exprs=roots))
        lines.append("ctx.axpy(%s,%s,%s);" % (out,scalar_cpp(row.coefficient),temporary))
    return prepared_var


def _emit_diffusive_accepted(v, prepared_var, lines, temporal_weight_cpp, evaluation_context,
                             program_block=0):
    """Publish only the selected accepted quadrature; no residual or seed contributes."""
    view = v.attrs["physical_balance"]
    handle = view.balance.handle
    source = handle if handle.is_resolved else handle._resolved(handle.owner_path.canonical())
    block = v.block if v.block.is_resolved else v.block._resolved(v.block.owner_path.canonical())
    operation_data = {"source": source.canonical_identity(), "block": block.canonical_identity()}
    operation = make_identity("diffusive-operation", operation_data).token
    from pops.codegen.program_emit_transport_exchanges import emit_transport_exchanges
    for row in view.occurrences:
        if row.kind == "flux":
            occurrence = operation+"/occurrence:"+str(row.identity[1])
            lines.extend(emit_transport_exchanges(
                prepared_var+"_transport_faces", operation, occurrence, evaluation_context,
                temporal_weight_cpp, program_block=program_block,
                active_field=prepared_var+".prototype()"))
    if v.attrs.get("fitted",False):
        ordinals=tuple(row.ordinal for row in view.occurrences if row.kind in {"drift","diffusion"})
        occurrence=operation+"/joint-occurrences:"+",".join(map(str,ordinals))
        lines.append("%s.stage_accepted_exchanges(ctx,%d,%s,%s,%s,%s);" % (
            prepared_var,program_block,json.dumps(operation),json.dumps(occurrence),json.dumps(evaluation_context),temporal_weight_cpp))
        return
    for row in view.occurrences:
        if row.kind != "diffusion":
            continue
        occurrence = operation+"/occurrence:"+str(row.identity[1])
        lines.append("%s.stage_accepted_exchanges(ctx,%d,%s,%s,%s,(%s)*%s);" % (
            prepared_var,program_block,json.dumps(operation),json.dumps(occurrence),
            json.dumps(evaluation_context),temporal_weight_cpp,scalar_cpp(row.coefficient)))
