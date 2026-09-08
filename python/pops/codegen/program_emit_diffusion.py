"""Program dispatch for the prepared physical diffusive face operator."""
from __future__ import annotations
import json
from pops.identity.scalar import scalar_cpp
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
    return impl,selected,rows


def _boundary_cpp(law):
    rows=[]
    for row in law.boundaries:
        slope=row.slope or (0,)*law.dimension
        rows.append("{pops::runtime::program::DiffusiveBoundaryKind::%s, %s, {%s}}" % (
            row.kind,scalar_cpp(row.value),", ".join(scalar_cpp(x) for x in slope)))
    return "std::array<pops::runtime::program::DiffusiveBoundary<pops::kNativeDimension>, 2*pops::kNativeDimension>{{%s}}" % ", ".join(rows)


def _emit_diffusive_preparation(v, state_var, prepared_var, node_model,
                                provider_plans=None, bidx=0, target="system"):
    if target != "system":
        raise ValueError("AMR diffusion execution is unavailable before composite face/exchange integration")
    _,selected,_=_selected(v,node_model)
    return ["pops::runtime::program::PreparedDiffusion<pops::kNativeDimension> %s(ctx, %s, %s);" % (
        prepared_var,state_var,_boundary_cpp(selected["physical"]))]


def _emit_diffusive_rhs(v, var, lines, node_model, provider_plans, bidx, target,
                        prepared_var=None):
    if target != "system":
        raise ValueError("AMR diffusion execution is unavailable before composite face/exchange integration")
    impl,selected,rows=_selected(v,node_model)
    state_var=var[v.inputs[0].id]
    out="diffusive_rhs_%d" % v.id
    var[v.id]=out
    if prepared_var is None:
        prepared_var="diffusion_prepared_%d" % v.id
        lines.extend(_emit_diffusive_preparation(v,state_var,prepared_var,node_model,
                                                 provider_plans,bidx,target))
    lines.append("pops::MultiFab<pops::kNativeDimension>& %s = ctx.rhs_scratch(%d, 0, %s);" % (
        out,v.id,state_var))
    exprs=(selected["variable"],*selected["diagonal"],selected["derivative"])
    sources=tuple(row for row in v.attrs["physical_balance"].occurrences if row.kind=="source")
    roots=list(exprs)
    for row in sources:
        roots.extend(impl._source_terms[row.payload.reg_name])
    qid=program_provider_consumer_qid(node_model,v.id,v.block)
    binding=_provider_binding(impl,roots,provider_plans,qid)
    impl.assign_runtime_indices()
    lines.extend(_prepare_provider_values(binding,bidx,state_var))
    lines.append("%s.apply(%s, %s, [&](std::size_t li) {" % (prepared_var,state_var,out))
    lines.append("  const auto %sA=std::as_const(%s).fab(li).view();" % (state_var,state_var))
    if binding["count"]:
        lines.append("  const auto providers=ctx.template provider_values_view<%d>(%s,%d,li);" % (
            binding["count"],json.dumps(binding["qid"]),bidx))
    if _has_runtime_param(roots):
        lines.append("  const auto params=ctx.program_params(%d);" % bidx)
    lines.append("  return [=] POPS_HD(const pops::Index<pops::kNativeDimension>& index) {")
    lines.extend("    "+line for line in _cell_locals(impl,exprs,state_var,with_cons=True,
                 with_prim=True,provider_binding=binding))
    lines.append("    return std::array<pops::Real,pops::kNativeDimension+2>{%s};" %
                 ", ".join(expr.to_cpp() for expr in exprs))
    lines.extend(("  };", "});"))
    coefficient=sum(row.coefficient for row in rows)
    if coefficient != 1:
        lines.append("pops::scale(%s,%s);" % (out,scalar_cpp(coefficient)))
    for ordinal,row in enumerate(sources,1):
        temporary="diffusive_source_%d_%d" % (v.id,ordinal)
        lines.append("auto& %s=ctx.rhs_scratch(%d,%d,%s);" % (temporary,v.id,ordinal,state_var))
        lines.extend(_emit_source_kernel(node_model,row.payload.reg_name,state_var,temporary,bidx,
            provider_plans=provider_plans,consumer_qid=qid,plan_exprs=roots))
        lines.append("ctx.axpy(%s,%s,%s);" % (out,scalar_cpp(row.coefficient),temporary))
    return prepared_var
