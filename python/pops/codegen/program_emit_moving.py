"""Coupled SSA lowering through the owned native interval producer.

The realization consumes original principal/source bodies. It does not evaluate
an immutable-grid residual and reinterpret that residual as a moving flux.
"""
from __future__ import annotations

import json

from pops.identity.scalar import scalar_cpp
from ._compiled_parameter import _decode_scalar, _thaw_json


def moving_updates(program):
    from .program_lowerability import all_ops
    return tuple(value for value in all_ops(program) if value.op == "reynolds_update")


def deferred_moving_rates(program):
    updates = moving_updates(program)
    rates=set()
    def collect(rate):
        rates.add(rate.id)
        for item in rate.inputs:
            if item.vtype=="rhs": collect(item)
    for value in updates:
        for rate in value.inputs[1:]: collect(rate)
    from .program_lowerability import all_ops
    for value in all_ops(program):
        if value.op != "reynolds_update" and value.id not in rates and any(item.id in rates for item in value.inputs):
            raise NotImplementedError("a moving face/source evaluation shared with another consumer "
                                      "needs a prepared interval projection provider")
    return frozenset(rates)


def _entry(value, authority):
    from .program_models import model_for_node
    from .principal_lowering import principal_for_value
    physical = value.inputs[1]
    if physical.op not in {"principal_rate", "rhs"}:
        raise NotImplementedError("moving physical transport needs an authenticated principal "
                                  "FiniteVolume provider; a legacy rhs is not a relative face flux")
    model=model_for_node(authority,physical)
    if physical.op == "principal_rate":
        entry = principal_for_value(model, physical)
    else:
        from pops.numerics.principal import PrincipalGroup
        from pops.numerics.spatial import FiniteVolume
        from .program_models import ProgramModelGraph
        if type(authority) is not ProgramModelGraph:
            raise ValueError("moving single-state transport requires resolved numerical authority")
        rows=[row for row in authority.numerics_for_block(physical.block).rates
              if row.rate.registered_operator_name==physical.attrs["operator_handle"].registered_operator_name]
        if len(rows)!=1 or type(rows[0].method) is not FiniteVolume:
            raise NotImplementedError("moving physical rate needs its declared FiniteVolume realization")
        if not physical.attrs.get("flux") or physical.attrs.get("sources"):
            raise ValueError("moving physical rate must contain only its original conservative flux")
        from pops.time.references import canonical_handle
        state=canonical_handle(physical.inputs[0].attrs["state"])
        group=PrincipalGroup((state,),(canonical_handle(rows[0].rate),),(rows[0].method,),1)
        module=authority.source_module_for_owner(physical.block.model_owner_path)
        view=module.operator_registry().get(rows[0].rate.registered_operator_name).lowering.get("physical_balance")
        if view is None or not view.accumulation.is_identity or len(view.occurrences)!=1 or \
                view.occurrences[0].kind!="flux" or view.occurrences[0].coefficient!=-1:
            raise ValueError("moving transport requires the original identity-accumulation -div(flux) Equation")
        if module.primitive_coordinates():
            raise NotImplementedError("single-state moving physical coordinates need a prepared conversion provider")
        from pops._ir import Var
        from pops._ir.application import substitute_quantities
        from pops._ir.primitive_expansion import expand_primitive_recipes
        from pops._ir.visitors import _children
        impl=model._m
        def native(body):
            expanded=expand_primitive_recipes(body,impl.prim_defs)
            pending=list(expanded.values()) if isinstance(expanded,dict) else list(expanded)
            replacements={}; seen=set()
            while pending:
                node=pending.pop()
                if isinstance(node,(tuple,list)): pending.extend(node); continue
                if id(node) in seen: continue
                seen.add(id(node))
                if isinstance(node,Var):
                    if node.kind!="cons" or node.name not in impl.cons_names:
                        raise NotImplementedError("moving physical flux needs an interval-qualified provider for %s"%node.name)
                    replacements[id(node)]=Var("pops_principal_%d"%impl.cons_names.index(node.name),"cons")
                pending.extend(_children(node))
            state_handle=module.state_handle((state.declaration_ref or state).space)
            quantities={(state_handle,i):Var("pops_principal_%d"%i,"cons")
                        for i in range(group.component_count)}
            return substitute_quantities(expanded,quantities,expression_bindings=replacements)
        body=module.operator_registry().get(view.occurrences[0].payload.reg_name).body
        if tuple(body) != ("x",):
            raise NotImplementedError("moving transport needs the prepared 1D physical face provider")
        if not getattr(impl,"_eig",None):
            raise ValueError("moving transport requires its authored physical wave-speed bound")
        entry={"group":group,"fluxes":(native(body),),"waves":native(impl._eig),
            "axes":tuple(body),"conversion":None,
            "cpp_name":"PoPSPrincipal_"+group.identity.token.split(":")[-1][:24],
            "parameter_indices":tuple((node.name,index) for index,node in enumerate(impl.assign_runtime_indices()))}
    group = entry["group"]
    if group.dimension != 1 or len(group.states) != 1:
        raise NotImplementedError("this moving face provider realizes one complete 1D state; "
                                  "joint-state/higher-dimensional geometry providers are not installed")
    method = group.methods[0]
    if method.reconstruction.name != "firstorder":
        raise NotImplementedError("the selected reconstruction needs a moving stencil metric provider")
    if method.riemann.native_id != "pops::RusanovFlux":
        raise NotImplementedError("the selected numerical flux needs a moving FaceContext provider")
    return entry


def _law(value):
    from pops.analytic import ScalarExpr
    from ._analytic_expression_lowering import lower_analytic_components
    evolution = value.inputs[0].attrs["evolution"]
    expressions = tuple(ScalarExpr.from_data(_thaw_json(item)) for item in evolution["coordinate_map"])
    if len(expressions) != 1 or any(expr.input_references() for expr in expressions):
        raise NotImplementedError("moving coordinates need a 1D analytic law; discrete inputs "
                                  "need an interval-qualified coordinate provider")
    ((operations, literals),) = lower_analytic_components(
        tuple(expr.to_data() for expr in expressions), frame_id=evolution["frame_id"],
        time_clock_id=value.clock.qualified_id)
    # This realization initializes physical state/geometry on the reference
    # mesh. Establish identity at the initial time by exact symbolic zero/one
    # elimination; other initial maps need a moving initial-projection provider.
    initial=[]
    for op,literal in zip(operations,literals,strict=True):
        if op=="constant": result=float(literal)
        elif op=="x": result=("reference",)
        elif op=="input": result=0.
        elif op in {"add","sub","mul","div"}:
            right,left=initial.pop(),initial.pop()
            if op=="add": result=left if right==0 else right if left==0 else (op,left,right)
            elif op=="sub": result=left if right==0 else 0. if left==right else (op,left,right)
            elif op=="mul": result=0. if left==0 or right==0 else left if right==1 else right if left==1 else (op,left,right)
            else: result=left if right==1 else (op,left,right)
        elif op in {"where","between"}:
            third,second,first=initial.pop(),initial.pop(),initial.pop(); result=(op,first,second,third)
        elif op in {"pow","atan2","hypot","minimum","maximum","eq","ne","lt","le","gt","ge","and","or"}:
            right,left=initial.pop(),initial.pop(); result=(op,left,right)
        else: result=(op,initial.pop())
        initial.append(result)
    if initial != [("reference",)]:
        raise NotImplementedError("the initial coordinate map needs a prepared physical-state "
                                  "projection provider; this provider requires identity at time zero")
    stack, lines = [], []
    binary = {"add":"+", "sub":"-", "mul":"*", "div":"/", "eq":"==", "ne":"!=",
              "lt":"<", "le":"<=", "gt":">", "ge":">=", "and":"&&", "or":"||"}
    unary = {name:name for name in ("sqrt","abs","sin","cos","exp","log","erf","erfc")}
    pairs = {"pow":"pow", "atan2":"atan2", "hypot":"hypot", "minimum":"fmin", "maximum":"fmax"}
    for ordinal,(op,literal) in enumerate(zip(operations,literals,strict=True)):
        if op == "constant": expr = "pops::Real(%s)" % float(literal).hex()
        elif op == "x": expr = "reference"
        elif op == "input": expr = "physical_time"
        elif op in binary:
            right,left=stack.pop(),stack.pop(); expr="(%s %s %s)"%(left,binary[op],right)
        elif op in pairs:
            right,left=stack.pop(),stack.pop(); expr="Kokkos::%s(%s,%s)"%(pairs[op],left,right)
        elif op in unary: expr="Kokkos::%s(%s)"%(unary[op],stack.pop())
        elif op in {"neg","not"}: expr="(%s%s)"%("-" if op=="neg" else "!",stack.pop())
        elif op == "where":
            no,yes,predicate=stack.pop(),stack.pop(),stack.pop(); expr="(%s?%s:%s)"%(predicate,yes,no)
        elif op == "between":
            upper,lower,item=stack.pop(),stack.pop(),stack.pop(); expr="(%s<=%s&&%s<=%s)"%(lower,item,item,upper)
        else: raise NotImplementedError("moving analytic opcode %r has no device provider"%op)
        name="coordinate_%d"%ordinal
        lines.append("    const pops::Real %s=%s;"%(name,expr)); stack.append(name)
    if len(stack)!=1: raise ValueError("moving coordinate law has an invalid postfix stack")
    lines.append("    return %s;"%stack[0])
    return lines


def _source(value, authority):
    count=len(value.space.components)
    if len(value.inputs)==2: return [], ["pops::Real(0)"]*count, "false"
    source=value.inputs[2]
    if source.op=="linear_combine" and source.attrs.get("physical_balance") is not None:
        terms=[]
        for item,coefficient in zip(source.inputs,source.attrs["coeffs"],strict=True):
            if any(power!=0 for power in coefficient):
                raise ValueError("moving physical source may not contain an extra duration weight")
            terms.append((item,coefficient.get(0,0)))
    else: terms=[(source,1)]
    if any(item.op!="source" for item,weight in terms):
        raise NotImplementedError("moving source projection needs an authenticated named source "
                                  "body; general RHS/provider projection is not installed")
    from .program_models import model_for_node
    from .program_emit_kernels import _model_impl
    from pops._ir.primitive_expansion import expand_primitive_recipes
    from pops._ir.visitors import _children
    from pops._ir.expr import Var
    from .cpp_writer import _cse_emit
    impl=_model_impl(model_for_node(authority,source)); impl.assign_runtime_indices()
    roots=None
    for item,weight in terms:
        body=tuple(expand_primitive_recipes(impl._source_terms[item.attrs["source"]],impl.prim_defs))
        roots=tuple(weight*expr for expr in body) if roots is None else tuple(
            old+weight*expr for old,expr in zip(roots,body,strict=True))
    pending=list(roots); seen=set()
    while pending:
        node=pending.pop()
        if id(node) in seen: continue
        seen.add(id(node))
        if isinstance(node,Var) and (node.kind!="cons" or node.name not in impl.cons_names):
            raise NotImplementedError("moving source requires an interval-qualified provider "
                                      "for coordinate %s"%node.name)
        pending.extend(_children(node))
    bindings={name:"u[%d]"%i for i,name in enumerate(impl.cons_names)}
    lines,values,observed=_cse_emit(roots,"pops::Real","    ",materialize_all=True,
                                    return_names=True,scalar_bindings=bindings)
    if len(values)!=count: raise ValueError("moving source body does not cover its exact StateSpace")
    invalid=" || ".join("!Kokkos::isfinite(%s)"%item for item in observed) or "false"
    return lines,values,invalid


def check_moving_program(program, authority, target):
    updates=moving_updates(program)
    if not updates: return
    if target != "system":
        raise NotImplementedError("moving AMR needs a geometry transfer/reflux connector; "
                                  "the Uniform1D provider cannot certify AMR motion")
    deferred_moving_rates(program)
    for value in updates:
        if program._commits.get(value.state_ref) is not value:
            raise ValueError("each moving interval candidate must have one terminal coupled commit")
        _entry(value,authority); _law(value); _source(value,authority)


def emit_moving_helpers(program, authority):
    lines=[]
    emitted=set()
    for value in moving_updates(program):
        entry=_entry(value,authority); name="PoPSMoving_%d"%value.id; base=entry["cpp_name"]
        if value.inputs[1].op=="rhs" and base not in emitted:
            from .program_emit_principal import emit_principal_model
            lines.append(emit_principal_model(entry)); emitted.add(base)
        from .program_emit_principal import _checked
        wave_lines,wave_values,wave_invalid=_checked(entry["waves"]["x"],entry["group"].component_count,indent="    ")
        lines += ["struct %sLaw {"%name,
            "  POPS_HD pops::Real operator()(pops::Real reference,pops::Real physical_time) const {",
            *_law(value),"  }","};",
            "struct %sRelative : %s {"%(name,base),"  pops::Real mesh_speed{};",
            "  template<int Axis> POPS_HD State flux(const State& u) const {",
            "    auto result=%s::template flux<Axis>(u);"%base,
            "    for(int c=0;c<n_vars;++c) result[c]-=mesh_speed*u[c]; return result; }",
            "  template<int Axis> POPS_HD pops::Real max_wave_speed(const State& u) const {",
            "    static_assert(Axis==0); const auto params=parameter_sets[0]; pops::Real result=0;",
            *wave_lines,
            "    if(%s) return std::numeric_limits<pops::Real>::quiet_NaN();"%wave_invalid,
            *("    result=Kokkos::max(result,Kokkos::abs((%s)-mesh_speed));"%wave for wave in wave_values),
            "    return result; }",
            "  template<int Axis> POPS_HD void wave_speeds(const State& u,pops::Real& lo,pops::Real& hi) const {",
            "    %s::template wave_speeds<Axis>(u,lo,hi); lo-=mesh_speed; hi-=mesh_speed; }"%base,"};"]
    return "\n".join(lines)+("\n" if lines else "")


def emit_moving_op(program,value,var,lines,prelude,authority,blocks):
    if value.op=="geometry_state":
        updates=[item for item in moving_updates(program) if item.inputs[0] is value]
        if len(updates)!=1: raise ValueError("geometry binding requires one complete interval realization")
        from pops.time.references import canonical_handle
        identity="pops.moving:"+canonical_handle(value.state_ref).qualified_id
        var[value.id]=json.dumps(identity)
        tolerance=scalar_cpp(_decode_scalar(updates[0].attrs["geometry_tolerance"]))
        prelude.append("ctx.initialize_moving_interval_geometry(%s,%d,%s,%s);"%
            (var[value.id],blocks[value.block],json.dumps(value.space.frame),tolerance))
        name="PoPSMoving_%d"%updates[0].id
        prelude += ["{", "  std::exception_ptr initial_geometry_error; pops::Real initial_geometry_invalid=0;",
            "  try {", "    const auto& moving=ctx.moving_interval_geometry(%s);"%var[value.id],
            "    const auto reference=ctx.geometry(); const auto time=ctx.physical_time(); %sLaw law;"%name,
            "    for(std::size_t patch=0;patch<moving.measures.local_size();++patch) {",
            "      const auto box=moving.measures.box(patch); const auto position=moving.coordinates[patch].template field<0>().view();",
            "      initial_geometry_invalid=Kokkos::max(initial_geometry_invalid,pops::for_each_cell_reduce_max(pops::nd::face_box(box,0),",
            "        [=] POPS_HD(const pops::Index<pops::kNativeDimension>& face) {",
            "          const auto mapped=law(reference.face_coordinate(0,face[0]),time);",
            "          return !Kokkos::isfinite(mapped)||Kokkos::abs(mapped-position(face))>%s?pops::Real(1):pops::Real(0); }));"%tolerance,
            "    } Kokkos::fence();", "  } catch(...) { initial_geometry_error=std::current_exception(); }",
            "  pops::collectively_rethrow_exception(initial_geometry_error,ctx.prepared_execution_lane(),\"moving initial projection failed collectively\");",
            "  if(pops::all_reduce_max(double(initial_geometry_invalid),ctx.prepared_execution_lane())!=0)",
            "    throw std::invalid_argument(\"moving initial coordinate law differs from reference state projection\");", "}"]
        return
    entry=_entry(value,authority); count=entry["group"].component_count
    name="PoPSMoving_%d"%value.id; token="moving_%d"%value.id; geometry=value.inputs[0]
    tolerance=scalar_cpp(_decode_scalar(value.attrs["geometry_tolerance"]))
    face_weights=[scalar_cpp(_decode_scalar(item)) for item in value.attrs["projection"]["face_weights"]]
    measure_weights=[scalar_cpp(_decode_scalar(item)) for item in value.attrs["projection"]["source_measure_weights"]]
    source_lines,source_values,source_invalid=_source(value,authority)
    lines += ["ctx.set_stage_time(0,1);", "%sRelative %s_model;"%(name,token),
        "%s_model.parameter_sets[0]=ctx.program_params(%d);"%(token,blocks[value.block]),
        "const auto %s_reference=ctx.geometry();"%token,
        "const auto %s_params=ctx.program_params(%d);"%(token,blocks[value.block]),
        "auto %s_evaluation=ctx.template project_moving_interval<%d>("%(token,count),
        "  %s,%d,%s,%s,%sLaw{},"%(var[geometry.id],blocks[value.block],json.dumps(value.space.frame),
            json.dumps("left_endpoint@1/exact_endpoint_displacement@1"),name),
        "  [=] POPS_HD(const auto& left,const auto& right,const auto& face,const auto& time) {",
        "    auto model=%s_model; model.mesh_speed=face.swept_volume/time.duration;"%token,
        "    %sLaw law; const auto reference=%s_reference;"%(name,token),
        "    const auto x=face.reference, dx=reference.spacing(0);",
        "    const auto lower=reference.lower()[0], upper=reference.upper()[0];",
        "    const auto xl=(x<=lower?upper-dx:x-dx), xr=(x>=upper?lower+dx:x+dx);",
        "    const auto vl=law(x<=lower?upper:x,time.begin)-law(xl,time.begin);",
        "    const auto vr=law(xr,time.begin)-law(x>=upper?lower:x,time.begin);",
        "    const auto metric=Kokkos::min(vl,vr);",
        "    const auto flux=pops::nd::evaluate_axis_flux<0>(pops::RusanovFlux{},model,left,right,pops::Real(1),metric);",
        "    pops::runtime::program::MovingFaceEvaluation<%d> result{};"%count,
        "    if(!flux.succeeded() || !(metric>0) || time.duration*flux.stability.value>metric) {",
        "      for(int c=0;c<%d;++c) result.physical_amount[c]=std::numeric_limits<pops::Real>::quiet_NaN(); return result; }"%count,
        "    for(int c=0;c<%d;++c) {"%count,
        "      result.density[c]=%s*left[c]+%s*right[c];"%tuple(face_weights),
        "      result.physical_amount[c]=time.duration*flux.checked_density().value[c]+result.density[c]*face.swept_volume; }",
        "    return result; },",
        "  [=] POPS_HD(const auto& u,const auto& cell,const auto& time) {",
        "    const auto params=%s_params; pops::StateVec<%d> result{};"%(token,count),
        *source_lines,
        "    const auto measure=%s*cell.previous_measure+%s*cell.measure;"%tuple(measure_weights),
        "    if(%s) { for(int c=0;c<%d;++c) result[c]=std::numeric_limits<pops::Real>::quiet_NaN(); return result; }"%(source_invalid,count),
        *("    result[%d]=time.duration*measure*(%s);"%(c,expr) for c,expr in enumerate(source_values)),
        "    return result; },%s);"%tolerance,
        "auto %s=ctx.prepare_moving_interval_update(%s_evaluation,%s);"%(token,token,tolerance)]
    var[value.id]=token
