"""Uniform lowering for contiguous distributed domains and local payload algebra.

No collective executes inside a local body. Scalar/Product/Variant leaves share
the Continuation codec; unsupported transport authorities and AMR are refused.
"""
import json
from pops.identity.scalar import scalar_cpp
from pops.time._program.distributed_arrays import signature, transportable, local_effects, OWNERSHIP, LOCAL_REGIONS, GLOBAL_OPS
from pops.time._program.structured_control import value_signature, success_signature, failure_signature

NS="pops::runtime::program"
DN=NS+"::distributed"
CV=NS+"::ContinuationValue<pops::kNativeDimension>"
REF=NS+"::ContinuationValueRef<pops::kNativeDimension>"
OPS=GLOBAL_OPS
LOCAL_FAILURE=DN+"::LocalPayloadFailure<pops::kNativeDimension>"



def _requests_signature(sig):
    return (sig["vtype"]=="array" and sig.get("ownership")==OWNERSHIP and
            sig["element"]["vtype"]=="index" and sig["element"].get("ownership")==OWNERSHIP)


def _array_native(sig, token):
    return DN+"::requests_array<pops::kNativeDimension>(%s)"%token if _requests_signature(sig) else token


def _array_public(sig, token, lines, name, *, condition=None):
    if not _requests_signature(sig):return token
    lines.append(DN+"::Requests "+name+";")
    statement=name+"="+DN+"::requests_from_array<pops::kNativeDimension>("+token+");"
    if condition is not None:statement="if ("+condition+") {"+statement+"}"
    lines.append(DN+"::phase(distributed_lane,[&]{ "+statement+" });")
    return name


def _local_signature(value):
    if value.vtype in ("result", "iteration_result"):
        return {"vtype":"variant", "members":(("err",failure_signature(value)),("ok",success_signature(value)))}
    return value_signature(value)


def _error_key(sig):
    # Runtime error identity excludes authoring clocks, but includes the complete
    # native schema; two distinct payload structures cannot be confused.
    return json.dumps(schema(sig))


def _inject_error(source, token, target):
    if schema(source)==schema(target):return token
    if target["vtype"]!="variant":raise TypeError("local failure cannot erase its payload type")
    arms=dict(target["members"]);keys=list(arms)
    if source["vtype"]!="variant":
        if "user" not in arms or schema(source)!=schema(arms["user"]):
            raise TypeError("local error is absent from its inferred union")
        return CV+"::variant(%d,%s)"%(keys.index("user"),token)
    cases=[]
    for n,(tag,child) in enumerate(source["members"]):
        if tag not in arms or schema(child)!=schema(arms[tag]):
            raise TypeError("local error union changes a payload type")
        cases.append("case %d: return %s::variant(%d,(%s)->members()[0]);"%(n,CV,keys.index(tag),token))
    return "[&]() -> "+REF+" {switch (("+token+")->arm()) {"+" ".join(cases)+" default: throw "+DN+"::Failure("+DN+"::Code::wire);}}()"


def _raised_errors(block):
    out={}
    for node in block:
        if node.op in ("variant_require_success","result_success","iteration_success"):
            sig=failure_signature(node.inputs[0]);out[_error_key(sig)]=sig
        if node.op not in ("capture_result","structured_iterate"):
            for key in LOCAL_REGIONS:out.update(_raised_errors(node.attrs.get(key,())))
    return out


def _runtime_error(v, code, iterations="0"):
    sig=failure_signature(v)
    if v.vtype=="variant":
        keys=[k for k,_ in sig["members"]]
        runtime=CV+"::product({"+CV+"::integer("+code+"),"+CV+"::integer(2)})"
        return CV+"::variant(%d,%s)"%(keys.index("runtime"),runtime)
    values={"code":code,"origin":"2","iterations":iterations}
    return CV+"::product({"+", ".join(CV+"::integer("+values[k]+")" for k,_ in sig["members"])+"})"


def _local_catches(v, block, lines, *, iterations="0", extra=()):
    errors=_raised_errors(block)
    for sig in extra:errors[_error_key(sig)]=sig
    lines.append("} catch (const "+LOCAL_FAILURE+"& e) {")
    for key,sig in errors.items():
        if v.vtype in ("result","iteration_result") and sig["vtype"]=="product":
            # Legacy Result's declared ABI has code/origin only. IterationResult
            # additionally reports this loop's count, matching the public ABI.
            fields=[k for k,_ in sig["members"]]
            if not {"code","origin"}<=set(fields):raise TypeError("legacy local capture cannot erase typed failure")
            values={k:"e.payload->members()[%d]"%fields.index(k) for k in ("code","origin")}
            values["iterations"]=CV+"::integer("+iterations+")"
            error=CV+"::product({"+", ".join(values[k] for k,_ in failure_signature(v)["members"])+"})"
        else:error=_inject_error(sig,"e.payload",failure_signature(v))
        lines.append("if (e.schema==%s) return %s::variant(0,%s);"%(key,CV,error))
    lines.append("throw "+DN+"::Failure("+DN+"::Code::wire);")
    lines.append("} catch (const "+DN+"::Failure&) {")
    lines.append("return "+CV+"::variant(0,"+_runtime_error(v,"3",iterations)+");")
    lines.append("} catch (const std::exception&) {")
    lines.append("return "+CV+"::variant(0,"+_runtime_error(v,"3",iterations)+"); }")


def schema(sig):
    if sig["vtype"] in ("index_set","index","array") and sig.get("ownership")==OWNERSHIP:
        native={"index_set":"DistributedDomain","index":"DistributedIndex","array":"DistributedArray"}[sig["vtype"]]
        children=schema(sig["element"]) if sig["vtype"]=="array" else ""
        return NS+"::ContinuationType{"+NS+"::ContinuationKind::"+native+",{"+children+"},{},%d,%d}"%(sig["family"],sig["capacity"])
    if sig["vtype"] in ("index_set","index","array") and sig.get("ownership")=="replicated":
        from .program_emit_versioned import schema_cpp
        return schema_cpp(sig,{})
    kind={"bool":"Boolean","int64":"Int64","scalar":"Float64","product":"Product","variant":"Variant"}[sig["vtype"]]
    members=sig.get("members",())
    return NS+"::ContinuationType{"+NS+"::ContinuationKind::"+kind+", {"+", ".join(schema(s) for _,s in members)+"}, {"+", ".join(json.dumps(k) for k,_ in members)+"}}"


def unpack(sig,ref):
    kind=sig["vtype"]
    if kind=="product":return {k:unpack(s,"(%s)->members()[%d]"%(ref,n)) for n,(k,s) in enumerate(sig["members"])}
    if kind=="variant":return {"tag":"std::int64_t((%s)->arm())"%ref,"payloads":{k:unpack(s,"(%s)->members()[0]"%ref) for k,s in sig["members"]}}
    if kind in ("index_set","index","array") and sig.get("ownership")==OWNERSHIP:
        if kind=="index_set":return "(%s)->distributed_domain_ref()"%ref
        if kind=="index":return DN+"::Index{(%s)->distributed_domain_ref(),(%s)->distributed_index_offset()}"%(ref,ref)
        if (sig["element"]["vtype"]=="index" and sig["element"].get("ownership")==OWNERSHIP):return DN+"::requests_from_array<pops::kNativeDimension>((%s)->distributed_array_ref())"%ref
        return "(%s)->distributed_array_ref()"%ref
    if kind in ("index_set","index","array") and sig.get("ownership")=="replicated":
        from .program_emit_versioned import _unpack
        lines=[];expression=_unpack(sig,ref,lines,"distributed_unpacked")
        if not lines:return expression
        return "[&](){"+" ".join(lines)+" return "+expression+";}()"
    return "(%s)->%s()"%(ref,{"scalar":"real","int64":"integer","bool":"boolean"}[kind])


def pack(sig,token,lines,name):
    kind=sig["vtype"]
    if kind=="product":
        children=[pack(s,token[k],lines,name+"_"+str(n)) for n,(k,s) in enumerate(sig["members"])]
        return CV+"::product({"+", ".join(children)+"})"
    if kind=="variant":
        lines.append(REF+" "+name+";")
        lines.append("switch ("+token["tag"]+") {")
        for n,(key,child) in enumerate(sig["members"]):
            lines.append("case %d: {"%n)
            value=pack(child,token["payloads"][key],lines,name+"_arm"+str(n))
            lines.append("%s = %s::variant(%d,%s); break; }"%(name,CV,n,value))
        lines.append("default: throw "+DN+"::Failure("+DN+"::Code::wire); }")
        return name
    if kind in ("index_set","index","array") and sig.get("ownership")=="replicated":
        from .program_emit_versioned import pack_local
        return pack_local(sig,token,lines,name,{})
    if kind in ("index_set","index","array") and sig.get("ownership")==OWNERSHIP:
        if kind=="index_set":return CV+"::distributed_domain("+token+")"
        if kind=="index":return CV+"::distributed_index((%s).domain,(%s).gid)"%(token,token)
        value=DN+"::requests_array<pops::kNativeDimension>(%s)"%token if (sig["element"]["vtype"]=="index" and sig["element"].get("ownership")==OWNERSHIP) else token
        return CV+"::distributed_array("+value+")"
    transportable(sig)
    return CV+"::"+{"scalar":"real","int64":"integer","bool":"boolean"}[kind]+"("+token+")"


def _local(block,bindings,lines):
    """Map the closed finite local graph to direct scalar C++; no ctx/collective dispatch."""
    local_effects(block);env=dict(bindings)
    for v in block:
        from .program_source_map import operation
        if v.id in env:
            # Only the declared region arguments are introduced at this lexical
            # binding site. Captured computed values keep their producer map.
            if v.op in ("finite_index_argument", "distributed_index_argument",
                        "distributed_offset_argument", "region_argument", "region_index"):
                with operation(v, lines, environment=env,
                               channel="distributed_local_formal", binding_role="lexical_formal"):
                    pass
            continue
        with operation(v, lines, environment=env, channel="distributed_local"):
            op=v.op;t="local_%d"%v.id;args=[env[x.id] for x in v.inputs]
            if op in ("finite_index_argument","distributed_index_argument","distributed_offset_argument","region_argument","region_index"):
                raise ValueError("local formal lacks exact lexical binding")
            if op == "provenance_bind":
                from pops.ledger_provenance import validate_contract_operation
                validate_contract_operation(v.inputs, v.attrs, value_signature(v))
                env[v.id] = args[0]
                continue
            if op=="int64_constant":
                n=v.attrs["value"];e="std::numeric_limits<std::int64_t>::min()" if n==-(1<<63) else "std::int64_t(%d)"%n
                expr=CV+"::integer("+e+")"
            elif op=="bool_constant":expr=CV+"::boolean("+("true" if v.attrs["value"] else "false")+")"
            elif op=="distributed_gid":expr=CV+"::integer(("+args[0]+")->distributed_index_offset())"
            elif op=="distributed_rank":expr=CV+"::integer(distributed_lane.rank())"
            elif op=="distributed_index":expr=CV+"::distributed_index((%s)->distributed_domain_ref(),(%s)->integer())"%tuple(args)
            elif op=="distributed_local_at":expr=DN+"::at_local((%s)->distributed_array_ref(),%s::Index{(%s)->distributed_domain_ref(),(%s)->distributed_index_offset()})"%(args[0],DN,args[1],args[1])
            elif op=="distributed_size":expr=CV+"::integer((%s)->distributed_domain_ref()->size())"%args[0]
            elif op=="distributed_value_domain":
                expr=CV+"::distributed_domain((%s)->distributed_domain_ref())"%args[0]
            elif op=="finite_range_set":
                sig=value_signature(v);capacity=sig["capacity"];family=sig["family"]
                if sig["ownership"]!="replicated":raise TypeError("local range requires finite local storage")
                if len(args)>1:
                    witness="(%s)->domain_ref()"%args[1]
                    lines.append(DN+"::local_require("+witness+" && "+witness+"->visible() && "+witness+"->family==%d && "%family+witness+"->capacity==%d,"%capacity+DN+"::Code::domain);")
                expr=CV+"::domain(ctx.continuation_new_local_domain((%s)->integer(),%d,%d))"%(args[0],capacity,family)
            elif op=="finite_size":
                if value_signature(v.inputs[0]).get("ownership")==OWNERSHIP:
                    expr=CV+"::integer((%s)->distributed_domain_ref()->size())"%args[0]
                else:expr=CV+"::integer((%s)->domain_ref()->size)"%args[0]
            elif op=="finite_domain":expr=CV+"::domain((%s)->domain_ref())"%args[0]
            elif op=="finite_index":
                lines.append(DN+"::local_require((%s)->integer()>=0 && (%s)->integer()<(%s)->domain_ref()->size,"%(args[1],args[1],args[0])+DN+"::Code::index);")
                expr=CV+"::index((%s)->domain_ref(),(%s)->integer())"%(args[0],args[1])
            elif op=="finite_index_value":expr=CV+"::integer((%s)->index_offset())"%args[0]
            elif op=="finite_same_domain":
                if value_signature(v.inputs[0]).get("ownership")==OWNERSHIP:
                    lines.append("(%s)->distributed_domain_ref()->require_live(); (%s)->distributed_domain_ref()->require_live();"%(args[0],args[1]));expr=CV+"::boolean((%s)->distributed_domain_ref()==(%s)->distributed_domain_ref())"%tuple(args)
                else:
                    lines.append("(%s)->domain_ref()->require_live(); (%s)->domain_ref()->require_live();"%tuple(args))
                    expr=CV+"::boolean((%s)->domain_ref()==(%s)->domain_ref())"%tuple(args)
            elif op=="finite_at":
                lines.append(DN+"::local_require((%s)->domain_ref()==(%s)->domain_ref() && (%s)->index_offset()>=0 && (%s)->index_offset()<std::int64_t((%s)->members().size()),"%(args[0],args[1],args[1],args[1],args[0])+DN+"::Code::index);")
                expr="(%s)->members()[std::size_t((%s)->index_offset())]"%(args[0],args[1])
            elif op in ("finite_map","finite_scan"):
                scan=op=="finite_scan";domain=args[1] if scan else args[0]
                lines.append("const auto %s=[&]() -> %s {"%(t,REF))
                lines.append("const auto domain=(%s)->domain_ref(); domain->require_live();"%domain)
                lines.append(DN+"::local_require(domain->size>=0 && domain->size<=domain->capacity,"+DN+"::Code::capacity);")
                lines.append("std::vector<%s> values; values.reserve(std::size_t(domain->size));"%REF)
                if scan:lines.append(REF+" carry="+args[0]+";")
                lines.append("for(std::int64_t k=0;k<domain->size;++k) {")
                local=dict(env);local[v.attrs["body_index"].id]=CV+"::index(domain,k)"
                if scan:local[v.attrs["body_argument"].id]="carry"
                inner=_local(v.attrs["body_block"],local,lines);result=inner[v.attrs["body"].id]
                if scan:
                    keys=[k for k,_ in value_signature(v.attrs["body"])["members"]]
                    lines.append("carry=(%s)->members()[%d]; values.push_back((%s)->members()[%d]);"%(result,keys.index("state"),result,keys.index("value")))
                else:lines.append("values.push_back(%s);"%result)
                lines.append("}");array=CV+"::array(domain,std::move(values))"
                if scan:
                    fields={"state":"carry","values":array};array=CV+"::product({"+", ".join(fields[k] for k,_ in value_signature(v)["members"])+"})"
                lines.append("return %s; }();"%array);env[v.id]=t;continue
            elif op=="product":expr=CV+"::product({"+", ".join(args)+"})"
            elif op=="product_get":
                keys=[k for k,_ in value_signature(v.inputs[0])["members"]]
                expr="(%s)->members()[%d]"%(args[0],keys.index(v.attrs["field"]))
            elif op=="int64_op":
                operation={"add":0,"sub":1,"mul":2,"floordiv":3,"mod":4}[v.attrs["operation"]]
                expr=CV+"::integer(%s::integer(%d,(%s)->integer(),(%s)->integer()))"%(DN,operation,*args)
            elif op=="int64_compare":expr=CV+"::boolean((%s)->integer() %s (%s)->integer())"%(args[0],v.attrs["comparison"],args[1])
            elif op=="finite_real":
                source="double((%s)->%s())"%(args[0],"integer" if v.inputs[0].vtype=="int64" else "real") if args else float(v.attrs["value"]).hex()
                expr=CV+"::real("+source+")"
            elif op=="scalar_op":
                parts=["(%s)->real()"%args[n] if k=="v" else scalar_cpp(n) for k,n in v.attrs["operands"]]
                expression="(%s %s %s)"%(parts[0],{"add":"+","sub":"-","mul":"*","div":"/"}[v.attrs["fn"]],parts[1])
                lines.append("const double %s_value = %s;"%(t,expression))
                lines.append(DN+"::local_require(std::isfinite("+t+"_value),"+DN+"::Code::arithmetic);")
                expr=CV+"::real("+t+"_value)"
            elif op=="scalar_expression":
                from .program_emit_expressions import checked_expression_dag
                body,outputs,invalid=checked_expression_dag(v.attrs["expressions"],v.attrs["expression_nodes"],
                    tuple(("(%s)->real()"%x,) for x in args))
                if len(outputs)!=1:raise TypeError("local scalar expression must produce one scalar")
                lines.append("const auto %s_value=[&](){"%t)
                lines.extend(body)
                lines.append(DN+"::local_require(!("+invalid+"),"+DN+"::Code::arithmetic);")
                lines.append("return %s; }();"%outputs[0]);expr=CV+"::real("+t+"_value)"
            elif op=="compare":
                rhs="(%s)->real()"%args[1] if len(args)==2 else scalar_cpp(v.attrs["rhs"])
                expr=CV+"::boolean((%s)->real() %s %s)"%(args[0],v.attrs["cmp"],rhs)
            elif op=="scalar_isfinite":expr=CV+"::boolean(std::isfinite((%s)->real()))"%args[0]
            elif op=="variant_unreachable":
                expr="[&]() -> "+REF+" {throw "+DN+"::Failure("+DN+"::Code::wire); }()"
            elif op=="variant_construct":
                keys=[k for k,_ in value_signature(v)["members"]]
                expr=CV+"::variant(%d,%s)"%(keys.index(v.attrs["tag"]),args[0])
            elif op=="variant_is":
                keys=[k for k,_ in value_signature(v.inputs[0])["members"]]
                expr=CV+"::boolean((%s)->arm()==%d)"%(args[0],keys.index(v.attrs["tag"]))
            elif op=="variant_payload":
                if env.get(("guard",v.inputs[0].id))!=v.attrs["tag"]:
                    raise ValueError("unguarded local Variant projection")
                expr="(%s)->members()[0]"%args[0]
            elif op=="result_ok":expr=CV+"::variant(1,"+args[0]+")"
            elif op=="result_error":
                lines.append(DN+"::local_require(("+args[1]+")->integer()!=0,"+DN+"::Code::arithmetic);")
                expr=CV+"::variant(0,"+CV+"::product({"+args[1]+","+CV+"::integer(1)}))"
            elif op in ("result_is_ok","iteration_is_ok"):
                expr=CV+"::boolean(("+args[0]+")->arm()==1)"
            elif op in ("variant_require_success","result_success","iteration_success"):
                lines.append("if ((%s)->arm()!=1) throw %s{(%s)->members()[0],%s};"%(args[0],LOCAL_FAILURE,args[0],_error_key(failure_signature(v.inputs[0]))))
                expr="(%s)->members()[0]"%args[0]
            elif op in ("result_payload","iteration_payload"):
                required=1 if v.attrs["variant"]=="ok" else 0
                lines.append(DN+"::local_require((%s)->arm()==%d,%s::Code::wire);"%(args[0],required,DN))
                expr="(%s)->members()[0]"%args[0]
            elif op=="structured_fold":
                lines.append("const auto %s = [&]() -> %s {"%(t,REF))
                lines.append("const auto count=(%s)->integer(), budget=(%s)->integer();"%(args[1],args[2]))
                lines.append(DN+"::local_require(count>=0 && budget>=0 && count<=budget,"+DN+"::Code::capacity);")
                lines.append("%s carry=%s; for (std::int64_t k=0;k<count;++k) {"%(REF,args[0]))
                local=dict(env);local[v.attrs["body_argument"].id]="carry";local[v.attrs["body_index"].id]=CV+"::integer(k)"
                inner=_local(v.attrs["body_block"],local,lines)
                lines.append("carry=%s; } return carry; }();"%inner[v.attrs["body"].id]);env[v.id]=t;continue
            elif op=="structured_iterate":
                lines.append("const auto %s = [&]() -> %s {"%(t,REF))
                lines.append("std::int64_t iterations=0; const auto budget=(%s)->integer(); %s carry=%s; try {"%(args[1],REF,args[0]))
                lines.append("if (budget<0) return "+CV+"::variant(0,"+_runtime_error(v,"2","iterations")+");")
                lines.append("while (true) {")
                local=dict(env);local[v.attrs["cond_argument"].id]="carry"
                condition=_local(v.attrs["cond_block"],local,lines)[v.attrs["cond"].id]
                fields={"state":"carry","iterations":CV+"::integer(iterations)"}
                payload=CV+"::product({"+", ".join(fields[k] for k,_ in success_signature(v)["members"])+"})"
                lines.append("if ((%s)->boolean()) return %s::variant(1,%s);"%(condition,CV,payload))
                lines.append("if (iterations>=budget) return "+CV+"::variant(0,"+_runtime_error(v,"1","iterations")+");")
                local=dict(env);local[v.attrs["body_argument"].id]="carry"
                inner=_local(v.attrs["body_block"],local,lines);result=inner[v.attrs["body"].id];extras=[]
                if v.attrs["body_is_result"] or v.attrs.get("body_is_variant_result"):
                    error=failure_signature(v.attrs["body"]);extras.append(error)
                    lines.append("if ((%s)->arm()!=1) throw %s{(%s)->members()[0],%s};"%(result,LOCAL_FAILURE,result,_error_key(error)))
                    result="(%s)->members()[0]"%result
                lines.append("carry=%s; ++iterations; }"%result)
                _local_catches(v,v.attrs["cond_block"]+v.attrs["body_block"],lines,iterations="iterations",extra=extras)
                lines.append("}();");env[v.id]=t;continue
            elif op=="capture_result":
                lines.append("const auto %s = [&]() -> %s { try {"%(t,REF))
                inner=_local(v.attrs["body_block"],env,lines);result=inner[v.attrs["body"].id];extras=[]
                if v.attrs["body_is_result"] or v.attrs.get("body_is_variant_result"):
                    error=failure_signature(v.attrs["body"]);extras.append(error)
                    lines.append("if ((%s)->arm()!=1) throw %s{(%s)->members()[0],%s};"%(result,LOCAL_FAILURE,result,_error_key(error)))
                    result="(%s)->members()[0]"%result
                lines.append("return %s::variant(1,%s);"%(CV,result))
                _local_catches(v,v.attrs["body_block"],lines,extra=extras)
                lines.append("}();");env[v.id]=t;continue
            elif op in ("branch","result_match"):
                lines.append("const auto %s = [&]() {"%t)
                predicate=args[0] if op=="branch" else args[1]
                lines.append("if ((%s)->boolean()) {"%predicate);yes=dict(env)
                condition=v.inputs[0] if op=="branch" else v.inputs[1]
                if condition.op=="variant_is":yes[("guard",condition.inputs[0].id)]=condition.attrs["tag"]
                y=_local(v.attrs["true_block"],yes,lines)
                lines.append("return %s; } else {"%y[v.attrs["true_result"].id])
                n=_local(v.attrs["false_block"],env,lines)
                lines.append("return %s; } }();"%n[v.attrs["false_result"].id]);env[v.id]=t;continue
            else:raise NotImplementedError(op)
            lines.append("const auto %s = %s;"%(t,expr));env[v.id]=t
    return env


def _captures(v,var):
    block=v.attrs["body_block"];known={x.id for x in block};env={}
    def gather(nodes):
        for node in nodes:
            for item in node.inputs:
                if item.id in var and item.id not in known:env[item.id]=item
            for key in LOCAL_REGIONS:gather(node.attrs.get(key,()))
    gather(block)
    returned=v.attrs["body"]
    if returned.id not in known and returned.id in var:env[returned.id]=returned
    return env


def _body(v,var,lines,formal_name):
    block=v.attrs["body_block"];env=_captures(v,var)
    for key,item in list(env.items()):
        env[key]=pack(value_signature(item),var[key],lines,"capture_%d_%d"%(v.id,key))
    if isinstance(formal_name,dict):
        for key,name in formal_name.items():env[v.attrs[key].id]=name
    else:
        formal=v.attrs.get("body_index",v.attrs.get("body_offset"))
        env[formal.id]=(CV+"::distributed_index((%s).domain,(%s).gid)"%(formal_name,formal_name)
                        if formal.vtype=="index" else formal_name)
    return _local(block,env,lines)[v.attrs["body"].id]


def emit_distributed_op(program,v,var,model,lines,prelude,block_idx,field_plans,target):
    if target!="system":raise NotImplementedError("distributed arrays initially require Uniform Program; AMR invocation calendar is not qualified")
    if v.op not in OPS and v.op!="finite_same_domain":raise NotImplementedError(v.op)
    token="distributed_%d"%v.id
    # Context hook has persistent, non-rollback lifetime; no invocation-local counter is created.
    lines.append("auto& %s_registry = ctx.distributed_registry();"%token)
    registry=token+"_registry"
    lines.append("const auto& distributed_lane = %s.lane();"%registry) if not var.get(("distributed_lane",)) else None
    var[("distributed_lane",)]=True
    if v.op in GLOBAL_OPS:
        lines.append("%s.configure(%d,%d);"%(registry,v.attrs.get("storage_bytes",64*1024*1024),v.attrs.get("storage_versions",65536)))
    session=token+"_transport"
    lines.append("auto& %s=ctx.continuation_transport_session();"%session)
    lines.append(DN+"::phase(distributed_lane,[&]{ "+session+".bind_distributed_registry("+registry+
        ",[registry_ptr=&"+registry+"](auto version,const auto& domain,const auto& element){return "+
        DN+"::resolve_array<pops::kNativeDimension>(*registry_ptr,version,domain,element);}); });")
    if v.op in GLOBAL_OPS:
        lines.append(DN+"::collect_garbage<pops::kNativeDimension>("+registry+");")
        lines.append(session+".collect_expired_aliases();")
    if "body_block" in v.attrs:
        captures=_captures(v,var)
        # Seed replicated captures on every rank, including empty owners, before
        # any payload is decoded. Equal metadata never substitutes its authority.
        def finite_capture(sig):
            if sig["vtype"] in ("index_set","index","array"):return sig.get("ownership") in ("replicated",OWNERSHIP)
            return any(finite_capture(child) for _,child in sig.get("members",()))
        candidates=[item for item in captures.values() if finite_capture(value_signature(item))]
        if candidates:
            lines.append(DN+"::phase(distributed_lane,[&]{")
            for item in candidates:
                value=pack(value_signature(item),var[item.id],lines,"seed_%d_%d"%(v.id,item.id))
                lines.append("%s.capture(%s);"%(session,value))
            lines.append("});")
    args=[var[x.id] for x in v.inputs]
    if v.op=="finite_same_domain":
        for source in v.inputs:
            if signature(source).get("ownership")!=OWNERSHIP:
                raise TypeError("distributed same_domain requires two distributed domain authorities")
        lines.append("%s.common(%s); %s.common(%s);"%(registry,args[0],registry,args[1]))
        lines.append("const bool %s = (%s)==(%s);"%(token,args[0],args[1]))
    elif v.op=="distributed_size":
        lines.append("%s.common(%s);"%(registry,args[0]))
        lines.append("const std::int64_t %s=(%s)->size();"%(token,args[0]))
    elif v.op=="distributed_value_domain":
        source=signature(v.inputs[0]);is_requests=source["vtype"]=="array" and (source["element"]["vtype"]=="index" and source["element"].get("ownership")==OWNERSHIP)
        access="." if source["vtype"]=="index" or is_requests else "->"
        lines.append("const auto %s=(%s)%sdomain;"%(token,args[0],access))
        lines.append("%s.common(%s);"%(registry,token))
    elif v.op=="distributed_domain":
        sig=signature(v)
        offsets=args[:v.attrs["rank_count"]+1]
        if v.attrs.get("family_witness"):
            witness=args[-1]
            lines.append("%s.common(%s);"%(registry,witness))
            lines.append(DN+"::phase(distributed_lane,[&]{ "+DN+"::local_require(("+witness+")->family()=="+str(sig["family"])+" && ("+witness+")->capacity()=="+str(sig["capacity"])+","+DN+"::Code::domain); });")
        lines.append("std::vector<std::int64_t> %s_offsets;"%token)
        lines.append(DN+"::phase(distributed_lane,[&]{ "+token+"_offsets={"+", ".join(offsets)+"}; });")
        lines.append("const auto %s = %s.issue(%d,%d,%s_offsets);"%(token,registry,sig["family"],sig["capacity"],token))
    elif v.op in ("distributed_map","distributed_requests"):
        body=[];result=_body(v,var,body,"distributed_index")
        if v.op=="distributed_map":
            lines.append(NS+"::ContinuationType "+token+"_element, "+token+"_error;")
            lines.append(DN+"::phase(distributed_lane,[&]{ "+token+"_element="+schema(v.attrs["element"])+"; "+token+"_error="+schema(v.attrs["error"])+"; });")
            lines.append("const auto %s = %s::map<pops::kNativeDimension>(%s,%s,%s,%s,%d,[&](auto distributed_index){"%(token,DN,registry,args[0],token+"_element",token+"_error",v.attrs["byte_capacity"]))
        else:
            lines.append("auto %s = %s::requests(%s,%s,%s,[&](auto distributed_index){"%(token,DN,registry,*args))
            result=unpack(value_signature(v.attrs["body"]),result)
        escaping=v.attrs.get("escaping_errors",())
        if escaping:lines.append("try {")
        lines.extend("  "+x for x in body);lines.append("  return %s;"%result)
        if escaping:
            lines.append("} catch (const "+LOCAL_FAILURE+"& e) {")
            for error in escaping:
                lines.append("if (e.schema==%s) return %s::variant(0,%s);"%(_error_key(error),CV,_inject_error(error,"e.payload",v.attrs["error"])))
            lines.append("throw "+DN+"::Failure("+DN+"::Code::wire); }")
        lines.append("}, &%s);"%session if v.op=="distributed_map" else "});")
        if v.op=="distributed_requests":
            lines.append(DN+"::publish_requests<pops::kNativeDimension>(%s,%s,%d,&%s);"%(registry,token,v.attrs.get("storage_bytes",64*1024*1024),session))
        if v.op=="distributed_map":
            errsig=dict(signature(v)["members"])["err"]
            err={"rank":token+".rank","gid":token+".gid","reason":{"tag":"(%s.code==%s::Code::user?std::int64_t(1):std::int64_t(0))"%(token,DN),"payloads":{"runtime":{"code":"std::int64_t(%s.code)"%token,"origin":"std::int64_t(3)"},"user":unpack(v.attrs["error"],token+".payload")}}}
            ok=_array_public(dict(signature(v)["members"])["ok"],token+".value",lines,token+"_requests",condition=token+".ok()")
            var[v.id]={"tag":"(%s.ok()?std::int64_t(1):std::int64_t(0))"%token,"payloads":{"ok":ok,"err":err}};return
    elif v.op=="distributed_import":
        args[0]=_array_native(signature(v.inputs[0]),args[0])
        lines.append("const auto %s = %s::import_values(%s,%s,%s,%d,&%s);"%(token,DN,registry,*args,v.attrs["byte_capacity"],session))
    elif v.op in ("distributed_reduce","distributed_scatter_reduce","distributed_scan"):
        args[0]=_array_native(signature(v.inputs[0]),args[0])
        state=v.attrs["state_signature"]
        errors=v.attrs["error_signature"]
        native_error=errors if errors is not None else {"vtype":"product","members":()}
        lines.append(NS+"::ContinuationType "+token+"_state, "+token+"_error; "+REF+" "+token+"_initial;")
        lines.append(DN+"::phase(distributed_lane,[&]{")
        initial=pack(state,args[-1],lines,token+"_initial_pack")
        lines.append(token+"_state="+schema(state)+"; "+token+"_error="+schema(native_error)+"; "+token+"_initial="+initial+"; });")
        body=[];result=_body(v,var,body,{"body_argument":"combine_state","body_value":"combine_value"})
        name={"distributed_reduce":"reduce","distributed_scan":"scan","distributed_scatter_reduce":"scatter_reduce"}[v.op]
        arguments=args[:-1]+[token+"_state",token+"_initial",token+"_error"]
        if v.op=="distributed_scan":
            lines.append(NS+"::ContinuationType "+token+"_element;")
            lines.append(DN+"::phase(distributed_lane,[&]{ "+token+"_element="+schema(v.attrs["element"])+"; });")
            arguments.append(token+"_element")
            keys=[k for k,_ in v.attrs["body_signature"]["members"]]
            result="std::make_pair((%s)->members()[%d],(%s)->members()[%d])"%(result,keys.index("state"),result,keys.index("value"))
        arguments.append(str(v.attrs["byte_capacity"]))
        result_type=REF if v.op=="distributed_reduce" else DN+("::Scan" if v.op=="distributed_scan" else "::ArrayRef")+"<pops::kNativeDimension>"
        lines.append(result_type+" "+token+"; try {")
        lines.append("%s=%s::%s(%s,%s,[&](const auto& combine_state,const auto& combine_value){"%(token,DN,name,registry,",".join(arguments)))
        if errors is not None:lines.append("try {")
        lines.extend("  "+x for x in body);lines.append("return %s;"%result)
        if errors is not None:
            lines.append("} catch (const "+LOCAL_FAILURE+"& e) {")
            for error in v.attrs["escaping_errors"]:
                lines.append("if (e.schema==%s) throw %s{%s,%s};"%(_error_key(error),LOCAL_FAILURE,_inject_error(error,"e.payload",errors),_error_key(errors)))
            lines.append("throw "+DN+"::Failure("+DN+"::Code::wire); }")
        lines.append("}, &%s); } catch (const "%session+DN+"::CollectivePayloadFailure<pops::kNativeDimension>& failure) {")
        if errors is not None:
            from .program_emit_structured_control import _propagate_typed_error
            _propagate_typed_error(errors,unpack(errors,"failure.payload"),var,lines,block_idx,target)
        else:lines.append('throw std::logic_error("undeclared distributed combiner failure");')
        lines.append("}")
        if v.op in ("distributed_reduce","distributed_scan"):
            from .program_emit_versioned import unpack_owned
            reference=token if v.op=="distributed_reduce" else token+".state"
            value=unpack_owned(state,reference,lines,token+"_unpack_state")
            if v.op=="distributed_reduce":var[v.id]=value
            else:
                array_sig=dict(signature(v)["members"])["values"]
                array=_array_public(array_sig,token+".values",lines,token+"_requests")
                var[v.id]={"state":value,"values":array}
            return
    elif v.op=="distributed_csr":
        body=[];result=_body(v,var,body,CV+"::integer(distributed_offset)")
        explicit_target=args[2] if len(args)==3 else "("+args[1]+").target"
        lines.append("const auto %s = %s::csr_offsets(%s,%s,%s,%s,%d,[&](std::int64_t distributed_offset){"%(token,DN,registry,args[0],explicit_target,args[1],v.attrs["nnz_capacity"]))
        lines.extend("  "+x for x in body);lines.append("  return (%s)->integer(); });"%result)
    else:
        args[0]=_array_native(signature(v.inputs[0]),args[0])
        lines.append("const auto %s = %s::import_csr(%s,%s,%s,%d,&%s);"%(token,DN,registry,*args,v.attrs["byte_capacity"],session))
    var[v.id]=_array_public(signature(v),token,lines,token+"_requests") if v.op in ("distributed_import","distributed_scatter_reduce") else token
