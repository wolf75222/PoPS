"""Finite distributed structural algebra on genuine Program SSA and Uniform.

Local bodies elaborate once without runtime Python or collectives. Nested finite
values retain exact domain and immutable version authorities through a persistent
transport session. Ordered reductions/scans/scatter use global source-gid order.
Bytes and retained domain/version counts are bounded by explicit Program policy;
collective liveness frontiers reclaim unreachable immutable versions. There is
no AMR invocation-calendar adaptation.
Distributed gid/index/rank/at_local require a lexical local region.
"""
from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import new_typed, value_signature, same_signature, scalar_signature, failure_signature, success_signature
from pops.time._program.finite_arrays import _record_body, finite_signature
from pops.time._program.value_validation import require_owned, validate_input_regions
from pops.time.variant_types import ValueType, Product, Int64, Variant, RuntimeFailure, instantiate

OWNERSHIP = "contiguous_distributed"
GLOBAL_OPS=frozenset(("distributed_domain","distributed_map","distributed_requests","distributed_import",
    "distributed_csr","distributed_import_csr","distributed_reduce","distributed_scatter_reduce",
    "distributed_scan","distributed_value_domain","distributed_size"))
LOCAL_OPS = frozenset(("distributed_index_argument", "finite_index_argument", "distributed_offset_argument", "distributed_index",
    "distributed_gid", "distributed_local_at", "distributed_rank", "product", "product_get", "provenance_bind",
    "int64_constant", "bool_constant", "int64_op", "int64_compare", "finite_real", "scalar_op",
    "compare", "scalar_isfinite", "scalar_expression", "branch", "variant_construct", "variant_is", "variant_payload", "variant_unreachable",
    "region_argument", "region_index", "structured_fold", "structured_iterate",
    "result_ok", "result_error", "result_success", "result_is_ok", "result_payload", "result_match",
    "iteration_success", "iteration_is_ok", "iteration_payload", "variant_require_success", "capture_result"))
LOCAL_OPS = LOCAL_OPS | frozenset(("finite_range_set","finite_size","finite_domain","finite_index",
    "finite_index_value","finite_at","finite_map","finite_scan","finite_same_domain",
    "distributed_size","distributed_value_domain"))

LOCAL_REGIONS = ("true_block", "false_block", "body_block", "cond_block")


def signature(value):
    return value.attrs["signature"] if value.vtype == "csr" else value_signature(value)


def transportable(sig):
    if sig["vtype"] in ("product", "variant"):
        for _,child in sig["members"]: transportable(child)
    elif sig["vtype"] in ("array","index_set","index") and sig.get("ownership") in ("replicated",OWNERSHIP):
        if type(sig.get("family")) is not int or type(sig.get("capacity")) is not int:
            raise TypeError("distributed structural transport requires an exact bounded domain family")
        if sig["vtype"]=="array":transportable(sig["element"])
    elif sig["vtype"] not in ("scalar", "int64", "bool"):
        raise TypeError("distributed transport requires finite structural values with bounded domain authority; legacy fields are not transport payloads")


def local_effects(block):
    for node in block:
        if node.op not in LOCAL_OPS:
            raise TypeError("distributed local body cannot execute collective or unsupported operation: " + node.op)
        for key in LOCAL_REGIONS:
            local_effects(node.attrs.get(key, ()))


def local_escaping_errors(block):
    errors=[]
    for node in block:
        if node.op in ("variant_require_success","result_success","iteration_success"):
            errors.append(failure_signature(node.inputs[0]))
        if node.op not in ("capture_result","structured_iterate"):
            for key in LOCAL_REGIONS:errors.extend(local_escaping_errors(node.attrs.get(key,())))
    return tuple(errors)


def _require_error_union(source,target):
    if same_signature(source,target):return
    if target["vtype"]=="variant":
        members=dict(target["members"])
        incoming=dict(source["members"]) if source["vtype"]=="variant" else {"user":source}
        if all(k in members and same_signature(v,members[k]) for k,v in incoming.items()):return
    raise TypeError("local_map error type must retain every escaping failure; return p.capture(body) with its inferred typed Result")


def _require_local_region(program, operation):
    for block in program._recording:
        for node in block:
            if node.op == "distributed_offset_argument" or (
                    node.op in ("finite_index_argument", "distributed_index_argument")
                    and value_signature(node).get("ownership") == OWNERSHIP):
                return
    raise TypeError(operation + " requires a lexical distributed local body; global distributed indices and local reads are unsupported")


def _capacity(value):
    if type(value) is not int or not 0 < value <= (1 << 31)-1:
        raise ValueError("capacity must be a positive exact Int32-range constant")
    return value


def _distributed(value, kind=None):
    sig=signature(value)
    if (kind is not None and value.vtype!=kind) or sig.get("ownership")!=OWNERSHIP:
        raise TypeError("operation requires an authenticated contiguous distributed value")
    return sig


def _new(program,sig,op,inputs,attrs):
    if op in GLOBAL_OPS:
        storage_bytes,storage_versions=program.__dict__.get("_distributed_storage_policy",(64*1024*1024,65536))
        attrs=dict(attrs,storage_bytes=storage_bytes,storage_versions=storage_versions,
                   storage_policy_binding=("explicit" if program.__dict__.get(
                       "_distributed_storage_explicit", False) else "inherited"))
    if sig["vtype"]=="csr":
        return program._new("csr",op,inputs,dict(attrs,signature=sig),None,None,
                            point=sig.get("point"),inherit_state_ref=False)
    return new_typed(program,sig,op,inputs,attrs,point=sig.get("point"))


def _combine(program, values, initial, body, *, scan=False):
    """Record one equation-defined combiner, never a runtime Python callback."""
    require_owned(program,values,"distributed reduction values")
    require_owned(program,initial,"distributed reduction initial")
    state=value_signature(initial);element=_distributed(values,"array")["element"]
    transportable(state);transportable(element)
    if not callable(body):raise TypeError("distributed reduction body must be callable")
    block=[];region=program._region_for_block(block)
    for parent in program._recording:program._allow_region_capture(program._region_for_block(parent),region)
    for item in (values,initial):
        if item.region:program._allow_region_capture(item.region,region)
    program._recording.append(block)
    try:
        argument=new_typed(program,state,"region_argument",(initial,),{},point=initial.point)
        value=new_typed(program,element,"region_argument",(values,),{},point=values.point)
        output=body(program,argument,value)
        require_owned(program,output,"distributed combiner result")
        validate_input_regions(program,(output,),region,"distributed combiner result")
    finally:program._recording.pop()
    output_signature=value_signature(output)
    if scan:
        if output_signature["vtype"]!="product" or set(dict(output_signature["members"]))!={"state","value"}:
            raise TypeError("distributed scan body must return Product(state=...,value=...)")
        output_state=dict(output_signature["members"])["state"]
        transportable(dict(output_signature["members"])["value"])
    else:output_state=output_signature
    if not same_signature(state,output_state):
        raise TypeError("distributed combiner must preserve the exact accumulator type and authority")
    local_effects(block)
    errors={}
    for error in local_escaping_errors(block):
        members=dict(error["members"]) if error["vtype"]=="variant" else {"user":error}
        for key,payload in members.items():
            if key in errors and not same_signature(errors[key],payload):
                raise TypeError("distributed combiner failure tags have conflicting payloads")
            errors[key]=payload
    return dict(body_block=block,body_region=region,body_argument=argument,
                body_value=value,body=output,state_signature=state,
                body_signature=output_signature,
                escaping_errors=local_escaping_errors(block),
                error_signature={"vtype":"variant","members":tuple(sorted(errors.items()))} if errors else None)


class Distributed:
    """Bound authoring namespace; no alternative evaluator or runtime.

    The immutable distributed value vault has explicit program-wide resource
    bounds. Bytes conservatively charge encoded local shards, schemas and domain tables;
    versions bound each retained domain and published-array table. Exhaustion rejects before publishing a value.
    They are resource bounds, independent of the physical model and equations.
    """
    def __init__(self,program,*,storage_bytes=None,storage_versions=None):
        self.program=program
        previous=program.__dict__.get("_distributed_storage_policy")
        base=previous or (64*1024*1024,65536)
        policy=(_capacity(base[0] if storage_bytes is None else storage_bytes),
                _capacity(base[1] if storage_versions is None else storage_versions))
        if previous is not None and policy!=previous:
            raise ValueError("a Program has one immutable distributed storage policy")
        program.__dict__["_distributed_storage_policy"]=policy
        if storage_bytes is not None or storage_versions is not None:
            program.__dict__["_distributed_storage_explicit"]=True

    def domain_of(self,value):
        p=self.program;require_owned(p,value,"distributed domain projection")
        sig=_distributed(value)
        if value.vtype not in ("array","index"):raise TypeError("distributed domain_of requires Array or Index")
        domain=dict(sig,vtype="index_set");domain.pop("element",None)
        return _new(p,domain,"distributed_value_domain",(value,),{})

    def size(self,domain):
        p=self.program;require_owned(p,domain,"distributed size");_distributed(domain,"index_set")
        return _new(p,dict(scalar_signature("int64"),point=domain.point),"distributed_size",(domain,),{})

    def reduce(self,values,initial,body,*,order,byte_capacity):
        p=self.program
        with _transaction(p):
            if order!="global_gid_ascending":raise ValueError("distributed reduction requires order='global_gid_ascending'")
            _capacity(byte_capacity);attrs=_combine(p,values,initial,body)
            attrs.update(order=order,byte_capacity=byte_capacity)
            return _new(p,value_signature(initial),"distributed_reduce",(values,initial),attrs)

    def scatter_reduce(self,values,indices,domain,initial,body,*,order,byte_capacity):
        p=self.program
        with _transaction(p):
            if order!="global_gid_ascending":raise ValueError("distributed scatter requires order='global_gid_ascending'")
            _capacity(byte_capacity)
            for item in (values,indices,domain,initial):require_owned(p,item,"distributed scatter input")
            source=_distributed(values,"array");requests=_distributed(indices,"array");target=_distributed(domain,"index_set")
            a=dict(source);a.pop("element");b=dict(requests);b.pop("element")
            if not same_signature(a,b):raise TypeError("scatter values and indices must have the same exact source domain family")
            if not same_signature(requests["element"],dict(target,vtype="index")):
                raise TypeError("scatter indices must address the exact target domain family")
            attrs=_combine(p,values,initial,body);attrs.update(order=order,byte_capacity=byte_capacity)
            return _new(p,dict(target,vtype="array",element=value_signature(initial)),
                        "distributed_scatter_reduce",(values,indices,domain,initial),attrs)

    def scan(self,values,initial,body,*,order,byte_capacity):
        p=self.program
        with _transaction(p):
            if order!="global_gid_ascending":raise ValueError("distributed scan requires order='global_gid_ascending'")
            _capacity(byte_capacity);attrs=_combine(p,values,initial,body,scan=True)
            element=dict(attrs["body_signature"]["members"])["value"]
            attrs.update(order=order,byte_capacity=byte_capacity,element=element)
            array=dict(_distributed(values,"array"),element=element)
            sig={"vtype":"product","members":(("state",value_signature(initial)),("values",array))}
            return _new(p,sig,"distributed_scan",(values,initial),attrs)

    def domain(self,offsets,*,capacity=None,family=None):
        """Issue a fresh instance; an optional live witness preserves only its type family."""
        p=self.program
        with _transaction(p):
            if not isinstance(offsets,(tuple,list)) or len(offsets)<2:
                raise TypeError("partition requires one ordered offset per rank plus the terminal offset")
            values=tuple(p.int64(x) for x in offsets)
            if family is None:
                capacity=_capacity(capacity)
                family_id=getattr(p,"_next_finite_domain_identity",0);p._next_finite_domain_identity=family_id+1
                inputs=values
            else:
                require_owned(p,family,"distributed family witness")
                witness=_distributed(family,"index_set")
                if capacity is not None:_capacity(capacity)
                if capacity is not None and capacity!=witness["capacity"]:
                    raise TypeError("distributed family witness preserves its exact capacity")
                capacity=witness["capacity"];family_id=witness["family"]
                inputs=values+(family,)
            sig=dict(vtype="index_set",family=family_id,capacity=capacity,ownership=OWNERSHIP,point=values[0].point)
            return _new(p,sig,"distributed_domain",inputs,{"rank_count":len(offsets)-1,"family_witness":family is not None})

    def gid(self,index):
        p=self.program;_require_local_region(p,"distributed gid")
        require_owned(p,index,"distributed gid");_distributed(index,"index")
        return new_typed(p,scalar_signature("int64"),"distributed_gid",(index,),{},point=index.point)

    def index(self,domain,gid):
        p=self.program;_require_local_region(p,"distributed index")
        require_owned(p,domain,"distributed index");sig=_distributed(domain,"index_set")
        return _new(p,dict(sig,vtype="index"),"distributed_index",(domain,p.int64(gid)),{})

    def rank(self):
        p=self.program
        _require_local_region(p,"distributed rank")
        return new_typed(p,scalar_signature("int64"),"distributed_rank",(),{})

    def at_local(self,array,index):
        p=self.program;_require_local_region(p,"distributed local access")
        require_owned(p,array,"distributed local array");require_owned(p,index,"distributed local index")
        a=_distributed(array,"array");i=_distributed(index,"index")
        if not same_signature(dict(a,vtype="index",element=None),dict(i,element=None)):
            raise TypeError("local access requires the same declared domain family")
        return _new(p,a["element"],"distributed_local_at",(array,index),{})

    def local_map(self,domain,body,*,byte_capacity):
        p=self.program
        with _transaction(p):
            ds=_distributed(domain,"index_set");_capacity(byte_capacity)
            block,region,_,index,result=_record_body(p,domain,None,body)
            # _record_body is shared authoring only; local lowering owns its finite effect set.
            local_effects(block)
            rs=value_signature(result)
            if result.vtype=="result":
                members={"ok":success_signature(result),"err":failure_signature(result)}
            elif result.vtype=="variant" and set(dict(rs["members"]))=={"ok","err"}:
                members=dict(rs["members"])
            else:
                raise TypeError("local_map body must return explicit Result(ok=T, err=E)")
            transportable(members["ok"]);transportable(members["err"])
            escaping=local_escaping_errors(block)
            for error in escaping:_require_error_union(error,members["err"])
            array=dict(ds,vtype="array",element=members["ok"])
            reason=Variant(runtime=instantiate(RuntimeFailure,domain.point),user=ValueType(members["err"]))
            err=Product(rank=instantiate(Int64,domain.point),gid=instantiate(Int64,domain.point),reason=reason)
            sig=Variant(ok=ValueType(array),err=err).signature
            return _new(p,sig,"distributed_map",(domain,),dict(body_block=block,body_region=region,
                body_index=index,body=result,element=members["ok"],error=members["err"],escaping_errors=escaping,byte_capacity=byte_capacity))

    def requests(self,domain,target,body):
        p=self.program
        with _transaction(p):
            ds=_distributed(domain,"index_set");ts=_distributed(target,"index_set")
            block,region,_,index,result=_record_body(p,domain,None,body)
            local_effects(block)
            if not same_signature(value_signature(result),dict(ts,vtype="index")):
                raise TypeError("requests body must return Index of the exact target family")
            sig=dict(ds,vtype="array",element=dict(ts,vtype="index"))
            return _new(p,sig,"distributed_requests",(domain,target),dict(body_block=block,
                body_region=region,body_index=index,body=result))

    def import_values(self,source,requests,*,byte_capacity):
        p=self.program;_capacity(byte_capacity)
        require_owned(p,source,"distributed source");require_owned(p,requests,"distributed requests")
        a=_distributed(source,"array");r=_distributed(requests,"array");transportable(a["element"])
        target=dict(a,vtype="index");target.pop("element")
        if not same_signature(target,r["element"]):raise TypeError("import requires exact source/target domain family")
        sig=dict(r,element=a["element"])
        return _new(p,sig,"distributed_import",(source,requests),{"byte_capacity":byte_capacity})

    def csr(self,rows,requests,offset_body,*,nnz_capacity,target=None):
        p=self.program
        with _transaction(p):
            _capacity(nnz_capacity);row_sig=_distributed(rows,"index_set");req=_distributed(requests,"array")
            if req["element"]["vtype"]!="index":raise TypeError("CSR columns require typed target indices")
            inputs=(rows,requests)
            if target is not None:
                require_owned(p,target,"CSR explicit target")
                target_sig=_distributed(target,"index_set")
                if not same_signature(dict(target_sig,vtype="index"),req["element"]):
                    raise TypeError("CSR explicit target must preserve the exact index family")
                inputs+=(target,)
            block=[];region=p._region_for_block(block)
            for parent in p._recording:p._allow_region_capture(p._region_for_block(parent),region)
            p._recording.append(block)
            try:
                offset=new_typed(p,scalar_signature("int64"),"distributed_offset_argument",(),{},point=rows.point)
                result=offset_body(p,offset);require_owned(p,result,"CSR offset")
                validate_input_regions(p,(result,),region,"CSR offset")
            finally:p._recording.pop()
            if result.vtype!="int64":raise TypeError("CSR offsets require exact Int64")
            local_effects(block)
            sig=dict(vtype="csr",rows=row_sig,columns=req,ownership=OWNERSHIP,point=rows.point)
            return _new(p,sig,"distributed_csr",inputs,dict(body_block=block,body_region=region,
                body_offset=offset,body=result,nnz_capacity=nnz_capacity))

    def import_csr(self,source,csr,*,byte_capacity):
        p=self.program;require_owned(p,source,"CSR source");require_owned(p,csr,"CSR incidence")
        a=_distributed(source,"array");c=_distributed(csr,"csr");_capacity(byte_capacity);transportable(a["element"])
        target=dict(a,vtype="index");target.pop("element")
        if not same_signature(target,c["columns"]["element"]):raise TypeError("CSR target does not match source family")
        return _new(p,dict(c["columns"],element=a["element"]),"distributed_import_csr",(source,csr),{"byte_capacity":byte_capacity})


from pops.time._authoring import authoring_transaction as _transaction
