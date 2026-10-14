"""C++ value/region lowering for replicated finite arrays in the existing Program engine."""
from __future__ import annotations

from pops.time._program.structured_control import value_signature


FINITE_OPS = frozenset(("finite_real", "finite_range_set", "finite_size", "finite_domain", "finite_same_domain",
                        "finite_index", "finite_index_value", "finite_index_argument", "finite_at",
                        "finite_map", "finite_scan"))


def finite_cpp_helpers():
    # These are immutable artifact-local values, not an alternate execution runtime. All
    # iteration, ownership, transaction and collective services remain those of Program.
    return r'''
#include <memory>
#include <vector>
#include <tuple>
#include <variant>
#include <bit>
#include <new>
#include <pops/runtime/program/continuation_authority.hpp>
namespace pops_algebra_finite {
static_assert(sizeof(pops::Real) == sizeof(std::int64_t) && std::numeric_limits<pops::Real>::is_iec559,
              "finite Float64 arrays require IEEE binary64 native Real");
using Domain = pops::runtime::program::FiniteDomain;
using DomainRef = pops::runtime::program::FiniteDomainRef;
struct Index {
  DomainRef domain;
  std::int64_t offset = 0;
};
template<class Element> struct Array {
  DomainRef domain;
  std::shared_ptr<const std::vector<Element>> values;
};
}
'''


def finite_cpp_type(signature):
    kind = signature["vtype"]
    if signature.get("ownership") == "contiguous_distributed":
        ns = "pops::runtime::program::distributed::"
        if kind == "array":
            element = signature["element"]
            requests = (element["vtype"] == "index" and
                        element.get("ownership") == "contiguous_distributed")
            return ns + ("Requests" if requests else "ArrayRef<pops::kNativeDimension>")
        return ns + {"index_set":"DomainRef", "index":"Index"}[kind]
    if kind.startswith("continuation_"):
        from .program_emit_versioned import capability_cpp_type
        return capability_cpp_type(signature)
    if kind == "product":
        return "std::tuple<%s>" % ", ".join(finite_cpp_type(item) for _,item in signature["members"])
    if kind == "variant":
        return "std::variant<%s>" % ", ".join(finite_cpp_type(item) for _,item in signature["members"])
    if kind == "array":
        return "pops_algebra_finite::Array<%s>" % finite_cpp_type(signature["element"])
    if kind in ("state", "rhs", "scalar_field"):
        return "std::shared_ptr<pops::MultiFab<pops::kNativeDimension>>"
    return {"scalar":"pops::Real", "int64":"std::int64_t", "bool":"bool",
            "index_set":"pops_algebra_finite::DomainRef", "index":"pops_algebra_finite::Index"}[kind]


def _packed(signature, token, lines, prefix):
    if signature["vtype"] == "product":
        return "std::make_tuple(%s)" % ", ".join(
            _packed(child, token[key], lines, "%s_member%d" % (prefix, n))
            for n, (key, child) in enumerate(signature["members"]))
    if signature["vtype"] == "variant":
        output = prefix + "_sum"
        lines.append("%s %s;" % (finite_cpp_type(signature), output))
        lines.append("switch (%s) {" % token["tag"])
        for n, (tag, child) in enumerate(signature["members"]):
            lines.append("case %d: {" % n)
            body = []
            packed = _packed(child, token["payloads"][tag], body, "%s_arm%d" % (prefix, n))
            body.append("%s.emplace<%d>(%s);" % (output, n, packed))
            lines.extend("  " + line for line in body)
            lines.append("  break; }")
        lines.append('default: throw std::logic_error("invalid Variant element discriminant");')
        lines.append("}")
        return output
    if signature["vtype"] in ("state", "rhs", "scalar_field"):
        from .program_emit_structured_control import _copy
        output = prefix + "_snapshot"
        lines.append("%s %s;" % (finite_cpp_type(signature), output))
        _allocate("std::make_shared<pops::MultiFab<pops::kNativeDimension>>(ctx.scratch_state_like(%s))" % token,
                  output, lines)
        _copy(signature, "(*%s)" % output, token, lines)
        return output
    return token


def _unpacked(signature, token):
    if signature["vtype"] == "product":
        return {key:_unpacked(child,"std::get<%d>(%s)" % (n,token))
                for n,(key,child) in enumerate(signature["members"])}
    if signature["vtype"] == "variant":
        return {"tag": "std::int64_t((%s).index())" % token,
                "payloads": {key: _unpacked(child, "std::get<%d>(%s)" % (n, token))
                             for n, (key, child) in enumerate(signature["members"])}}
    if signature["vtype"] in ("state", "rhs", "scalar_field"):
        return "(*(%s))" % token
    return token


def _uniform_element(signature, token, lines, depth=0):
    from .program_emit_structured_control import _reject_if, _uniform_int
    kind = signature["vtype"]
    if kind == "product":
        for key,child in signature["members"]:
            _uniform_element(child,token[key],lines,depth)
    elif kind == "variant":
        _uniform_int(token["tag"], lines, "replicated Variant tag differs across execution ranks")
        _reject_if("%s < 0 || %s >= %d" % (token["tag"], token["tag"], len(signature["members"])),
                   lines, "replicated Variant has an invalid discriminant")
        lines.append("switch (%s) {" % token["tag"])
        for n, (tag, child) in enumerate(signature["members"]):
            lines.append("case %d: {" % n)
            body = []
            _uniform_element(child, token["payloads"][tag], body, depth)
            lines.extend("  " + line for line in body)
            lines.append("  break; }")
        lines.append("}")
    elif kind == "array":
        _check_domain(signature, "(%s).domain" % token, lines)
        _reject_if("!(%s).values || (%s).values->size() != std::size_t((%s).domain->size)" % ((token,)*3),
                   lines, "replicated nested Array has an invalid buffer")
        # A nested scope gives each recursive array level its own loop variable.
        lines.append("{")
        index = "element_i_%d" % depth
        lines.append("for (std::int64_t %s = 0; %s < (%s).domain->size; ++%s) {" % (index,index,token,index))
        body = []
        _uniform_element(signature["element"], _unpacked(signature["element"],
                          "(*(%s).values)[std::size_t(%s)]" % (token,index)), body, depth+1)
        lines.extend("  " + line for line in body)
        lines.extend(("}", "}"))
    elif kind == "index_set":
        _check_domain(signature, token, lines)
    elif kind in ("state", "rhs", "scalar_field"):
        # The outer finite index is replicated; each field element retains its actual distributed
        # MultiFab ownership. Its data must not be compared as if every rank held a replica.
        # The authoring signature authenticates the state/support, and packing snapshots storage.
        pass
    elif kind == "scalar":
        _reject_if("!std::isfinite(%s)" % token,lines,"replicated array has a nonfinite element")
        # Compare the exact IEEE payload as signed Int64. This preserves signed zero and
        # cannot silently erase a NaN through floating min/max; non-finite is rejected first.
        _uniform_int("std::bit_cast<std::int64_t>(pops::Real(%s))" % token,lines,
                     "replicated array Float64 bits differ across execution ranks")
    elif kind in ("int64", "bool"):
        _uniform_int("std::int64_t(%s)" % token,lines,"replicated array element differs across execution ranks")
    elif kind == "index":
        _reject_if("!(%s).domain || !(%s).domain->visible() || (%s).offset < 0 || (%s).offset >= (%s).domain->size" % ((token,)*5),
                   lines,"replicated array contains an invalid Index")
        for expression in ("(%s).offset"%token,"(%s).domain->family"%token,
                           "(%s).domain->version"%token,"(%s).domain->size"%token):
            _uniform_int(expression,lines,"replicated Index instance differs across execution ranks")
    else:
        raise TypeError("unsupported replicated array element type %r" % kind)


def _allocate(expression, token, lines):
    """Collective allocation rendezvous before any peer can enter a body collective."""
    from .program_emit_structured_control import _reject_if
    lines += ["bool %s_exhausted = false;"%token, "std::exception_ptr %s_error;"%token,
              "try { %s = %s; }"%(token,expression),
              "catch (const std::bad_alloc&) { %s_exhausted = true; }"%token,
              "catch (const std::length_error&) { %s_exhausted = true; }"%token,
              "catch (...) { %s_error = std::current_exception(); }"%token,
              'pops::collectively_rethrow_exception(%s_error, ctx.prepared_execution_lane(), "finite array allocation");'%token]
    _reject_if("%s_exhausted"%token,lines,"finite array allocation capacity exhausted")


def _check_domain(signature, token, lines):
    from .program_emit_structured_control import _reject_if, _uniform_int
    if signature["ownership"] != "replicated":
        raise NotImplementedError("finite array ownership must be replicated")
    _reject_if("!(%s) || !(%s)->visible() || (%s)->size < 0 || (%s)->size > %d || (%s)->capacity != %d || (%s)->family != %d"
               % (token,token,token,token,signature["capacity"],token,signature["capacity"],token,signature["family"]),
               lines,"invalid finite domain instance")
    for expression in ("(%s)->size"%token,"(%s)->version"%token):
        _uniform_int(expression,lines,"replicated finite domain differs across execution ranks")


def _emit_array_region(program,value,var,model,lines,prelude,block_idx,field_plans,target):
    from .program_emit_structured_control import _check_region_effects, _emit_region, _slots, _copy, _storage_width
    _check_region_effects(value.attrs["body_block"])
    scan = value.op == "finite_scan"
    domain = value.inputs[1] if scan else value.inputs[0]
    signature = value.attrs["array_signature"] if scan else value_signature(value)
    element = signature["element"]
    domain_token = var[domain.id]
    _check_domain(value_signature(domain),domain_token,lines)
    prefix = "finite_%d"%value.id
    buffer = prefix+"_buffer"
    lines.append("std::shared_ptr<std::vector<%s>> %s;"%(finite_cpp_type(element),buffer))
    _allocate("std::make_shared<std::vector<%s>>(std::size_t((%s)->size))"%(finite_cpp_type(element),domain_token),buffer,lines)
    if scan:
        initial = value.inputs[0]
        state_signature = value_signature(initial)
        current = _slots(state_signature,var[initial.id],value.id,"scan_carry",lines)
        pending = _slots(state_signature,var[initial.id],value.id,"scan_next",lines,
                         slot_offset=_storage_width(state_signature))
        _copy(state_signature,current,var[initial.id],lines)
    index = prefix+"_index"
    lines.append("for (std::int64_t %s = 0; %s < (%s)->size; ++%s) {"%(index,index,domain_token,index))
    body = ["const pops_algebra_finite::Index %s_checked{%s, %s};"%(prefix,domain_token,index)]
    sub = dict(var)
    sub[value.attrs["body_index"].id] = prefix+"_checked"
    if scan:
        sub[value.attrs["body_argument"].id] = current
    _emit_region(program,value.attrs["body_block"],sub,model,body,prelude,block_idx,field_plans,target)
    result = sub[value.attrs["body"].id]
    emitted = result["value"] if scan else result
    _uniform_element(element,emitted,body)
    packed = _packed(element, emitted, body, prefix+"_element")
    body.append("(*%s)[std::size_t(%s)] = %s;"%(buffer,index,packed))
    if scan:
        _copy(state_signature,pending,result["state"],body)
        _copy(state_signature,current,pending,body)
    lines.extend("  "+line for line in body)
    lines.append("}")
    # The only publication freezes the private buffer after every element succeeded.
    output = prefix+"_array"
    lines.append("const %s %s{%s, std::move(%s)};"%(finite_cpp_type(signature),output,domain_token,buffer))
    var[value.id] = {"state":current,"values":output} if scan else output


def emit_finite_op(program,value,var,model,lines,prelude,block_idx,field_plans,target):
    if target != "system":
        raise NotImplementedError(
            "finite arrays currently require one Uniform Program invocation; "
            "AMR level invocation and collective ownership are not qualified")
    from .program_emit_structured_control import _reject_if, _uniform_int
    op = value.op
    token = "finite_%d"%value.id
    if op in ("finite_map", "finite_scan"):
        _emit_array_region(program,value,var,model,lines,prelude,block_idx,field_plans,target)
        return
    if op == "finite_index_argument":
        if value.id not in var:
            raise ValueError("finite Index argument is outside its recorded native region")
        return
    if op == "finite_real":
        expression = ("static_cast<pops::Real>(%s)"%var[value.inputs[0].id] if value.inputs else
                      "pops::Real(%s)"%float(value.attrs["value"]).hex())
        lines.append("const pops::Real %s = %s;"%(token,expression))
        _reject_if("!std::isfinite(%s)"%token,lines,"nonfinite Float64 construction")
    elif op == "finite_range_set":
        signature = value_signature(value)
        if signature["ownership"] != "replicated":
            raise NotImplementedError("finite array distributed ownership is not implemented")
        count = var[value.inputs[0].id]
        if len(value.inputs) == 2:
            _check_domain(value_signature(value.inputs[1]), var[value.inputs[1].id], lines)
        _uniform_int(count,lines,"replicated range cardinality differs across execution ranks")
        _reject_if("%s < 0 || %s > %d"%(count,count,signature["capacity"]),lines,
                   "range cardinality exceeds declared capacity")
        lines.append("pops_algebra_finite::DomainRef %s;"%token)
        lines.append("%s = ctx.continuation_new_domain(%s, %d, %d);"
                     %(token,count,signature["capacity"],signature["family"]))
    elif op == "finite_size":
        domain = var[value.inputs[0].id]
        _check_domain(value_signature(value.inputs[0]),domain,lines)
        lines.append("const std::int64_t %s = (%s)->size;"%(token,domain))
    elif op == "finite_same_domain":
        left, right = (var[item.id] for item in value.inputs)
        for item in value.inputs:
            _check_domain(value_signature(item), var[item.id], lines)
        lines.append("const bool %s = (%s == %s);" % (token, left, right))
    elif op == "finite_domain":
        lines.append("const auto %s = (%s).domain;"%(token,var[value.inputs[0].id]))
        _check_domain(value_signature(value),token,lines)
    elif op == "finite_index":
        domain,offset = (var[item.id] for item in value.inputs)
        _check_domain(value_signature(value.inputs[0]),domain,lines)
        _uniform_int(offset,lines,"replicated Index offset differs across execution ranks")
        _reject_if("%s < 0 || %s >= (%s)->size"%(offset,offset,domain),lines,"Index offset is outside its domain")
        lines.append("const pops_algebra_finite::Index %s{%s, %s};"%(token,domain,offset))
    elif op == "finite_index_value":
        index = var[value.inputs[0].id]
        _uniform_element(value_signature(value.inputs[0]),index,lines)
        lines.append("const std::int64_t %s = (%s).offset;"%(token,index))
    elif op == "finite_at":
        array,index = (var[item.id] for item in value.inputs)
        _reject_if("!(%s).domain || !(%s).domain->visible() || !(%s).values || (%s).domain != (%s).domain || (%s).offset < 0 || "
                   "(%s).offset >= (%s).domain->size || (%s).values->size() != std::size_t((%s).domain->size)"
                   %(array,array,array,index,array,index,index,array,array,array),lines,
                   "Array access requires the same live domain instance and an in-range Index")
        lines.append("const auto %s = (*(%s).values)[std::size_t((%s).offset)];"%(token,array,index))
        signature = value_signature(value)
        unpacked = _unpacked(signature,token)
        def has_field(item):
            if item["vtype"] in ("state", "rhs", "scalar_field"):
                return True
            return item["vtype"] in ("product", "variant") and any(has_field(child) for _, child in item["members"])
        if has_field(signature):
            # A field pointer must not outlive the array's shared ownership after a branch/capture
            # ends. Materialize the selected fields into Program-owned storage before projection.
            from .program_emit_structured_control import _slots, _copy
            output = _slots(signature, unpacked, value.id, "array_read", lines)
            _copy(signature, output, unpacked, lines)
            var[value.id] = output
        else:
            var[value.id] = unpacked
        return
    else:
        raise NotImplementedError("finite array operation %r"%op)
    var[value.id] = token
