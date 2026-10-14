"""Lower owned continuation values and dynamic capabilities to the actual Program context."""
from __future__ import annotations

import json

from pops.time._program.structured_control import value_signature
from pops.time._program.versioned_values import CAPABILITY_TYPES, VERSIONED_OPS, _digest, _at
from pops.bundles import BUNDLE_OPS, validate_bundle_signature

_NS = "pops::runtime::program::"
_DIM = "pops::kNativeDimension"
_VALUE = _NS + "ContinuationValue<" + _DIM + ">"
_REF = _NS + "ContinuationValueRef<" + _DIM + ">"
_AUTHORITY = _NS + "ContinuationAuthority<" + _DIM + ">"


def capability_cpp_type(signature):
    return _AUTHORITY + "::" + {"continuation_base": "Base", "continuation_candidate": "Candidate",
                                "continuation_assessment": "Assessment", "continuation_accepted": "Accepted"}[signature["vtype"]]


def _indices(signature, path):
    indices = []
    for key in path:
        members = tuple(signature["members"])
        index = tuple(name for name, _ in members).index(key)
        indices.append(index)
        signature = members[index][1]
    return "{" + ",".join(map(str, indices)) + "}"


def _dependent_indices(signature, path):
    indices = []
    for key in path:
        if signature["vtype"] == "array" and key == "[]":
            indices.append(_NS + "continuation_each")
            signature = signature["element"]
        else:
            members = tuple(signature["members"])
            index = tuple(name for name, _ in members).index(key)
            indices.append(str(index))
            signature = members[index][1]
    return "{" + ",".join(indices) + "}"


def _roles(schema, prefix=()):
    return {(*prefix, *path): _digest((schema["identity"], path)) for _, path in schema["fields"]}


def _field_contract(signature, path, roles):
    return roles.get(path, _digest(signature))


def schema_cpp(signature, block_idx, *, roles=None, path=()):
    roles = {} if roles is None else roles
    kind = signature["vtype"]
    names, children = [], []
    family, capacity, owner, components, contract = -1, 0, -1, 0, ""
    if kind in ("product", "variant"):
        for name, child in signature["members"]:
            names.append(json.dumps(name))
            children.append(schema_cpp(child, block_idx, roles=roles, path=(*path, name)))
    elif kind == "array":
        children.append(schema_cpp(signature["element"], block_idx, roles=roles, path=(*path, "[]")))
    if kind in ("array", "index", "index_set"):
        family, capacity = signature["family"], signature["capacity"]
    if kind in ("state", "scalar_field"):
        owner = block_idx[signature["block"]]
        components = len(signature["space"].components)
        contract = _field_contract(signature, path, roles)
    native = {"bool": "Boolean", "int64": "Int64", "scalar": "Float64", "product": "Product",
              "variant": "Variant", "array": "Array", "index_set": "Domain", "index": "Index",
              "state": "Field", "scalar_field": "Field"}[kind]
    if kind in ("array", "index", "index_set") and signature.get("ownership") == "contiguous_distributed":
        native = "Distributed" + native
    dependencies = []
    bundle = signature.get("bundle") is not None
    if bundle:
        dependency_contract = validate_bundle_signature(signature)
        for dependent, domain in dependency_contract["bindings"]:
            dependencies.append(_NS + "ContinuationDomainBinding{" + _dependent_indices(signature, dependent)
                                + "," + _indices(signature, domain) + "}")
    return (_NS + "ContinuationType{" + _NS + "ContinuationKind::" + native + ",{" + ",".join(children)
            + "},{" + ",".join(names) + "},%d,%d,%d,%d,%s,{%s},%s}" % (
                family, capacity, owner, components, json.dumps(contract),
                ",".join(dependencies), "true" if bundle else "false"))


def configuration_cpp(schema, block_idx, *, distributed_storage=(0, 0)):
    from pops.time.points import point_clock
    fields = []
    for state_ref, path in schema["fields"]:
        read, write = _at(schema["read"], path), _at(schema["write"], path)
        fields.append(_NS + "ContinuationFieldBinding{" + _indices(schema["read"], path)
                      + ",%d,%s,%s,%s}" % (block_idx[state_ref.block_ref],
                          json.dumps(point_clock(read["point"], "continuation binding").qualified_id),
                          json.dumps(_digest(read)), json.dumps(_digest(write))))
    return (_NS + "ContinuationConfiguration{" + schema_cpp(schema["read"], block_idx, roles=_roles(schema))
            + ",%s,%d,%s,%s,{%s},%d,%d}" % (json.dumps(schema["identity"]), schema["byte_capacity"],
              _indices(schema["read"], schema["time_path"]), _indices(schema["read"], schema["step_path"]),
              ",".join(fields), *distributed_storage))


def _pack(signature, token, body, prefix, block_idx, roles, path=()):
    kind = signature["vtype"]
    if signature.get("ownership") == "contiguous_distributed":
        from .program_emit_distributed_arrays import pack
        return pack(signature, token, body, prefix)
    if kind == "product":
        children = [_pack(child, token[name], body, prefix + "_%d" % n, block_idx, roles, (*path, name))
                    for n, (name, child) in enumerate(signature["members"])]
        return _VALUE + "::product({" + ",".join(children) + "})"
    if kind == "variant":
        out = prefix + "_variant"
        body.append(_REF + " " + out + ";")
        body.append("switch (%s) {" % token["tag"])
        for n, (name, child) in enumerate(signature["members"]):
            arm = []
            value = _pack(child, token["payloads"][name], arm, prefix + "_arm%d" % n,
                          block_idx, roles, (*path, name))
            body.append("case %d: {" % n)
            body.extend("  " + line for line in arm)
            body.append("  %s = %s::variant(%d,%s); break; }" % (out, _VALUE, n, value))
        body += ['default: throw std::logic_error("invalid continuation Variant tag");', "}"]
        return out
    if kind == "array":
        from .program_emit_finite_arrays import _unpacked
        values = prefix + "_elements"
        body += ['if (!(%s).domain || !(%s).values) throw std::logic_error("invalid finite Array");' % (token, token),
                 "(%s).domain->require_live();" % token,
                 "std::vector<%s> %s;" % (_REF, values),
                 "%s.reserve((%s).values->size());" % (values, token),
                 "for (const auto& item_%s : *(%s).values) {" % (prefix, token)]
        element_body = []
        element = _pack(signature["element"], _unpacked(signature["element"], "item_" + prefix),
                        element_body, prefix + "_item", block_idx, roles, (*path, "[]"))
        body.extend("  " + line for line in element_body)
        body += ["  %s.push_back(%s);" % (values, element), "}"]
        return _VALUE + "::array((%s).domain,std::move(%s))" % (token, values)
    if kind == "index_set":
        return _VALUE + "::domain(%s)" % token
    if kind == "index":
        return _VALUE + "::index((%s).domain,(%s).offset)" % (token, token)
    if kind in ("state", "scalar_field"):
        return _VALUE + "::field(%s,%d,%s)" % (token, block_idx[signature["block"]],
                    json.dumps(_field_contract(signature, path, roles)))
    return _VALUE + "::%s(%s)" % ({"scalar": "real", "int64": "integer", "bool": "boolean"}[kind], token)


def _unpack(signature, reference, body, prefix):
    """Return a packed C++ value with owned field copies and shared restored domain aliases."""
    from .program_emit_finite_arrays import finite_cpp_type
    kind = signature["vtype"]
    if signature.get("ownership") == "contiguous_distributed":
        from .program_emit_distributed_arrays import unpack
        return unpack(signature, reference)
    if kind == "product":
        return "std::make_tuple(%s)" % ",".join(_unpack(child, "(%s)->members().at(%d)" % (reference, n),
                            body, prefix + "_%d" % n) for n, (_, child) in enumerate(signature["members"]))
    if kind == "variant":
        out = prefix + "_variant"
        body += [finite_cpp_type(signature) + " " + out + ";", "switch ((%s)->arm()) {" % reference]
        for n, (_, child) in enumerate(signature["members"]):
            arm = []
            payload = _unpack(child, "(%s)->members().at(0)" % reference, arm, prefix + "_arm%d" % n)
            body.append("case %d: {" % n)
            body.extend("  " + line for line in arm)
            body.append("  %s.emplace<%d>(%s); break; }" % (out, n, payload))
        body += ['default: throw std::logic_error("invalid owned continuation Variant tag");', "}"]
        return out
    if kind == "array":
        element_type = finite_cpp_type(signature["element"])
        values = prefix + "_elements"
        body += ["std::vector<%s> %s;" % (element_type, values),
                 "%s.reserve((%s)->members().size());" % (values, reference),
                 "for (const auto& item_%s : (%s)->members()) {" % (prefix, reference)]
        elements = []
        value = _unpack(signature["element"], "item_" + prefix, elements, prefix + "_item")
        body.extend("  " + line for line in elements)
        body += ["  %s.push_back(%s);" % (values, value), "}"]
        return "%s{(%s)->domain_ref(),std::make_shared<const std::vector<%s>>(std::move(%s))}" % (
            finite_cpp_type(signature), reference, element_type, values)
    if kind == "index_set":
        return "(%s)->domain_ref()" % reference
    if kind == "index":
        return "pops_algebra_finite::Index{(%s)->domain_ref(),(%s)->index_offset()}" % (reference, reference)
    if kind in ("state", "scalar_field"):
        return "std::make_shared<pops::MultiFab<%s>>((%s)->field())" % (_DIM, reference)
    accessor = {"scalar": "real", "int64": "integer", "bool": "boolean"}[kind]
    expression = "(%s)->%s()" % (reference, accessor)
    if kind == "scalar":
        # The structural codec is binary64. This executable profile refuses a narrower native Real.
        body.append('static_assert(sizeof(pops::Real)==sizeof(double), "continuation Float64 requires binary64 Real");')
    return expression


def _checked(lines, token, cpp_type, body, expression, *, message):
    from .program_emit_structured_control import _reject_if
    error = token + "_error"
    lines += ["%s %s{};" % (cpp_type, token), "std::exception_ptr %s;" % error, "try {"]
    lines.extend("  " + line for line in body)
    if expression is not None:
        lines.append("  %s = %s;" % (token, expression))
    lines += ["} catch (...) { %s=std::current_exception(); }" % error,
              "try { Kokkos::fence(); } catch (...) { if (!%s) %s=std::current_exception(); }" % (error, error)]
    _reject_if("bool(%s)" % error, lines, message)


def pack_local(signature, tokens, lines, prefix, block_indices, *, roles=None):
    """Generate only local allocation/copy work; caller must converge errors before collectives."""
    return _pack(signature, tokens, lines, prefix, block_indices, {} if roles is None else roles)


def pack_owned(signature, tokens, lines, prefix, block_indices, *, roles=None):
    """Generate an owned ValueRef with one checked local preparation/fence/consensus boundary."""
    body = []
    expression = pack_local(signature, tokens, body, prefix, block_indices, roles=roles)
    _checked(lines, prefix, _REF, body, expression, message="continuation value packing failed")
    return prefix


def unpack_owned(signature, reference, lines, prefix):
    """Materialize an immutable native value into owned, ordinary Program value tokens."""
    from .program_emit_finite_arrays import finite_cpp_type, _unpacked
    body = []
    expression = _unpack(signature, reference, body, prefix + "_unpacked")
    _checked(lines, prefix, finite_cpp_type(signature), body, expression,
             message="continuation capability read or owned materialization failed")
    return _unpacked(signature, prefix)


def emit_versioned_op(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    if value.op in BUNDLE_OPS:
        return _emit_bundle_op(value, var, lines, block_idx)
    if target != "system":
        raise NotImplementedError("continuation persistent publication currently requires fixed-topology Uniform")
    from .program_emit_finite_arrays import finite_cpp_type, _unpacked
    signature = value_signature(value)
    op, token = value.op, "continuation_%d" % value.id
    args = [var[item.id] for item in value.inputs]
    authority = "ctx.continuation_authority()"
    if op == "continuation_open":
        schema = value.attrs["schema"]
        if prelude is None:
            raise ValueError("continuation schema requires the generated artifact installer")
        # Argument construction allocates: converge its failures before entering Context collectives.
        from .program_lowerability import all_ops
        policies = {(v.attrs["storage_bytes"], v.attrs["storage_versions"])
                    for v in all_ops(program) if "storage_bytes" in v.attrs}
        if len(policies) > 1:
            raise ValueError("Continuation has conflicting distributed storage policies")
        configuration = configuration_cpp(schema, block_idx,
            distributed_storage=next(iter(policies), (0, 0)))
        for destination, name in ((prelude, token + "_installed_configuration"),
                                  (lines, token + "_configuration")):
            destination.append("auto %s = ctx.continuation_local_phase([&] { return %s; }, "
                               '"continuation configuration construction");' % (name, configuration))
            destination.append("ctx.continuation_configure(std::move(%s));" % name)
        lines.append("if (!ctx.continuation_initialized()) {")
        initial = []
        owned = pack_owned(schema["read"], args[0], initial, token + "_initial", block_idx, roles=_roles(schema))
        initial.append("ctx.continuation_initialize(%s);" % owned)
        lines.extend("  " + line for line in initial)
        lines.append("}")
        expression = "ctx.continuation_read()"
    elif op in ("continuation_read", "continuation_candidate_value", "continuation_proposal", "continuation_assessment_checks"):
        getter = {"continuation_read": "root", "continuation_candidate_value": "root",
                  "continuation_proposal": "proposal", "continuation_assessment_checks": "checks"}[op]
        body = []
        reference = token + "_reference"
        body.append("const auto& %s = (%s).%s();" % (reference, args[0], getter))
        expression = _unpack(signature, reference, body, token + "_unpacked")
        _checked(lines, token, finite_cpp_type(signature), body, expression,
                 message="continuation capability read or owned materialization failed")
        var[value.id] = _unpacked(signature, token)
        return
    elif op == "continuation_freeze":
        schema = signature["base"]["schema"]
        proposal = pack_owned(value_signature(value.inputs[1]), args[1], lines, token + "_proposal", block_idx)
        envelope = pack_owned(signature["envelope"], args[2], lines, token + "_envelope", block_idx,
                                roles=_roles(schema, ("continuation",)))
        expression = "%s.freeze(%s,%s,%s)" % (authority, args[0], proposal, envelope)
    elif op == "continuation_candidate_base":
        expression = "(%s).base()" % args[0]
    elif op == "continuation_attest":
        from .program_emit_structured_control import _uniform_int
        _uniform_int("std::int64_t(%s)" % args[2], lines, "continuation assessment decision differs across ranks")
        checks = pack_owned(signature["checks"], args[1], lines, token + "_checks", block_idx)
        expression = "%s.attest(%s,%s,%s,%s)" % (authority, args[0], checks, args[2],
                                               json.dumps(value.attrs["assessment_identity"]))
    elif op == "continuation_assessment_ok":
        expression = "(%s).accepted()" % args[0]
    elif op == "continuation_accept":
        expression = "%s.accept(%s,%s)" % (authority, args[0], args[1])
    elif op == "continuation_accepted_candidate":
        expression = "(%s).candidate()" % args[0]
    elif op == "continuation_child_base":
        expression = "%s.provisional_base(%s)" % (authority, args[0])
    elif op == "continuation_transform":
        _checked(lines, token + "_supersede", "bool", ["%s.supersede(%s);" % (authority, args[0])],
                 "true", message="continuation transformation supersedes an invalid candidate")
        expression = args[1]
    elif op == "continuation_consume_increment":
        reference = token + "_reference"
        _checked(lines, reference, _REF, [], "%s.consume_increment(%s)" % (authority, args[0]),
                 message="child increment was already consumed or belongs to another scope")
        lines.append("ctx.continuation_control_consensus();")
        var[value.id] = unpack_owned(signature, reference, lines, token)
        return
    elif op == "continuation_publish":
        # Context publication contains its own ordered collective phases; no outer local catch.
        lines.append("ctx.continuation_publish(%s);" % args[0])
        expression = args[0]
    else:
        raise NotImplementedError("unknown continuation operation " + op)
    cpp_type = capability_cpp_type(signature) if signature["vtype"] in CAPABILITY_TYPES else finite_cpp_type(signature)
    _checked(lines, token, cpp_type, [], expression, message="continuation capability operation failed")
    if signature["vtype"] in CAPABILITY_TYPES:
        lines.append("ctx.continuation_control_consensus();")
    var[value.id] = token


def _emit_bundle_op(value, var, lines, block_idx):
    """Validate the same native immutable values which the following graph consumes."""
    signature = value_signature(value)
    validate_bundle_signature(signature)
    token = "bundle_%d" % value.id
    inputs = [var[item.id] for item in value.inputs]
    owned = []
    for index, item in enumerate(value.inputs):
        owned.append(pack_owned(signature, inputs[index], lines, token + "_input%d" % index, block_idx))
    body = ["auto bundle_schema = %s;" % schema_cpp(signature, block_idx),
            "const auto no_fields = [](int) -> const pops::MultiFab<%s>& {" % _DIM,
            '  throw std::logic_error("Bundle has no legacy field prototype"); };',
            "%sContinuationCodec<%s> codec(std::move(bundle_schema),%s,UINT64_C(67108864),no_fields);" % (
                _NS, _DIM, json.dumps("bundle/" + _digest(signature)))]
    body += ["codec.validate(%s);" % reference for reference in owned]
    if value.op == "bundle_remap_validate":
        before, after = owned
        source = _indices(signature, value.attrs["source_instance"])
        current = _indices(signature, ("instance",))
        getter = ("distributed_domain_ref" if _at(signature, ("instance",)).get("ownership") == "contiguous_distributed"
                  else "domain_ref")
        body += ["const auto old_instance = %scontinuation_at<%s>(%s,%s)->%s();" % (
                    _NS, _DIM, before, current, getter),
                 "const auto new_instance = %scontinuation_at<%s>(%s,%s)->%s();" % (
                    _NS, _DIM, after, current, getter),
                 "const auto source_instance = %scontinuation_at<%s>(%s,%s)->%s();" % (
                    _NS, _DIM, after, source, getter),
                 'if (new_instance == old_instance || source_instance != old_instance) '
                 'throw std::invalid_argument("remap must create a new instance and retain its exact source");']
    _checked(lines, token + "_validated", "bool", body, "true",
             message="Bundle dependency or remap validation failed")
    var[value.id] = inputs[-1]
