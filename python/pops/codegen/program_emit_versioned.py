"""Lower owned continuation values and dynamic capabilities to the actual Program context."""
from __future__ import annotations

import json
from typing import Any

from pops.time._program.structured_control import value_signature
from pops.time._program.versioned_values import CAPABILITY_TYPES, _digest, _at
from pops.time._program.versioned_values import VERSIONED_OPS as VERSIONED_OPS
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
    """Construct the native tree in postorder, without nested C++ initializers.

    The temporary vector is only an authoring artifact's construction workspace:
    children are moved into their parent and the native schema remains the sole
    installed authority. No C++ template type grows with mathematical depth.
    """
    roles = {} if roles is None else roles
    lines = ["[&]() {", "std::vector<%sContinuationType> schema_nodes;" % _NS]
    # Frames borrow signatures and keep only their parent edge, not an O(depth)
    # tuple at each level. A full path is materialized only for a Field contract.
    frames = [{"signature": signature, "parent": None, "edge": None, "children": [], "next": 0}]
    active = {id(signature)}
    next_node = 0
    while frames:
        frame = frames[-1]
        node = frame["signature"]
        kind = node["vtype"]
        members: Any = (node["members"] if kind in ("product", "variant") else
                   (("[]", node["element"]),) if kind == "array" else ())
        if frame["next"] < len(members):
            edge, child = members[frame["next"]]
            frame["next"] += 1
            if id(child) in active:
                raise ValueError("continuation schema is cyclic")
            active.add(id(child))
            frames.append({"signature": child, "parent": frame, "edge": edge,
                           "children": [], "next": 0})
            continue
        family, capacity, owner, components, contract = -1, 0, -1, 0, ""
        if kind in ("array", "index", "index_set"):
            family, capacity = node["family"], node["capacity"]
        if kind in ("state", "rhs", "scalar_field"):
            owner = block_idx[node["block"]]
            components = len(node["space"].components)
            edges, cursor = [], frame
            while cursor["parent"] is not None:
                edges.append(cursor["edge"])
                cursor = cursor["parent"]
            contract = _field_contract(node, (*path, *reversed(edges)), roles)
        native = {"bool": "Boolean", "int64": "Int64", "scalar": "Float64",
                  "product": "Product", "variant": "Variant", "array": "Array",
                  "index_set": "Domain", "index": "Index", "state": "Field",
                  "rhs": "Field", "scalar_field": "Field"}[kind]
        if kind in ("array", "index", "index_set") and node.get("ownership") == "contiguous_distributed":
            native = "Distributed" + native
        dependencies = []
        bundle = node.get("bundle") is not None
        if bundle:
            dependency_contract = validate_bundle_signature(node)
            for dependent, domain in dependency_contract["bindings"]:
                dependencies.append(_NS + "ContinuationDomainBinding{" + _dependent_indices(node, dependent)
                                    + "," + _indices(node, domain) + "}")
        labels = ",".join(json.dumps(name) for name, _ in members) if kind in ("product", "variant") else ""
        lines.append("schema_nodes.emplace_back(%sContinuationKind::%s," % (_NS, native)
                     + "std::vector<%sContinuationType>{},std::vector<std::string>{%s}," % (_NS, labels)
                     + "%d,%d,%d,%d,%s,std::vector<%sContinuationDomainBinding>{%s},%s);" % (
                         family, capacity, owner, components, json.dumps(contract), _NS,
                         ",".join(dependencies), "true" if bundle else "false"))
        if frame["children"]:
            lines.append("schema_nodes.back().children.reserve(%d);" % len(frame["children"]))
            for child in frame["children"]:
                lines.append("schema_nodes.back().children.push_back(std::move(schema_nodes[%d]));" % child)
        completed = next_node
        next_node += 1
        active.remove(id(node))
        frames.pop()
        if frames:
            frames[-1]["children"].append(completed)
    lines.extend(["return std::move(schema_nodes.back());", "}()"])
    return "\n".join(lines)


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


def _limits(capacity):
    return _NS + "finite_structural::Limits{" + ",".join([str(capacity)] * 3) + "}"


def _local_storage(signature, tokens, body, prefix, capacity):
    """Build borrowed flat cells locally; no collective is legal inside this preparation."""
    from .program_emit_finite_arrays import finite_cpp_type
    from .program_structural_storage import slots

    def leaf(sig, token):
        if sig["vtype"] in ("state", "rhs", "scalar_field"):
            return "std::cref(%s)" % token
        if sig["vtype"] in ("bool", "int64", "scalar"):
            return "%s(%s)" % (finite_cpp_type(sig), token)
        return _NS + "finite_structural::source_cell(%s)" % token

    if signature["vtype"] not in ("product", "variant"):
        return "std::any(%s)" % leaf(signature, tokens)
    layout: list[Any] = list(slots(signature, tokens))
    output = prefix + "_storage"
    body += ["if (std::uint64_t(%d) > std::uint64_t(%s)/sizeof(std::any)) "
             'throw std::length_error("finite structural source exceeds declared storage budget");'
             % (len(layout), capacity),
             "pops_algebra_finite::Record %s{std::vector<std::any>(%d)};" % (output, len(layout))]
    active: dict[tuple[Any, ...], str] = {(): "true"}
    for index, sig, token, guards in layout:
        for depth in range(1, len(guards) + 1):
            path = guards[:depth]
            if path not in active:
                slot, arm = path[-1]
                name = "%s_active_%d" % (prefix, len(active))
                body.append("const bool %s = %s && std::any_cast<std::int64_t>(%s.values.at(%d)) == %d;"
                            % (name, active[path[:-1]], output, slot, arm))
                active[path] = name
        if sig["vtype"] == "variant":
            value = "std::int64_t(%s)" % token["tag"]
            body.append("if (%s && (%s < 0 || %s >= %d)) "
                        'throw std::invalid_argument("invalid structural Variant discriminant");'
                        % (active[guards], value, value, len(sig["members"])))
        else:
            value = leaf(sig, token)
        body.append("if (%s) %s.values[%d] = %s;" % (active[guards], output, index, value))
    return "std::any(std::move(%s))" % output


def _pack(signature, token, body, prefix, block_idx, roles, path=(), *, capacity):
    storage = _local_storage(signature, token, body, prefix, capacity)
    schema = schema_cpp(signature, block_idx, roles=roles, path=path)
    return (_NS + "finite_structural::pack<" + _DIM + ">(" + schema + "," + storage
            + "," + _limits(capacity)
            + ",[&](const auto& source, const auto& destination, const auto& lifetime, const auto&... metadata) { "
              "return ctx.continuation_bind_field_provenance(source, destination, lifetime, metadata...); })")


def _unpack(signature, reference, body, prefix, *, capacity, block_idx=None, roles=None):
    """Project an authenticated native value through bounded iterative storage conversion."""
    from .program_emit_finite_arrays import finite_cpp_type
    schema = schema_cpp(signature, {} if block_idx is None else block_idx, roles=roles)
    expression = (_NS + "finite_structural::unpack<" + _DIM + ">(" + schema + "," + reference
                  + "," + _limits(capacity)
                  + ",[&](const auto& source, const auto& destination, const auto& lifetime, const auto&... metadata) { "
                    "return ctx.continuation_bind_field_provenance(source, destination, lifetime, metadata...); })")
    return "std::any_cast<%s>(%s)" % (finite_cpp_type(signature), expression)


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


def pack_local(signature, tokens, lines, prefix, block_indices, *, capacity, roles=None):
    """Generate only local allocation/copy work; caller must converge errors before collectives."""
    return _pack(signature, tokens, lines, prefix, block_indices, {} if roles is None else roles,
                 capacity=capacity)


def pack_owned(signature, tokens, lines, prefix, block_indices, *, capacity, roles=None):
    """Generate an owned ValueRef with one checked local preparation/fence/consensus boundary."""
    body = []
    expression = pack_local(signature, tokens, body, prefix, block_indices, capacity=capacity, roles=roles)
    _checked(lines, prefix, _REF, body, expression, message="continuation value packing failed")
    return prefix


def unpack_owned(signature, reference, lines, prefix, *, capacity, block_indices=None):
    """Materialize an immutable native value into owned, ordinary Program value tokens."""
    from .program_emit_finite_arrays import finite_cpp_type, _unpacked
    body = []
    expression = _unpack(signature, reference, body, prefix + "_unpacked",
                         capacity=capacity, block_idx=block_indices)
    _checked(lines, prefix, finite_cpp_type(signature), body, expression,
             message="continuation capability read or owned materialization failed")
    return _unpacked(signature, prefix)


def emit_versioned_op(program, value, var, model, lines, prelude, block_idx, field_plans, target):
    if value.op in BUNDLE_OPS:
        return _emit_bundle_op(value, var, lines, block_idx)
    from pops.time._program.continuation_forest import FOREST_OPS, forest_node
    forest = forest_node(program) if target == "amr_system" else None
    if target != "system" and forest is None:
        raise NotImplementedError("AMR continuation requires an explicit complete acceptance forest")
    from .program_emit_finite_arrays import finite_cpp_type, _unpacked
    signature = value_signature(value)
    op, token = value.op, "continuation_%d" % value.id
    args = [var[item.id] for item in value.inputs]
    authority = "ctx.continuation_authority()"
    capacity = "ctx.continuation_byte_capacity()"
    if op in FOREST_OPS and forest is None:
        raise ValueError("AMR forest operation lacks its closed parent/child graph")
    if op in ("continuation_source_topology", "continuation_target_topology"):
        dimension = signature["dimension"]
        block = block_idx[signature["field_signature"]["block"]]
        lines.append('static_assert(pops::kNativeDimension == %d, "regrid topology dimension mismatch");' % dimension)
        method = "continuation_source_topology" if op == "continuation_source_topology" else "continuation_target_topology"
        expression = "ctx.%s(%d, %s, %s)" % (method, block, args[1], args[0])
        _checked(lines, token, finite_cpp_type(signature), [], expression,
                 message="authenticated regrid topology view")
        var[value.id] = token
        return
    if op == "continuation_regular_phase":
        expression = "!ctx.continuation_regrid_phase()"
    elif op == "continuation_regrid_source_count":
        expression = "ctx.continuation_regrid_source_count()"
    elif op in ("continuation_regrid_base", "continuation_regrid_parent_open"):
        expression = ("ctx.continuation_regrid_base(%s)" % args[0]
                      if op == "continuation_regrid_base" else "ctx.continuation_read()")
        lines.append("const auto %s = %s;" % (token, expression))
        lines.append("ctx.continuation_control_consensus();")
        var[value.id] = token
        return
    elif op == "continuation_regrid_source":
        reference = token + "_reference"
        lines.append("const auto %s = ctx.continuation_regrid_source(%s);" % (reference, args[0]))
        expression = reference
        body = []
        expression = _unpack(signature, expression, body, token + "_unpacked", capacity=capacity,
                             block_idx=block_idx, roles=_roles(value.attrs["schema"]))
        _checked(lines, token, finite_cpp_type(signature), body, expression,
                 message="immutable complete regrid source image")
        var[value.id] = _unpacked(signature, token)
        return
    elif op in ("continuation_forest_child_phase", "continuation_forest_level",
              "continuation_forest_level_count"):
        expression = {"continuation_forest_child_phase": "!ctx.continuation_parent_phase()",
                      "continuation_forest_level": "std::int64_t(ctx.level())",
                      "continuation_forest_level_count": "std::int64_t(ctx.program_resource_topology().levels)"}[op]
    elif op == "continuation_forest_interval":
        var[value.id] = {"start": "ctx.continuation_macro_start()", "end": "ctx.continuation_macro_end()",
                         "duration": "(ctx.continuation_macro_end()-ctx.continuation_macro_start())"}
        return
    elif op in ("continuation_synchronized_fields", "continuation_regrid_templates"):
        var[value.id] = {"field_%d" % i: "ctx.state(%d)" % block_idx[state_ref.block_ref]
                         for i, (state_ref, _) in enumerate(value.attrs["schema"]["fields"])}
        return
    elif op in ("continuation_working_value", "continuation_child_reports"):
        expression = ("ctx.continuation_working_value()" if op == "continuation_working_value"
                      else "ctx.continuation_child_reports(%s)" % args[0])
        body = []
        expression = _unpack(signature, expression, body, token + "_unpacked", capacity=capacity,
                             block_idx=block_idx,
                             roles=_roles(value.attrs["schema"]) if op == "continuation_working_value" else None)
        _checked(lines, token, finite_cpp_type(signature), body, expression,
                 message="AMR complete continuation view failed")
        var[value.id] = _unpacked(signature, token)
        return
    elif op == "continuation_sync_open":
        lines.append("ctx.begin_level_continuation_sync(ctx.continuation_macro_end());")
        expression = "ctx.continuation_read()"
    elif op in ("continuation_open", "continuation_parent_open"):
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
        destinations = ((lines, token + "_configuration"),)
        if target == "system":
            destinations = ((prelude, token + "_installed_configuration"), *destinations)
        for destination, name in destinations:
            destination.append("auto %s = ctx.continuation_local_phase([&] { return %s; }, "
                               '"continuation configuration construction");' % (name, configuration))
            destination.append("ctx.continuation_configure(std::move(%s));" % name)
        lines.append("if (!ctx.continuation_initialized()) {")
        initial = []
        owned = pack_owned(schema["read"], args[0], initial, token + "_initial", block_idx, capacity=capacity, roles=_roles(schema))
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
        source_type = value_signature(value.inputs[0])
        roles = (_roles(source_type["schema"]) if op == "continuation_read" else
                 _roles(source_type["base"]["schema"], ("continuation",))
                 if op == "continuation_candidate_value" else {})
        expression = _unpack(signature, reference, body, token + "_unpacked", capacity=capacity,
                             block_idx=block_idx, roles=roles)
        _checked(lines, token, finite_cpp_type(signature), body, expression,
                 message="continuation capability read or owned materialization failed")
        var[value.id] = _unpacked(signature, token)
        return
    elif op == "continuation_freeze":
        schema = signature["base"]["schema"]
        proposal = pack_owned(value_signature(value.inputs[1]), args[1], lines, token + "_proposal", block_idx, capacity=capacity)
        envelope = pack_owned(signature["envelope"], args[2], lines, token + "_envelope", block_idx, capacity=capacity,
                                roles=_roles(schema, ("continuation",)))
        dependencies = ",ctx.continuation_child_dependencies()" if forest is not None else ""
        expression = "%s.freeze(%s,%s,%s%s)" % (authority, args[0], proposal, envelope, dependencies)
    elif op == "continuation_candidate_base":
        expression = "(%s).base()" % args[0]
    elif op == "continuation_attest":
        from .program_emit_structured_control import _uniform_int
        _uniform_int("std::int64_t(%s)" % args[2], lines, "continuation assessment decision differs across ranks")
        checks = pack_owned(signature["checks"], args[1], lines, token + "_checks", block_idx, capacity=capacity)
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
        var[value.id] = unpack_owned(signature, reference, lines, token, capacity=capacity, block_indices=block_idx)
        return
    elif op in ("continuation_publish", "continuation_parent_publish", "continuation_sync_publish",
                "continuation_regrid_child_publish", "continuation_regrid_parent_publish"):
        # Context publication contains its own ordered collective phases; no outer local catch.
        method = ("prepare_level_continuation_sync" if op == "continuation_sync_publish" else
                  "prepare_continuation_regrid_child" if op == "continuation_regrid_child_publish" else
                  "continuation_publish")
        lines.append("ctx.%s(%s);" % (method, args[0]))
        if forest is not None:
            var[value.id] = args[0]
            return
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
    capacity = value.attrs["byte_capacity"]
    for index in range(len(value.inputs)):
        owned.append(pack_owned(signature, inputs[index], lines, token + "_input%d" % index, block_idx, capacity=capacity))
    body = ["auto bundle_schema = %s;" % schema_cpp(signature, block_idx),
            "const auto no_fields = [](int) -> const pops::MultiFab<%s>& {" % _DIM,
            '  throw std::logic_error("Bundle has no legacy field prototype"); };',
            "%sContinuationCodec<%s> codec(std::move(bundle_schema),%s,UINT64_C(%d),no_fields);" % (
                _NS, _DIM, json.dumps("bundle/" + _digest(signature)), capacity)]
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
