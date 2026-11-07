"""Type/effect-directed Kokkos realization of pure finite map and ordered regions.

This emits kernels into the existing Program DSO. It neither identifies a model
nor changes the finite value/continuation ABI. Replicated Arrays of scalar and
Product elements are device columns (DM7): kernels read and write them in place,
and only O(1) control scalars reach the host (DM8). Unsupported regions keep the
ordinary host realization, with an explicit placement reason in generated code.
"""

from __future__ import annotations

import json

from pops.identity.scalar import scalar_cpp
from pops.time._program.structured_control import value_signature

_NS = "pops::runtime::program::finite_kokkos::"
_TYPES = {"scalar": "pops::Real", "int64": "std::int64_t", "bool": "bool"}
_OPS = frozenset(
    (
        "finite_index_argument",
        "finite_index",
        "finite_index_value",
        "finite_size",
        "finite_domain",
        "finite_same_domain",
        "finite_at",
        "finite_real",
        "int64_constant",
        "bool_constant",
        "bool_op",
        "int64_op",
        "int64_compare",
        "int64_cast",
        "scalar_op",
        "scalar_expression",
        "scalar_isfinite",
        "compare",
        "product",
        "product_get",
        "branch",
        "provenance_bind",
        "distributed_index_argument",
        "distributed_index",
        "distributed_gid",
        "distributed_local_at",
        "distributed_rank",
        "distributed_size",
        "distributed_value_domain",
    )
)


def _leaves(signature, token):
    from .program_emit_structured_control import _leaves as leaves

    return [(sig, value) for _, sig, value in leaves(signature, token)]


def _tree(signature, callback, path=()):
    from .program_emit_structured_control import _leaves as leaves, _repack

    values = {where: callback(sig, where) for where, sig, _ in leaves(signature, None, path)}
    return _repack(signature, values, path)


def _flat(signature):
    from .program_emit_structured_control import _leaves as leaves

    return all(sig["vtype"] in _TYPES for _, sig, _ in leaves(signature, None))


def device_array(signature):
    """A replicated Array of scalar/Product elements is stored as device columns."""
    return (
        signature["vtype"] == "array"
        and signature.get("ownership") != "contiguous_distributed"
        and _flat(signature["element"])
    )


def leaf_count(signature):
    return sum(1 for _ in _leaves(signature, None))


def storage_type(signature):
    """Column value type of a scalar leaf; Bool columns hold unsigned char."""
    return "unsigned char" if signature["vtype"] == "bool" else _TYPES[signature["vtype"]]


_ARRAY_LABEL = "pops_finite_array:"
_CONTROL_LABEL = "pops_finite_control:"
_DEVICE_STORAGE = "device columns in the default memory space; no host staging"


def result_storage(signature, backend):
    """Placement text for the storage of a region's Array result."""
    if not device_array(signature):
        return "host structural cells"
    if backend == "host":
        return "host evaluation; device columns published once (counted host phase)"
    return _DEVICE_STORAGE


def _capture_type(signature):
    kind = signature["vtype"]
    if _flat(signature):
        return True
    return signature.get("ownership") == "replicated" and (
        kind in ("index_set", "index") or (kind == "array" and _flat(signature["element"]))
    )


def _walk(block):
    for node in block:
        yield node
        for key in ("true_block", "false_block"):
            yield from _walk(node.attrs.get(key, ()))


def _captures(nodes, roots, *, arguments=()):
    """Region exits are uses even when an arm contains no producing operation."""
    internal = {node.id for node in nodes} | {node.id for node in arguments}
    used = [item for node in nodes for item in node.inputs]
    used.extend(roots)
    for node in nodes:
        if node.op == "branch":
            used.extend(node.attrs[key] for key in ("true_result", "false_result"))
    return {item.id: item for item in used if item.id not in internal}


def admission(value):
    """Return a stable placement explanation and the complete capture set."""
    if value.op != "finite_map":
        return "ordered scan has a host realization", ()
    if not _flat(value_signature(value)["element"]):
        return "result requires structural storage beyond scalar/Product leaves", ()
    nodes = tuple(_walk(value.attrs["body_block"]))
    for node in nodes:
        if node.op not in _OPS:
            return "missing device lowering: " + node.op, ()
        if node.op == "branch" and not _flat(value_signature(node)):
            return "branch result requires structural storage", ()
        if node.op == "scalar_expression" and any(
            item[0] in ("finite_linear_v1", "finite_projection_v1")
            for item in node.attrs["expression_nodes"]
        ):
            return "finite linear expression has a host realization", ()
    captures = _captures(nodes, (value.attrs["body"],))
    captures[value.inputs[0].id] = value.inputs[0]
    if any(not _capture_type(value_signature(item)) for item in captures.values()):
        return "capture requires structural/field storage", ()
    return None, tuple(captures[key] for key in sorted(captures))


def _declare(signature, prefix, lines):
    serial = [0]

    def leaf(sig, _path):
        token = prefix + "_" + str(serial[0])
        serial[0] += 1
        lines.append("%s %s{};" % (_TYPES[sig["vtype"]], token))
        return token

    return _tree(signature, leaf)


def _assign(signature, target, source, lines):
    for (_, left), (_, right) in zip(
        _leaves(signature, target), _leaves(signature, source), strict=True
    ):
        lines.append("%s = %s;" % (left, right))


def _failure(condition, lines, failure, index):
    lines.append("if (%s) { %s(%s) = 1; return; }" % (condition, failure, index))


def _emit_block(block, env, lines, failure, index, constants, *, kernel=None):
    from .program_source_map import operation

    for value in block:
        scope = (
            operation(value, lines, environment=env, channel="finite_kokkos_kernel")
            if kernel is None
            else kernel.operation(value, env)
        )
        with scope:
            if value.id in env:
                continue
            op = value.op
            args = [env[item.id] for item in value.inputs]
            token = "device_%d" % value.id if kernel is None else kernel.local(value)
            if op == "product":
                env[value.id] = dict(zip(value.attrs["fields"], args, strict=True))
                continue
            if op == "product_get":
                env[value.id] = args[0][value.attrs["field"]]
                continue
            if op == "provenance_bind":
                from pops.ledger_provenance import validate_contract_operation

                validate_contract_operation(value.inputs, value.attrs, value_signature(value))
                env[value.id] = args[0]
                continue
            if op == "branch":
                signature = value_signature(value)
                result = _declare(signature, token, lines)
                lines.append("if (%s) {" % args[0])
                for arm in ("true", "false"):
                    if arm == "false":
                        lines.append("} else {")
                    sub = dict(env)
                    body = []
                    _emit_block(
                        value.attrs[arm + "_block"],
                        sub,
                        body,
                        failure,
                        index,
                        constants,
                        kernel=kernel,
                    )
                    _assign(signature, result, sub[value.attrs[arm + "_result"].id], body)
                    lines.extend("  " + line for line in body)
                lines.append("}")
                env[value.id] = result
                continue
            if op in ("finite_index", "distributed_index"):
                _failure(
                    "%s < 0 || %s >= %s" % (args[1], args[1], args[0]["size"]),
                    lines,
                    failure,
                    index,
                )
                env[value.id] = {"domain": args[0], "offset": args[1]}
                continue
            if op in ("finite_domain", "distributed_value_domain"):
                env[value.id] = args[0]["domain"]
                continue
            if op == "finite_same_domain":
                expression = "(%s == %s)" % (args[0]["identity"], args[1]["identity"])
            elif op in ("finite_size", "distributed_size"):
                expression = args[0]["size"]
            elif op in ("finite_index_value", "distributed_gid"):
                expression = args[0]["offset"]
            elif op == "distributed_rank":
                expression = env[("rank",)]
            elif op in ("finite_at", "distributed_local_at"):
                a, i = args
                lower = a["domain"]["begin"] if op == "distributed_local_at" else "0"
                upper = a["domain"]["end"] if op == "distributed_local_at" else a["domain"]["size"]
                _failure(
                    "%s != %s || %s < %s || %s >= %s"
                    % (
                        a["domain"]["identity"],
                        i["domain"]["identity"],
                        i["offset"],
                        lower,
                        i["offset"],
                        upper,
                    ),
                    lines,
                    failure,
                    index,
                )
                offset = "(%s-%s)" % (i["offset"], lower)
                env[value.id] = _tree(
                    value_signature(value),
                    lambda _s, path, a=a, offset=offset: (
                        _path(a["columns"], path) + "(" + offset + ")"
                    ),
                )
                continue
            elif op == "int64_constant":
                expression = (
                    "std::numeric_limits<std::int64_t>::min()"
                    if value.attrs["value"] == -(1 << 63)
                    else "std::int64_t(%d)" % value.attrs["value"]
                )
            elif op == "bool_constant":
                expression = "true" if value.attrs["value"] else "false"
            elif op == "bool_op":
                from .program_emit_structured_control import boolean_expression
                expression = boolean_expression(value, args)
            elif op == "finite_real":
                expression = "pops::Real(%s)" % args[0] if args else constants[value.id]
            elif op == "int64_op":
                code = {"add": 0, "sub": 1, "mul": 2, "floordiv": 3, "mod": 4}[
                    value.attrs["operation"]
                ]
                lines.append("std::int64_t %s{};" % token)
                _failure(
                    "!%sinteger(%d, %s, %s, %s)" % (_NS, code, *args, token), lines, failure, index
                )
                env[value.id] = token
                continue
            elif op == "int64_cast":
                a = args[0]
                _failure(
                    "!Kokkos::isfinite(%s) || %s < -9223372036854775808.0 || "
                    "%s >= 9223372036854775808.0 || Kokkos::trunc(%s) != %s" % ((a,) * 5),
                    lines,
                    failure,
                    index,
                )
                expression = "static_cast<std::int64_t>(%s)" % a
            elif op == "int64_compare":
                expression = "(%s %s %s)" % (args[0], value.attrs["comparison"], args[1])
            elif op == "compare":
                expression = "(%s %s %s)" % (
                    args[0],
                    value.attrs["cmp"],
                    args[1] if len(args) == 2 else scalar_cpp(value.attrs["rhs"]),
                )
            elif op == "scalar_isfinite":
                expression = "Kokkos::isfinite(%s)" % args[0]
            elif op == "scalar_op":
                parts = [
                    args[n] if kind == "v" else scalar_cpp(n) for kind, n in value.attrs["operands"]
                ]
                symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[value.attrs["fn"]]
                expression = "(%s %s %s)" % (parts[0], symbol, parts[1])
            elif op == "scalar_expression":
                from .program_emit_expressions import checked_expression_dag

                body, outputs, invalid = checked_expression_dag(
                    value.attrs["expressions"],
                    value.attrs["expression_nodes"],
                    tuple((arg,) for arg in args),
                )
                if len(outputs) != 1:
                    raise ValueError("device scalar expression requires one output")
                lines.append("pops::Real %s{}; {" % token)
                lines.extend("  " + line for line in body)
                _failure(invalid, lines, failure, index)
                lines.extend(("  %s = %s;" % (token, outputs[0]), "}"))
                env[value.id] = token
                continue
            else:
                raise ValueError("unadmitted finite Kokkos operation: " + op)
            lines.append("const %s %s = %s;" % (_TYPES[value.vtype], token, expression))
            if value.vtype == "scalar":
                _failure("!Kokkos::isfinite(%s)" % token, lines, failure, index)
            env[value.id] = token


def _path(tree, path) -> str:
    for key in path:
        tree = tree[key]
    if not isinstance(tree, str):
        raise TypeError("device leaf requires one concrete C++ binding")
    return tree


def _stage(prefix, captures, var, block, lines, serial):
    """Bind captures before dispatch: scalars by value, device Arrays in place.

    Host domain pointers are compared only here. The device receives
    process-local equivalence-class ordinals, never pointers or forged offsets.
    """
    from .program_emit_finite_arrays import _check_domain, _collective_call, _uniform_element

    domains = []

    def descriptor(token, sig):
        _check_domain(sig, token, lines)
        n = len(domains)
        name = prefix + "_domain_%d" % n
        lines.append("const std::int64_t %s_size = (%s)->size;" % (name, token))
        identity = str(n)
        for previous, old in reversed(list(enumerate(domains))):
            identity = "((%s) == (%s) ? %d : %s)" % (token, old, previous, identity)
        lines.append("const std::int64_t %s_identity = %s;" % (name, identity))
        domains.append(token)
        return {"size": name + "_size", "identity": name + "_identity"}

    env = {}
    constants = {}
    for capture in captures:
        sig = value_signature(capture)
        token = var[capture.id]
        name = prefix + "_capture_%d" % capture.id
        if _flat(sig):
            _uniform_element(sig, token, lines)

            def leaf(s, path, name=name, token=token):
                name_leaf = name + "_" + str(serial[0])
                serial[0] += 1
                lines.append(
                    "const %s %s = %s;" % (_TYPES[s["vtype"]], name_leaf, _path(token, path))
                )
                return name_leaf

            env[capture.id] = _tree(sig, leaf)
        elif sig["vtype"] == "index_set":
            env[capture.id] = descriptor(token, sig)
        elif sig["vtype"] == "index":
            _uniform_element(sig, token, lines)
            env[capture.id] = {
                "domain": descriptor("(%s).domain" % token, sig),
                "offset": name + "_offset",
            }
            lines.append("const std::int64_t %s_offset = (%s).offset;" % (name, token))
        else:
            # A device Array is read in place: its fingerprint, then its columns.
            _uniform_element(sig, token, lines)
            description = descriptor("(%s).domain" % token, sig)
            position = [0]

            def column(s, path, name=name, token=token, position=position):
                local = "%s_column_%d" % (name, position[0])
                lines.append("%sView<const %s> %s;" % (_NS, storage_type(s), local))
                _collective_call("capture", (local, token, "std::size_t(%d)" % position[0]), lines)
                position[0] += 1
                return local

            env[capture.id] = {"domain": description, "columns": _tree(sig["element"], column)}
    # Literal producers stay host-visible for exact replication validation. This
    # also keeps rank-local fault injection outside the device capture list.
    for node in _walk(block):
        if node.op == "finite_real" and not node.inputs:
            token = prefix + "_literal_%d" % node.id
            lines.append(
                "const pops::Real %s = pops::Real(%s);" % (token, float(node.attrs["value"]).hex())
            )
            _uniform_element(value_signature(node), token, lines)
            constants[node.id] = token
    return env, constants


def _device_constants(kernel, block, constants):
    return {
        node.id: kernel.capture(
            "pops::Real", constants[node.id], identity=("literal", float(node.attrs["value"]).hex())
        )
        for node in _walk(block)
        if node.id in constants
    }


def _output_column(signature, prefix, count, lines, serial):
    from .program_emit_finite_arrays import _collective_call

    token = prefix + "_" + str(serial[0])
    serial[0] += 1
    lines.append("%sView<%s> %s;" % (_NS, storage_type(signature), token))
    _collective_call("allocate_view", (token, count, json.dumps(_ARRAY_LABEL + token)), lines)
    return token


def _publish(signature, output, domain_token, columns, lines):
    """One immutable device Array from kernel-written columns, then its fingerprint."""
    from .program_emit_finite_arrays import _uniform_element, finite_cpp_type

    lines.append("%s %s;" % (finite_cpp_type(signature), output))
    lines.append(
        "pops::runtime::program::finite_collective::publish(%s);"
        % ", ".join((output, domain_token, "ctx.prepared_execution_lane()", *columns))
    )
    _uniform_element(signature, output, lines)


def emit_map(value, var, lines):
    """Append a complete realization only after closed admission; return False on fallback."""
    if value.op == "finite_scan":
        return emit_ordered(value, var, lines)
    reason, captures = admission(value)
    signature = value_signature(value)
    backend = "host" if reason else "Kokkos::DefaultExecutionSpace"
    placement = {
        "ssa": value.id,
        "backend": backend,
        "reason": reason,
        "storage": result_storage(signature, backend),
    }
    lines.append("// pops-finite-map-placement@1 " + json.dumps(placement, sort_keys=True))
    if reason is not None:
        return False
    from .program_emit_finite_arrays import _check_domain, _collective_call, finite_cpp_type
    from .program_emit_structured_control import _reject_if

    start = len(lines)
    prefix = "finite_%d" % value.id
    element = signature["element"]
    domain = value.inputs[0]
    domain_token = var[domain.id]
    _check_domain(value_signature(domain), domain_token, lines)
    count = prefix + "_count"
    lines.append("const std::int64_t %s = (%s)->size;" % (count, domain_token))
    serial = [0]
    env, constants = _stage(prefix, captures, var, value.attrs["body_block"], lines, serial)
    outputs = _tree(
        element, lambda s, path: _output_column(s, prefix + "_output", count, lines, serial)
    )
    failure = prefix + "_failure"
    lines.append("%sView<unsigned char> %s;" % (_NS, failure))
    _collective_call("allocate_view", (failure, count, json.dumps(_CONTROL_LABEL + failure)), lines)
    lines.append("std::exception_ptr %s_kernel_error;" % prefix)
    lines.append("try { if (%s > 0) {" % count)
    from .program_kernel_registry import DeviceKernel, device_environment

    kernel = DeviceKernel()
    device_env = device_environment(kernel, captures, env, value_signature)
    device_constants = _device_constants(kernel, value.attrs["body_block"], constants)
    device_outputs = _tree(
        element,
        lambda sig, path: kernel.capture(
            _NS + "View<" + storage_type(sig) + ">", _path(outputs, path)
        ),
    )
    device_failure = kernel.capture(_NS + "View<unsigned char>", failure)
    body = ["%s(index) = 0;" % device_failure]
    device_env[value.attrs["body_index"].id] = {"domain": device_env[domain.id], "offset": "index"}
    _emit_block(
        value.attrs["body_block"],
        device_env,
        body,
        device_failure,
        "index",
        device_constants,
        kernel=kernel,
    )
    result = device_env[value.attrs["body"].id]
    for (_, col), (_, source) in zip(
        _leaves(element, device_outputs), _leaves(element, result), strict=True
    ):
        body.append("%s(index) = %s;" % (col, source))
    invocation = kernel.invocation(body)
    lines.append(
        'Kokkos::parallel_for("pops_finite_map_%d", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(0,%s), %s);'
        % (value.id, count, invocation)
    )
    lines.extend(
        (
            'Kokkos::fence("pops finite map completion");',
            "pops_finite_kernel_launches.fetch_add(1, std::memory_order_relaxed);",
            "} } catch (...) { %s_kernel_error = std::current_exception(); }" % prefix,
            'pops::collectively_rethrow_exception(%s_kernel_error, ctx.prepared_execution_lane(), "finite Kokkos kernel");'
            % prefix,
        )
    )
    # The first failing lane is reduced on the device: one control scalar.
    bad = prefix + "_first_failure"
    lines.append(
        "const std::int64_t %s = pops::runtime::program::finite_collective::first_failure(%s, %s, ctx.prepared_execution_lane());"
        % (bad, failure, count)
    )
    lines.append(
        "if (%s != %s) pops_finite_kernel_last_failure.store(%s, std::memory_order_relaxed);"
        % (bad, count, bad)
    )
    _reject_if(
        "%s != %s" % (bad, count),
        lines,
        "finite Kokkos map failed checked arithmetic or index validation",
    )
    output = prefix + "_array"
    _publish(
        signature, output, domain_token, [col for _, col in _leaves(element, outputs)], lines
    )
    lines.append("return %s;" % output)
    body = lines[start:]
    lines[start:] = ["const auto %s = [&]() POPS_FINITE_HOST_NOINLINE -> %s {"
                     % (output, finite_cpp_type(signature))]
    lines.extend("  " + line for line in body)
    lines.append("}();")
    var[value.id] = output
    return True


_ORDERED_OPS = _OPS | {"region_argument", "region_index"}


def reads_arrays(value):
    """Whether a fold carries or reads an Array, so its placement is worth a row."""
    if any(sig["vtype"] == "array" for sig, _ in _leaves(value_signature(value.inputs[0]), None)):
        return True
    return any(
        item.vtype == "array" for node in _walk(value.attrs["body_block"]) for item in node.inputs
    )


def ordered_admission(value):
    """Ordered folds and scans with flat carries run as one serial device region (IR4)."""
    scan = value.op == "finite_scan"
    state = value_signature(value.inputs[0])
    if not _flat(state):
        if any(sig["vtype"] == "array" for sig, _ in _leaves(state, None)):
            # The carry is a handle: each iteration's array operations keep their own placement.
            return "ordered carry holds an Array: the host sequences the iterations", ()
        return "ordered carry requires structural storage", ()
    if scan and not _flat(value.attrs["array_signature"]["element"]):
        return "result requires structural storage beyond scalar/Product leaves", ()
    nodes = tuple(_walk(value.attrs["body_block"]))
    for node in nodes:
        if node.op not in _ORDERED_OPS:
            return "missing device lowering: " + node.op, ()
        if node.op == "branch" and not _flat(value_signature(node)):
            return "branch result requires structural storage", ()
        if node.op == "scalar_expression" and any(
            item[0] in ("finite_linear_v1", "finite_projection_v1")
            for item in node.attrs["expression_nodes"]
        ):
            return "finite linear expression has a host realization", ()
    captures = _captures(nodes, (value.attrs["body"],))
    if any(not _capture_type(value_signature(item)) for item in captures.values()):
        return "capture requires structural/field storage", ()
    # A scalar fold that reads no device Array has nothing to keep on the device.
    if not scan and not any(device_array(value_signature(item)) for item in captures.values()):
        return "ordered fold reads no device Array", ()
    return None, tuple(captures[key] for key in sorted(captures))


def _bits(signature, token):
    if signature["vtype"] == "scalar":
        return "Kokkos::bit_cast<std::int64_t>(%s)" % token
    return "static_cast<std::int64_t>(%s)" % token


def _from_bits(signature, token):
    if signature["vtype"] == "scalar":
        return "std::bit_cast<pops::Real>(%s)" % token
    if signature["vtype"] == "bool":
        return "(%s != 0)" % token
    return token


def emit_ordered(value, var, lines):
    """One serial device region for an ascending fold or scan; False keeps the host loop."""
    scan = value.op == "finite_scan"
    reason, captures = ordered_admission(value)
    backend = "host" if reason else "Kokkos::DefaultExecutionSpace"
    placement = {
        "ssa": value.id,
        "backend": backend,
        "reason": reason,
        "schedule": "host loop" if reason else "serial region (ordered, IR4)",
    }
    if scan:
        placement["storage"] = result_storage(value.attrs["array_signature"], backend)
    lines.append("// pops-finite-ordered-placement@1 " + json.dumps(placement, sort_keys=True))
    if reason is not None:
        return False
    from .program_emit_finite_arrays import (
        _check_domain,
        _collective_call,
        _uniform_element,
        finite_cpp_type,
    )
    from .program_emit_structured_control import _reject_if, _uniform_int

    prefix = "finite_%d" % value.id
    block = value.attrs["body_block"]
    initial = value.inputs[0]
    state = value_signature(initial)
    if scan:
        array_signature = value.attrs["array_signature"]
        element = array_signature["element"]
        domain_token = var[value.inputs[1].id]
        _check_domain(value_signature(value.inputs[1]), domain_token, lines)
        count = prefix + "_count"
        lines.append("const std::int64_t %s = (%s)->size;" % (count, domain_token))
    else:
        bound, budget = (var[item.id] for item in value.inputs[1:])
        _uniform_int(bound, lines, "fold count differs across execution lane")
        _uniform_int(budget, lines, "fold capacity differs across execution lane")
        _reject_if("%s < 0 || %s < 0 || %s > %s" % (bound, budget, bound, budget),
                   lines, "fold count outside declared capacity")
        count = bound
    serial = [0]
    results = []

    def result_leaf(sig, path):
        token = "%s_result_%d" % (prefix, len(results))
        results.append((sig, token, _path(var[initial.id], path)))
        lines.append("%s %s = %s;" % (_TYPES[sig["vtype"]], token, _path(var[initial.id], path)))
        return token

    carried = _tree(state, result_leaf)
    published = prefix + "_values"
    if scan:
        # The published Array outlives the staging scope below.
        lines.append("%s %s;" % (finite_cpp_type(array_signature), published))
    lines.append("{")
    env, constants = _stage(prefix, captures, var, block, lines, serial)
    outputs = (
        _tree(element, lambda s, path: _output_column(s, prefix + "_output", count, lines, serial))
        if scan else None
    )
    # control(0): failed, control(1): iteration, then the final carry bits.
    control = prefix + "_control"
    lines.append("%sView<std::int64_t> %s;" % (_NS, control))
    _collective_call(
        "allocate_view", (control, str(2 + len(results)), json.dumps(_CONTROL_LABEL + control)), lines
    )
    lines.append("if (%s > 0) {" % count)
    lines.append("std::exception_ptr %s_kernel_error;" % prefix)
    lines.append("try {")
    from .program_kernel_registry import DeviceKernel, device_environment

    kernel = DeviceKernel()
    device_env = device_environment(kernel, captures, env, value_signature)
    device_constants = _device_constants(kernel, block, constants)
    device_control = kernel.capture(_NS + "View<std::int64_t>", control)
    device_count = kernel.capture("std::int64_t", count)
    device_outputs = (
        _tree(element, lambda sig, path: kernel.capture(
            _NS + "View<" + storage_type(sig) + ">", _path(outputs, path)))
        if scan else None
    )
    initial_device = device_env[initial.id]
    body = ["%s(0) = 0;" % device_control]
    carry_tokens = []

    def carry_leaf(sig, path):
        token = "carry_%d" % len(carry_tokens)
        carry_tokens.append((sig, token))
        body.append("%s %s = %s;" % (_TYPES[sig["vtype"]], token, _path(initial_device, path)))
        return token

    carry = _tree(state, carry_leaf)
    body.append("for (std::int64_t step = 0; step < %s; ++step) {" % device_count)
    inner = ["%s(1) = step;" % device_control]
    loop_env = dict(device_env)
    loop_env[value.attrs["body_argument"].id] = carry
    index = value.attrs["body_index"]
    loop_env[index.id] = (
        {"domain": device_env[value.inputs[1].id], "offset": "step"} if scan else "step"
    )
    _emit_block(block, loop_env, inner, device_control, "0", device_constants, kernel=kernel)
    produced = loop_env[value.attrs["body"].id]
    next_state = produced["state"] if scan else produced
    if scan:
        for (_, col), (_, source) in zip(
            _leaves(element, device_outputs), _leaves(element, produced["value"]), strict=True
        ):
            inner.append("%s(step) = %s;" % (col, source))
    # All carried leaves are updated simultaneously, as in the host loop.
    staged = []
    for n, ((sig, _), (_, source)) in enumerate(
        zip(carry_tokens, _leaves(state, next_state), strict=True)
    ):
        staged.append("next_%d" % n)
        inner.append("const %s next_%d = %s;" % (_TYPES[sig["vtype"]], n, source))
    for (_, token), pending in zip(carry_tokens, staged, strict=True):
        inner.append("%s = %s;" % (token, pending))
    body.extend("  " + line for line in inner)
    body.append("}")
    for n, (sig, token) in enumerate(carry_tokens):
        body.append("%s(%d) = %s;" % (device_control, 2 + n, _bits(sig, token)))
    invocation = kernel.invocation(body)
    lines.append(
        'Kokkos::parallel_for("pops_finite_ordered_%d", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(0,1), %s);'
        % (value.id, invocation)
    )
    lines.extend(
        (
            'Kokkos::fence("pops finite ordered region completion");',
            "pops_finite_kernel_launches.fetch_add(1, std::memory_order_relaxed);",
            "} catch (...) { %s_kernel_error = std::current_exception(); }" % prefix,
            'pops::collectively_rethrow_exception(%s_kernel_error, ctx.prepared_execution_lane(), "finite Kokkos kernel");'
            % prefix,
        )
    )
    # O(1) control scalars: the failure, its iteration and the final carry.
    lines.append("decltype(%s)::host_mirror_type %s_host;" % (control, control))
    _collective_call("mirror_copy", (control + "_host", control), lines)
    lines.append(
        "if (%s_host(0) != 0) pops_finite_kernel_last_failure.store(%s_host(1), std::memory_order_relaxed);"
        % (control, control)
    )
    _reject_if(
        "%s_host(0) != 0" % control,
        lines,
        "finite Kokkos ordered region failed checked arithmetic or index validation",
    )
    for n, (sig, token, _) in enumerate(results):
        lines.append("%s = %s;" % (token, _from_bits(sig, "%s_host(%d)" % (control, 2 + n))))
    lines.append("}")
    _uniform_element(state, carried, lines)
    if scan:
        output = prefix + "_array"
        _publish(
            array_signature, output, domain_token, [col for _, col in _leaves(element, outputs)],
            lines,
        )
        lines.append("%s = std::move(%s);" % (published, output))
    lines.append("}")
    var[value.id] = {"state": carried, "values": published} if scan else carried
    return True


# Host loops (PF2, C34): a map or scan placed on the host and a fold sequenced on the host copy each
# device Array that they read but do not redefine once per entry, at its first read, instead of
# reading it one element per iteration. The loop's own carry changes every iteration: its elements
# stay one-element reads.
HOST_COLUMNS = "finite_host_columns"
_ARGUMENTS = ("body_argument", "body_index", "cond_argument")
_HOST_REGIONS = frozenset(("branch", "result_match", "capture_result", "structured_iterate"))


def _host_reads(block):
    """finite_at operations that the host executes in a block, through nested host regions."""
    stack = [iter(block)]
    while stack:
        node = next(stack[-1], None)
        if node is None:
            stack.pop()
            continue
        if node.op == "finite_at":
            yield node
            continue
        if node.op in _HOST_REGIONS:
            keys = ("cond_block", "body_block", "true_block", "false_block")
        elif node.op == "finite_map":
            keys = ("body_block",) if admission(node)[0] is not None else ()
        elif node.op == "finite_scan":
            keys = ("body_block",) if ordered_admission(node)[0] is not None else ()
        elif node.op == "structured_fold":
            device = reads_arrays(node) and ordered_admission(node)[0] is None
            keys = () if device else ("body_block",)
        else:
            keys = ()
        stack.extend(iter(node.attrs[key]) for key in reversed(keys) if key in node.attrs)


def _defined(region):
    """Ids of the values that a region defines: its arguments and its nested operations."""
    defined, stack = set(), [region]
    while stack:
        node = stack.pop()
        defined.update(node.attrs[key].id for key in _ARGUMENTS if key in node.attrs)
        for key, block in node.attrs.items():
            if key.endswith("_block") and isinstance(block, (list, tuple)):
                defined.update(item.id for item in block)
                stack.extend(block)
    return defined


def _invariant(value, var, defined):
    """The binding of a value that the region reads without redefining it, else None."""
    path = []
    while value.id in defined:
        if value.op != "product_get":
            return None
        path.append(value.attrs["field"])
        value = value.inputs[0]
    binding = var.get(value.id)
    for field in reversed(path):
        if not isinstance(binding, dict):
            return None
        binding = binding.get(field)
    return binding


def host_snapshots(region, var, lines):
    """Declare the host copies of the device Arrays that a host loop reads without redefining.

    Returns their bindings for the loop body: finite_at fills the copy at the first read of each
    entry into the loop, so an Array that no iteration reads is never copied.
    """
    defined = _defined(region)
    snapshots = {}
    for node in _host_reads(region.attrs["body_block"]):
        signature = value_signature(node.inputs[0])
        if not device_array(signature):
            continue
        token = _invariant(node.inputs[0], var, defined)
        key = (HOST_COLUMNS, token)
        if not isinstance(token, str) or key in var or key in snapshots:
            continue
        prefix = "finite_%d_host_read_%d" % (region.id, len(snapshots))
        columns = []
        for leaf, _ in _leaves(signature["element"], None):
            columns.append("%s_%d" % (prefix, len(columns)))
            lines.append(
                "pops::runtime::program::finite_device::HostView<%s> %s;"
                % (storage_type(leaf), columns[-1])
            )
        lines.append("bool %s_ready = false;" % prefix)
        snapshots[key] = {"columns": tuple(columns), "ready": prefix + "_ready"}
    return snapshots
