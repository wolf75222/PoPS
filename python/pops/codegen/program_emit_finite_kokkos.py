"""Type/effect-directed Kokkos realization of pure finite map regions.

This emits kernels into the existing Program DSO. It neither identifies a model
nor changes the finite value/continuation ABI. Unsupported regions keep the
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


def emit_map(value, var, lines):
    """Append a complete realization only after closed admission; return False on fallback."""
    reason, captures = admission(value)
    placement = {
        "ssa": value.id,
        "backend": "host" if reason else "Kokkos::DefaultExecutionSpace",
        "reason": reason,
        "storage": "host immutable value with explicit device staging",
    }
    lines.append("// pops-finite-map-placement@1 " + json.dumps(placement, sort_keys=True))
    if reason is not None:
        return False
    from .program_emit_finite_arrays import (
        _allocate,
        _check_domain,
        _uniform_element,
        _packed,
        _unpacked_cell,
        _store_cell,
        finite_cpp_type,
    )
    from .program_emit_structured_control import _reject_if

    start = len(lines)
    prefix = "finite_%d" % value.id
    signature = value_signature(value)
    element = signature["element"]
    domain = value.inputs[0]
    domain_token = var[domain.id]
    _check_domain(value_signature(domain), domain_token, lines)
    count = prefix + "_count"
    lines.append("const std::int64_t %s = (%s)->size;" % (count, domain_token))
    # Host domain pointers are compared only before dispatch. The device receives
    # process-local equivalence-class ordinals, never pointers or forged offsets.
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
    staging = []
    constants = {}
    for capture in captures:
        sig = value_signature(capture)
        token = var[capture.id]
        name = prefix + "_capture_%d" % capture.id
        if _flat(sig):
            _uniform_element(sig, token, lines)

            def leaf(s, path, name=name, token=token):
                name_leaf = name + "_" + str(len(staging))
                staging.append(name_leaf)
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
            _uniform_element(sig, token, lines)
            description = descriptor("(%s).domain" % token, sig)
            columns = _tree(
                sig["element"],
                lambda s, path, name=name, token=token: _column(
                    s, path, name, "(%s).domain->size" % token, lines, staging
                ),
            )
            k = name + "_i"
            lines.append("for (std::int64_t %s=0; %s<(%s).domain->size; ++%s) {" % (k, k, token, k))
            unpacked = _unpacked_cell(
                sig["element"], "(*(%s).values)[std::size_t(%s)]" % (token, k)
            )
            for (_, col), (_, source) in zip(
                _leaves(sig["element"], columns), _leaves(sig["element"], unpacked), strict=True
            ):
                lines.append("  %s_host(%s) = %s;" % (col, k, source))
            lines.append("}")
            for _, col in _leaves(sig["element"], columns):
                _copy_device(col, lines, prefix)
            env[capture.id] = {"domain": description, "columns": columns}
    # Literal producers stay host-visible for exact replication validation. This
    # also keeps rank-local fault injection outside the device capture list.
    for node in _walk(value.attrs["body_block"]):
        if node.op == "finite_real" and not node.inputs:
            token = prefix + "_literal_%d" % node.id
            lines.append(
                "const pops::Real %s = pops::Real(%s);" % (token, float(node.attrs["value"]).hex())
            )
            _uniform_element(value_signature(node), token, lines)
            constants[node.id] = token
    outputs = _tree(
        element, lambda s, path: _column(s, path, prefix + "_output", count, lines, staging)
    )
    failure = prefix + "_failure"
    lines.append("%sView<unsigned char> %s;" % (_NS, failure))
    _allocate('%sallocate<unsigned char>(%s, "%s")' % (_NS, count, failure), failure, lines)
    lines.append("std::exception_ptr %s_kernel_error;" % prefix)
    lines.append("try { if (%s > 0) {" % count)
    index = prefix + "_index"
    from .program_kernel_registry import DeviceKernel, device_environment

    kernel = DeviceKernel()
    device_env = device_environment(kernel, captures, env, value_signature)
    device_constants = {
        node.id: kernel.capture(
            "pops::Real", constants[node.id], identity=("literal", float(node.attrs["value"]).hex())
        )
        for node in _walk(value.attrs["body_block"])
        if node.id in constants
    }
    device_outputs = _tree(
        element,
        lambda sig, path: kernel.capture(
            _NS
            + "View<"
            + ("unsigned char" if sig["vtype"] == "bool" else _TYPES[sig["vtype"]])
            + ">",
            _path(outputs, path),
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
    lines.append("decltype(%s)::host_mirror_type %s_host;" % (failure, failure))
    _allocate(
        "Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, %s)" % failure,
        failure + "_host",
        lines,
    )
    bad = prefix + "_first_failure"
    lines.append("std::int64_t %s = %s;" % (bad, count))
    lines.append(
        "for (std::int64_t i=0;i<%s;++i) if (%s_host(i)) { %s=i; break; }" % (count, failure, bad)
    )
    lines.append(
        "%s = pops::all_reduce_min(static_cast<long>(%s), ctx.prepared_execution_lane());"
        % (bad, bad)
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
    for _, col in _leaves(element, outputs):
        _allocate(
            "Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, %s)" % col,
            col + "_host",
            lines,
        )
    buffer = prefix + "_buffer"
    lines.append("std::shared_ptr<std::vector<std::any>> %s;" % buffer)
    _allocate("pops_algebra_finite::make_values(std::size_t(%s))" % count, buffer, lines)
    lines.append("for (std::int64_t %s=0;%s<%s;++%s) {" % (index, index, count, index))
    projected = _tree(element, lambda _s, path: _path(outputs, path) + "_host(" + index + ")")
    publication = []
    _uniform_element(element, projected, publication)
    packed = _packed(element, projected, publication, prefix + "_element")
    _store_cell(
        packed, "(*%s)[std::size_t(%s)]" % (buffer, index), publication, prefix + "_publication"
    )
    lines.extend("  " + line for line in publication)
    lines.append("}")
    output = prefix + "_array"
    lines.append(
        "return %s{%s,std::move(%s)};" % (finite_cpp_type(signature), domain_token, buffer)
    )
    body = lines[start:]
    lines[start:] = ["const auto %s = [&]() -> %s {" % (output, finite_cpp_type(signature))]
    lines.extend("  " + line for line in body)
    lines.append("}();")
    var[value.id] = output
    return True


def _column(signature, path, prefix, count, lines, staging):
    from .program_emit_finite_arrays import _allocate

    token = prefix + "_" + str(len(staging))
    staging.append(token)
    dtype = "unsigned char" if signature["vtype"] == "bool" else _TYPES[signature["vtype"]]
    lines.append("%sView<%s> %s;" % (_NS, dtype, token))
    _allocate('%sallocate<%s>(%s,"%s")' % (_NS, dtype, count, token), token, lines)
    lines.append("decltype(%s)::host_mirror_type %s_host;" % (token, token))
    _allocate("Kokkos::create_mirror_view(%s)" % token, token + "_host", lines)
    return token


def _copy_device(token, lines, prefix):
    lines.append("std::exception_ptr %s_copy_error;" % token)
    lines.extend(
        (
            "try { Kokkos::deep_copy(%s,%s_host); } catch (...) { %s_copy_error=std::current_exception(); }"
            % (token, token, token),
            'pops::collectively_rethrow_exception(%s_copy_error, ctx.prepared_execution_lane(), "finite Kokkos capture copy");'
            % token,
        )
    )
