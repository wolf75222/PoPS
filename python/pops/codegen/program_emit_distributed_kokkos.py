"""Kokkos numerical regions composed with the existing distributed map protocol."""

from __future__ import annotations

import json

from pops.time._program.structured_control import value_signature
from .program_emit_finite_kokkos import (
    _NS,
    _TYPES,
    _walk,
    _flat,
    _tree,
    _path,
    _leaves,
    _emit_block,
    _captures,
)
from pops._ir.data_model import is_finite_array, is_finite_index, value_kind
from pops._ir.effects import device_lowered

_DN = "pops::runtime::program::distributed::"
_CV = "pops::runtime::program::ContinuationValue<pops::kNativeDimension>"


_DESCRIPTOR = ("identity", "size", "begin", "end")


def _identity(pointer, earlier):
    """The index of the first of the @p earlier domains equal to @p pointer, else their count."""
    identity = str(len(earlier))
    for n, old in reversed(list(enumerate(earlier))):
        identity = "((%s)==(%s) ? %d : %s)" % (pointer, old, n, identity)
    return identity


def _array_staging(element, capacity, *, lane, registry, source, earlier, descriptor, columns,
                   labels, index, declare):
    """Return the statements that stage one captured distributed Array after its null check.

    In order: authenticate its domain; compute the descriptor the kernel reads (identity among the
    @p earlier domains of the map, size, begin, end); check the local value count; allocate one
    device column per element leaf, then its host mirror; unpack every local value into the
    mirrors; copy each mirror to its column. @p columns lists ``(type, name)`` per leaf. With
    @p declare, each descriptor value and view is declared where it is first written (the inline
    form); without, the statements assign variables its caller declared (a shared routine body).
    """
    from .program_emit_distributed_arrays import unpack

    statements = []

    def phase(statement):
        statements.append(_DN + "phase(" + lane + ",[&]{ " + statement + " });")

    pointer = "(%s)->domain" % source
    phase("%s.authenticate(%s);" % (registry, pointer))
    values = (_identity(pointer, earlier),) + tuple(
        "(%s)->%s()" % (pointer, member) for member in _DESCRIPTOR[1:]
    )
    for member, expression in zip(_DESCRIPTOR, values, strict=True):
        form = "const std::int64_t %s = %s;" if declare else "%s = %s;"
        statements.append(form % (descriptor[member], expression))
    length = "(%s)->domain->local_size()" % source
    phase(
        "%slocal_require((%s)->values.size()==std::size_t(%s),%sCode::domain);"
        % (_DN, source, length, _DN)
    )
    for (dtype, column), label in zip(columns, labels, strict=True):
        if declare:
            statements.append("%sView<%s> %s;" % (_NS, dtype, column))
        phase("%s = %sallocate<%s>(%s,%s);" % (column, _NS, dtype, length, label))
        if declare:
            statements.append("decltype(%s)::host_mirror_type %s_host;" % (column, column))
    for _, column in columns:
        phase("%s_host = Kokkos::create_mirror_view(%s);" % (column, column))
    statements.append(_DN + "phase(" + lane + ",[&]{")
    statements.append("for(std::int64_t %s=0;%s<%s;++%s) {" % (index, index, length, index))
    decoded = unpack(element, "(%s)->values[std::size_t(%s)]" % (source, index), capacity=capacity)
    for (_, column), (_, scalar) in zip(columns, _leaves(element, decoded), strict=True):
        statements.append("  %s_host(%s) = %s;" % (column, index, scalar))
    statements.append("} });")
    for _, column in columns:
        phase("Kokkos::deep_copy(%s,%s_host);" % (column, column))
    return statements


def _shared_array_staging(element, capacity, *, lane, registry, source, earlier, descriptor,
                          columns):
    """Return the call of the shared routine staging one captured Array, or None outside a scope.

    The routine body is what `_array_staging` emits for the routine parameters. The call site
    declares the descriptor values and the views first; none of these declarations is an
    operation (a default View holds no allocation), so the operations, their order and their
    operands are those of the inline form. The Array and the earlier domains are passed by
    reference after the inline null check: each names an object every inline spelling reads.
    When the Array is the typed read of a continuation cell
    (``std::any_cast<const ArrayRef<...>&>((continuation_N).values.at(k))``), the call evaluates
    that read once more, outside a phase. It cannot fail there: the same read has just succeeded
    in the phase of the null check, nothing runs in between but declarations, and the inline
    form already evaluated it outside a phase, in its descriptor lines.
    """
    from .program_shared_definitions import staging_routine

    parameters = [
        "const pops::ExecutionLane& pops_lane",
        _DN + "Registry& pops_registry",
        "const %sArrayRef<pops::kNativeDimension>& pops_source" % _DN,
    ]
    parameters += ["const %sDomainRef& pops_domain_%d" % (_DN, n) for n in range(len(earlier))]
    parameters += ["std::int64_t& pops_%s" % member for member in _DESCRIPTOR]
    routine_columns = []
    for n, (dtype, _) in enumerate(columns):
        view = "%sView<%s>" % (_NS, dtype)
        parameters += [
            "%s& pops_column_%d" % (view, n),
            "%s::host_mirror_type& pops_column_%d_host" % (view, n),
            "const char* pops_label_%d" % n,
        ]
        routine_columns.append((dtype, "pops_column_%d" % n))
    body = _array_staging(
        element,
        capacity,
        lane="pops_lane",
        registry="pops_registry",
        source="pops_source",
        earlier=["pops_domain_%d" % n for n in range(len(earlier))],
        descriptor={member: "pops_" + member for member in _DESCRIPTOR},
        columns=routine_columns,
        labels=["pops_label_%d" % n for n in range(len(columns))],
        index="pops_index",
        declare=False,
    )
    name = staging_routine(parameters, ["  " + line for line in body])
    if name is None:
        return None
    statements = [
        "std::int64_t %s;" % ", ".join("%s = 0" % descriptor[member] for member in _DESCRIPTOR)
    ]
    arguments = [lane, registry, source, *earlier] + [descriptor[member] for member in _DESCRIPTOR]
    for dtype, column in columns:
        statements.append("%sView<%s> %s;" % (_NS, dtype, column))
        statements.append("decltype(%s)::host_mirror_type %s_host;" % (column, column))
        arguments += [column, column + "_host", '"%s"' % column]
    statements.append("%s(%s);" % (name, ",".join(arguments)))
    return statements


def admission(value):
    result = value.attrs["body"]
    if (
        result.op != "capture_result"
        or result.attrs["body_is_result"]
        or result.attrs.get("body_is_variant_result")
    ):
        return "device local_map requires capture of a pure value body", None
    outer = value.attrs["body_block"]
    formal = value.attrs["body_index"]
    if any(node.id not in (formal.id, result.id) for node in outer):
        return "local_map contains operations outside its captured pure body", None
    raw = result.attrs["body"]
    block = result.attrs["body_block"]
    if not _flat(value_signature(raw)):
        return "local result requires structural storage", None
    nodes = tuple(_walk(block))
    for node in nodes:
        if not device_lowered(node.op):
            return "missing device lowering: " + node.op, None
        if node.op == "branch" and not _flat(value_signature(node)):
            return "branch result requires structural storage", None
        if node.op == "scalar_expression" and any(
            item[0] in ("finite_linear_v1", "finite_projection_v1")
            for item in node.attrs["expression_nodes"]
        ):
            return "finite linear expression has a host realization", None
    captures = _captures(nodes, (raw,), arguments=(formal,))
    captures[value.inputs[0].id] = value.inputs[0]
    for item in captures.values():
        sig = value_signature(item)
        kind = sig["vtype"]
        if _flat(sig):
            continue
        if sig.get("ownership") != "contiguous_distributed" or not is_finite_index(kind):
            return "capture requires unsupported structural/replicated storage", None
        if is_finite_array(kind) and not _flat(sig["element"]):
            return "captured distributed Array requires non-flat elements", None
    return None, (result, raw, block, formal, tuple(captures[key] for key in sorted(captures)))


def emit(value, var, lines, *, token, registry, session, lane):
    reason, plan = admission(value)
    lines.append(
        "// pops-distributed-map-placement@1 "
        + json.dumps(
            dict(
                ssa=value.id,
                backend="host" if reason else "Kokkos::DefaultExecutionSpace",
                reason=reason,
            ),
            sort_keys=True,
        )
    )
    if reason:
        return False
    assert plan is not None
    from .program_emit_distributed_arrays import schema, _runtime_error
    from .program_emit_versioned import _local_storage, _limits
    from .program_source_map import operation

    captured, raw, block, formal, captures = plan
    element = value.attrs["element"]
    error = value.attrs["error"]
    domain = var[value.inputs[0].id]
    lines.append("const auto %s = [&]() -> %sOutcome<pops::kNativeDimension> {" % (token, _DN))
    body = ["try {"]

    def phase(statement):
        body.append(_DN + "phase(" + lane + ",[&]{ " + statement + " });")

    def allocate(expression, name):
        phase(name + " = " + expression + ";")

    body.append("%s.common(%s);" % (registry, domain))
    body.append("pops::runtime::program::ContinuationType %s_element, %s_error;" % (token, token))
    phase("%s_element=%s; %s_error=%s;" % (token, schema(element), token, schema(error)))
    count = token + "_count"
    begin = token + "_begin"
    body.extend(
        (
            "const std::int64_t %s = (%s)->local_size();" % (count, domain),
            "const std::int64_t %s = (%s)->begin();" % (begin, domain),
        )
    )
    domains = []
    columns_all = []
    serial = [0]

    def descriptor(pointer):
        name = token + "_domain_" + str(len(domains))
        phase("%s.authenticate(%s);" % (registry, pointer))
        identity = _identity(pointer, domains)
        domains.append(pointer)
        body.append("const std::int64_t %s_identity = %s;" % (name, identity))
        for member in ("size", "begin", "end"):
            body.append("const std::int64_t %s_%s = (%s)->%s();" % (name, member, pointer, member))
        return {member: name + "_" + member for member in ("identity", "size", "begin", "end")}

    def column_name(prefix):
        name = prefix + "_" + str(serial[0])
        serial[0] += 1
        columns_all.append(name)
        return name

    def column_type(sig):
        return "unsigned char" if sig["vtype"] == "bool" else _TYPES[sig["vtype"]]

    def column(sig, path, prefix, length):
        name = prefix + "_" + str(serial[0])
        serial[0] += 1
        dtype = column_type(sig)
        body.append("%sView<%s> %s;" % (_NS, dtype, name))
        allocate('%sallocate<%s>(%s,"%s")' % (_NS, dtype, length, name), name)
        body.append("decltype(%s)::host_mirror_type %s_host;" % (name, name))
        columns_all.append(name)
        return name

    env = {}
    constants = {}
    for item in captures:
        sig = value_signature(item)
        kind = sig["vtype"]
        source = var[item.id]
        name = token + "_capture_" + str(item.id)
        if _flat(sig):

            def leaf(s, path, name=name, source=source):
                local = name + "_" + str(serial[0])
                serial[0] += 1
                body.append("const %s %s = %s;" % (_TYPES[s["vtype"]], local, _path(source, path)))
                return local

            env[item.id] = _tree(sig, leaf)
        elif value_kind(kind) == "index_set":
            env[item.id] = descriptor(source)
        elif value_kind(kind) == "index":
            env[item.id] = {"domain": descriptor("(%s).domain" % source), "offset": name + "_gid"}
            body.append("const std::int64_t %s_gid = (%s).gid;" % (name, source))
        else:
            phase(
                "%slocal_require(bool(%s)&&bool((%s)->domain),%sCode::domain);"
                % (_DN, source, source, _DN)
            )
            prefix = token + "_domain_" + str(len(domains))
            desc = {member: prefix + "_" + member for member in _DESCRIPTOR}
            earlier = list(domains)
            domains.append("(%s)->domain" % source)
            columns = _tree(sig["element"], lambda _s, _path, name=name: column_name(name))
            typed = [
                (column_type(leaf_sig), col)
                for (leaf_sig, _), (_, col) in zip(
                    _leaves(sig["element"], None), _leaves(sig["element"], columns), strict=True
                )
            ]
            staging = dict(lane=lane, registry=registry, source=source, earlier=earlier,
                           descriptor=desc, columns=typed)
            shared = _shared_array_staging(sig["element"], value.attrs["byte_capacity"], **staging)
            if shared is None:
                shared = _array_staging(
                    sig["element"], value.attrs["byte_capacity"], **staging,
                    labels=['"%s"' % col for _, col in typed], index=name + "_i", declare=True,
                )
            body.extend(shared)
            env[item.id] = {"domain": desc, "columns": columns}
    for node in _walk(block):
        if node.op == "finite_real" and not node.inputs:
            literal = token + "_literal_" + str(node.id)
            body.append(
                "const pops::Real %s = pops::Real(%s);"
                % (literal, float(node.attrs["value"]).hex())
            )
            constants[node.id] = literal
    rank = token + "_rank"
    body.append("const std::int64_t %s = %s.rank();" % (rank, lane))
    env[("rank",)] = rank
    outputs = _tree(element, lambda s, path: column(s, path, token + "_output", count))
    for _, col in _leaves(element, outputs):
        phase("Kokkos::deep_copy(%s,0);" % col)
    failure = token + "_failure"
    body.append("%sView<unsigned char> %s;" % (_NS, failure))
    allocate('%sallocate<unsigned char>(%s,"%s")' % (_NS, count, failure), failure)
    body.append("decltype(%s)::host_mirror_type %s_host;" % (failure, failure))
    phase_open = _DN + "phase(" + lane + ",[&]{ if (%s>0) {" % count
    body.append(phase_open)
    from .program_kernel_registry import DeviceKernel, device_environment

    kernel = DeviceKernel()
    device_env = device_environment(kernel, captures, env, value_signature)
    device_env[("rank",)] = kernel.capture("std::int64_t", rank)
    device_constants = {
        node.id: kernel.capture(
            "pops::Real", constants[node.id], identity=("literal", float(node.attrs["value"]).hex())
        )
        for node in _walk(block)
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
    device = ["%s(index)=0;" % device_failure]
    device_env[formal.id] = {
        "domain": device_env[value.inputs[0].id],
        "offset": "(%s+index)" % kernel.capture("std::int64_t", begin),
    }
    with kernel.operation(formal, device_env):
        pass
    _emit_block(block, device_env, device, device_failure, "index", device_constants, kernel=kernel)
    for (_, col), (_, scalar) in zip(
        _leaves(element, device_outputs), _leaves(element, device_env[raw.id]), strict=True
    ):
        device.append("%s(index)=%s;" % (col, scalar))
    invocation = kernel.invocation(device)
    body.append(
        'Kokkos::parallel_for("pops_distributed_map_%d", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,Kokkos::IndexType<std::int64_t>>(0,%s), %s);'
        % (value.id, count, invocation)
    )
    body.extend(
        (
            'Kokkos::fence("pops distributed map completion");',
            "pops_finite_kernel_launches.fetch_add(1,std::memory_order_relaxed);",
            "} });",
        )
    )
    allocate(
        "Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},%s)" % failure, failure + "_host"
    )
    # Failed device lanes did not initialize outputs; copy only initialized
    # output storage after zero-filling every scalar leaf before the dispatch.
    for _, col in _leaves(element, outputs):
        allocate("Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},%s)" % col, col + "_host")
    packer_type = "pops::runtime::program::finite_structural::ScalarProductPack<pops::kNativeDimension>"
    packer = token + "_packer"
    body.append("std::unique_ptr<%s> %s;" % (packer_type, packer))
    body.append(
        "const auto outcome = %smap<pops::kNativeDimension>(%s,%s,%s_element,%s_error,%d,[&](auto distributed_index){"
        % (_DN, registry, domain, token, token, value.attrs["byte_capacity"])
    )
    host = []
    offset = "(distributed_index.gid-%s)" % begin
    result = token + "_result"
    callback_env = {}
    with operation(captured, host, environment=callback_env, channel="distributed_kokkos_result"):
        host.append("try {")
        host.append(
            "if (%s_host(%s)) return %s::variant(0,%s);"
            % (failure, offset, _CV, _runtime_error(captured, "3"))
        )
        projected = _tree(element, lambda _s, path: _path(outputs, path) + "_host(" + offset + ")")
        capacity = value.attrs["byte_capacity"]
        storage = _local_storage(element, projected, host, result, capacity)
        # Prepare lazily at the first successful lane. Empty ranks and a failed
        # first lane retain the same local error and collective publication path.
        host.append("if (!%s) %s = std::make_unique<%s>(%s_element,%s);"
                    % (packer, packer, packer_type, token, _limits(capacity)))
        packed = "%s->pack(%s)" % (packer, storage)
        callback_env[captured.id] = _CV + "::variant(1," + packed + ")"
        host.append("return " + callback_env[captured.id] + ";")
        host.append(
            "} catch(const std::exception&) { return %s::variant(0,%s); }"
            % (_CV, _runtime_error(captured, "3"))
        )
    body.extend("  " + line for line in host)
    body.append("}, &%s);" % session)
    body.append(
        "if(!outcome.ok()) pops_finite_kernel_last_failure.store(outcome.gid,std::memory_order_relaxed);"
    )
    body.append("return outcome;")
    body.extend(
        (
            "} catch(const %sFailure& failure) {" % _DN,
            "%sOutcome<pops::kNativeDimension> rejected;" % _DN,
            "rejected.code=failure.code;rejected.rank=failure.rank;rejected.gid=failure.gid;return rejected;",
            "}",
        )
    )
    # Value staging belongs to this host scope and dies after publication/error.
    lines.extend("  " + line for line in body)
    lines.append("}();")
    return True
