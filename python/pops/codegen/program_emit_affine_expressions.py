"""Retain affine provenance of the exact pointwise equation, without recomputing it."""
from pops.time._program.affine_expressions import affine_field_terms, coefficient_dag


def emit_uniform_coefficients(trees, var, lines, *, name):
    from pops.codegen.program_emit_expressions import checked_expression_dag
    from pops.codegen.program_emit_finite_arrays import _uniform_element
    from pops.codegen.program_emit_structured_control import _reject_if
    roots, nodes, scalar_ids = coefficient_dag(trees)
    body, values, invalid = checked_expression_dag(roots, nodes,
                                                  tuple((var[key],) for key in scalar_ids))
    lines += ["std::array<pops::Real, %d> %s{};" % (len(trees), name), "{"]
    lines.extend(body)
    _reject_if(invalid, lines, "nonfinite affine coefficient expression")
    for index, token in enumerate(values):
        lines.append("%s[%d] = %s;" % (name, index, token))
        _uniform_element({"vtype": "scalar"}, "%s[%d]" % (name, index), lines)
    lines.append("}")
    return tuple("%s[%d]" % (name, index) for index in range(len(trees)))


def emit_affine_expression(value, var, lines, *, block_index, explicit):
    terms = affine_field_terms(value)
    if terms is None:
        return
    from pops.codegen.program_emit_expressions import pointwise_scalar_aliases
    # Use the same authenticated primitive reads as the numerical kernel. A scalar
    # SSA may denote a PreparedIntegralCapture or host structural projection;
    # neither is a coefficient until its exact consumer boundary has read it.
    coefficients = {**var, **pointwise_scalar_aliases(value)}
    tokens = emit_uniform_coefficients(tuple(tree for _, tree in terms), coefficients, lines,
                                       name="affine_coefficients_%d" % value.id)
    if explicit and value.id not in var.get(("authored_stability_updates",), ()):
        emit_runtime_affine_stability(value, tuple(zip((x[0] for x in terms), tokens, strict=True)),
                                      var, lines, block_index=block_index)
    alias = "affine_term_%d" % value.id
    lines.append("using %s = typename std::remove_reference_t<decltype(ctx)>::UniformAffineTerm;" % alias)
    rows = ", ".join("%s{&%s, %s, %s}" % (alias, var[node.id], token,
                       "true" if node.vtype == "rhs" else "false")
                     for (node, _), token in zip(terms, tokens, strict=True))
    lines.append("const std::array<%s, %d> affine_terms_%d{%s};" % (alias, len(terms), value.id, rows))
    lines.append("ctx.attach_affine_flux_expression(%s, affine_terms_%d);" % (var[value.id], value.id))


def emit_runtime_affine_stability(value, terms, var, lines, *, block_index):
    """Default sufficient spatial certificate; signs remain legal in the algebra."""
    from pops.codegen.program_partition_stability import _is_local_source, _rate_state
    def selected(node):
        return any((kind, node.id) in var for kind in (
            "user_face_frequency", "principal_frequency", "path_frequency", "partition_frequency"))
    if not any(selected(node) for node, _ in terms):
        return
    states = {node.id: token for node, token in terms if node.vtype == "state"}
    invalid = ["!std::isfinite(dt)", "dt < 0"]
    invalid.extend("%s < 0" % token for node, token in terms if node.vtype == "state")
    total = " + ".join(states.values()) or "pops::Real(0)"
    invalid.append("std::abs((%s) - pops::Real(1)) > 32 * std::numeric_limits<pops::Real>::epsilon()" % total)
    groups = {}
    for node, token in terms:
        if node.vtype != "rhs" or _is_local_source(node):
            continue
        source = _rate_state(node)
        frequency = var.get(("partition_frequency", node.id))
        if frequency is None or source.id not in states:
            raise ValueError("runtime affine spatial update lacks its exact input-state budget or frequency")
        invalid.append("%s < 0" % token)
        groups.setdefault(source.id, []).append("%s * %s" % (token, frequency))
    for source, frequencies in groups.items():
        frequency = " + ".join(frequencies)
        invalid += ["!std::isfinite(%s)" % frequency,
                    "(%s) > %s * static_cast<pops::Real>(ctx.numerical_face_courant()) * "
                    "(1 + 32 * std::numeric_limits<pops::Real>::epsilon())" % (frequency, states[source])]
    lines.append('ctx.consume_pointwise_evaluation_status(%d,%d,(%s) ? 1 : 0,'
                 '"runtime_affine_spatial_stability",503);'
                 % (block_index, value.id, " || ".join("(" + x + ")" for x in invalid)))
