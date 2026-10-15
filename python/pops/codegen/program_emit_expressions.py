"""Lower closed Program mathematical expressions to native cell kernels."""

def expression_cpp(node, inputs):
    """Revalidate a leaf against its exact captured components or scalar metadata."""
    from pops.time.expression_dag import validate_expression_leaf

    value = validate_expression_leaf(node, inputs)
    if node[0] == "input":
        return value
    if node[0] == "literal":
        return "static_cast<pops::Real>(%s)" % value.to_cpp()
    from pops.codegen.program_emit_kernels import _coeff_cpp
    return "(%s)" % _coeff_cpp(value)


def pointwise_output_template(value):
    """One authenticated output authority for allocation, mask and cell evaluation."""
    from pops.time.expressions import component_names

    if "numerical_pointwise_rate" in value.attrs:
        from pops.time._program.expressions import validate_pointwise_rate

        return validate_pointwise_rate(value)
    if "cell_mean_projection" in value.attrs:
        from pops.time._program.spatial_interaction import validate_cell_mean

        return validate_cell_mean(value)
    if value.vtype != "state" and "field_product_seed" not in value.attrs:
        raise ValueError("typed pointwise output lost its field product authority")
    if "field_product_seed" in value.attrs:
        index = value.attrs.get("field_product_template_index")
        from pops.model.spaces import FieldSpace

        if (
            type(index) is not int
            or not 0 <= index < len(value.inputs)
            or value.inputs[index].vtype != "state"
            or value.block != value.inputs[index].block
            or value.state_ref is not None
            or type(value.space) is not FieldSpace
            or type(value.attrs.get("ncomp")) is not int
            or value.attrs["ncomp"] != len(value.space.components)
        ):
            raise ValueError("typed field seed allocation authority changed")
        return value.inputs[index]
    if "global_template_index" in value.attrs:
        index = value.attrs["global_template_index"]
        if (
            type(index) is not int
            or not 0 <= index < len(value.inputs)
            or value.inputs[index].vtype != "state"
            or value.inputs[index].block != value.block
            or value.inputs[index].state_ref != value.state_ref
        ):
            raise ValueError("global expression output authority changed")
        return value.inputs[index]
    if "finite_support_v1" in value.attrs:
        from pops.linalg.finite import FiniteSupport

        support = FiniteSupport(*value.attrs["finite_support_v1"])
        index = value.attrs.get("finite_template_index")
        if (
            type(index) is not int
            or not 0 <= index < len(value.inputs)
            or support.dofs != component_names(value)
            or support.dofs != component_names(value.inputs[index])
            or value.inputs[index].block != value.block
            or value.inputs[index].state_ref != value.state_ref
        ):
            raise ValueError("finite materialization output authority changed")
        return value.inputs[index]
    return next(item for item in value.inputs if item.vtype == "state")


def checked_pointwise_rows(value, inputs):
    """Restore the closed DAG into the common CSE emitter and observe every result."""
    from pops.time.expressions import component_names

    if len(inputs) != len(value.inputs):
        raise ValueError("pointwise expression input arity changed")
    for original, row in zip(value.inputs, inputs, strict=True):
        if len(row) != len(component_names(original)):
            raise ValueError("pointwise expression component Space changed")
    pointwise_output_template(value)
    expressions = value.attrs["expressions"]
    if len(expressions) != len(component_names(value)):
        raise ValueError("pointwise expression output Space changed")
    return checked_expression_dag(expressions, value.attrs["expression_nodes"], inputs)


def checked_expression_dag(expressions, nodes, inputs):
    """One checked, lazy evaluation policy for scalar fields and local products."""
    from pops.time.expression_dag import restore_expression_dag
    from pops.codegen.cpp_writer import _cse_emit

    restored, bindings = restore_expression_dag(expressions, nodes, inputs, bind_leaf=expression_cpp)
    lines, values, names = _cse_emit(
        restored,
        "pops::Real",
        "",
        materialize_all=True,
        return_names=True,
        scalar_bindings=bindings,
    )
    return lines, values, " || ".join("!Kokkos::isfinite(%s)" % name for name in names) or "false"


def pointwise_scalar_aliases(value):
    """The exact POD names shared by numerical equations and their affine proof."""
    return {
        item.id: ("global_value_%d_%d" if item.op == "integral_candidate"
                  else "expression_scalar_%d_%d") % (value.id, item.id)
        for item in value.inputs if item.vtype == "scalar"
    }


def pointwise_scalar_bindings(value, variables):
    aliases = pointwise_scalar_aliases(value)
    return [
        "const pops::Real %s = %s;" % (aliases[item.id], prepared_scalar_input(item, variables))
        for item in value.inputs if item.vtype == "scalar"
    ]


def emit_pointwise_kernel(value, variables, output, *, block_index, status, scalars_prepared=False):
    from pops.codegen.program_emit_kernels import _kernel_open, _kernel_close
    from pops.time.expressions import component_names

    names = [variables[item.id] for item in value.inputs]
    field_names = [variables[item.id] for item in value.inputs if item.vtype != "scalar"]
    # Storage can be a C++ expression such as (*sf6), not an identifier. Give
    # those expressions local reference names before deriving Array4 names.
    aliases = {
        name: (name if name.isidentifier() else "expression_input_%d_%d" % (value.id, index))
        for index, name in enumerate(dict.fromkeys(field_names))
    }
    references = [
        "const auto& %s = %s;" % (alias, name) for name, alias in aliases.items() if alias != name
    ]
    names = [aliases.get(name, name) for name in names]
    field_names = [aliases[name] for name in field_names]
    template_name = aliases[variables[pointwise_output_template(value).id]]
    body = _kernel_open(output, template_name)
    index = next(i for i, line in enumerate(body) if "pops::for_each_cell" in line)
    views = []
    for name in dict.fromkeys(field_names):
        if name != template_name:
            views.append("  const auto %sA = std::as_const(%s).fab(li).view();" % (name, name))
    mask = "expression_active_%d" % value.id
    body.insert(
        0, "const auto* %s = ctx.pointwise_active_mask(%d, %s);" % (mask, block_index, output)
    )
    index += 1
    views.extend(
        [
            "  const auto expression_status_ = %s.fab(li).view();" % status,
            "  const bool expression_has_mask_ = %s != nullptr;" % mask,
            "  const auto expression_mask_ = expression_has_mask_ ? std::as_const(*%s).fab(li).view()"
            " : pops::FieldView<const pops::Real, pops::kNativeDimension>{};" % mask,
        ]
    )
    body[index:index] = views
    # A scalar binding can project from host-only structural storage (std::any,
    # vector, or an authenticated native capability). Resolve that binding on
    # the host; device closures capture only the resulting primitive value.
    scalar_aliases = pointwise_scalar_aliases(value)
    scalar_captures = {
        item.id: (scalar_aliases[item.id], prepared_scalar_input(item, variables))
        for item in value.inputs if item.vtype == "scalar"
    }
    rows = [
        [scalar_captures[item.id][0]]
        if item.vtype == "scalar"
        else ["%sA(index, %d)" % (name, c) for c in range(len(component_names(item)))]
        for item, name in zip(value.inputs, names, strict=True)
    ]
    temporaries, rendered, invalid = checked_pointwise_rows(value, rows)
    body.append(
        "    if (expression_has_mask_ && !(expression_mask_(index, 0) >= pops::Real(0.5))) {"
    )
    for c in range(len(rendered)):
        body.append(
            "      outA(index, %d) = pops::Real(0);" % c
            if "field_product_seed" in value.attrs
            else "      outA(index, %d) = %sA(index, %d);" % (c, template_name, c)
        )
    body.extend(["      expression_status_(index, 0) = pops::Real(0);", "      return;", "    }"])
    body.extend("    " + line for line in temporaries)
    body.append(
        "    expression_status_(index, 0) = (%s) ? pops::Real(1) : pops::Real(0);" % invalid
    )
    body.extend("    outA(index, %d) = %s;" % (c, expr) for c, expr in enumerate(rendered))
    if "finite_support_v1" in value.attrs:
        vote = ["{", "long finite_layout_error_ = 0;"]
        for name in dict.fromkeys(field_names):
            vote.append(
                "finite_layout_error_ |= (%s.layout() != %s.layout() || "
                "%s.distribution() != %s.distribution() || %s.local_rank() != %s.local_rank());"
                % (name, output, name, output, name, output)
            )
        vote += [
            "if (pops::all_reduce_max(finite_layout_error_, ctx.prepared_execution_lane()))",
            '  throw std::runtime_error("finite materialization requires co-located layouts, distributions and ranks");',
        ]
        body = vote + ["}"] + body
    captures = [] if scalars_prepared else pointwise_scalar_bindings(value, variables)
    return references + captures + body + _kernel_close()


def prepared_scalar_input(value, var):
    """Read a prepared global capture through its exact live owner/point contract."""
    if value.op != "integral_candidate":
        return var[value.id]
    import json
    from pops.codegen.program_integral_transfers import integral_identity

    return "ctx.integral_candidate_value(%s,%s,%s)" % (
        var[value.id],
        json.dumps(integral_identity(value.prog, value.attrs["integral"])),
        json.dumps(value.attrs["units"]),
    )
