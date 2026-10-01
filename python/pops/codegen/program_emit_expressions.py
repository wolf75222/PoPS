"""Lower closed Program mathematical expressions to native cell kernels."""
from pops.identity.scalar import ScalarLiteral


def expression_cpp(node, inputs):
    """Revalidate a leaf against its exact captured components or scalar metadata."""
    if not isinstance(node, (list, tuple)) or not node:
        raise TypeError("pointwise expression requires a closed AST")
    op = node[0]
    if op == "input" and len(node) == 3:
        index, component = node[1:]
        if type(index) is not int or not 0 <= index < len(inputs):
            raise ValueError("pointwise input outside captured values")
        if type(component) is not int or not 0 <= component < len(inputs[index]):
            raise ValueError("pointwise component outside its exact Space")
        return inputs[index][component]
    if op == "literal" and len(node) == 2 and type(node[1]) is ScalarLiteral:
        if node[1].kind not in ("integer", "rational", "decimal", "binary64"):
            raise TypeError("pointwise literal requires an exact numeric scalar")
        return "static_cast<pops::Real>(%s)" % node[1].to_cpp()
    if op == "coefficient" and len(node) == 2:
        from pops.codegen.program_emit_kernels import _coeff_cpp
        from pops.time.value_metadata import CoeffPolynomial
        if type(node[1]) is not CoeffPolynomial:
            raise TypeError("pointwise method coefficient requires an exact polynomial")
        return "(%s)" % _coeff_cpp(node[1])
    raise ValueError("pointwise expression has an unsupported leaf or arity")


def pointwise_output_template(value):
    """One authenticated output authority for allocation, mask and cell evaluation."""
    from pops.time.expressions import component_names
    if value.vtype != "state" and "field_product_seed" not in value.attrs:
        raise ValueError("typed pointwise output lost its field product authority")
    if "field_product_seed" in value.attrs:
        index = value.attrs.get("field_product_template_index")
        from pops.model.spaces import FieldSpace
        if type(index) is not int or not 0 <= index < len(value.inputs) or value.inputs[index].vtype != "state" or \
                value.block != value.inputs[index].block or value.state_ref is not None or type(value.space) is not FieldSpace or \
                type(value.attrs.get("ncomp")) is not int or value.attrs["ncomp"] != len(value.space.components):
            raise ValueError("typed field seed allocation authority changed")
        return value.inputs[index]
    if "global_template_index" in value.attrs:
        index = value.attrs["global_template_index"]
        if (type(index) is not int or not 0 <= index < len(value.inputs)
                or value.inputs[index].vtype != "state"
                or value.inputs[index].block != value.block
                or value.inputs[index].state_ref != value.state_ref):
            raise ValueError("global expression output authority changed")
        return value.inputs[index]
    if "finite_support_v1" in value.attrs:
        from pops.linalg.finite import FiniteSupport
        support = FiniteSupport(*value.attrs["finite_support_v1"])
        index = value.attrs.get("finite_template_index")
        if (type(index) is not int or not 0 <= index < len(value.inputs)
                or support.dofs != component_names(value)
                or support.dofs != component_names(value.inputs[index])
                or value.inputs[index].block != value.block
                or value.inputs[index].state_ref != value.state_ref):
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
    from pops._ir import expr as ir
    from pops._ir.control_expr import Where, Rounded
    from pops.codegen.cpp_writer import _cse_emit
    binary = {"add": ir.Add, "sub": ir.Sub, "mul": ir.Mul, "div": ir.Div,
              "pow": ir.Pow, "minimum": ir.Minimum, "maximum": ir.Maximum,
              "and": ir.BooleanAnd, "or": ir.BooleanOr}
    unary = {"neg": ir.Neg, "abs": ir.Abs, "sqrt": ir.Sqrt, "exp": ir.Exp, "sign": ir.Sign,
             "not": ir.BooleanNot, "rounded": Rounded}
    restored, bindings = [], {}
    for index, node in enumerate(nodes):
        if not isinstance(node, (tuple, list)) or not node:
            raise TypeError("pointwise expression requires a closed DAG")
        operation = node[0]
        if operation in ("input", "literal", "coefficient"):
            name = "expression_leaf_%d_" % index
            bindings[name] = expression_cpp(node, inputs)
            restored.append(ir.Var(name, "program_expression"))
            continue
        if operation in ("finite_linear_v1", "finite_projection_v1"):
            from pops._ir.finite_linear import FiniteApplication, FiniteProjection
            if operation == "finite_linear_v1" and len(node) == 6:
                children = node[5]
                if any(type(child) is not int or not 0 <= child < index for child in children):
                    raise ValueError("finite map DAG must reference earlier inputs")
                restored.append(FiniteApplication(*node[1:5], tuple(restored[c] for c in children)))
            elif operation == "finite_projection_v1" and len(node) == 3:
                if type(node[1]) is not int or not 0 <= node[1] < index:
                    raise ValueError("finite projection must reference an earlier map")
                restored.append(FiniteProjection(restored[node[1]], node[2]))
            else:
                raise ValueError("invalid finite map DAG arity")
            continue
        children = node[2:] if operation == "compare" else node[1:]
        arity = (3 if operation == "where" else 2 if operation == "compare"
                 or operation in binary else 1 if operation in unary else None)
        if arity is None or len(children) != arity:
            raise ValueError("pointwise expression has unsupported operation or arity")
        if any(type(child) is not int or not 0 <= child < index for child in children):
            raise ValueError("pointwise expression DAG must reference earlier nodes")
        arguments = tuple(restored[child] for child in children)
        if operation == "compare":
            result = ir.Compare(node[1], *arguments)
        else:
            constructor = Where if operation == "where" else binary.get(operation, unary.get(operation))
            result = constructor(*arguments)
        restored.append(result)
    if any(type(root) is not int or not 0 <= root < len(restored) for root in expressions):
        raise ValueError("pointwise expression output is outside its closed DAG")
    lines, values, names = _cse_emit([restored[root] for root in expressions],
                                    "pops::Real", "", materialize_all=True, return_names=True,
                                    scalar_bindings=bindings)
    return lines, values, " || ".join(
        "!Kokkos::isfinite(%s)" % name for name in names) or "false"


def emit_pointwise_kernel(value, variables, output, *, block_index, status):
    from pops.codegen.program_emit_kernels import _kernel_open, _kernel_close
    from pops.time.expressions import component_names
    names = [variables[item.id] for item in value.inputs]
    field_names = [variables[item.id] for item in value.inputs if item.vtype != "scalar"]
    template_name = variables[pointwise_output_template(value).id]
    body = _kernel_open(output, template_name)
    index = next(i for i, line in enumerate(body) if "pops::for_each_cell" in line)
    views = []
    for name in dict.fromkeys(field_names):
        if name != template_name:
            views.append("  const auto %sA = std::as_const(%s).fab(li).view();" % (name, name))
    mask = "expression_active_%d" % value.id
    body.insert(0, "const auto* %s = ctx.pointwise_active_mask(%d, %s);"
                % (mask, block_index, output))
    index += 1
    views.extend([
        "  const auto expression_status_ = %s.fab(li).view();" % status,
        "  const bool expression_has_mask_ = %s != nullptr;" % mask,
        "  const auto expression_mask_ = expression_has_mask_ ? std::as_const(*%s).fab(li).view()"
        " : pops::FieldView<const pops::Real, pops::kNativeDimension>{};" % mask,
    ])
    body[index:index] = views
    rows = [["global_value_%d" % item.id] if item.op == "integral_candidate" else
            [name] if item.vtype == "scalar" else
            ["%sA(index, %d)" % (name, c) for c in range(len(component_names(item)))]
            for item, name in zip(value.inputs, names, strict=True)]
    temporaries, rendered, invalid = checked_pointwise_rows(value, rows)
    body.append("    if (expression_has_mask_ && !(expression_mask_(index, 0) >= pops::Real(0.5))) {")
    for c in range(len(rendered)):
        body.append("      outA(index, %d) = pops::Real(0);" % c if "field_product_seed" in value.attrs else
                    "      outA(index, %d) = %sA(index, %d);" % (c, template_name, c))
    body.extend(["      expression_status_(index, 0) = pops::Real(0);", "      return;", "    }"])
    body.extend("    " + line for line in temporaries)
    body.append("    expression_status_(index, 0) = (%s) ? pops::Real(1) : pops::Real(0);" % invalid)
    body.extend("    outA(index, %d) = %s;" % (c, expr) for c, expr in enumerate(rendered))
    if "finite_support_v1" in value.attrs:
        vote = ["{", "long finite_layout_error_ = 0;"]
        for name in dict.fromkeys(field_names):
            vote.append("finite_layout_error_ |= (%s.layout() != %s.layout() || "
                        "%s.distribution() != %s.distribution() || %s.local_rank() != %s.local_rank());"
                        % (name, output, name, output, name, output))
        vote += ["if (pops::all_reduce_max(finite_layout_error_, ctx.prepared_execution_lane()))",
                 '  throw std::runtime_error("finite materialization requires co-located layouts, distributions and ranks");']
        body = vote + ["}"] + body
    captures = []
    for item, name in zip(value.inputs, names, strict=True):
        if item.op == "integral_candidate":
            import json
            from pops.codegen.program_integral_transfers import integral_identity
            captures.append("const pops::Real global_value_%d = ctx.integral_candidate_value(%s,%s,%s);" %
                (item.id, name, json.dumps(integral_identity(item.prog,item.attrs["integral"])),
                 json.dumps(item.attrs["units"])))
    return captures + body + _kernel_close()
