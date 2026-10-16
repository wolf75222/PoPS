"""One closed scalar DAG language for authoring validation and target lowering."""
from collections.abc import Mapping
from typing import Any

from pops._ir import expr as ir
from pops._ir.control_expr import Where, Rounded
from pops.identity import canonical_bytes
from pops.identity.scalar import ScalarLiteral, parse_canonical_integer
from pops.time.value_metadata import CoeffPolynomial


def decode_expression_literal(data: Any) -> ScalarLiteral:
    if not isinstance(data, Mapping) or data.get("kind") not in (
            "integer", "rational", "decimal", "binary64"):
        raise TypeError("field expression requires an exact numeric literal without raw C++")
    kind = data["kind"]
    if kind == "integer":
        payload = parse_canonical_integer(data["value"])
    elif kind == "rational":
        payload = (parse_canonical_integer(data["numerator"]), parse_canonical_integer(data["denominator"]))
    else:
        payload = data["value"]
    literal = ScalarLiteral(kind, payload, data.get("unit"), data.get("target"))
    if canonical_bytes(literal.to_data()) != canonical_bytes(dict(data)):
        raise ValueError("field expression literal is not canonical")
    return literal



def validate_expression_leaf(node, inputs) -> Any:
    """Validate the mathematical leaf without invoking any target compiler."""
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
        return node[1]
    if op == "coefficient" and len(node) == 2:
        if type(node[1]) is not CoeffPolynomial:
            raise TypeError("pointwise method coefficient requires an exact polynomial")
        return node[1]
    raise ValueError("pointwise expression has an unsupported leaf or arity")


def restore_expression_dag(expressions, nodes, inputs, *, bind_leaf=validate_expression_leaf):
    """Restore exact IR nodes; target-specific leaf bindings are provided by the caller."""
    binary = {
        "add": ir.Add,
        "sub": ir.Sub,
        "mul": ir.Mul,
        "div": ir.Div,
        "pow": ir.Pow,
        "minimum": ir.Minimum,
        "maximum": ir.Maximum,
        "and": ir.BooleanAnd,
        "or": ir.BooleanOr,
    }
    unary = {
        "neg": ir.Neg,
        "abs": ir.Abs,
        "sqrt": ir.Sqrt,
        "exp": ir.Exp,
        "sin": ir.Sin,
        "cos": ir.Cos,
        "log": ir.Log,
        "expm1": ir.Expm1,
        "sign": ir.Sign,
        "not": ir.BooleanNot,
        "rounded": Rounded,
    }
    restored, bindings = [], {}
    for index, node in enumerate(nodes):
        if not isinstance(node, (tuple, list)) or not node:
            raise TypeError("pointwise expression requires a closed DAG")
        operation = node[0]
        if operation in ("input", "literal", "coefficient"):
            name = "expression_leaf_%d_" % index
            bindings[name] = bind_leaf(node, inputs)
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
        arity = (
            3
            if operation == "where"
            else 2
            if operation == "compare" or operation in binary
            else 1
            if operation in unary
            else None
        )
        if arity is None or len(children) != arity:
            raise ValueError("pointwise expression has unsupported operation or arity")
        if any(type(child) is not int or not 0 <= child < index for child in children):
            raise ValueError("pointwise expression DAG must reference earlier nodes")
        arguments = tuple(restored[child] for child in children)
        if operation == "compare":
            result = ir.Compare(node[1], *arguments)
        else:
            constructor = (
                Where if operation == "where" else binary.get(operation, unary.get(operation))
            )
            if constructor is None:
                raise ValueError("pointwise expression has unsupported operation or arity")
            result = constructor(*arguments)
        restored.append(result)
    if any(type(root) is not int or not 0 <= root < len(restored) for root in expressions):
        raise ValueError("pointwise expression output is outside its closed DAG")
    return [restored[root] for root in expressions], bindings
