"""Derive complete-field affine structure from the executable expression DAG.

This is a proof over the existing equations, never an author-supplied bypass or
a numerical-method classifier. Uniform coefficients retain their exact SSA
dependencies. Their signs are unrestricted; stability certificates are separate.
"""
from __future__ import annotations

from pops.identity.scalar import ScalarLiteral, scalar_literal

ZERO = ("literal", scalar_literal(0))
ONE = ("literal", scalar_literal(1))


def literal_is(tree, value):
    return (tree[0] == "literal" and type(tree[1]) is ScalarLiteral
            and tree[1].unit is None and tree[1].target is None
            and tree[1].kind in ("integer", "rational", "decimal", "binary64")
            and tree[1].to_python() == value)


def coefficient(operation, *children):
    if operation == "mul":
        if any(literal_is(x, 0) for x in children): return ZERO
        if literal_is(children[0], 1): return children[1]
        if literal_is(children[1], 1): return children[0]
    elif operation == "add":
        if literal_is(children[0], 0): return children[1]
        if literal_is(children[1], 0): return children[0]
    elif operation == "sub" and literal_is(children[1], 0): return children[0]
    elif operation == "div" and literal_is(children[1], 1): return children[0]
    return (operation, *children)


def coefficient_polynomial(polynomial):
    """Keep the existing exact ambient-dt descriptor as one uniform leaf."""
    from pops.time.value_metadata import CoeffPolynomial
    if not polynomial:
        return ZERO
    if set(polynomial) == {0}:
        return ("literal", scalar_literal(polynomial[0]))
    return ("coefficient", polynomial if type(polynomial) is CoeffPolynomial else CoeffPolynomial(polynomial))


def affine_field_terms(value):
    """Return (field, uniform coefficient tree) pairs, or None for a non-affine DAG.

    Every output component must use its same-numbered input components with
    identical uniform coefficients. Cross-component maps remain ordinary
    pointwise equations and do not manufacture a face-ledger proof.
    """
    if value.op != "pointwise_expression" or value.vtype != "state":
        return None
    from pops.time.expressions import component_names
    outputs = value.attrs.get("expressions", ())
    nodes = value.attrs.get("expression_nodes", ())
    if len(outputs) != len(component_names(value)):
        return None
    fields = {item.id: item for item in value.inputs if item.vtype in ("state", "rhs")}
    common = None
    for component, root in enumerate(outputs):
        parsed = []
        for position, node in enumerate(nodes):
            op = node[0]
            if op == "input":
                index, slot = node[1:]
                if type(index) is not int or not 0 <= index < len(value.inputs): return None
                original = value.inputs[index]
                if original.vtype == "scalar" and slot == 0:
                    parsed.append(("uniform", ("scalar", original.id)))
                elif (original.id in fields and slot == component
                      and len(component_names(original)) == len(outputs)):
                    parsed.append(("field", {original.id: ONE}))
                else:
                    parsed.append(None)
                continue
            if op in ("literal", "coefficient"):
                parsed.append(("uniform", tuple(node)))
                continue
            child_ids = node[2:] if op == "compare" else node[1:]
            if any(type(index) is not int or not 0 <= index < position for index in child_ids):
                parsed.append(None); continue
            children = [parsed[index] for index in child_ids]
            if any(child is None for child in children):
                parsed.append(None); continue
            if all(child[0] == "uniform" for child in children):
                tree = ((op, node[1], *(child[1] for child in children)) if op == "compare"
                        else (op, *(child[1] for child in children)))
                parsed.append(("uniform", tree)); continue
            result = None
            if op in ("add", "sub") and len(children) == 2:
                left, right = children
                if left[0] == "uniform" and literal_is(left[1], 0): left = ("field", {})
                if right[0] == "uniform" and literal_is(right[1], 0): right = ("field", {})
                if left[0] == right[0] == "field":
                    result = dict(left[1])
                    for key, factor in right[1].items():
                        result[key] = coefficient(op, result.get(key, ZERO), factor)
            elif op == "neg" and len(children) == 1:
                result = {key: coefficient("neg", factor) for key, factor in children[0][1].items()}
            elif op == "mul" and len(children) == 2:
                left, right = children
                if left[0] == "uniform" and right[0] == "field":
                    result = {key: coefficient("mul", left[1], factor) for key, factor in right[1].items()}
                elif right[0] == "uniform" and left[0] == "field":
                    result = {key: coefficient("mul", factor, right[1]) for key, factor in left[1].items()}
            elif op == "div" and len(children) == 2 and children[1][0] == "uniform":
                result = {key: coefficient("div", factor, children[1][1]) for key, factor in children[0][1].items()}
            parsed.append(None if result is None else ("field", result))
        if type(root) is not int or not 0 <= root < len(parsed): return None
        result = parsed[root]
        if result is None or result[0] != "field": return None
        terms = {key: tree for key, tree in result[1].items() if not literal_is(tree, 0)}
        if common is None: common = terms
        elif common != terms: return None
    if not common: return None
    return tuple((fields[key], tree) for key, tree in common.items())


def coefficient_dag(trees):
    """Encode the derived uniform coefficients for the ordinary checked emitter."""
    nodes, scalar_ids = [], []
    def emit(tree):
        op = tree[0]
        if op == "scalar":
            if tree[1] not in scalar_ids: scalar_ids.append(tree[1])
            node = ("input", scalar_ids.index(tree[1]), 0)
        elif op in ("literal", "coefficient"): node = tree
        elif op == "compare": node = (op, tree[1], *(emit(child) for child in tree[2:]))
        else: node = (op, *(emit(child) for child in tree[1:]))
        nodes.append(node)
        return len(nodes)-1
    roots = tuple(emit(tree) for tree in trees)
    return roots, tuple(nodes), tuple(scalar_ids)
