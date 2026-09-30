"""Scientific scalar expressions reading exact temporal SSA components.

These are authoring nodes in the existing mathematical Expr algebra. They carry no
buffers and execute no Python per cell. Materialization authenticates every read
and captures a closed expression tree in the ordinary Program IR.
"""
from __future__ import annotations

from typing import Any

from pops._ir import expr as ir
from pops._ir.control_expr import Where, Rounded
from pops._ir.symbolic import ImmutableSymbolic


def component_names(value: Any) -> tuple[str, ...]:
    if value.vtype == "scalar":
        return ("scalar",)
    space = value.space
    if getattr(space, "kind", None) == "rate":
        space = space.base_space
    names = tuple(getattr(space, "components", ()))
    if not names:
        raise TypeError("pointwise expressions require a complete component Space")
    return names


class ProgramComponent(ir.Expr):
    """A component of one exact ProgramValue, not a declaration-name lookup."""

    def __init__(self, value: Any, component: str | int) -> None:
        from pops.time.value_support import _ProgramValueBase as ProgramValue
        if not isinstance(value, ProgramValue) or not value.is_field():
            raise TypeError("component expressions require a readable Program field value")
        names = component_names(value)
        if type(component) is str:
            if component not in names:
                raise KeyError(component)
            component = names.index(component)
        elif type(component) is not int:
            raise TypeError("component must be an exact name or integer")
        if not 0 <= component < len(names):
            raise IndexError("component is outside the value Space")
        self.value = value
        self.component = component

    def __pops_ir_children__(self):
        return ()

    def __pops_ir_key__(self, recurse):
        return ("program_component", str(self.value.prog.owner_path),
                self.value.id, self.component)

    def eval(self, env):
        return env[(self.value.id, self.component)]

    def to_cpp(self):
        raise TypeError("a Program component requires an authenticated Program lowering")

    def _str(self):
        return "%s[%d]" % (self.value.name, self.component)


class ProgramScalar(ir.Expr):
    """One exact collective Scalar value broadcast into a native cell expression."""

    def __init__(self, value):
        self.value = value

    def __pops_ir_children__(self):
        return ()

    def __pops_ir_key__(self, recurse):
        return ("program_scalar", str(self.value.prog.owner_path), self.value.id)

    def to_cpp(self):
        raise TypeError("a Program Scalar requires an authenticated Program lowering")

    def _str(self):
        return self.value.name


class ProgramExpression(ImmutableSymbolic):
    """A componentwise expression product retaining its output StateSpace."""

    def __init__(self, components, template):
        self.components = tuple(components)
        self.template = template

    def __getitem__(self, key):
        if type(key) is str:
            key = component_names(self.template).index(key)
        if type(key) is not int:
            raise TypeError("component must be an exact name or integer")
        if not 0 <= key < len(self.components):
            raise IndexError("component is outside the expression Space")
        return self.components[key]

    def _binary(self, other, operation, reverse=False):
        right = as_expression(other, template=self.template)
        if len(right.components) != len(self.components):
            raise ValueError("pointwise expression component counts differ")
        pairs = zip(self.components, right.components, strict=True)
        return ProgramExpression(tuple(operation(b, a) if reverse else operation(a, b)
                                       for a, b in pairs), self.template)

    def __add__(self, other): return self._binary(other, ir.Add)
    def __radd__(self, other): return self._binary(other, ir.Add, True)
    def __sub__(self, other): return self._binary(other, ir.Sub)
    def __rsub__(self, other): return self._binary(other, ir.Sub, True)
    def __mul__(self, other): return self._binary(other, ir.Mul)
    def __rmul__(self, other): return self._binary(other, ir.Mul, True)
    def __truediv__(self, other): return self._binary(other, ir.Div)
    def __rtruediv__(self, other): return self._binary(other, ir.Div, True)
    def __pow__(self, other): return self._binary(other, ir.Pow)
    def __neg__(self):
        return ProgramExpression(tuple(ir.Neg(x) for x in self.components), self.template)


def as_expression(value, *, template=None):
    from pops.time.value_support import (
        _ProgramValueBase as ProgramValue, _AffineExpressionBase as _Affine,
        _MethodCoefficientBase as _Coeff, resolve_temporal_handle as _resolve_handle,
    )
    value = _resolve_handle(value)
    if isinstance(value, ProgramExpression):
        return value
    if isinstance(value, ProgramValue) and value.vtype == "scalar":
        if template is None:
            raise TypeError("a broadcast Program Scalar needs a State template")
        return ProgramExpression((ProgramScalar(value),) * len(component_names(template)), template)
    if isinstance(value, ProgramValue):
        return ProgramExpression(tuple(ProgramComponent(value, c)
                                       for c in range(len(component_names(value)))), value)
    if isinstance(value, _Affine):
        if not value.terms:
            raise TypeError("an empty affine expression has no component Space")
        result = None
        for item, coefficient in value.terms:
            term = as_expression(item) * coefficient
            result = term if result is None else result + term
        return result
    if template is None:
        raise TypeError("a literal pointwise expression needs a template")
    if isinstance(value, _Coeff):
        # dt is an explicitly captured method coefficient, not an equation seed.
        scalar = CoefficientExpression(value.to_polynomial())
    else:
        scalar = ir._wrap(value)
    return ProgramExpression((scalar,) * len(component_names(template)), template)


class CoefficientExpression(ir.Expr):
    def __init__(self, polynomial):
        self.polynomial = polynomial

    def __pops_ir_children__(self):
        return ()

    def __pops_ir_key__(self, recurse):
        import json
        return ("method_coefficient", json.dumps(self.polynomial.to_data(),
                                                 sort_keys=True, separators=(",", ":")))

    def to_cpp(self):
        raise TypeError("a method coefficient requires an authenticated Program lowering")

    def _str(self):
        return "method_coefficient(%r)" % self.polynomial


_BINARY = {ir.Add: "add", ir.Sub: "sub", ir.Mul: "mul", ir.Div: "div",
           ir.Pow: "pow", ir.Minimum: "minimum", ir.Maximum: "maximum",
           ir.BooleanAnd: "and", ir.BooleanOr: "or"}
_UNARY = {ir.Neg: "neg", ir.Sqrt: "sqrt", ir.Exp: "exp", ir.Abs: "abs", ir.Sign: "sign",
          ir.BooleanNot: "not", Rounded: "rounded"}


def encode_expressions(expressions, program):
    """Capture exact reads and bounded scalar operations, without simplification."""
    from pops.time.value_support import require_owned_value as require_owned
    from pops._ir.finite_linear import FiniteApplication, FiniteProjection
    inputs = []
    by_id = {}
    nodes = []
    memo = {}

    def encode(node):
        node = ir._wrap(node)
        if id(node) in memo:
            return memo[id(node)][1]
        if type(node) is ProgramScalar:
            value = require_owned(program, node.value, "pointwise scalar expression")
            if value.vtype != "scalar":
                raise TypeError("pointwise scalar expression requires an exact Scalar")
            if value.id not in by_id:
                by_id[value.id] = len(inputs)
                inputs.append(value)
            encoded = ("input", by_id[value.id], 0)
        elif type(node) is ProgramComponent:
            value = require_owned(program, node.value, "pointwise expression")
            if value.id not in by_id:
                by_id[value.id] = len(inputs)
                inputs.append(value)
            encoded = ("input", by_id[value.id], node.component)
        elif type(node) is ir.Const:
            encoded = ("literal", node.literal)
        elif type(node) is CoefficientExpression:
            encoded = ("coefficient", node.polynomial)
        elif type(node) is FiniteApplication:
            encoded = ("finite_linear_v1", node.operation, node.source, node.target,
                       node.coefficients, tuple(encode(x) for x in node.inputs))
        elif type(node) is FiniteProjection:
            encoded = ("finite_projection_v1", encode(node.application), node.index)
        elif type(node) is Where:
            encoded = ("where", encode(node.test), encode(node.yes), encode(node.no))
        elif type(node) is ir.Compare:
            encoded = ("compare", node.comparison, encode(node.a), encode(node.b))
        elif type(node) in _BINARY:
            encoded = (_BINARY[type(node)], encode(node.a), encode(node.b))
        elif type(node) in _UNARY:
            encoded = (_UNARY[type(node)], encode(node.a))
        else:
            raise TypeError("pointwise expression cannot lower %s" % type(node).__name__)
        index = len(nodes)
        nodes.append(encoded)
        memo[id(node)] = (node, index)
        return index

    encoded = tuple(encode(expression) for expression in expressions)
    return encoded, tuple(nodes), tuple(inputs)
