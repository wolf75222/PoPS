"""Joint native finite map application; support order is part of every identity."""
import math
from .expr import Expr, _wrap


def _support(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("finite support requires a name and ordered DOF labels")
    name, dofs = value[0], tuple(value[1])
    if type(name) is not str or not name or not dofs or \
            any(type(x) is not str or not x for x in dofs) or len(set(dofs)) != len(dofs):
        raise ValueError("finite support requires unique nonempty name/DOF labels")
    return name, dofs


class FiniteApplication(Expr):
    def __init__(self, operation, source, target, coefficients, inputs):
        source, target = _support(source), _support(target)
        coefficients = tuple(tuple(row) for row in coefficients)
        if len(coefficients) != len(target[1]) or any(len(row) != len(source[1]) for row in coefficients):
            raise ValueError("finite map coefficients differ from source/target shape")
        if any(type(x) not in (int, float) or not math.isfinite(x) for row in coefficients for x in row):
            raise ValueError("finite map coefficients must be finite numeric constants")
        if operation not in ("apply", "solve"):
            raise ValueError("unknown finite map operation")
        if operation == "solve" and len(source[1]) != len(target[1]):
            raise ValueError("finite solve requires a square map")
        self.operation = operation
        self.source, self.target = source, target
        self.coefficients = coefficients
        self.inputs = tuple(_wrap(x) for x in inputs)
        if len(self.inputs) != len((source if operation == "apply" else target)[1]):
            raise ValueError("finite map input width differs from its support")
        self.width = len((target if operation == "apply" else source)[1])

    def __pops_ir_children__(self): return self.inputs
    def __pops_ir_key__(self, recurse):
        return ("finite_linear_v1", self.operation, self.source, self.target,
                self.coefficients, tuple(recurse(x) for x in self.inputs))
    def _str(self): return "finite_" + self.operation


class FiniteProjection(Expr):
    def __init__(self, application, index):
        if not isinstance(application, FiniteApplication) or type(index) is not int or not 0 <= index < application.width:
            raise ValueError("invalid finite map projection")
        self.application, self.index = application, index
    def __pops_ir_children__(self): return (self.application,)
    def __pops_ir_key__(self, recurse): return ("finite_projection_v1", recurse(self.application), self.index)
    def _str(self): return "finite_component_%d" % self.index


def lower_finite_scalars(values):
    """Consume immutable finite declarations through their versioned scalar protocol.

    One conversion owns the memo, so all projections share one native application.
    No lower algebra module imports or constructs compiler expressions.
    """
    from . import expr

    binary = {"add": expr.Add, "sub": expr.Sub, "mul": expr.Mul, "div": expr.Div,
              "pow": expr.Pow, "and": expr.BooleanAnd, "or": expr.BooleanOr}
    unary = {"neg": expr.Neg, "abs": expr.Abs, "not": expr.BooleanNot}
    comparisons = {"eq", "ne", "lt", "le", "gt", "ge"}
    memo, active = {}, set()

    def application(value):
        if id(value) in memo:
            return memo[id(value)]
        hook = getattr(value, "__pops_finite_application__", None)
        row = hook() if callable(hook) else None
        if type(row) is not tuple or len(row) != 6 or row[0] != "pops.finite-application-plan@1":
            raise TypeError("finite application has an invalid immutable declaration protocol")
        if id(value) in active:
            raise ValueError("finite scalar declaration contains a cycle")
        active.add(id(value))
        result = FiniteApplication(*row[1:5], tuple(lower(x) for x in row[5]))
        active.remove(id(value))
        memo[id(value)] = result
        return result

    def lower(value):
        if id(value) in memo:
            return memo[id(value)]
        hook = getattr(value, "__pops_scalar_plan__", None)
        if not callable(hook):
            return _wrap(value)
        if id(value) in active:
            raise ValueError("finite scalar declaration contains a cycle")
        row = hook()
        if type(row) is not tuple or len(row) != 3 or row[0] != "pops.finite-scalar-plan@1" \
                or type(row[1]) is not str or type(row[2]) is not tuple:
            raise TypeError("finite scalar has an invalid immutable declaration protocol")
        op, arguments = row[1:]
        active.add(id(value))
        if op == "literal" and len(arguments) == 1:
            result = expr.Const(value)
        elif op == "number" and len(arguments) == 1:
            result = expr.Const(arguments[0])
        elif op == "projection" and len(arguments) == 2:
            result = FiniteProjection(application(arguments[0]), arguments[1])
        elif op in binary and len(arguments) == 2:
            result = binary[op](*(lower(x) for x in arguments))
        elif op in comparisons and len(arguments) == 2:
            result = expr.Compare(op, *(lower(x) for x in arguments))
        elif op in unary and len(arguments) == 1:
            result = unary[op](lower(arguments[0]))
        else:
            raise ValueError("unknown finite scalar operation or operand count")
        active.remove(id(value))
        memo[id(value)] = result
        return result

    return tuple(lower(value) for value in values)
