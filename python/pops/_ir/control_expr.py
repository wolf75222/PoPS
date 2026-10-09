"""Scientific control and evaluation boundaries in the common expression IR."""
from __future__ import annotations

from .expr import Expr, _boolean_operand, _wrap


class Where(Expr):
    """Only the selected branch is evaluated; both bodies are authored once."""

    def __init__(self, test, yes, no):
        self.test = _boolean_operand(test, where="where")
        self.yes = _wrap(yes)
        self.no = _wrap(no)

    def __pops_ir_children__(self):
        return (self.test, self.yes, self.no)

    def __pops_ir_key__(self, recurse):
        return ("where", recurse(self.test), recurse(self.yes), recurse(self.no))

    def __pops_ir_diff__(self, *, recurse, target, definitions):
        # Piecewise derivative with the predicate held fixed. No differentiability
        # at a switching surface is asserted by this local symbolic operation.
        return Where(self.test, recurse(self.yes), recurse(self.no))

    def eval(self, env):
        return (self.yes if self.test.eval(env) else self.no).eval(env)

    def deps(self):
        from .visitors import _dependencies
        return _dependencies((self.test, self.yes, self.no))

    def to_cpp(self):
        from pops.codegen.cpp_writer import _cse_emit
        lines, (value,) = _cse_emit([self], "double", "", materialize_all=True)
        return "([&]() { %s return %s; }())" % (" ".join(lines), value)

    def _str(self):
        return "where(%s, %s, %s)" % (self.test, self.yes, self.no)


class Rounded(Expr):
    """An explicit binary64 store, retained as a distinct expression boundary."""

    def __init__(self, value):
        self.a = _wrap(value)

    def __pops_ir_children__(self):
        return (self.a,)

    def __pops_ir_key__(self, recurse):
        return ("rounded", "binary64", recurse(self.a))

    def __pops_ir_diff__(self, *, recurse, target, definitions):
        return recurse(self.a)

    def eval(self, env):
        return float(self.a.eval(env))

    def deps(self):
        from .visitors import _dependencies
        return _dependencies(self.a)

    @staticmethod
    def emit(value):
        return ("([&]() { static_assert(std::numeric_limits<double>::is_iec559 && "
                "std::numeric_limits<double>::digits == 53 && "
                "std::numeric_limits<double>::max_exponent == 1024, "
                "\"rounded requires IEEE binary64 double\"); "
                "volatile double rounded_value_ = static_cast<double>(%s); "
                "return static_cast<double>(rounded_value_); }())") % value

    def to_cpp(self):
        return self.emit(self.a.to_cpp())

    def _str(self):
        return "rounded(%s)" % self.a


def where(test, yes, no):
    """Author two nullary bodies once; evaluate only the selected native branch."""
    if not callable(yes) or not callable(no):
        raise TypeError("where requires callable yes/no expression bodies")
    _boolean_operand(test, where="where")
    return Where(test, yes(), no())


def rounded(value):
    """Retain one binary64 evaluation boundary; formal differentiation is real identity."""
    return Rounded(value)


def has_evaluation_boundary(expressions):
    from .visitors import _children
    pending = [expressions] if isinstance(expressions, Expr) else list(expressions)
    seen = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (Where, Rounded)):
            return True
        pending.extend(_children(node))
    return False


__all__ = ["Where", "Rounded", "where", "rounded", "has_evaluation_boundary"]
