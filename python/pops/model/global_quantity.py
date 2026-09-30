"""Physical, scalar global inputs. Their values belong to a temporal consumer."""
from __future__ import annotations

from pops._ir.expr import Expr, Const
from pops._ir.quantity import PhysicalDimension, expression_handle_key
from .handles import Handle


class GlobalQuantityHandle(Handle):
    __slots__ = ("units",)

    def __init__(self, name, *, owner, units):
        if not isinstance(units, PhysicalDimension):
            raise TypeError("global quantity requires explicit PhysicalDimension units")
        super().__init__(name, kind="global_quantity", owner=owner)
        object.__setattr__(self, "units", units)

    @property
    def _node(self):
        return GlobalQuantityRef(self)

    def __mul__(self, other): return self._node * other
    def __rmul__(self, other): return other * self._node
    def __add__(self, other): return self._node + other
    def __radd__(self, other): return other + self._node
    def __sub__(self, other): return self._node - other
    def __rsub__(self, other): return other - self._node
    def __truediv__(self, other): return self._node / other
    def __rtruediv__(self, other): return other / self._node
    def __neg__(self): return -self._node

    def declaration_data(self):
        return {"version": 1, "scope": "global", "units": self.units.to_data()}


class GlobalQuantityRef(Expr):
    """A model-declared input, never a RuntimeParam or a spatial field."""
    def __init__(self, handle):
        if not isinstance(handle, GlobalQuantityHandle):
            raise TypeError("physical global read requires a GlobalQuantityHandle")
        self.handle, self.units = handle, handle.units

    def __pops_ir_children__(self): return ()
    def __pops_ir_key__(self, recurse):
        return ("physical_global.v1", expression_handle_key(self.handle), self.units.powers)
    def __pops_ir_diff__(self, *, recurse, target, definitions):
        handle = target.handle if isinstance(target, GlobalQuantityRef) else target
        return Const(int(isinstance(handle, GlobalQuantityHandle) and handle == self.handle))
    def to_cpp(self):
        raise NotImplementedError("physical global quantity requires an explicit Program source binding; FieldProblem/flux consumers are not implemented")
    def eval(self, env): return env[self.handle.qualified_id]
    def _str(self): return "Global(%s)" % self.handle.local_id


def global_references(expressions):
    from pops._ir.visitors import _children
    found, seen = [], set()
    def visit(node):
        if isinstance(node, (tuple, list)):
            for item in node: visit(item)
        elif isinstance(node, Expr) and id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, GlobalQuantityRef): found.append(node)
            for child in _children(node): visit(child)
    visit(expressions)
    return tuple(found)
