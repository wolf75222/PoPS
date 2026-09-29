"""Joint native finite map application; support order is part of every identity."""
from .expr import Expr, _wrap


class FiniteApplication(Expr):
    def __init__(self, operation, source, target, coefficients, inputs):
        from pops.linalg.finite import FiniteSupport, FiniteLinearMap
        mapping = FiniteLinearMap(FiniteSupport(*source), FiniteSupport(*target), coefficients)
        if operation not in ("apply", "solve"):
            raise ValueError("unknown finite map operation")
        if operation == "solve" and len(mapping.source.dofs) != len(mapping.target.dofs):
            raise ValueError("finite solve requires a square map")
        self.operation = operation
        self.source, self.target = mapping.source.contract, mapping.target.contract
        self.coefficients = mapping.coefficients
        self.inputs = tuple(_wrap(x) for x in inputs)
        if len(self.inputs) != len((mapping.source if operation == "apply" else mapping.target).dofs):
            raise ValueError("finite map input width differs from its support")
        self.width = len((mapping.target if operation == "apply" else mapping.source).dofs)

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
