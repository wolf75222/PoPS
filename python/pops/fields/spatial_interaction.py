"""Inspectable physical kernels and explicit direct spatial quadrature contracts."""
from __future__ import annotations

from dataclasses import dataclass
import math

from pops._ir import expr as ast
from pops._ir.quantity import PhysicalDimension
from pops.identity.scalar import scalar_literal


def _tree(value, dimension):
    if type(value) is ast.Const:
        if value.handle is not None or type(value.value) not in (int, float):
            raise TypeError("interaction constants must be unannotated real literals")
        number = float(value.value)
        if not math.isfinite(number) or number != value.value:
            raise ValueError("interaction constants require finite lossless binary64 literals")
        return ("constant", number.hex())
    if type(value) is ast.Var:
        if value.kind not in ("interaction_target", "interaction_source"):
            raise ValueError("interaction kernel reads a foreign physical variable")
        prefix = "x" if value.kind == "interaction_target" else "y"
        if value.name not in tuple(prefix + str(i) for i in range(dimension)):
            raise ValueError("interaction coordinate is outside its declared dimension")
        return (prefix, int(value.name[1:]))
    binary = {ast.Add: "add", ast.Sub: "sub", ast.Mul: "mul", ast.Div: "div", ast.Pow: "pow"}
    unary = {ast.Neg: "neg", ast.Abs: "abs", ast.Sqrt: "sqrt", ast.Exp: "exp"}
    if type(value) in binary:
        return (binary[type(value)], _tree(value.a, dimension), _tree(value.b, dimension))
    if type(value) in unary:
        return (unary[type(value)], _tree(value.a, dimension))
    raise TypeError("interaction kernel requires an inspectable scalar coordinate Expr")


def kernel_cpp(tree, dimension):
    """Validate serialized kernels again, including detached/loaded Program graphs."""
    if not isinstance(tree, (tuple, list)) or not tree:
        raise ValueError("invalid spatial interaction kernel tree")
    op = tree[0]
    if op == "constant" and len(tree) == 2 and isinstance(tree[1], str):
        number = float.fromhex(tree[1])
        if not math.isfinite(number) or number.hex() != tree[1]:
            raise ValueError("invalid canonical interaction literal")
        return tree[1]
    if op in ("x", "y") and len(tree) == 2 and type(tree[1]) is int and 0 <= tree[1] < dimension:
        return "%s[%d]" % (op, tree[1])
    if op in ("add", "sub", "mul", "div", "pow") and len(tree) == 3:
        a, b = (kernel_cpp(part, dimension) for part in tree[1:])
        if op == "pow":
            return "Kokkos::pow(%s, %s)" % (a, b)
        return "(%s %s %s)" % (a, {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op], b)
    if op in ("neg", "abs", "sqrt", "exp") and len(tree) == 2:
        a = kernel_cpp(tree[1], dimension)
        return "(-%s)" % a if op == "neg" else "Kokkos::%s(%s)" % ({"abs": "fabs"}.get(op, op), a)
    raise ValueError("invalid spatial interaction kernel operation/arity")


@dataclass(frozen=True, init=False)
class SpatialInteractionKernel:
    """Physical W(x,y); builder runs once during authoring, never at runtime.

    Coordinates are physical cell centers. Periodic images or distance conventions
    must be authored in W; no implicit minimum-image or symmetry substitution exists.
    A scalar W acts independently on each explicitly selected source component.
    """
    dimension: int
    tree: tuple
    units: PhysicalDimension | None

    def __init__(self, dimension, builder, *, units=None):
        if type(dimension) is not int or not 1 <= dimension <= 3:
            raise ValueError("interaction kernel dimension must be 1, 2 or 3")
        if not callable(builder):
            raise TypeError("interaction kernel requires a build-time coordinate builder")
        if units is not None and type(units) is not PhysicalDimension:
            raise TypeError("interaction kernel units require a PhysicalDimension")
        x = tuple(ast.Var("x%d" % i, "interaction_target") for i in range(dimension))
        y = tuple(ast.Var("y%d" % i, "interaction_source") for i in range(dimension))
        expression = builder(x, y)
        if type(expression) in (int, float):
            expression = ast.Const(expression)
        tree = _tree(expression, dimension)
        kernel_cpp(tree, dimension)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "units", units)

    def to_data(self):
        return {"contract": "pops.spatial-interaction-kernel@1", "dimension": self.dimension,
                "tree": self.tree, "units": None if self.units is None else self.units.to_data()}


@dataclass(frozen=True)
class CellVolumeMeasure:
    """Physical Cartesian cell volume times the prepared EB volume fraction."""
    __pops_ir_immutable__ = True
    coordinate_units: tuple = ()

    def __post_init__(self):
        units = tuple(self.coordinate_units)
        if any(type(unit) is not PhysicalDimension for unit in units):
            raise TypeError("cell measure coordinate units require PhysicalDimensions")
        object.__setattr__(self, "coordinate_units", units)

    def to_data(self):
        return {"contract": "pops.cell-volume-eb@1", "coordinate_units": tuple(unit.to_data() for unit in self.coordinate_units)}


@dataclass(frozen=True)
class CellMidpoint:
    """Piecewise constant source values; W evaluated at physical cell centers."""
    __pops_ir_immutable__ = True

    def to_data(self):
        return {"contract": "pops.cell-midpoint@1"}


@dataclass(frozen=True)
class DirectSpatialInteraction:
    __pops_ir_immutable__ = True
    max_workspace_bytes: int

    def __post_init__(self):
        if type(self.max_workspace_bytes) is not int or not 0 < self.max_workspace_bytes < 2**64:
            raise ValueError("direct interaction budget requires a positive exact uint64")


@dataclass(frozen=True)
class FieldInteractionQuadrature:
    """Numerical realization of physical interactions inside the original F(q)."""
    __pops_ir_immutable__ = True
    measure: CellVolumeMeasure
    quadrature: CellMidpoint
    method: DirectSpatialInteraction

    def to_data(self):
        if type(self.measure) is not CellVolumeMeasure or type(self.quadrature) is not CellMidpoint or type(self.method) is not DirectSpatialInteraction:
            raise TypeError("original field interaction requires explicit volume/midpoint/direct descriptors")
        self.measure.__post_init__()
        self.method.__post_init__()
        return {"contract": "pops.original-field-interaction-realization@1",
                "measure": self.measure.to_data(), "quadrature": "pops.cell-midpoint@1",
                "method": "pops.direct-spatial-interaction@1",
                "max_workspace_bytes": (self.method.max_workspace_bytes if self.method.max_workspace_bytes < 2**63
                                        else scalar_literal(self.method.max_workspace_bytes).to_data())}
