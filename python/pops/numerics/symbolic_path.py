"""Authored state paths lowered through the ordinary PoPS expression compiler.

The callback is expanded once when the method is declared. It is never called
from a cell/face loop. The physical product remains a separate declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SymbolicPath:
    """A numerical integral of ``B(Psi) dPsi`` and a whole-path speed bound.

    The path is ``left + s*jump + s*(1-s)*bend(left,right,s,axis) @ jump``.
    ``bend`` returns a square matrix of expressions; omitting it selects the
    straight path. This factorization preserves a constant path when the jump
    vanishes. A finite sequence of
    ``(node, weight)`` pairs declares its quadrature on [0, 1]; weights sum to
    one. PoPS integrates ``B(left)*jump`` exactly and applies the quadrature to
    ``(B(Psi)-B(left))*dPsi``. Thus a constant physical matrix is exact even if
    the chosen quadrature does not integrate a curved path's tangent exactly.
    An arbitrary integral cannot erase the physical product.

    ``speed(left, right, axis)`` must bound the complete ``DF + B`` along the
    chosen path. This is an author's mathematical obligation, not an inferred
    certificate; endpoint speeds alone do not prove it. Bodies construct
    ordinary expressions, with no callbacks during native execution.

    This native realization accepts state coordinates and captured runtime
    parameters. Spatially varying auxiliary inputs require an explicit trace
    realization and are rejected here. The finite-volume method owns its order,
    CFL check, shared conservative transfer and separate side contributions.
    """

    product: Any
    frame: Any
    integrals: tuple
    speeds: tuple
    left_symbols: tuple
    right_symbols: tuple
    covectors: tuple
    quadrature: tuple
    curves: tuple
    __pops_ir_immutable__ = True

    def __init__(self, product: Any, *, frame: Any, quadrature: Any,
                 speed: Any, bend: Any = None) -> None:
        from pops._ir.application import substitute_quantities
        from pops._ir.expr import Const, Var, _wrap
        from pops._ir.lowering import diff
        from pops.physics.nonconservative import NonconservativeProductHandle

        if not isinstance(product, NonconservativeProductHandle):
            raise TypeError("SymbolicPath requires a declared nonconservative product")
        if (tuple(axis.name for axis in frame.axes) != product.law.axes
                or frame.canonical_id != product.state.space.frame):
            raise ValueError("SymbolicPath requires the product's exact physical frame")
        if not callable(speed) or (bend is not None and not callable(bend)):
            raise TypeError("path bend and speed must be expression-building callables")
        try:
            rule = tuple((float(node), float(weight)) for node, weight in quadrature)
        except (TypeError, ValueError) as error:
            raise ValueError("path quadrature requires finite (node, weight) pairs") from error
        if (not rule or any(not math.isfinite(node) or not 0 <= node <= 1
                            or not math.isfinite(weight) for node, weight in rule)
                or not math.isclose(math.fsum(weight for _, weight in rule), 1.,
                                    rel_tol=0., abs_tol=8 * math.ulp(1.))):
            raise ValueError("path quadrature needs nodes in [0, 1] and weights summing to one")
        size = len(product.state.components)
        left = tuple(Var("pops_path_left_%d" % k, "path_left") for k in range(size))
        right = tuple(Var("pops_path_right_%d" % k, "path_right") for k in range(size))
        coordinate = Var("pops_path_coordinate", "path_coordinate")
        integrals, speeds, curves = [], [], []
        jump = tuple(b - a for a, b in zip(left, right, strict=True))
        for axis, matrix in enumerate(product.law.matrices):
            try:
                bending = (((Const(0),) * size,) * size if bend is None else
                           tuple(tuple(map(_wrap, row))
                                 for row in bend(left, right, coordinate, axis)))
            except TypeError as error:
                raise ValueError("path bend must be a square matrix on the declared state") from error
            if len(bending) != size or any(len(row) != size for row in bending):
                raise ValueError("path bend must be a square matrix on the declared state")
            _validate_body(tuple(value for row in bending for value in row),
                           (*left, coordinate), right)
            deviation = tuple(sum((value * delta for value, delta in
                                   zip(row, jump, strict=True)), Const(0)) for row in bending)
            curve = tuple(a + coordinate * delta
                          + coordinate * (1 - coordinate) * c
                          for a, delta, c in zip(left, jump, deviation, strict=True))
            tangent = tuple(diff(value, coordinate) for value in curve)
            baseline = substitute_quantities(matrix, {
                (product.state, k): value for k, value in enumerate(left)})
            terms = []
            for node, weight in rule:
                replacements = {id(value): value for value in (*left, *right)}
                replacements[id(coordinate)] = Const(node)
                states, tangents = substitute_quantities(
                    (curve, tangent), {}, expression_bindings=replacements)
                substituted = substitute_quantities(matrix, {
                    (product.state, k): value for k, value in enumerate(states)})
                terms.append(tuple(weight * sum(
                    ((coefficient - base) * delta for coefficient, base, delta in
                     zip(row, reference, tangents, strict=True)), Const(0))
                    for row, reference in zip(substituted, baseline, strict=True)))
            constant_part = tuple(sum((coefficient * delta for coefficient, delta
                                       in zip(row, jump, strict=True)), Const(0))
                                  for row in baseline)
            values = tuple(base + sum(row, Const(0))
                           for base, row in zip(constant_part, zip(*terms, strict=True), strict=True))
            bound = _wrap(speed(left, right, axis))
            _validate_body((*values, bound), left, right)
            integrals.append(values)
            speeds.append(bound)
            curves.append(curve)
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "integrals", tuple(integrals))
        object.__setattr__(self, "speeds", tuple(speeds))
        object.__setattr__(self, "left_symbols", left)
        object.__setattr__(self, "right_symbols", right)
        object.__setattr__(self, "quadrature", rule)
        object.__setattr__(self, "curves", tuple(curves))
        object.__setattr__(self, "covectors", tuple(
            tuple(Const(int(i == j)) for j in range(len(frame.axes)))
            for i in range(len(frame.axes))))

    def validate_flux(self, flux: Any) -> None:
        from pops._ir.quantity import QuantityRef
        from pops._ir.visitors import _children
        from pops._ir.expr import Var
        from pops.physics.board_handles import FluxHandle

        model = self.product._model_ref()
        if (not isinstance(flux, FluxHandle) or model is None
                or flux.owner_path != self.product.owner_path
                or model.fluxes.get(flux.name) != flux):
            raise ValueError("path and flux must belong to the same declaring Model")
        pending = [value for values in model.module.operator_registry().get(flux.reg_name).body.values()
                   for value in values]
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if ((isinstance(value, QuantityRef) and value.handle != self.product.state)
                    or isinstance(value, Var)):
                raise NotImplementedError("symbolic path flux needs explicit traces for non-state inputs")
            pending.extend(_children(value))

    def declaration_references(self) -> tuple:
        from pops._ir.expr_references import collect_reference_value
        result = [self.product]
        collect_reference_value((self.integrals, self.speeds, self.curves), result, set())
        return tuple(result)

    def resolve_references(self, resolver: Any) -> SymbolicPath:
        from pops._ir.expr_references import resolve_reference_value
        result = object.__new__(type(self))
        object.__setattr__(result, "product", resolver(self.product))
        object.__setattr__(result, "frame", self.frame)
        for key in ("integrals", "speeds", "left_symbols", "right_symbols", "covectors",
                    "quadrature", "curves"):
            object.__setattr__(result, key, resolve_reference_value(
                getattr(self, key), resolver, {}, allow_formula_vars=True))
        return result

    def native_kernel(self) -> dict:
        return {"kind": "symbolic_path", "integrals": self.integrals,
                "speeds": self.speeds, "left_symbols": self.left_symbols,
                "right_symbols": self.right_symbols,
                "parameter_expressions": (*[value for row in self.integrals for value in row],
                                          *self.speeds),
                "identity_namespace": "numerics.symbolic-path"}

    def validate_native(self, *, law: Any, flux_body: Any, native: Any) -> None:
        # The integral was constructed from this exact retained product. The common
        # lowerer authenticates the resolved law and method identities separately.
        from pops._ir.expr import Var
        from pops._ir.visitors import _children
        pending = [item for row in native(flux_body).values() for item in row]
        while pending:
            node = pending.pop()
            if isinstance(node, Var) and node.kind != "cons":
                raise NotImplementedError("symbolic path flux needs explicit traces for auxiliary inputs")
            pending.extend(_children(node))

    def to_data(self) -> dict:
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {"kind": "symbolic_path", "schema_version": 1,
                "product": _handle_data(self.product), "frame": self.frame.to_dict(),
                "integral": canonical_hash_data(self.integrals),
                "curve": canonical_hash_data(self.curves),
                "quadrature": self.quadrature,
                "quadrature_form": "exact_constant_matrix_plus_path_difference",
                "speed": canonical_hash_data(self.speeds),
                "face_geometry": "coordinate_direction",
                "stability": "authored_whole_path_bound"}


def _validate_body(expressions: tuple, left: tuple, right: tuple) -> None:
    from pops._ir.expr import Const, Var
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children

    arguments = {id(value) for value in (*left, *right)}
    pending, seen = list(expressions), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        children = _children(node)
        if isinstance(node, Var) and id(node) not in arguments:
            raise ValueError("path body contains a free variable instead of a path argument")
        if not children and id(node) not in arguments and not isinstance(node, (Const, RuntimeParamRef)):
            raise NotImplementedError("symbolic path body needs explicit traces for captured quantities")
        pending.extend(children)


__all__ = ["SymbolicPath"]
