"""Version-one coordinated conservative/nonconservative face construction."""
from dataclasses import dataclass
from typing import Any

from .symbolic_path import SymbolicPath, _validate_body


@dataclass(frozen=True, slots=True)
class FaceBalance:
    """Flux density and already-signed sources for the left and right cells.

    ``stability`` is the author's nonnegative speed bound for the COMPLETE
    update, including both sources. It is not inferred from the physical flux.
    """
    flux: Any
    left: Any
    right: Any
    stability: Any


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CoordinatedFace:
    """One authored face body bound to an exact physical flux and product.

    ``body(left, right, axis) -> FaceBalance`` constructs ordinary PoPS IR once
    per positive coordinate direction. States are the adjacent stored cell
    averages. Runtime callbacks, auxiliary traces and within-cell reconstructions
    are not implied. Mathematical consistency with the declared PDE is the
    author's obligation; the compiler authenticates dependencies and coverage.
    """
    flux: Any
    product: Any
    frame: Any
    balances: tuple
    left_symbols: tuple
    right_symbols: tuple
    covectors: tuple
    __pops_ir_immutable__ = True

    def __init__(self, *, flux, product, frame, body):
        from pops.model.expression_language import Const, Var, _wrap
        from pops.physics.nonconservative import NonconservativeProductHandle
        if not isinstance(product, NonconservativeProductHandle):
            raise TypeError("CoordinatedFace requires an exact nonconservative product")
        if (tuple(axis.name for axis in frame.axes) != product.law.axes
                or frame.canonical_id != product.state.space.frame):
            raise ValueError("coordinated face requires the product's exact frame")
        if not callable(body):
            raise TypeError("coordinated face body must be an expression-building callable")
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "frame", frame)
        self.validate_flux(flux)
        size = len(product.state.components)
        left = tuple(Var("pops_coord_left_%d" % k, "coordinated_left") for k in range(size))
        right = tuple(Var("pops_coord_right_%d" % k, "coordinated_right") for k in range(size))
        balances = []
        for axis in range(len(frame.axes)):
            output = body(left, right, axis)
            if type(output) is not FaceBalance:
                raise TypeError("coordinated face body must return FaceBalance")
            rows = []
            for name in ("flux", "left", "right"):
                values = getattr(output, name)
                if not isinstance(values, tuple) or len(values) != size:
                    raise TypeError("FaceBalance.%s requires one scalar per state component" % name)
                rows.append(tuple(map(_wrap, values)))
            speed = _wrap(output.stability)
            _validate_body((*rows[0], *rows[1], *rows[2], speed), left, right)
            for component in product.law.conservative_components:
                index = product.state.components.index(component)
                if any(not isinstance(row[index], Const) or row[index].value != 0
                       for row in rows[1:]):
                    raise ValueError("conservative component %s requires literal zero side sources" % component)
            balances.append(FaceBalance(*rows, speed))
        object.__setattr__(self, "balances", tuple(balances))
        object.__setattr__(self, "left_symbols", left)
        object.__setattr__(self, "right_symbols", right)
        object.__setattr__(self, "covectors", tuple(tuple(Const(int(i == j))
            for j in range(len(frame.axes))) for i in range(len(frame.axes))))

    def validate_flux(self, flux):
        if flux != self.flux:
            raise ValueError("coordinated face changes its exact physical flux")
        SymbolicPath.validate_flux(self, flux)

    def _expressions(self):
        return tuple(value for row in self.balances
                     for value in (*row.flux, *row.left, *row.right, row.stability))

    def declaration_references(self):
        from pops.model.expression_language import collect_reference_value
        result = [self.flux, self.product]
        collect_reference_value(self._expressions(), result, set())
        return tuple(result)

    def resolve_references(self, resolver):
        from pops.model.expression_language import resolve_reference_value
        result = object.__new__(type(self))
        for name in ("flux", "product"):
            object.__setattr__(result, name, resolver(getattr(self, name)))
        object.__setattr__(result, "frame", self.frame)
        for name in ("left_symbols", "right_symbols", "covectors"):
            object.__setattr__(result, name, getattr(self, name))
        def resolve(value):
            return resolve_reference_value(value, resolver, {}, allow_formula_vars=True)
        object.__setattr__(result, "balances", tuple(FaceBalance(
            resolve(row.flux), resolve(row.left), resolve(row.right), resolve(row.stability))
            for row in self.balances))
        return result

    def native_kernel(self):
        return {"kind": "coordinated_face", "interface_contract": 1,
                "balances": self.balances, "left_symbols": self.left_symbols,
                "right_symbols": self.right_symbols,
                "parameter_expressions": self._expressions(),
                "identity_namespace": "numerics.coordinated-face.v1"}

    def validate_native(self, *, law, flux_body, native):
        SymbolicPath.validate_native(self, law=law, flux_body=flux_body, native=native)

    def to_data(self):
        from pops.model.expression_language import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {"kind": "coordinated_face", "schema_version": 1,
                "flux": _handle_data(self.flux), "product": _handle_data(self.product),
                "frame": self.frame.to_dict(),
                "body": canonical_hash_data(self._expressions()),
                "orientation": "positive_axis_left_to_right",
                "side_sign": "already_signed_cell_rhs",
                "stencil": "two_adjacent_cell_averages",
                "stability": "authored_complete_face_speed_bound",
                "publication": "atomic_shared_flux_two_sides_speed"}


__all__ = ["FaceBalance", "CoordinatedFace"]
