"""Declared raw-moment paths and first-order path-conservative finite volumes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from pops.model import Handle
from .spatial import FiniteVolume, _brick_data, _resolved_brick

if TYPE_CHECKING:
    from pops._ir.expr import Expr
    from pops.physics.nonconservative import NonconservativeProductHandle


def path_balance_supported(view: Any) -> bool:
    """Exact equation shape supported by the single-flux/product native route."""
    if view is None or not view.accumulation.is_identity:
        return False
    rows = view.occurrences
    return (len(rows) == 2
            and sum(row.kind == "flux" for row in rows) == 1
            and sum(row.kind == "nonconservative" for row in rows) == 1
            and all(row.coefficient == -1 for row in rows))


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FanLi15RawMomentPath:
    """Straight complete-raw-state path with the analytic Fan–Li15 integral.

    This names a weak-solution path, not a closure substitution. Its physical
    matrix is authenticated against the Fan–Li expressions for the exact supplied
    face covectors. The density-oriented polynomial/logarithm integral is part of
    its versioned numerical identity.
    """

    product: NonconservativeProductHandle
    frame: Any
    covectors: tuple[tuple[Expr, Expr], ...]
    __pops_ir_immutable__ = True

    def __init__(self, product: Any, *, frame: Any, covectors: Any) -> None:
        from collections.abc import Mapping
        from pops._ir.expr import _wrap
        from pops.physics.nonconservative import NonconservativeProductHandle

        # Physical declarations are authenticated when a path is constructed. Importing the
        # numerical descriptor catalog itself does not enter that authoring phase.
        if not isinstance(product, NonconservativeProductHandle):
            raise TypeError("FanLi15RawMomentPath requires a physical nonconservative product")
        if (not hasattr(frame, "axes") or len(frame.axes) != 2
                or tuple(axis.name for axis in frame.axes) != product.law.axes
                or frame.canonical_id != product.state.space.frame):
            raise ValueError("FanLi15RawMomentPath requires the product's exact two-dimensional frame")
        if not isinstance(covectors, Mapping) or set(covectors) != set(frame.axes):
            raise ValueError("path covectors must cover the two exact physical frame axes")
        values = []
        for axis in frame.axes:
            row = covectors[axis]
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("each Fan–Li path covector must contain two Cartesian velocity components")
            values.append(tuple(_wrap(value) for value in row))
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "covectors", tuple(values))
        from pops._ir.expr import Var
        from pops._ir.visitors import _children
        from pops._ir.quantity import QuantityRef
        pending = [value for row in values for value in row]
        while pending:
            node = pending.pop()
            if ((isinstance(node, Var) and node.kind != "aux")
                    or (isinstance(node, QuantityRef) and node.handle.kind == "state")):
                raise ValueError("Fan–Li path covectors must be state-independent geometry")
            pending.extend(_children(node))
        self.validate_product()

    def validate_product(self) -> None:
        from pops._ir.expr import _wrap
        from pops._ir.quantity import local_expression_identity
        from pops.model.hash_data import canonical_hash_data
        from pops.moments.fan_li import fan_li15_expressions, FAN_LI15_REGULARIZED_COMPONENTS
        from pops.moments.model_builder import moment_names

        law = self.product.law
        if tuple(self.product.state.components) != tuple(moment_names(4)):
            raise ValueError("Fan–Li15 path requires the exact q-outer raw moment ordering")
        conserved = tuple(name for slot, name in enumerate(moment_names(4))
                          if slot not in FAN_LI15_REGULARIZED_COMPONENTS)
        if law.conservative_components != conserved:
            raise ValueError("Fan–Li15 path requires all ten conservative moment rows explicitly")
        expected = fan_li15_expressions(law.variables)
        matrices = tuple(tuple(tuple(_wrap(value) for value in row)
                               for row in expected.directional_nonconservative_matrix(g))
                         for g in self.covectors)
        with local_expression_identity(self.product.owner_path):
            if canonical_hash_data(matrices) != canonical_hash_data(law.matrices):
                raise ValueError("analytic Fan–Li path integral does not match the declared physical product")

    def validate_flux(self, flux: Handle) -> None:
        """The analytic speed proof applies to DF_Grad+B, not an arbitrary flux."""
        from pops._ir.quantity import local_expression_identity
        from pops.model.hash_data import canonical_hash_data
        from pops.moments.fan_li import fan_li15_expressions
        model = self.product._model_ref()
        if model is None:
            raise ValueError("Fan–Li15 path flux authentication requires its declaring Model")
        if flux.owner_path != self.product.owner_path or model._fluxes.get(flux.name) != flux:
            raise ValueError("Fan–Li15 path requires its exact Model's physical flux")
        module = model.module
        body = module.operator_registry().get(flux.reg_name).body
        physical = fan_li15_expressions(self.product.law.variables)
        expected = {axis: physical.directional_flux(g)
                    for axis, g in zip(self.product.law.axes, self.covectors, strict=True)}
        with local_expression_identity(self.product.owner_path):
            if canonical_hash_data(body) != canonical_hash_data(expected):
                raise ValueError("Fan–Li15 path speed requires the complete declared Grad flux plus B")

    def declaration_references(self) -> tuple[Handle, ...]:
        from pops._ir.expr_references import collect_reference_value
        result = [self.product]
        collect_reference_value(self.covectors, result, set())
        return tuple(result)

    def resolve_references(self, resolver: Any) -> FanLi15RawMomentPath:
        from pops._ir.expr_references import resolve_reference_value
        result = object.__new__(type(self))
        object.__setattr__(result, "product", resolver(self.product))
        object.__setattr__(result, "frame", self.frame)
        object.__setattr__(result, "covectors", resolve_reference_value(
            self.covectors, resolver, {}, allow_formula_vars=True))
        return result

    def to_data(self) -> dict[str, Any]:
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {"kind": "fan_li15_straight_raw_moment_path", "schema_version": 1,
                "product": _handle_data(self.product), "frame": self.frame.to_dict(),
                "covectors": canonical_hash_data(self.covectors),
                "integral": "density_oriented_polynomial_logarithm_v1",
                "face_geometry": "arithmetic_trace_covector",
                "stability": "whole_path_raw_second_moment_bound"}


class PathConservativeFiniteVolume(FiniteVolume):
    """Shared conservative flux plus distinct signed nonconservative side terms.

    The first executable route uses complete raw states and no spatial slopes.
    Higher-order reconstruction requires an additional within-cell path integral.
    """

    category = "path_conservative_finite_volume"

    def __init__(self, *, flux: Any, path: Any, variables: Any,
                 reconstruction: Any, riemann: Any, zero_measure_faces: Any = ()) -> None:
        if type(path) is not FanLi15RawMomentPath:
            raise TypeError("this path-conservative route requires a typed FanLi15RawMomentPath")
        super().__init__(flux=flux, variables=variables, reconstruction=reconstruction, riemann=riemann)
        if (isinstance(flux, tuple) or flux.owner_path != path.product.owner_path
                or self.variables.options.get("state") != path.product.state):
            raise ValueError("path, physical flux and reconstruction must own the same exact state")
        if (self.formal_order != 1 or self.reconstruction.name != "firstorder"
                or self.variables.scheme != "conservative"
                or self.riemann.native_id != "pops::RusanovFlux"):
            raise ValueError("raw path-conservative transport currently requires FirstOrder and Rusanov")
        path.validate_flux(flux)
        self.path = path
        from pops.domain.rectangle import DomainBoundary
        faces = tuple(zero_measure_faces)
        expected = () if not faces else path.frame.boundaries.all
        if (any(type(face) is not DomainBoundary or face not in expected for face in faces)
                or len(set(faces)) != len(faces)):
            raise ValueError("zero_measure_faces must name distinct exact boundaries of the path frame")
        # This is an explicit physical degeneracy contract, independent of a NoFlux policy.
        # Bounded physical moments at these zero-measure faces give zero F and path terms.
        self.zero_measure_faces = faces

    def options(self) -> dict[str, Any]:
        return {**super().options(), "path": self.path, "zero_measure_faces": self.zero_measure_faces}

    def validate_rate_contract(self, contract: Any) -> bool:
        if (contract.get("flux") != self.flux
                or contract.get("state") != self.variables.options.get("state")
                or contract.get("nonconservative_products") != (self.path.product,)
                or contract.get("sources")):
            raise ValueError("path-conservative method must cover the exact flux/product/state balance")
        return True

    def validate_balance_view(self, view: Any) -> bool:
        if not path_balance_supported(view):
            raise ValueError("path-conservative transport requires exactly -div(F) - B grad(U)")
        if any((row.kind == "flux" and row.payload != self.flux)
               or (row.kind == "nonconservative" and row.payload != self.path.product)
               for row in view.occurrences):
            raise ValueError("path-conservative construction does not match the retained balance")
        return True

    def resolve_references(self, resolver: Any) -> PathConservativeFiniteVolume:
        # Resolution preserves the captured law and path; it does not rebuild the
        # physical expressions using newly rebound, potentially different symbols.
        result = object.__new__(type(self))
        for name in ("variables", "reconstruction", "riemann"):
            setattr(result, name, _resolved_brick(getattr(self, name), resolver))
        result.flux = resolver(self.flux)
        result.path = self.path.resolve_references(resolver)
        result.positivity_floor = None
        result.zero_measure_faces = self.zero_measure_faces
        return result

    def to_data(self) -> dict[str, Any]:
        if not self.flux.is_resolved or not self.path.product.is_resolved:
            raise ValueError("PathConservativeFiniteVolume.to_data requires resolved physical handles")
        return {"schema_version": 1, "method": "path_conservative_finite_volume",
                "flux": self.flux.canonical_identity(), "path": self.path.to_data(),
                "variables": _brick_data(self.variables),
                "reconstruction": _brick_data(self.reconstruction),
                "riemann": _brick_data(self.riemann),
                "formal_order": self.formal_order, "ghost_depth": self.ghost_depth,
                "positivity_floor": None,
                "zero_measure_faces": [face.to_dict() for face in self.zero_measure_faces],
                "nonconservative_interfaces": "canonical_fine_subface_side_contributions"}


__all__ = ["FanLi15RawMomentPath", "PathConservativeFiniteVolume"]
