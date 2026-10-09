"""Fan–Li physical/method composition, outside the compiler and core numerics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from pops._ir.path_arithmetic import PathArithmeticComposition
if TYPE_CHECKING:
    from pops.model import Handle
    from pops._ir.expr import Expr
    from pops.physics.nonconservative import NonconservativeProductHandle


def fan_li15_path(product, *, frame, covectors, basis):
    """Public physical/method composition with explicit monomial storage binding.

    The state slot at position k represents ``basis.indices[k]``. No component
    spelling, first slot or canonical physical storage is inferred by the core.
    """
    from pops._ir.expr import _wrap
    from pops.numerics.normalized_polynomial_path import NormalizedPolynomialPath
    from .fan_li import FAN_LI15_INDICES, fan_li15_expressions, fan_li15_native_plan
    if set(basis.indices) != set(FAN_LI15_INDICES):
        raise ValueError("Fan–Li15 declares the complete degree-four monomial basis")
    canonical_slots = tuple(basis.index(index) for index in FAN_LI15_INDICES)
    authored_slots = tuple(FAN_LI15_INDICES.index(index) for index in basis.indices)
    physical = fan_li15_expressions(tuple(product.law.variables[k] for k in canonical_slots))
    matrices, flux = [], {}
    for axis in frame.axes:
        direction = covectors[axis]
        values = physical.directional_flux(direction)
        flux[axis.name] = tuple(_wrap(values[k]) for k in authored_slots)
        matrix = physical.directional_nonconservative_matrix(direction)
        matrices.append(tuple(tuple(_wrap(matrix[i][j]) for j in authored_slots)
                              for i in authored_slots))
    return NormalizedPolynomialPath(product, frame=frame, covectors=covectors,
        plan=fan_li15_native_plan(basis=basis), flux=flux, matrices=tuple(matrices))

@dataclass(frozen=True, slots=True, eq=False, init=False)
class FanLi15RawMomentPath(PathArithmeticComposition):
    """Straight complete-raw-state path with the analytic Fan–Li15 integral.

    This names a weak-solution path, not a closure substitution. Its physical
    matrix is authenticated against the Fan–Li expressions for the exact supplied
    face covectors. The density-oriented polynomial/logarithm integral is part of
    its versioned numerical identity.
    """

    product: NonconservativeProductHandle
    frame: Any
    covectors: tuple[tuple[Expr, Expr], ...]
    plan: Any
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
        from .fan_li import fan_li15_native_plan
        from pops.problem._detached import detached_frozen
        object.__setattr__(self, "plan", detached_frozen(fan_li15_native_plan()))

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
        from pops.physics.board_handles import FluxHandle

        if not isinstance(flux, FluxHandle):
            raise TypeError("Fan–Li15 path requires a physical FluxHandle")
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
        for name in ("plan",):
            object.__setattr__(result, name, resolve_reference_value(
                getattr(self, name), resolver, {}, allow_formula_vars=True))
        return result

    def to_data(self) -> dict[str, Any]:
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {"kind": "fan_li15_straight_raw_moment_path", "schema_version": 2,
                "product": _handle_data(self.product), "frame": self.frame.to_dict(),
                "covectors": canonical_hash_data(self.covectors),
                "integral": "density_oriented_polynomial_logarithm_v1",
                "face_geometry": "arithmetic_trace_covector",
                "stability": "whole_path_raw_second_moment_bound",
                "normalized_arithmetic": self.plan}

    def native_kernel(self) -> dict[str, Any]:
        """The scientific library owns this optimized constitutive realization."""
        from pops.moments.fan_li import fan_li15_native_plan
        return {"kind": "normalized_polynomial_path", "plan": fan_li15_native_plan(),
                "identity_namespace": "fan-li15.path-operator"}

    def validate_native(self, *, law: Any, flux_body: Any, native: Any) -> None:
        """Reauthenticate the optimized library kernel after instance resolution."""
        from pops._ir.expr import _wrap
        from pops.model.hash_data import canonical_hash_data
        from pops.moments.fan_li import fan_li15_expressions, FAN_LI15_REGULARIZED_COMPONENTS
        from pops.moments.model_builder import moment_names
        if tuple(law.state.space.components) != tuple(moment_names(4)):
            raise ValueError("Fan–Li15 native path requires complete q-outer raw moment storage")
        conserved = tuple(name for slot, name in enumerate(moment_names(4))
                          if slot not in FAN_LI15_REGULARIZED_COMPONENTS)
        if law.conservative_components != conserved:
            raise ValueError("Fan–Li15 native path lost a conservative row")
        physical = fan_li15_expressions(native(law.variables))
        covectors = native(self.covectors)
        expected_b = tuple(tuple(tuple(_wrap(value) for value in row)
                                 for row in physical.directional_nonconservative_matrix(g))
                           for g in covectors)
        if canonical_hash_data(native(law.matrices)) != canonical_hash_data(expected_b):
            raise ValueError("native Fan–Li15 matrix differs from its authenticated constitutive law")
        expected_f = {axis: physical.directional_flux(g)
                      for axis, g in zip(law.axes, covectors, strict=True)}
        if canonical_hash_data(native(flux_body)) != canonical_hash_data(expected_f):
            raise ValueError("native Fan–Li15 speed proof requires the complete Grad flux plus B")
