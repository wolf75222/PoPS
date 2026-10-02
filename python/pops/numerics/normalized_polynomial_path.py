"""Public authored normalized polynomial path realization, contract @1."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


class PathArithmeticComposition:
    """Common typed interface for public library path compositions."""
    __slots__ = ()


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NormalizedPolynomialPath(PathArithmeticComposition):
    """Bind explicit arithmetic to an exact physical flux and product.

    ``flux`` and ``matrices`` are the author's physical expressions. ``plan``
    is the separately authored normalized arithmetic realization. Equivalence
    and its whole-path bound are mathematical obligations of the library,
    exactly as the bound of SymbolicPath is an author's obligation. PoPS checks
    supports, identities, arithmetic degrees and retained physical expressions;
    it does not recognize a model or infer a closure from those expressions.
    """
    product: Any
    state: Any
    frame: Any
    covectors: tuple
    plan: Any
    flux: Any
    matrices: Any
    __pops_ir_immutable__ = True

    def __init__(self, product, *, frame, covectors, plan, flux, matrices):
        from collections.abc import Mapping
        from pops._ir.expr import _wrap
        from pops.physics.nonconservative import NonconservativeProductHandle
        from pops.codegen.moment_path_kernel import emit_moment_path_kernel
        if not isinstance(product, NonconservativeProductHandle):
            raise TypeError("normalized path requires a declared physical product")
        if (tuple(axis.name for axis in frame.axes) != product.law.axes
                or frame.canonical_id != product.state.space.frame or len(frame.axes) != 2):
            raise ValueError("normalized path requires its exact supported physical frame")
        if not isinstance(covectors, Mapping) or set(covectors) != set(frame.axes):
            raise ValueError("path covectors must cover the exact physical frame axes")
        rows = tuple(tuple(_wrap(value) for value in covectors[axis]) for axis in frame.axes)
        if any(len(row) != 2 for row in rows):
            raise ValueError("path covectors must contain two coordinate components")
        from pops._ir.expr import Var
        from pops._ir.quantity import QuantityRef
        from pops._ir.visitors import _children
        pending = [value for row in rows for value in row]
        while pending:
            value = pending.pop()
            if ((isinstance(value, Var) and value.kind != "aux")
                    or isinstance(value, QuantityRef) and value.handle.kind == "state"):
                raise ValueError("path covectors require state-independent geometry")
            pending.extend(_children(value))
        emit_moment_path_kernel(plan, "ContractCheck")
        if len(plan["indices"]) != len(product.state.components):
            raise ValueError("path basis must cover its exact physical state")
        from pops.problem._detached import detached_frozen
        for name, value in (("product", product), ("state", product.state), ("frame", frame), ("covectors", rows),
                            ("plan", detached_frozen(plan)), ("flux", flux), ("matrices", matrices)):
            object.__setattr__(self, name, value)
        self._authenticate(matrices, product.law.matrices)

    def _authenticate(self, expected, actual):
        from pops._ir.quantity import local_expression_identity
        from pops.model.hash_data import canonical_hash_data
        with local_expression_identity(self.product.owner_path):
            if canonical_hash_data(expected) != canonical_hash_data(actual):
                raise ValueError("path realization does not match its exact retained physical expressions")

    def validate_flux(self, flux):
        from pops.physics.board_handles import FluxHandle
        model = self.product._model_ref()
        if (not isinstance(flux, FluxHandle) or model is None
                or flux.owner_path != self.product.owner_path or model.fluxes.get(flux.name) != flux):
            raise ValueError("path and flux must belong to their exact declaring Model")
        self._authenticate(self.flux, model.module.operator_registry().get(flux.reg_name).body)

    def declaration_references(self):
        from pops._ir.expr_references import collect_reference_value
        result = [self.product, self.state]
        collect_reference_value((self.covectors, self.flux, self.matrices), result, set())
        return tuple(result)

    def resolve_references(self, resolver):
        from pops._ir.expr_references import resolve_reference_value
        result = object.__new__(type(self))
        for name in ("product", "state", "frame", "covectors", "plan", "flux", "matrices"):
            value = getattr(self, name)
            object.__setattr__(result, name, resolver(value) if name in ("product", "state") else
                               resolve_reference_value(value, resolver, {}, allow_formula_vars=True))
        return result

    def native_kernel(self):
        from collections.abc import Mapping
        def plain(value):
            if isinstance(value, Mapping):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(plain(item) for item in value)
            return value
        return {"kind": "normalized_polynomial_path", "plan": plain(self.plan),
                "identity_namespace": "numerics.normalized-polynomial-path"}

    def validate_native(self, *, law, flux_body, native):
        from pops._ir.application import substitute_quantities
        # The numerical declaration holds qualified instance quantities, whereas
        # the selected native module holds its authenticated declaration state.
        # Rebind by the exact retained handle and index, never by component names.
        if len(law.variables) != len(self.plan["indices"]):
            raise ValueError("native path cannot change its exact component support")
        bindings = {(self.state, k): variable for k, variable in enumerate(law.variables)}
        self._authenticate(native(substitute_quantities(self.matrices, bindings)), native(law.matrices))
        self._authenticate(native(substitute_quantities(self.flux, bindings)), native(flux_body))

    def to_data(self):
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {"kind": "normalized_polynomial_path", "schema_version": 1,
                "product": _handle_data(self.product), "frame": self.frame.to_dict(),
                "covectors": canonical_hash_data(self.covectors), "arithmetic": self.plan,
                "face_geometry": "arithmetic_trace_covector",
                "integral": "authored_density_polynomial_logarithm_v1",
                "stability": "authored_whole_path_endpoint_majorant"}
