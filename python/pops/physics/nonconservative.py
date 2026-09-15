"""Explicit physical products B(U, x) grad(U), independent of a weak-solution path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops import math as _math
from pops._ir.expr import Const, Expr, Partial, _wrap
from pops.model import Handle, Signature, FieldSpace


@dataclass(frozen=True, slots=True, eq=False)
class NonconservativeProductLaw:
    """One square state-space matrix for each physical coordinate derivative.

    The matrices retain the complete physical equation. A numerical path is a
    separate declaration and cannot replace this product with a flux divergence.
    """

    state: Handle
    variables: tuple[Expr, ...]
    matrices: tuple[tuple[tuple[Expr, ...], ...], ...]
    axes: tuple[str, ...]
    inputs: tuple[Any, ...]
    conservative_components: tuple[str, ...]
    __pops_ir_immutable__ = True

    @property
    def expressions(self) -> tuple[Expr, ...]:
        return tuple(value for matrix in self.matrices for row in matrix for value in row)

    def product_expressions(self) -> tuple[Expr, ...]:
        return tuple(sum((self.matrices[axis][row][column] * Partial(variable, axis)
                          for axis in range(len(self.axes))
                          for column, variable in enumerate(self.variables)), Const(0))
                     for row in range(len(self.variables)))

    def declaration_references(self) -> tuple[Handle, ...]:
        from pops._ir.expr_references import collect_reference_value
        result = [self.state]
        collect_reference_value((self.variables, self.expressions), result, set())
        return tuple(result)

    def resolve_references(self, resolver: Any) -> NonconservativeProductLaw:
        from pops._ir.expr_references import resolve_reference_value
        return NonconservativeProductLaw(
            resolver(self.state),
            resolve_reference_value(self.variables, resolver, {}, allow_formula_vars=True),
            resolve_reference_value(self.matrices, resolver, {}, allow_formula_vars=True),
            self.axes, self.inputs, self.conservative_components)

    def to_data(self) -> dict[str, Any]:
        from pops._ir.quantity import _hash_owner
        if not self.state.is_resolved and self.state.owner_path != _hash_owner.get():
            return self.resolve_references(
                lambda handle: handle._resolved(handle.owner_path.canonical())).to_data()
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {
            "kind": "nonconservative_product", "state": _handle_data(self.state),
            "variables": canonical_hash_data(self.variables),
            "directional_matrices": canonical_hash_data(self.matrices),
            "axes": self.axes, "inputs": [space.to_data() for space in self.inputs],
            "conservative_components": self.conservative_components,
        }


class NonconservativeProductHandle(Handle):
    """Owned identity usable as B grad(U) in a signed physical balance."""

    __slots__ = ("reg_name", "state", "law", "_model_ref")

    def __init__(self, name: str, law: NonconservativeProductLaw, *, model: Any) -> None:
        from weakref import ref
        super().__init__(name, kind="nonconservative_product", owner=model.owner_path)
        object.__setattr__(self, "reg_name", name)
        object.__setattr__(self, "state", law.state)
        object.__setattr__(self, "law", law)
        object.__setattr__(self, "_model_ref", ref(model))

    def __pops_rate_term__(self) -> Any:
        return _NonconservativeTerm(self)

    def __neg__(self) -> Any:
        return -self.__pops_rate_term__()

    def __add__(self, other: Any) -> Any:
        return self.__pops_rate_term__() + other

    def __radd__(self, other: Any) -> Any:
        return other + self.__pops_rate_term__()

    def __sub__(self, other: Any) -> Any:
        return self.__pops_rate_term__() - other

    def __rsub__(self, other: Any) -> Any:
        return other - self.__pops_rate_term__()

    def __mul__(self, coefficient: Any) -> Any:
        return self.__pops_rate_term__() * coefficient

    def __rmul__(self, coefficient: Any) -> Any:
        return self.__pops_rate_term__() * coefficient


class _NonconservativeTerm(_math.RateTerm):
    def __init__(self, handle: NonconservativeProductHandle) -> None:
        self.handle = handle

    def _rate_terms(self) -> Any:
        return [("nonconservative", self.handle, 1)]


def declare_nonconservative_product(model: Any, name: Any, *, state: Any,
                                    matrices: Any,
                                    conservative_components: Any = ()) -> NonconservativeProductHandle:
    from collections.abc import Mapping
    from ._board_contract import require_name
    from .board_handles import StateHandle
    from .diffusion import _authenticate_expression, _law_inputs

    model._guard_mutable("declare a physical nonconservative product")
    name = require_name(name, "nonconservative product name")
    if (not isinstance(state, StateHandle) or state.owner_path != model.owner_path
            or model._states.get(state.name) != state):
        raise ValueError("nonconservative product requires this Model's exact state declaration")
    if model.frame is None or not isinstance(matrices, Mapping):
        raise TypeError("nonconservative matrices require an explicit frame and typed axis mapping")
    if set(matrices) != set(model.frame.axes):
        raise ValueError("nonconservative matrices must cover every physical frame axis exactly once")
    size = len(state.components)
    tensors = []
    for axis in model.frame.axes:
        matrix = matrices[axis]
        if (not isinstance(matrix, (tuple, list)) or len(matrix) != size
                or any(not isinstance(row, (tuple, list)) or len(row) != size for row in matrix)):
            raise ValueError("each nonconservative matrix must have the complete state-space shape")
        tensors.append(tuple(tuple(_wrap(value) for value in row) for row in matrix))
    if isinstance(conservative_components, str):
        raise TypeError("conservative_components requires a sequence of exact component names")
    conserved = tuple(conservative_components)
    if len(set(conserved)) != len(conserved) or not set(conserved) <= set(state.components):
        raise ValueError("conservative components must be distinct names in the declared state")
    for component in conserved:
        row = state.components.index(component)
        if any(not isinstance(value, Const) or value.value != 0
               for matrix in tensors for value in matrix[row]):
            raise ValueError("a declared conservative component must have exactly zero product rows")
    expressions = tuple(value for matrix in tensors for row in matrix for value in row)
    for expression in expressions:
        _authenticate_expression(model, expression, state, label="nonconservative product")
    existing = getattr(model, "_nonconservative_products", {})
    registries = (existing, model._fluxes, model._sources,
                  getattr(model, "_diffusive_fluxes", {}), getattr(model, "_drift_fluxes", {}))
    if any(name in registry for registry in registries):
        raise ValueError("physical nonconservative product name is already declared")
    law = NonconservativeProductLaw(state, tuple(state), tuple(tensors),
        tuple(axis.name for axis in model.frame.axes), _law_inputs(model, state, expressions), conserved)
    handle = NonconservativeProductHandle(name, law, model=model)
    model._nonconservative_products = {**existing, name: handle}
    model._invalidate_authoring_views()
    return handle


def install_nonconservative_products(model: Any, module: Any) -> None:
    from pops.model.operators import Operator
    from pops.model import DeclarationIndex
    registry = module.operator_registry()
    for handle in getattr(model, "_nonconservative_products", {}).values():
        if handle.reg_name in registry.names():
            if registry.get(handle.reg_name).lowering.get("nonconservative_law") is handle.law:
                continue
            raise ValueError("nonconservative product collides with a registered operator")
        output = FieldSpace(handle.name + "_product", components=handle.state.components,
                            layout="cell", representation="nonconservative_product")
        registry.register(Operator(
            handle.reg_name, "expression", Signature(handle.law.inputs, output),
            body=handle.law.product_expressions(), lowering={"nonconservative_law": handle.law},
            requirements=model._dsl._m._aux_requirements(handle.law.expressions),
            capabilities={"local": False, "produces_rate": False}))
        declarations = module._register_operator_binding_authority(
            DeclarationIndex(owner=model.owner_path, handles=(handle,)))
        module._bind_operator(handle, module.operator_handle(handle.reg_name), declarations=declarations)


__all__ = ["NonconservativeProductHandle", "NonconservativeProductLaw"]
