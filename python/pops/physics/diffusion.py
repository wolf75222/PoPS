"""Physical constitutive diffusion declarations, independent of numerical stencils."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops._ir.expr import Const, Expr, Gradient, Mul, Partial, Var, _wrap
from pops._ir.elliptic import CoeffGradient
from pops._ir.quantity import QuantityRef
from pops._ir.visitors import _children
from pops.model import Handle, Signature, FieldSpace


class DiffusiveFluxHandle(Handle):
    """An owned constitutive flux; it is never a hyperbolic Riemann flux."""

    __slots__ = ("reg_name", "state", "law")

    def __init__(self, name: str, law: DiffusiveFluxLaw, *, owner: Any) -> None:
        super().__init__(name, kind="diffusive_flux", owner=owner)
        object.__setattr__(self, "reg_name", name)
        object.__setattr__(self, "state", law.state)
        object.__setattr__(self, "law", law)


@dataclass(frozen=True, slots=True, eq=False)
class DiffusiveFluxLaw:
    """The physical law Fd=A grad(W), retaining every gradient/coefficient dependency."""

    state: Handle
    variable: Expr
    coefficients: tuple[tuple[Expr, ...], ...]
    axes: tuple[str, ...]
    inputs: tuple[Any, ...]
    __pops_ir_immutable__ = True

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def expressions(self) -> tuple[Expr, ...]:
        return (self.variable, *(item for row in self.coefficients for item in row))

    def declaration_references(self) -> tuple[Handle, ...]:
        from pops._ir.expr_references import collect_reference_value
        result = [self.state]
        collect_reference_value(self.expressions, result, set())
        return tuple(result)

    def resolve_references(self, resolver: Any) -> DiffusiveFluxLaw:
        from pops._ir.expr_references import resolve_reference_value
        return DiffusiveFluxLaw(
            resolver(self.state),
            resolve_reference_value(self.variable, resolver, {}, allow_formula_vars=True),
            resolve_reference_value(self.coefficients, resolver, {}, allow_formula_vars=True),
            self.axes, self.inputs,
        )

    def to_data(self) -> dict[str, Any]:
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        return {
            "kind": "constitutive_diffusive_flux", "state": _handle_data(self.state),
            "gradient_variable": canonical_hash_data(self.variable),
            "coefficient_tensor": canonical_hash_data(self.coefficients),
            "axes": self.axes, "inputs": [space.to_data() for space in self.inputs],
        }

    def flux_expressions(self) -> tuple[Expr, ...]:
        return tuple(sum((coefficient * Partial(self.variable, axis)
                          for axis, coefficient in enumerate(row)), Const(0))
                     for row in self.coefficients)


def _gradient_law(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Gradient):
        return value.field, value.scale
    if isinstance(value, CoeffGradient):
        coefficient = value.coeff
        if value.scale != 1:
            if isinstance(coefficient, (tuple, list)):
                raise TypeError("scaled tensor gradients require explicitly scaled tensor entries")
            coefficient = _wrap(coefficient) * value.scale
        return value.field, coefficient
    # Expr.__mul__ deliberately retains arithmetic. Recognize exactly one
    # gradient factor here without invoking a discrete product or chain rule.
    if isinstance(value, Mul):
        if isinstance(value.b, (Gradient, CoeffGradient)):
            variable, coefficient = _gradient_law(value.b)
            return variable, value.a * _wrap(coefficient)
        if isinstance(value.a, (Gradient, CoeffGradient)):
            variable, coefficient = _gradient_law(value.a)
            return variable, _wrap(coefficient) * value.b
    raise TypeError("diffusive_flux value must explicitly declare A*grad(W)")


def _coefficient_tensor(value: Any, dimension: int) -> tuple[tuple[Expr, ...], ...]:
    if not isinstance(value, (tuple, list)):
        coefficient = _wrap(value)
        return tuple(tuple(coefficient if i == j else Const(0)
                           for j in range(dimension)) for i in range(dimension))
    if len(value) != dimension:
        raise ValueError("diffusion coefficient rank differs from its physical frame")
    if any(isinstance(item, (tuple, list)) for item in value):
        if any(not isinstance(row, (tuple, list)) or len(row) != dimension for row in value):
            raise ValueError("diffusion tensor must be a complete square matrix")
        return tuple(tuple(_wrap(item) for item in row) for row in value)
    return tuple(tuple(_wrap(value[i]) if i == j else Const(0)
                       for j in range(dimension)) for i in range(dimension))


def _authenticate_expression(model: Any, expression: Expr, state: Any) -> None:
    pending = [expression]
    primitives = set()
    while pending:
        node = pending.pop()
        if isinstance(node, QuantityRef):
            if node.handle.owner_path != model.owner_path:
                raise ValueError("diffusive law reads a foreign quantity owner")
            if node.handle.kind == "state" and node.handle != state:
                raise ValueError("diffusive law reads a different state declaration")
        elif isinstance(node, Var):
            if node.kind == "prim" and node.name in model._dsl._m.prim_defs:
                if node.name not in primitives:
                    primitives.add(node.name)
                    pending.append(model._dsl._m.prim_defs[node.name])
            elif node.kind != "aux" or node.name not in model._dsl._m._provider_components:
                raise ValueError("diffusive law requires qualified state or declared field quantities")
        pending.extend(_children(node))
    for reference in expression.declaration_references():
        if reference.owner_path != model.owner_path:
            raise ValueError("diffusive law reads a foreign declaration")


def declare_diffusive_flux(model: Any, name: Any, *, state: Any, value: Any) -> DiffusiveFluxHandle:
    from ._board_contract import require_name
    from .board_handles import StateHandle

    model._guard_mutable("declare a constitutive diffusive flux")
    name = require_name(name, "diffusive flux name")
    if not isinstance(state, StateHandle) or model._states.get(state.name) != state:
        raise ValueError("diffusive_flux state must be this Model's exact state declaration")
    if len(state.components) != 1:
        raise ValueError("the bounded diffusion declaration requires a scalar evolved state")
    if model._multi_module is not None:
        raise ValueError("diffusion of a multi-state Module has no selected native realization")
    variable, coefficient = _gradient_law(value)
    if isinstance(variable, StateHandle):
        if variable != state:
            raise ValueError("diffusive gradient variable names a foreign state")
        variable = tuple(variable)[0]
    if not isinstance(variable, Expr):
        raise TypeError("diffusive gradient variable must be an explicit scalar expression")
    if model.frame is None:
        raise ValueError("diffusive_flux requires an explicit physical Cartesian frame")
    axes = tuple(axis.name for axis in model.frame.axes)
    coefficients = _coefficient_tensor(coefficient, len(axes))
    expressions = (variable, *(item for row in coefficients for item in row))
    for expression in expressions:
        _authenticate_expression(model, expression, state)
    inputs = [state.space]
    if model._dsl._m._aux_requirements(expressions):
        inputs.append(model._dsl._m.field_space())
    for expression in expressions:
        for reference in expression.declaration_references():
            if reference.kind == "field" and reference.space not in inputs:
                inputs.append(reference.space)
    existing = getattr(model, "_diffusive_fluxes", {})
    if name in existing or name in model._fluxes:
        raise ValueError("physical flux %r is already declared" % name)
    law = DiffusiveFluxLaw(state, variable, coefficients, axes, tuple(inputs))
    handle = DiffusiveFluxHandle(name, law, owner=model.owner_path)
    model._diffusive_fluxes = {**existing, name: handle}
    model._invalidate_authoring_views()
    return handle


def install_diffusive_fluxes(model: Any, module: Any) -> None:
    from pops.model.operators import Operator
    from pops.model import DeclarationIndex
    registry = module.operator_registry()
    for handle in getattr(model, "_diffusive_fluxes", {}).values():
        if handle.reg_name in registry.names():
            previous = registry.get(handle.reg_name)
            if previous.lowering.get("diffusive_law") is handle.law:
                continue
            raise ValueError("diffusive flux collides with a registered operator")
        output = FieldSpace(handle.name + "_flux", components=handle.law.axes,
                            layout="face", representation="constitutive_flux")
        registry.register(Operator(
            handle.reg_name, "expression", Signature(handle.law.inputs, output),
            body=handle.law.flux_expressions(), lowering={"diffusive_law": handle.law},
            capabilities={"local": False, "produces_rate": False},
        ))
        declarations = module._register_operator_binding_authority(
            DeclarationIndex(owner=model.owner_path, handles=(handle,)))
        module._bind_operator(handle, module.operator_handle(handle.reg_name),
                              declarations=declarations)


__all__ = ["DiffusiveFluxHandle", "DiffusiveFluxLaw"]
