"""Residual-operator authoring for :class:`pops.time.Program`."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pops.time.program_transaction import atomic_authoring
from pops.time.program_value_validation import require_compatible_spaces, require_owned
from pops.time.solve_outcome import ResidualSolution, SolveOutcome
from pops.time.values import ProgramValue, _resolve_handle

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


def _product(values: Any, *, where: str,
             component_order: Any = None) -> tuple[ProgramValue, ...]:
    if isinstance(values, Mapping):
        if component_order is None:
            keys = tuple(sorted(values))
        else:
            keys = tuple(component_order)
            if set(values) != set(keys):
                raise ValueError(
                    "%s mapping keys must exactly match residual unknown components %r"
                    % (where, keys))
        raw = tuple(values[key] for key in keys)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw = tuple(values)
    else:
        raw = (values,)
    resolved = tuple(_resolve_handle(value) for value in raw)
    if not resolved:
        raise ValueError("%s requires a non-empty unknown product" % where)
    if any(not isinstance(value, ProgramValue) or not value.is_field() for value in resolved):
        raise ValueError("%s unknowns must be Program field values" % where)
    return resolved


def _require_residual_operator(operator: Any) -> None:
    from pops.time.residual import ResidualOperator
    if not isinstance(operator, ResidualOperator):
        raise TypeError("residual: operator must be an immutable pops.time.ResidualOperator")
    operator.validate()


def _require_canonical_descriptor(value: Any, *, where: str) -> None:
    if value is None:
        return
    canonical = getattr(value, "canonical_identity", None)
    if getattr(value, "__pops_ir_immutable__", False) is not True or not callable(canonical):
        raise TypeError("%s must be an immutable typed descriptor or None" % where)
    canonical()


class _ProgramResidual(_ProgramBase):
    """Build residual evaluations and solver references without executing either in Python."""

    @atomic_authoring
    def residual(self, operator: Any, unknowns: Any, name: Any = None) -> Any:
        _require_residual_operator(operator)
        components = operator.unknown_space.components
        product = _product(unknowns, where="residual", component_order=components)
        for value in product:
            require_owned(self, value, "residual")
        if len(components) != len(product):
            raise ValueError(
                "residual: operator expects %d unknown component(s), got %d"
                % (len(components), len(product)))
        return self._new(
            "residual", "residual_eval", product, {"operator": operator}, name,
            None, point=product[0].point)

    @atomic_authoring
    def solve_residual(self, residual: Any, *, initial: Any,
                       solver: Any = None, preconditioner: Any = None,
                       name: Any = None, at: Any = None, **options: Any) -> Any:
        residual = require_owned(self, residual, "solve_residual", vtype="residual")
        operator = residual.attrs["operator"]
        initial_product = _product(
            initial, where="solve_residual",
            component_order=operator.unknown_space.components)
        for value in initial_product:
            require_owned(self, value, "solve_residual")
        if len(initial_product) != len(residual.inputs):
            raise ValueError("solve_residual: initial product arity does not match residual unknowns")
        for index, (unknown, guess) in enumerate(zip(
                residual.inputs, initial_product, strict=True)):
            require_compatible_spaces(
                unknown.space, guess.space, "solve_residual initial %d" % index,
                typed_pair=True)
        _require_canonical_descriptor(solver, where="solve_residual solver")
        if preconditioner is not None:
            try:
                from pops.time.residual import PreconditionerContract
            except ImportError as exc:
                raise TypeError(
                    "solve_residual preconditioner requires a typed PreconditionerContract"
                ) from exc
            if not isinstance(preconditioner, PreconditionerContract):
                raise TypeError(
                    "solve_residual preconditioner must be a typed PreconditionerContract")
            preconditioner.validate_for(operator)
        attrs = {"solver": solver, "preconditioner": preconditioner, "options": options,
                 "unknown_count": len(initial_product)}
        if at is None:
            points = tuple(value.point for value in initial_product)
        elif isinstance(at, Sequence) and not isinstance(at, (str, bytes)):
            points = tuple(at)
            if len(points) != len(initial_product):
                raise ValueError("solve_residual: at product arity must match initial product")
        else:
            points = (at,) * len(initial_product)
        token = self._new(
            "residual_solution", "solve_residual", (residual, *initial_product), attrs,
            name, None, point=points[0])
        outcome_name = name or "residual_solution"

        def project(outcome: Any) -> Any:
            values = tuple(
                self._new(
                    initial_value.vtype, "solve_outcome_component", (outcome,),
                    {"index": index},
                    "%s_%d" % (outcome_name, index),
                    initial_value.block, space=initial_value.space, point=points[index])
                for index, initial_value in enumerate(initial_product)
            )
            return ResidualSolution(values)

        return SolveOutcome(self, token, project, outcome_name)


__all__ = ["ResidualSolution", "SolveOutcome", "_ProgramResidual"]
