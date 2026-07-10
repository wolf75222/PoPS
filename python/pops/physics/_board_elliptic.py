"""Atomic elliptic authoring for the blackboard physics facade."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .. import math as _bm
from ._board_contract import (atomic_attrs, normalize_sequence, normalize_string_mapping,
                              require_name)
from .board_handles import FieldHandle, FieldsHandle

if TYPE_CHECKING:
    from ._model_contract import _BoardModel
else:
    _BoardModel = object


class _EllipticAuthoringMixin(_BoardModel):
    """Build inert field problems and their operator-backed solve atomically."""

    def solve_field(self, name: Any, equation: Any = None, outputs: Any = None,
                    solver: Any = None) -> Any:
        """Declare ``-laplacian(field) == rhs`` and return its typed operator handle."""
        name = require_name(name, "solve_field name")
        self._validate_field_equation("solve_field", equation, require_laplacian=True)
        if name in self._fields or name in self._field_problems:
            raise ValueError("solve_field(%r) is already declared" % name)

        # Construct every public value first.  In particular, invalid output keys and
        # solver descriptors must fail before the elliptic RHS becomes visible.
        handle = FieldsHandle(
            name, outputs, solver, owner=self.owner_path,
            registered_operator_name="fields_from_state")
        normalized_outputs = dict(handle.outputs.items())
        for output in normalized_outputs.values():
            self._validate_output_field_reference(output, "solve_field output")
        model = self._dsl._m
        with atomic_attrs(
                (model, "aux_names"), (model, "aux_extra_names"), (model, "_elliptic"),
                (self, "_field_problems"), (self, "_fields"), (self, "_field_solvers")):
            rhs = self._to_expr(equation.rhs)
            if equation.lhs.scale > 0:
                rhs = -rhs
            problem = self._make_field_problem(
                name, equation, outputs=outputs, solver=solver)
            self._dsl.elliptic_rhs(rhs)
            self._field_problems[name] = problem
            self._fields[name] = handle
            if solver is not None:
                self._field_solvers[name] = solver
        return handle

    def field_problem(self, name: Any, equation: Any, outputs: Any = None, solver: Any = None,
                      bcs: Any = None, coefficients: Any = None) -> Any:
        """Construct and record one inert, inspectable elliptic problem descriptor."""
        name = require_name(name, "field_problem name")
        self._validate_field_equation("field_problem", equation, require_laplacian=False)
        if name in self._field_problems:
            raise ValueError("field_problem(%r) is already declared" % name)
        problem = self._make_field_problem(
            name, equation, outputs=outputs, solver=solver, bcs=bcs,
            coefficients=coefficients)
        self._field_problems[name] = problem
        return problem

    def _make_field_problem(self, name: str, equation: Any, *, outputs: Any = None,
                            solver: Any = None, bcs: Any = None,
                            coefficients: Any = None) -> Any:
        """Purely construct a descriptor; the caller decides when to publish it."""
        from pops import fields as _fields

        if isinstance(outputs, Mapping):
            outputs = normalize_string_mapping(outputs, "field problem outputs")
            for output in outputs.values():
                self._validate_output_field_reference(output, "field_problem output")
        bc_values = () if bcs is None else normalize_sequence(bcs, "field problem bcs")
        unknown = equation.lhs.field if isinstance(equation.lhs, _bm.Laplacian) else name
        cls = _fields.FieldProblem if coefficients is not None else _fields.PoissonProblem
        return cls(
            name=name, unknown=unknown, equation=equation, coefficients=coefficients,
            bcs=bc_values, outputs=outputs, solver=_fields.lower_field_solver(solver))

    def _validate_field_equation(self, where: str, equation: Any, *,
                                 require_laplacian: bool) -> None:
        if not isinstance(equation, _bm.Equation):
            raise TypeError(
                "%s expects a pops.math.Equation '-laplacian(field) == rhs'; got %r"
                % (where, type(equation).__name__))
        lhs = equation.lhs
        if require_laplacian and not isinstance(lhs, _bm.Laplacian):
            raise ValueError("%s left-hand side must be (-)laplacian(field); got %r"
                             % (where, lhs))
        if isinstance(lhs, _bm.Laplacian) and isinstance(lhs.field, FieldHandle):
            if (lhs.field.owner_path != self.owner_path
                    or self._fields.get(lhs.field.name) != lhs.field):
                raise ValueError(
                    "%s field handle %r belongs to another physics model"
                    % (where, lhs.field.name))

    def _validate_output_field_reference(self, value: Any, where: str) -> None:
        """Reject owner-mismatched field/gradient handles carried by output metadata."""
        field = None
        if isinstance(value, FieldHandle):
            field = value
        elif isinstance(value, (_bm.Partial, _bm.Gradient, _bm.Laplacian)):
            field = value.field
        if isinstance(field, FieldHandle):
            if (field.owner_path != self.owner_path
                    or self._fields.get(field.name) != field):
                raise ValueError(
                    "%s field handle %r belongs to another physics model"
                    % (where, field.name))


__all__ = ["_EllipticAuthoringMixin"]
