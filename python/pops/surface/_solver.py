"""``@pops.solver``: a solve algorithm written in the words of the surface (spec 0.3 SV9, SU15).

A solver is a function decorated with ``@pops.solver``. PoPS supplies its two entities, ``problem``
and ``region`` (2.6.4); every other parameter is an option given by keyword at the call, and the
call records its options and runs nothing (SU5, SU8): ``cg(max_iterations=500)``. The value it
returns is the realization of a solve, the value of ``solvers={key: ...}`` (SV4).

When the solve is lowered, PoPS runs the body once, with:

* ``problem``: an affine problem in one field unknown, read through the vector-space interface of
  SV19 and nothing else: ``apply(v)``, ``rhs``, ``guess``, ``pairing(a, b)``, ``request`` and
  ``require(...)`` (the protocol of a solver reference, C46, filled for this case only);
* ``region``: the smallest slice of the region level that a Krylov method needs (SU15):
  ``region.iterate(start=, step=, stop=, max_steps=)``.

In the body, ``pops.record``, ``pops.choose`` and ``pops.fail`` are the nodes ``Product``,
``cond`` and ``fail`` of IR4 (2.6.8), elaborated inside a solver only. The body lowers onto the
Program: one ``dot_all`` per pairing, ``iterate`` and ``branch``. The failures are variants of T39
(``pops.BudgetExhausted``, ``pops.SolveFailed``), each with its reason, the same on the native
route (N9 lot 4, C2 and D3). The function's own name enters no identity (O1).
"""
from __future__ import annotations

import inspect
import math as _pymath
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ._refusal import NotYetElaborated, SurfaceRefusal, SymbolicTruthValue, surface_entry
from . import _run
from ._run import BudgetExhausted, SolveFailed

_ENTITIES = ("problem", "region")
_N9_LATER = "a later lot of N9"
_N9_EVIDENCE = "the later N9 lot « Évidence des solves »"
_G10 = "G10 (the generated operator of a term)"


class PropertyNotDerived(SurfaceRefusal, ValueError):
    """A realization whose precondition is not derived (K4, SV2)."""

    rules = ("K4", "SV2", "IR16")


# --- the failures (T39) -------------------------------------------------------------------------

# The variants of T39 are the classes that a run raises (N15 lot G); a solver fails with them by
# name, and its outcome reports the same name and reason on the native route and the reference.
FAILURE_VARIANTS = (BudgetExhausted, SolveFailed)

# The failure codes of the lowered region and the (variant, reason) each one carries. Code 1 is
# the budget of Program.iterate itself; the reference never relies on it (it fails at its cap).
FAILURE_CODES = {
    (BudgetExhausted.__name__, "iteration limit"): 11,
    (SolveFailed.__name__, "breakdown"): 21,
    (SolveFailed.__name__, "invalid evaluation"): 22,
    (SolveFailed.__name__, "unverified solved value"): 23,
}
FAILURES_BY_CODE = {code: key for key, code in FAILURE_CODES.items()}
FAILURES_BY_CODE[1] = (BudgetExhausted.__name__, "iteration limit")
# Code 3 of iterate, origin 2: a native evaluation that the region rejected, a value not finite.
FAILURES_BY_CODE[3] = (SolveFailed.__name__, "invalid evaluation")


# --- the lowering context -----------------------------------------------------------------------

class _Lowering:
    """The Program and the names of one solver lowering."""

    def __init__(self, program: Any, name: str) -> None:
        self.program = program
        self.name = name
        self.count = 0
        self.templates: list = []
        # The block of the problem's fields, which owns the scalars that its pairings reduce, and
        # the point of the problem, at which the solver's literal scalars are made.
        self.block: Any = None
        self.point: Any = None

    def fresh(self, what: str) -> str:
        self.count += 1
        return "%s_%s%d" % (self.name, what, self.count)


_ACTIVE: list = []

# The lot that elaborates each word outside a solver; the surface fills it with its owners.
OUTSIDE_OWNERS: dict = {}


@contextmanager
def _lowering(context: _Lowering):
    _ACTIVE.append(context)
    try:
        yield context
    finally:
        _ACTIVE.pop()


def _active(word: str) -> _Lowering:
    if not _ACTIVE:
        owner = OUTSIDE_OWNERS.get(word, "a later lot")
        raise NotYetElaborated("%s outside a solver" % word,
                               "%s (it is elaborated inside @pops.solver only)" % owner)
    return _ACTIVE[-1]


# --- values inside a solver ---------------------------------------------------------------------

def _scalar_input(value: Any, where: str) -> Any:
    from ._geometry import MathConstant

    if isinstance(value, Scalar):
        return value.value
    if isinstance(value, MathConstant):
        return float(value.value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError("%s takes a scalar of the solver or a number, got %r" % (where, value))


class Scalar:
    """A Scalar or a Bool of the lowered region: a reduced value, identical on every rank."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def _context(self) -> _Lowering:
        return _active("a scalar of a solver")

    def _materialize(self, expression: Any, what: str) -> Scalar:
        context = self._context()
        return Scalar(context.program.scalar_expression(context.fresh(what), expression))

    def _binary(self, other: Any, fn: Any, swap: bool = False) -> Any:
        if isinstance(other, Field):
            return other._scaled_by(self, fn, swap)
        rhs = _scalar_input(other, "arithmetic")
        left, right = (rhs, self.value) if swap else (self.value, rhs)
        result = fn(left, right)
        if isinstance(result, type(self.value)):
            return Scalar(result)
        return self._materialize(result, "s")

    def __add__(self, other): return self._binary(other, lambda a, b: a + b)
    def __radd__(self, other): return self._binary(other, lambda a, b: a + b, True)
    def __sub__(self, other): return self._binary(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._binary(other, lambda a, b: a - b, True)
    def __mul__(self, other): return self._binary(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._binary(other, lambda a, b: a * b, True)
    def __truediv__(self, other): return self._binary(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._binary(other, lambda a, b: a / b, True)
    def __neg__(self): return self._binary(-1.0, lambda a, b: a * b)

    def _compare(self, other: Any, cmp: str) -> Scalar:
        program = self._context().program
        rhs = _scalar_input(other, "a comparison")
        if isinstance(rhs, float) and _pymath.isinf(rhs) and rhs > 0:
            if cmp == "<":
                # v < infinity holds for a finite value only; NaN and +inf fail it (D2).
                return Scalar(program.isfinite(self.value))
        return Scalar(program._compare(self.value, rhs, cmp))

    def __lt__(self, other): return self._compare(other, "<")
    def __le__(self, other): return self._compare(other, "<=")
    def __gt__(self, other): return self._compare(other, ">")
    def __ge__(self, other): return self._compare(other, ">=")
    def __eq__(self, other): return self._compare(other, "==")  # type: ignore[override]
    def __ne__(self, other): return self._compare(other, "!=")  # type: ignore[override]
    __hash__ = object.__hash__

    def __and__(self, other): return Scalar(self.value & _bool(other))
    def __rand__(self, other): return Scalar(_bool(other) & self.value)
    def __or__(self, other): return Scalar(self.value | _bool(other))
    def __ror__(self, other): return Scalar(_bool(other) | self.value)
    def __invert__(self): return Scalar(~self.value)

    def __abs__(self) -> Scalar:
        from pops._ir.ops import abs_
        return self._materialize(abs_(self.value), "abs")

    def __pops_elementary__(self, word: str) -> Scalar:
        if word != "sqrt":
            raise NotYetElaborated("%s of a scalar of a solver" % word, _N9_LATER)
        return self._materialize(self.value ** 0.5, "sqrt")

    def __bool__(self) -> bool:
        raise SymbolicTruthValue(
            "a scalar of a solver is computed when the solve runs and has no Python truth value",
            write="pops.choose(p, then=f, otherwise=g)")

    def __repr__(self) -> str:
        return "<solver scalar %s>" % getattr(self.value, "name", self.value)


def _bool(value: Any) -> Any:
    if isinstance(value, Scalar) and getattr(value.value, "vtype", None) == "bool":
        return value.value
    raise TypeError("& and | combine predicates of a solver, got %r" % (value,))


class Field:
    """A field of the lowered region (the unknown's space), or an affine combination of fields."""

    __slots__ = ("expression", "template")

    def __init__(self, expression: Any, template: Any) -> None:
        self.expression = expression
        self.template = template

    def value(self, what: str = "f") -> Any:
        """The field as one Program value, materialized at the template's point."""
        from pops.time.values import ProgramValue

        if isinstance(self.expression, ProgramValue):
            return self.expression
        context = _active("a field of a solver")
        return context.program.value(context.fresh(what), self.expression,
                                     at=self.template.point)

    def _combine(self, other: Any, sign: float) -> Field:
        if not isinstance(other, Field):
            raise TypeError("a field of a solver adds only another field, got %r" % (other,))
        if sign > 0:
            return Field(self.expression + other.expression, self.template)
        return Field(self.expression - other.expression, self.template)

    def __add__(self, other): return self._combine(other, 1.0)
    def __sub__(self, other): return self._combine(other, -1.0)

    def _scaled_by(self, factor: Scalar, fn: Any, swap: bool) -> Field:
        if fn(2.0, 3.0) != 6.0:
            raise TypeError("a field of a solver is scaled by a scalar only")
        return Field(factor.value * self.expression, self.template)

    def __truediv__(self, other):
        """v / s, each value divided by the scalar (the native's normalized difference)."""
        from pops.time.expressions import as_expression

        divisor = other.value if isinstance(other, Scalar) else \
            _scalar_input(other, "a divided field")
        return Field(as_expression(self.expression) / divisor, self.template)

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return Field(other.value * self.expression, self.template)
        return Field(_scalar_input(other, "a scaled field") * self.expression, self.template)

    __rmul__ = __mul__

    def __bool__(self) -> bool:
        raise SymbolicTruthValue("a field of a solver has no Python truth value",
                                 write="a pairing problem.pairing(a, b) compared with a number")

    def __repr__(self) -> str:
        return "<solver field>"


class Record:
    """``pops.record(...)`` inside a solver: named fields and scalars (IR4 ``Product``)."""

    __slots__ = ("_items",)

    def __init__(self, items: dict) -> None:
        object.__setattr__(self, "_items", dict(items))

    def __getattr__(self, name: str) -> Any:
        items = object.__getattribute__(self, "_items")
        if name in items:
            return items[name]
        raise AttributeError("the record has no field %r; it has %s" % (name, sorted(items)))

    def program_value(self) -> Any:
        context = _active("pops.record")
        members = {}
        for name, item in self._items.items():
            if isinstance(item, Field):
                members[name] = item.value(name)
            elif isinstance(item, Scalar):
                value = item.value
                if getattr(value, "state_ref", None) is not None:
                    # A scalar is a reduced number: it carries no State authority, so that the
                    # arms of pops.choose have one structural type (branch, iterate).
                    value = context.program.scalar_expression(context.fresh(name), value)
                members[name] = value
            else:
                members[name] = _number_scalar(context, item, name)
        return context.program.product(**members)

    @staticmethod
    def from_program(product: Any, template: Record) -> Record:
        context = _active("pops.record")
        items = {}
        for name, item in template._items.items():
            value = context.program._product_get(product, name)
            items[name] = Field(value, value) if isinstance(item, Field) else Scalar(value)
        return Record(items)


def _number_scalar(context: _Lowering, number: Any, name: str) -> Any:
    value = _scalar_input(number, "a record")
    return _owned_scalar(context, context.program.scalar_expression(context.fresh(name), value))


def _owned_scalar(context: _Lowering, value: Any) -> Any:
    """A literal Scalar of the solver in the block of the problem's fields, as its reduced
    scalars are, so that a record or a choice mixes them with one structural type."""
    if value.block is None and context.block is not None:
        # At the point of the problem: a solve at a stage point (an IMEX relation) mixes these
        # literals with its reduced scalars in a choice, which needs one exact point (lot 4 fix).
        return context.program._new("scalar", "scalar_expression", value.inputs,
                                    dict(value.attrs), context.fresh("owned"), context.block,
                                    point=context.point, inherit_state_ref=False)
    return value


@dataclass(frozen=True)
class _Failure:
    variant: type
    reason: str

    @property
    def code(self) -> int:
        return FAILURE_CODES[(self.variant.__name__, self.reason)]


@surface_entry
def record(**fields: Any) -> Record | _run.Record:
    """``pops.record(name=value, ...)``: a solver's record inside a solver (IR4 ``Product``), the
    run's record elsewhere (N15 lot G)."""
    if not _ACTIVE:
        # Outside a solver, pops.record is the record of the run (N15 lot G).
        return _run.record(**fields)
    for name, item in fields.items():
        if not isinstance(item, (Field, Scalar, int, float)) or isinstance(item, bool):
            raise TypeError("pops.record takes fields and scalars of the solver, got %s=%r"
                            % (name, item))
    return Record(fields)


@surface_entry
def fail(variant: Any, *, payload: str) -> _Failure:
    """``pops.fail(variant, payload=reason)`` in an arm of ``pops.choose`` (IR4 ``fail``, T39)."""
    _active("pops.fail")
    if variant not in FAILURE_VARIANTS:
        raise TypeError("pops.fail takes a variant of T39 (pops.SolveFailed, "
                        "pops.BudgetExhausted), got %r" % (variant,))
    if (variant.__name__, payload) not in FAILURE_CODES:
        raise NotYetElaborated("pops.fail(pops.%s, payload=%r)" % (variant.__name__, payload),
                               _N9_LATER)
    return _Failure(variant, payload)


def _lower_arm(context: _Lowering, value: Any, *, as_result: bool) -> Any:
    program = context.program
    if isinstance(value, _Failure):
        if not context.templates:
            raise TypeError("pops.fail is written in the step of region.iterate")
        return program.failure(context.templates[-1], value.code)
    if isinstance(value, Record):
        product = value.program_value()
        return program.success(product) if as_result else product
    from pops.time.values import ProgramValue
    if isinstance(value, ProgramValue):
        if value.vtype == "result" or not as_result:
            return value
        return program.success(value)
    raise TypeError("an arm of pops.choose returns pops.record(...) or pops.fail(...), got %r"
                    % (value,))


@surface_entry
def choose(condition: Any, *, then: Any, otherwise: Any) -> Any:
    """``pops.choose(p, then=f, otherwise=g)``: run one of two computations (IR4 ``cond``)."""
    context = _active("pops.choose")
    if not isinstance(condition, Scalar):
        raise TypeError("pops.choose takes a predicate of the solver, got %r" % (condition,))
    if not callable(then) or not callable(otherwise):
        raise TypeError("pops.choose takes two functions, then= and otherwise=")
    program = context.program
    results: dict = {}

    def arm(function: Any, key: str) -> Any:
        def build(_program: Any) -> Any:
            value = function()
            results[key] = value
            if isinstance(value, (Scalar, int, float)) and not isinstance(value, bool):
                # A choice of numbers is a number (IR4 cond): one reduced Scalar, without the
                # State authority of its inputs, so that both arms have one structural type. A
                # literal arm takes the block of the predicate, as its other arm has it.
                return _owned_scalar(context, program.scalar_expression(
                    context.fresh("chosen"), _scalar_input(value, "pops.choose")))
            return _lower_arm(context, value, as_result=True)
        return build

    chosen = program.branch(_bool(condition), arm(then, "then"), arm(otherwise, "otherwise"),
                            name=context.fresh("choose"))
    if chosen.vtype == "scalar":
        return Scalar(chosen)
    return chosen


# --- the entities -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    """The request of the solve (SV3): its accuracy and its norm."""

    rtol: float
    atol: float
    problem: Problem

    def norm(self, value: Any) -> Scalar:
        """The norm of the request: the square root of the pairing of a field with itself,
        measured scale-safely as the native's residual norm is.

        The largest magnitude m of the field is reduced first. Within [_NORM_LOW, _NORM_HIGH] the
        pairing of the field with itself cannot overflow or underflow, and the norm is
        sqrt(<v, v>), the native's rounding at ordinary scales. Outside it, the field is divided
        by m first, and the norm is m * sqrt(<v/m, v/m>). A field that holds a value that is not
        finite has the norm m = +inf, which the solver refuses, and is never paired (a pairing of
        such a field is a refusal of the run, not of the solve)."""
        program = self.problem._context.program
        field = value.value("n")
        largest = Scalar(program.norm_inf_all(field))

        def measured() -> Scalar:
            def ordinary() -> Scalar:
                return _sqrt(self.problem.pairing(value, value))

            def scaled() -> Scalar:
                divisor = Scalar(program.scalar_expression(
                    self.problem._context.fresh("norm_scale"),
                    _ir_maximum(largest.value, _MIN_NORMAL)))
                unit = value / divisor
                return largest * _sqrt(self.problem.pairing(unit, unit))

            return choose((largest >= _NORM_LOW) & (largest <= _NORM_HIGH),
                          then=ordinary, otherwise=scaled)

        return choose(largest < _pymath.inf, then=measured, otherwise=lambda: largest)

    def met(self, residual: Scalar) -> Scalar:
        """The stop of SV3: r <= max(atol, rtol * ref), the physical threshold of the native
        (make_normalization). No quotient: a zero reference is a threshold of zero, not 0 / 0."""
        absolute = residual <= self.atol
        if self.rtol <= 0:
            return absolute
        return absolute | (residual <= self.rtol * self.problem._reference())


def _sqrt(value: Scalar) -> Scalar:
    return value.__pops_elementary__("sqrt")


def _ir_maximum(left: Any, right: Any) -> Any:
    from pops._ir.ops import maximum
    return maximum(left, right)


# The scale-safe norm (Request.norm): within [_NORM_LOW, _NORM_HIGH] the squares of the values and
# their sum over up to 1e16 cells are normal doubles; outside it the field is divided by its
# largest magnitude first. _MIN_NORMAL keeps that division defined for the zero field.
_MIN_NORMAL = sys.float_info.min
_NORM_LOW = 2.0 ** -480
_NORM_HIGH = 2.0 ** 480


class Problem:
    """The entity ``problem`` of an affine problem in one field unknown (SV19, C46)."""

    def __init__(self, context: _Lowering, linear: Any, apply_body: Any, request: Any) -> None:
        self._context = context
        from pops.time.values import _resolve_handle

        self._linear = linear
        self._apply_body = apply_body
        program = context.program
        rhs = program._canonical_value(_resolve_handle(linear.rhs))
        self._guess = None if linear.initial_guess is None else \
            program._canonical_value(_resolve_handle(linear.initial_guess))
        self._rhs = Field(rhs, rhs)
        self._reference_norm = None
        self.request = Request(request[0], request[1], self)
        # The reference norm of the request is read in every region of the solve: reduce it once,
        # in the region of the solve, before the body runs (one collective).
        self._reference()

    @property
    def rhs(self) -> Field:
        return self._rhs

    @property
    def guess(self) -> Field:
        program = self._context.program
        rhs = self._rhs.expression
        if self._guess is not None:
            return Field(self._guess, rhs)
        return Field(program.value(self._context.fresh("guess"), 0.0 * rhs, at=rhs.point), rhs)

    def apply(self, value: Field) -> Field:
        """A v: the apply body of the operator, run again inside the solver's region."""
        if not isinstance(value, Field):
            raise TypeError("problem.apply takes a field of the solver, got %r" % (value,))
        state = value.value("v")
        return Field(self._apply_body(self._context, state), self._rhs.expression)

    def pairing(self, left: Field, right: Field) -> Scalar:
        """The declared pairing of the unknown's space: one reduction (SV20 duty 3)."""
        if not isinstance(left, Field) or not isinstance(right, Field):
            raise TypeError("problem.pairing takes two fields of the solver")
        return Scalar(self._context.program.dot_all(left.value("a"), right.value("b")))

    def _reference(self) -> Scalar:
        if self._reference_norm is None:
            self._reference_norm = self.request.norm(self.rhs)
        return self._reference_norm

    def require(self, *, symmetric: bool = False, positive_definite: bool = False) -> None:
        """Refuse the realization unless the problem's properties hold (IR16, K4)."""
        properties = self._linear.properties
        missing = []
        if symmetric and not properties.symmetric:
            missing.append("symmetric")
        if positive_definite and not properties.positive_definite:
            missing.append("positive definite")
        if missing:
            raise PropertyNotDerived(
                "the solver needs a %s operator, and the problem's operator is not derived %s"
                % (" ".join(missing), " ".join(missing)),
                write="a realization whose preconditions the problem meets, such as gmres()")

    def candidate(self, solution: Any, *, iterations: Any, residual_norm: Any) -> _Candidate:
        """The solver's candidate: the solution, the count and the norm of the last true
        residual, each a field of the result of ``region.iterate`` (SV5: the evidence reads them,
        so the record's layout stays the solver's)."""
        for name, item in (("solution", solution), ("residual_norm", residual_norm)):
            if not isinstance(item, _Deferred):
                raise TypeError("problem.candidate(%s=...) takes a field of the result of "
                                "region.iterate, got %r" % (name, item))
        return _Candidate(solution, iterations, residual_norm)

    def __getattr__(self, name: str) -> Any:
        raise NotYetElaborated("problem.%s, a reading of the problem of a solver" % name, _G10)


class _Deferred:
    """A field of the result of ``region.iterate``, read after the solve is consumed."""

    def __init__(self, iteration: _Iteration, name: str) -> None:
        self.iteration = iteration
        self.name = name


class _Iteration:
    def __init__(self, result: Any, template: Record) -> None:
        self.result = result
        self.template = template

    def __getattr__(self, name: str) -> _Deferred:
        if name == "iterations" or name in object.__getattribute__(self, "template")._items:
            return _Deferred(self, name)
        raise AttributeError(name)


class _FieldApply:
    """The apply of ``set_apply``, on scalar fields, run on a State v of the solver's region."""

    def __init__(self, body: Any) -> None:
        self.body = body

    def __call__(self, context: _Lowering, state: Any) -> Any:
        from pops._ir.data_model import field_role
        from pops.fields import CellMidpoint

        program = context.program
        # The unknown's space is a State and the body reads and writes scalar fields: the midpoint
        # sampling of a cell average and its quadrature map one onto the other with the same
        # values (pops.cell-midpoint-sampling@1, pops.cell-midpoint-quadrature@1).
        operand = program.cell_field(state, quadrature=CellMidpoint(),
                                     name=context.fresh("apply_in"))
        out = program.scalar_field(context.fresh("apply_out"), like=operand)
        result = self.body(_RegionScratch(program, operand), out, operand)
        result = out if result is None else result
        if field_role(result) != "field":
            result = program.value(context.fresh("apply_result"), result, at=state.point)
        return program.cell_mean(result, template=state, quadrature=CellMidpoint(),
                                 name=context.fresh("apply"))


class _RegionScratch:
    """The Program as an apply body sees it inside the solver's region: a scratch field that the
    body allocates without a prototype takes the operand's typed storage, which the region's
    Laplacian requires (a computed-cell prototype, pops.computed-cell-field@1)."""

    def __init__(self, program: Any, operand: Any) -> None:
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_operand", operand)

    def scalar_field(self, name: Any = None, ncomp: Any = None, *, like: Any = None,
                     **options: Any) -> Any:
        if like is None and not options and ncomp in (None, len(self._operand.space.components)):
            like = self._operand
            ncomp = None
        return self._program.scalar_field(name, ncomp, like=like, **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._program, name)


class StateOperator:
    """A linear operator given by its action on a State: ``apply(P, v)`` returns A v from the
    values of the Program, such as ``v - h * rate(v)``, whose rate reads the runtime parameters
    of its model. Only a reference solver (SV9) applies it, inside its own region, so nothing of
    it enters the IR until a solve runs it; a native Krylov solve needs a ``matrix_free_operator``
    and ``set_apply``. This is the provider's ``linear_apply`` of N15 A1, with the properties that
    the provider states in the ``LinearProblem`` (R1)."""

    __slots__ = ("name", "apply")

    def __init__(self, name: str, apply: Any) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("StateOperator takes a nonempty name")
        if not callable(apply):
            raise TypeError("StateOperator takes apply(P, v), a callable")
        self.name = name
        self.apply = apply

    def __repr__(self) -> str:
        return "<StateOperator %s>" % self.name


class _StateApply:
    """The apply of a ``StateOperator``, on the State v itself (a model rate and its parameters)."""

    def __init__(self, body: Any) -> None:
        self.body = body

    def __call__(self, context: _Lowering, state: Any) -> Any:
        program = context.program
        result = self.body(program, state)
        return program.value(context.fresh("apply"), result, at=state.point)


@dataclass(frozen=True)
class _Candidate:
    solution: Any
    iterations: Any
    residual_norm: Any


class Region:
    """The entity ``region`` of a solver: the slice of SU15 that a Krylov method needs."""

    def __init__(self, context: _Lowering) -> None:
        self._context = context

    def iterate(self, *, start: Record, step: Any, stop: Any, max_steps: Any) -> _Iteration:
        context = self._context
        program = context.program
        if not isinstance(start, Record):
            raise TypeError("region.iterate takes start=pops.record(...)")
        initial = start.program_value()

        def step_body(_program: Any, state: Any) -> Any:
            context.templates.append(state)
            try:
                value = step(Record.from_program(state, start))
                return _lower_arm(context, value, as_result=True)
            finally:
                context.templates.pop()

        def stop_body(_program: Any, state: Any) -> Any:
            return _bool(stop(Record.from_program(state, start)))

        raw = program.iterate(initial, step_body, stop_body, max_steps=max_steps)
        # An iteration result is not carried out of a region: the same sum as an ordinary Result
        # (success: the state and the count; failure: the code), so that the solve's region
        # captures it with the failures of what precedes the loop.
        template = program.product(state=initial, iterations=program.int64(0))

        def ok(_program: Any, payload: Any) -> Any:
            return program.success(payload)

        def err(_program: Any, payload: Any) -> Any:
            return program.failure(template, program._product_get(payload, "code"))

        return _Iteration(program.match(raw, ok=ok, err=err), start)

    def __getattr__(self, name: str) -> Any:
        raise NotYetElaborated("region.%s in a solver" % name, _N9_LATER)


# --- the decorator, the value and the outcome ---------------------------------------------------

class SolverDefinition:
    """A function decorated with ``@pops.solver``."""

    def __init__(self, function: Any) -> None:
        parameters = inspect.signature(function).parameters
        missing = [name for name in _ENTITIES if name not in parameters]
        if missing:
            raise TypeError("@pops.solver supplies %s; %s does not take %s"
                            % (" and ".join(_ENTITIES), function.__name__, " and ".join(missing)))
        self.function = function
        self.options = {name: parameter.default for name, parameter in parameters.items()
                        if name not in _ENTITIES}
        self.relation_native: Any = None
        self.requires: tuple = ()
        self.relation_description: str | None = None

    def bind_relation_native(self, native: Any, *, requires: tuple, description: str) -> Any:
        """The bound native that a relation solve of a traced method runs under this solver
        (N15 lot « solves », §5.5): ``native(options, request)`` gives its b6 descriptor and
        ``requires`` the properties of the operator that it needs (IR16, K4). A bridge of MG9
        until the N9 lot « La référence sous les solves de relation » runs the reference there.
        """
        if not callable(native):
            raise TypeError("bind_relation_native(native, ...) takes native(options, request)")
        self.relation_native = native
        self.requires = tuple(requires)
        self.relation_description = description
        return self

    @surface_entry
    def __call__(self, *args: Any, **options: Any) -> SolverValue:
        if args:
            raise TypeError("a solver takes its options by keyword (SU8): %s(%s)"
                            % (self.function.__name__,
                               ", ".join("%s=..." % name for name in self.options)))
        unknown = sorted(set(options) - set(self.options))
        if unknown:
            raise TypeError("%s has no option %s; its options are %s"
                            % (self.function.__name__, ", ".join(unknown),
                               ", ".join(sorted(self.options))))
        if options.get("preconditioner") is not None:
            # The protocol of a solver reference has no preconditioner apply yet (C46, SV19).
            raise NotYetElaborated("%s(preconditioner=...), a preconditioned solver"
                                   % self.function.__name__, _G10)
        merged = dict(self.options)
        merged.update(options)
        required = [name for name, value in merged.items() if value is inspect.Parameter.empty]
        if required:
            raise TypeError("%s requires the option %s" % (self.function.__name__,
                                                           ", ".join(required)))
        return SolverValue(self, merged)

    def __repr__(self) -> str:
        return "<pops.solver %s>" % self.function.__name__


class SolverValue:
    """A called solver: its options, and the request that the solve gives it (SV3)."""

    def __init__(self, definition: SolverDefinition, options: dict,
                 request: tuple | None = None) -> None:
        self.definition = definition
        self.options = dict(options)
        self.request = request

    def with_request(self, *, rtol: float, atol: float) -> SolverValue:
        """The same solver with the request of the solve that runs it (SV3, SU9)."""
        rtol, atol = float(rtol), float(atol)
        if not (0.0 <= rtol < 1.0) or not atol >= 0.0 or not (_pymath.isfinite(atol)):
            raise ValueError("a request has rtol in [0, 1) and a finite atol >= 0")
        if rtol == 0.0 and atol == 0.0:
            raise ValueError("a request needs rtol > 0 or atol > 0")
        return SolverValue(self.definition, self.options, (rtol, atol))

    def prepare_program_solve(self) -> SolverValue:
        if self.request is None:
            raise ValueError("%r has no request; the solve gives it one (tolerance=, SV3)" % self)
        return self

    def build_program_solve(self, *, program: Any, problem: Any, name: Any = None) -> Any:
        return lower_solver(self, program, problem, name)

    # The protocol of a relation solve of a traced method (N15 lot « solves », §5.5): the
    # properties of the operator that the solver needs, and the bound native that it runs.

    @property
    def requires(self) -> tuple:
        """The properties of the operator that this solver needs (IR16): the relation solve
        refuses an operator that its claims do not derive with ``PropertyNotDerived`` (K4)."""
        return self.definition.requires

    def linear_solver(self, *, request: Any) -> Any:
        """The b6 descriptor that a relation solve runs under this solver, with its request."""
        native = self.definition.relation_native
        if native is None:
            raise NotYetElaborated(
                "%r under a relation solve of a traced method, whose operator only the bound "
                "native of cg() solves in this lot" % (self,),
                "the N9 lot « La référence sous les solves de relation »")
        return native(self.options, request)

    def describe(self) -> str:
        description = self.definition.relation_description
        return "%r, the reference of @pops.solver%s" % (
            self, "" if description is None else "; a relation solve runs %s" % description)

    def __repr__(self) -> str:
        return "%s(%s)" % (self.definition.function.__name__,
                           ", ".join("%s=%r" % item for item in sorted(self.options.items())))


def solver(function: Any) -> SolverDefinition:
    """``@pops.solver``: a solve algorithm of ``pops.lib.solve`` or of the case (SV9, SV-T4)."""
    if not callable(function):
        raise TypeError("@pops.solver decorates a function")
    return SolverDefinition(function)


# The status of each failure of a solver's region, by the reason of its variant (§3.2 of the lot
# « La référence sous les solves de relation »): the table of solve failures gives each one its
# action and its variant, as on the native route. A failure that the region rejects at runtime
# (origin 2, such as a value that is not finite) is an invalid evaluation, with its code.
_FAILURE_STATUSES = {
    "iteration limit": "iteration_limit",
    "breakdown": "breakdown",
    "invalid evaluation": "invalid_evaluation",
    "unverified solved value": "invalid_evaluation",
}


def _failure_rows() -> list:
    """``[code, status, reason]`` of each user failure (origin 1) of a solver's region."""
    return [[code, _FAILURE_STATUSES[reason], reason]
            for code, (_variant, reason) in sorted(FAILURES_BY_CODE.items())]


def lower_solver(value: SolverValue, program: Any, linear: Any, name: Any = None) -> Any:
    """Lower ``value`` for the linear problem ``linear``: one ``solve_linear`` node (SV12) whose
    realization is the reference, with the solver's body as its region (lot « La référence sous
    les solves de relation », §3.1), and its outcome, b6's ``SolveOutcome``."""
    from pops.linalg import LinearProblem

    if not isinstance(linear, LinearProblem):
        raise NotYetElaborated("a solver of a problem other than one affine field unknown", _G10)
    if linear.nullspace is not None:
        raise NotYetElaborated("a solver of a singular problem (a nullspace)", _N9_LATER)
    from pops.time.values import _resolve_handle

    if isinstance(linear.operator, StateOperator):
        raise NotYetElaborated(
            "a reference solve of a StateOperator, before the State operand of a "
            "matrix_free_operator", "the N9 lot « La référence sous les solves de relation »")
    operator = program._canonical_value(_resolve_handle(linear.operator))
    bodies = program.__dict__.get("_operator_apply_bodies", {})
    if operator.id not in bodies:
        raise TypeError("the operator of the problem has no apply; call P.set_apply first")
    apply = _FieldApply(bodies[operator.id])
    rhs = program._canonical_value(_resolve_handle(linear.rhs))
    guess = None if linear.initial_guess is None else \
        program._canonical_value(_resolve_handle(linear.initial_guess))
    at = rhs.point if linear.at is None else linear.at
    label = name or "reference_solve"
    context = _Lowering(program, label)
    returned: dict = {}

    def body(_program: Any) -> Any:
        with _lowering(context):
            context.point = at
            problem = Problem(context, linear, apply, value.request)
            returned["problem"] = problem
            context.block = problem.rhs.expression.block
            result = value.definition.function(problem=problem, region=Region(context),
                                               **value.options)
        if not isinstance(result, _Candidate) or not isinstance(result.solution, _Deferred):
            raise TypeError("a solver returns problem.candidate(x, iterations=k), x read from "
                            "the result of region.iterate")
        returned["candidate"] = result
        # The candidate, read inside the region: a failed iterate fails the region with its
        # code, and the solve's report reads the success's fields (the solver's names, SV5).
        success = program.require_success(result.solution.iteration.result)
        state = program._product_get(success, "state")
        solution = program._product_get(state, result.solution.name)
        gate = _gate_measurement(context, problem, solution)
        return program.product(
            solution=solution,
            iterations=program._product_get(success, "iterations"),
            residual_norm=program._product_get(state, result.residual_norm.name),
            gate_code=program._product_get(gate, "code"),
            gate_norm=program._product_get(gate, "norm"),
            gate_scale=program._product_get(gate, "scale"))

    # The whole solve is one closed region: a value that is not finite, in a pairing or in a
    # scalar, is then a failure of the solve (code 3, a rejected evaluation), before the loop as
    # inside it (SV5, T39). The region is the node's sub-block, as set_apply records apply_block.
    block: list = []
    region = program._region_for_block(block)
    for parent in program._recording:
        program._allow_region_capture(program._region_for_block(parent), region)
    program._recording.append(block)
    try:
        captured = program.capture(body)
    finally:
        program._recording.pop()
    _require_closed_region(program, block, operator, rhs, guess)
    if value.request is None:
        raise ValueError("%r has no request; the solve gives it one (tolerance=, SV3)" % value)
    rtol, atol = value.request
    ncomp = int(operator.attrs["ncomp"])
    attrs = {
        "realization": {"kind": "reference", "rtol": float(rtol).hex(),
                        "atol": float(atol).hex(), "failures": _failure_rows()},
        "body_block": block, "body_region": region, "body": captured,
        "has_guess": guess is not None, "ncomp": ncomp,
        "operator_properties": linear.properties.canonical_data(),
        "problem_kind": "matrix_free_linear",
    }
    result_type = ("state" if operator.attrs["domain"] == "state" and rhs.vtype == "state"
                   else "scalar_field")
    inputs = (operator, rhs) if guess is None else (operator, rhs, guess)
    token = program._new(result_type, "solve_linear", inputs, attrs, label, rhs.block,
                         space=rhs.space, point=at)
    from pops.time.solve_outcome import SolveOutcome

    def project(outcome: Any) -> Any:
        return program._new(result_type, "solve_outcome_component", (outcome,),
                            {"index": 0, "ncomp": ncomp}, label, rhs.block, space=rhs.space,
                            point=token.point)

    return SolveOutcome(program, token, project, label)


def _gate_measurement(context: _Lowering, problem: Problem, solution: Any) -> Any:
    """The measurement of the SV8 gate of a solved candidate u (§3.3 of the lot « La référence
    sous les solves de relation »): ||R(u)|| and the scale ||R(0)||, R(v) = b - A(v), in the
    request's norm, through the same apply as the body. The gate is the solve's, not the
    solver's: it runs after the solver's body returned its candidate, as its own captured region,
    so that a value it rejects is a failure of the measurement (code > 0), never of the body.
    The decision is the native's own C++ (``apply_measured_original_residual_gate``)."""
    program = context.program
    rhs = problem.rhs.expression

    def measure(_program: Any) -> Any:
        with _lowering(context):
            candidate = Field(solution, rhs)
            zero = Field(program.value(context.fresh("gate_zero"), 0.0 * rhs, at=rhs.point), rhs)
            norm = problem.request.norm(problem.rhs - problem.apply(candidate))
            # R(0) is computed with one apply, as the native does, not taken as b: the gate then
            # holds without the claim « linear and homogeneous ».
            scale = problem.request.norm(problem.rhs - problem.apply(zero))
            return Record({"norm": norm, "scale": scale}).program_value()

    measured = program.capture(measure)

    def ok(_program: Any, payload: Any) -> Any:
        with _lowering(context):
            return Record({"code": Scalar(program.int64(0)),
                           "norm": Scalar(program._product_get(payload, "norm")),
                           "scale": Scalar(program._product_get(payload, "scale"))}).program_value()

    def err(_program: Any, payload: Any) -> Any:
        with _lowering(context):
            return Record({"code": Scalar(program._product_get(payload, "code")),
                           "norm": 0.0, "scale": 0.0}).program_value()

    return program.match(measured, ok=ok, err=err)


# What a solver's region may read from outside it (§3.1): the node's inputs and what the apply of
# its operator reads from outside the apply (such as p.dt in b_k), and runtime parameters and
# constants. Any other value of the Program read inside the region is refused by name: the rule
# linear_solve of the accepted decomposition reads the operator and the right-hand side, never
# the solver, so a solver must not read another rate.
_CONSTANT_OPS = frozenset(("scalar_constant", "int64_constant", "bool_constant", "runtime_param",
                           "parameter"))


def _region_reads(block: Any) -> list:
    """The values that ``block`` and its nested sub-blocks read from outside them."""
    from pops.time._program.traversal import _STRUCTURED_BLOCK_KEYS

    inside: set = set()
    pending = [block]
    blocks = []
    while pending:
        current = pending.pop()
        blocks.append(current)
        for node in current:
            inside.add(node.id)
            for key in _STRUCTURED_BLOCK_KEYS:
                nested = node.attrs.get(key)
                if nested:
                    pending.append(nested)
    reads: dict = {}
    for current in blocks:
        for node in current:
            for item in node.inputs:
                if item.id not in inside:
                    reads.setdefault(item.id, item)
    return list(reads.values())


def _require_closed_region(program: Any, block: Any, operator: Any, rhs: Any, guess: Any) -> None:
    allowed = {item.id for item in (operator, rhs, guess) if item is not None}
    apply_block = operator.attrs.get("apply_block") or ()
    allowed.update(item.id for item in _region_reads(apply_block))
    refused = [item for item in _region_reads(block)
               if item.id not in allowed and item.op not in _CONSTANT_OPS]
    if refused:
        raise TypeError(
            "a solver's region reads, from outside it, only the problem (its operator, right-hand "
            "side and guess, and what the operator's apply reads); it reads %s"
            % ", ".join(sorted("%s (%s)" % (item.name, item.op) for item in refused)))


__all__ = [
    "BudgetExhausted", "FAILURE_CODES", "FAILURES_BY_CODE", "OUTSIDE_OWNERS", "PropertyNotDerived",
    "SolveFailed", "StateOperator", "SolverDefinition", "SolverValue",
    "choose", "fail",
    "lower_solver", "record", "solver",
]
