"""pops.time value algebra -- typed SSA handles and the affine/operator algebra.
A ``ProgramValue`` is a typed SSA node in a Program IR; field-like values support an affine algebra
(``U + dt * R``) and scalars compose into ``scalar_op`` nodes. ``_Coeff`` / ``_Affine`` /
``_Operator`` are the coefficient + linear-combination carriers; ``StageStateSet`` and
``_CoupledResult`` are multi-block grouping handles. Authoring + evaluation only: no codegen,
no _pops, no module-scope dsl import.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

from pops._ir import Equation
from pops.identity.scalar import scalar_data
from pops._ir.symbolic import ImmutableSymbolic
from pops.provenance import ProvenanceRecord
from pops.time.points import point_clock
from pops.time.value_support import (
    _ProgramValueBase,
    authoring_source_location as _authoring_source_location,  # noqa: F401
    resolve_temporal_handle as _resolve_handle,
)
from pops.time.value_metadata import (
    CoeffPolynomial, CoefficientLiteralError, _exact_add, _exact_divide, _exact_multiply,
    _exact_negate, _exact_number, _freeze_attr, validate_program_value_identity,
)


class _Coeff(ImmutableSymbolic):
    """Scalar coefficient: an exact polynomial in ``dt`` (``power -> scalar``).

    ``dt`` is ``_Coeff({1: 1})``; a plain number is ``_Coeff({0: c})``. Multiplying a coefficient by
    a State/RHS value yields an `_Affine` (one weighted term)."""

    def __init__(self, powers: Any) -> None:
        # Drop exact zeros without ever routing an integer, rational, decimal or binary64
        # authoring literal through ``float``.  Keeping the native exact Python value here lets
        # serialization retain its literal kind and defers target conversion to C++ lowering.
        exact = {}
        for power, coeff in powers.items():
            value = _exact_number(coeff)
            if value != 0:
                exact[int(power)] = value
        self.powers = MappingProxyType(exact)

    def _binop_number(self, x: Any) -> Any:
        try:
            return _Coeff({0: _exact_number(x)})
        except CoefficientLiteralError:
            raise
        except (TypeError, ValueError):
            return None

    def __add__(self, other: Any) -> Any:
        o = self._binop_number(other) if not isinstance(other, _Coeff) else other
        if o is None:
            return NotImplemented
        out = dict(self.powers)
        for p, c in o.powers.items():
            out[p] = _exact_add(out.get(p, 0), c)
        return _Coeff(out)

    __radd__ = __add__

    def __neg__(self) -> Any:
        return _Coeff({p: _exact_negate(c) for p, c in self.powers.items()})

    def __sub__(self, other: Any) -> Any:
        return self.__add__(-(other if isinstance(other, _Coeff) else _Coeff({0: _exact_number(other)})))

    def __mul__(self, other: Any) -> Any:
        if not isinstance(other, (_Coeff, ProgramValue, _Affine)):
            try:
                exact = _exact_number(other)
            except CoefficientLiteralError:
                raise
            except (TypeError, ValueError):
                exact = None
            if exact is not None:
                return _Coeff({p: _exact_multiply(c, exact) for p, c in self.powers.items()})
        if isinstance(other, _Coeff):
            out = {}
            for p1, c1 in self.powers.items():
                for p2, c2 in other.powers.items():
                    product = _exact_multiply(c1, c2)
                    out[p1 + p2] = _exact_add(out.get(p1 + p2, 0), product)
            return _Coeff(out)
        if isinstance(other, ProgramValue) and other.is_field():
            return _Affine([(other, self)])
        if isinstance(other, _Affine):
            return other.__mul__(self)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        try:
            exact = _exact_number(other)
        except CoefficientLiteralError:
            raise
        except (TypeError, ValueError):
            exact = None
        if exact is not None:
            return _Coeff({p: _exact_divide(c, exact) for p, c in self.powers.items()})
        return NotImplemented

    def as_dict(self) -> Any:
        return dict(self.powers)

    def to_polynomial(self) -> CoeffPolynomial:
        return CoeffPolynomial(self.powers)

    def _key(self) -> Any:
        return tuple((p, tuple(sorted(scalar_data(c).items())))
                     for p, c in sorted(self.powers.items()))


def _to_affine(x: Any) -> Any:
    x = _resolve_handle(x)
    if isinstance(x, _Affine):
        return x
    if isinstance(x, ProgramValue) and x.is_field():
        return _Affine([(x, _Coeff({0: 1}))])
    raise TypeError("expected a State/RHS value or an affine combination, got %r" % (x,))


def _is_field_value(x: Any) -> Any:
    """True for a grid-field ProgramValue (State / RHS / scalar_field) -- the values that carry an
    pops::MultiFab and support the affine algebra."""
    return isinstance(x, ProgramValue) and x.is_field()


def _affine_ids(aff: Any) -> Any:
    """Stable JSON-able form of an _Affine (the apply result of a matrix_free_operator): an ordered
    list of ``[value_id, sorted-coeff-powers]``."""
    return [[v.id, sorted((int(p), scalar_data(c)) for p, c in coeff.as_dict().items())]
            for v, coeff in aff._merge()]


def _residual_wants_guess(fn: Any) -> Any:
    """True if a `solve_local_nonlinear` residual callable takes the frozen guess as a third positional
    arg (``residual_fn(P, U, U0)``); False for the two-arg form (``residual_fn(P, U)``). A ``*args``
    callable is treated as wanting the guess (it can accept it)."""
    import inspect
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):  # builtins / C callables: pass the guess and let the call decide
        return True
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    return len(positional) >= 3


class _Affine(ImmutableSymbolic):
    """Affine combination of State/RHS values: ordered ``[(value, _Coeff)]`` terms. Built by the
    operator overloads on field values; consumed by `Program.linear_combine`."""

    def __init__(self, terms: Any) -> None:
        self.terms = tuple(terms)
        clocks = {
            term.clock for term, _ in self.terms
            if isinstance(term, ProgramValue)
        }
        if len(clocks) > 1:
            raise ValueError(
                "an affine expression cannot mix clocks; synchronize the foreign value first")

    def _merge(self) -> Any:
        # coalesce repeated values (sum their coefficient polynomials), preserve first-seen order
        order, acc = [], {}
        for v, c in self.terms:
            if v.id not in acc:
                order.append(v)
                acc[v.id] = (v, c)
            else:
                acc[v.id] = (v, acc[v.id][1] + c)
        return [acc[v.id] for v in order]

    def __add__(self, other: Any) -> Any:
        return _Affine(self.terms + _to_affine(other).terms)

    __radd__ = __add__

    def __neg__(self) -> Any:
        return _Affine([(v, -c) for v, c in self.terms])

    def __sub__(self, other: Any) -> Any:
        return _Affine(self.terms + (-_to_affine(other)).terms)

    def __rsub__(self, other: Any) -> Any:
        return _Affine((-self).terms + _to_affine(other).terms)

    def __mul__(self, other: Any) -> Any:
        if not isinstance(other, _Coeff):
            try:
                other = _Coeff({0: _exact_number(other)})
            except CoefficientLiteralError:
                raise
            except (TypeError, ValueError):
                pass
        if isinstance(other, _Coeff):
            return _Affine([(v, c * other) for v, c in self.terms])
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        try:
            exact = _exact_number(other)
        except CoefficientLiteralError:
            raise
        except (TypeError, ValueError):
            return NotImplemented
        return _Affine([(v, c / exact) for v, c in self.terms])


class _Operator(ImmutableSymbolic):
    """A LOCAL linear operator expression ``c_I * I + sum_k c_k * L_k`` (coefficients are `_Coeff`,
    polynomials in dt). Built by ``Program.I`` and ``a * Program.linear_source(handle)``; consumed by
    `Program.solve_local_linear` (operator ``I +/- a*L``). It is NOT a runtime field value -- it names
    the model linear source(s) and the scalar(s) that form the operator."""

    def __init__(self, identity: Any, terms: Any) -> None:
        self.identity = identity   # _Coeff: coefficient of the identity I
        self.terms = tuple(terms)   # [(ProgramValue(op='linear_source'), _Coeff)]

    def __add__(self, other: Any) -> Any:
        if not isinstance(other, _Operator):
            return NotImplemented
        return _Operator(self.identity + other.identity, self.terms + other.terms)

    def __sub__(self, other: Any) -> Any:
        if not isinstance(other, _Operator):
            return NotImplemented
        return _Operator(self.identity - other.identity,
                         self.terms + tuple((v, -c) for v, c in other.terms))

    def __neg__(self) -> Any:
        return _Operator(-self.identity, [(v, -c) for v, c in self.terms])

    def __mul__(self, other: Any) -> Any:
        if not isinstance(other, _Coeff):
            try:
                other = _Coeff({0: _exact_number(other)})
            except CoefficientLiteralError:
                raise
            except (TypeError, ValueError):
                pass
        if isinstance(other, _Coeff):
            return _Operator(self.identity * other, [(v, c * other) for v, c in self.terms])
        return NotImplemented

    __rmul__ = __mul__


class ProgramValue(ImmutableSymbolic, _ProgramValueBase):
    """A typed SSA node in a Program IR. Field-like values (State, RHS, scalar_field) support affine
    arithmetic. A ``scalar_field`` is a single-component grid field (the unknown / residual of a
    matrix-free linear solve), DISTINCT from the n_cons conservative ``state`` even though both lower
    to an pops::MultiFab; a ``matrix_free_op`` names a matrix-free operator A whose apply sub-block is
    recorded by ``set_apply`` and lowered to a C++ lambda the runtime Krylov loop calls."""

    _FIELD = ("state", "rhs", "scalar_field")
    _SCALAR = ("scalar", "bool")  # runtime scalars / predicates: never a Python bool / index

    def __init__(self, prog: Any, vid: Any, vtype: Any, op: Any, inputs: Any, attrs: Any,
                 name: Any, block: Any, *, space: Any = None, source_location: Any = None,
                 field_context: Any = None, region: int = 0, state_ref: Any = None,
                 point: Any, provenance: Any) -> None:
        inputs = validate_program_value_identity(
            vid, vtype, op, inputs, name, block, region, state_ref)
        self.prog = prog
        self.id = vid
        self.vtype = vtype
        self.op = op
        self.inputs = inputs
        self.attrs = _freeze_attr(dict(attrs))
        self.name = name
        self.block = block
        self.state_ref = state_ref
        self.region = region
        point_clock(point, "ProgramValue")
        self.point = point
        # Operator-first type tag (Spec 2): the pops.model space/operator-type this value lives over
        # (a StateSpace / RateSpace / FieldSpace / LocalLinearOperator); None skips space checks.
        if space is not None and getattr(space, "__pops_ir_immutable__", False) is not True:
            raise TypeError("ProgramValue space must be an immutable typed Space or None")
        self.space = space
        # OPTIONAL authoring source location: populated by _new when capture_source_locations() is on.
        # debug aid, INSPECTION-ONLY -- NEVER serialized into the IR / the hash. None by default.
        if source_location is not None \
                and (not isinstance(source_location, str) or not source_location):
            raise TypeError("ProgramValue source_location must be a non-empty string or None")
        self.source_location = source_location
        if not isinstance(provenance, ProvenanceRecord):
            raise TypeError("ProgramValue provenance must be a ProvenanceRecord")
        self.provenance = provenance
        if field_context is not None:
            from pops.time.field_context import FieldContext, FieldReadProvenance
            if not isinstance(field_context, (FieldContext, FieldReadProvenance)):
                raise TypeError("ProgramValue field_context must be typed immutable provenance")
        self.field_context = field_context

    @property
    def clock(self) -> Any:
        """The one logical clock owning this value's evaluation coordinate."""
        return point_clock(self.point, "ProgramValue.clock")

    def is_field(self) -> Any:
        return self.vtype in ProgramValue._FIELD

    def __eq__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self.prog._compare(self, other, "==")
        # Field-value equality is equation authoring syntax, never Python identity. Cross-Program
        # operands still build an Equation and are rejected by the consuming Program boundary.
        return Equation(self, other)

    def __ne__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self.prog._compare(self, other, "!=")
        # ``==`` has equation-building semantics for fields.  There is no meaningful
        # field-wide ``!=`` predicate in the Program IR; require the explicit per-cell reduction /
        # selection APIs instead of letting Python negate the Equation and truth-test it.
        raise TypeError(
            "field ProgramValue inequality is not a Python comparison; use P.cell_compare(...) "
            "for a per-cell predicate or compare an explicit scalar reduction")

    def __index__(self) -> Any:
        # range(scalar) / using a Scalar as a Python index is just as loud: the value is unknown until
        # the step runs.
        raise TypeError(
            "a Program %s (%r) cannot be used as a Python index; use P.while_ / P.branch for runtime "
            "control flow" % (self.vtype, self.name))

    def __len__(self) -> Any:
        # An IR ProgramValue has no Python length: its component / cell shape is a runtime grid property, not a
        # compile-time count. len(value) / iterating it would silently mis-read the grid, so refuse it
        # loudly (ADC-530) and point at the inspection-only logical_shape for the component layout.
        raise TypeError(
            "a Program %s value (%r) has no Python len(): its shape is a runtime grid property. Read "
            "its inspection-only logical_shape for the component layout; use P.while_ / P.branch / "
            "P.static_range for control flow." % (self.vtype, self.name))

    @property
    def logical_shape(self) -> Any:
        """The INSPECTION-ONLY logical shape of this value, derived on demand from its space (ADC-530).

        A plain dict ``{"vtype", "space", "n_comp", "layout"}`` naming the value's operator-first space
        (a StateSpace / RateSpace / FieldSpace) and its component count / storage layout when a space
        tag is present, else ``n_comp``/``layout`` = ``None`` (an untyped value). It is DERIVED rather
        than independently stored; the underlying immutable space does participate in the IR identity
        because its component order affects generated kernels. Purely a debug view."""
        space = self.space
        n_comp = None
        layout = None
        space_name = getattr(space, "name", None)
        components = getattr(space, "components", None)
        if space is not None and getattr(space, "kind", None) == "rate":
            components = space.base_space.components
        if components is not None:
            n_comp = len(components)
        layout = getattr(space, "layout", None)
        return {"vtype": self.vtype, "space": space_name, "n_comp": n_comp, "layout": layout}

    # --- scalar comparisons (scalar values only): build a Bool predicate, do not compare in Python ---
    def _compare(self, other: Any, cmp: Any) -> Any:
        if self.vtype != "scalar":
            raise TypeError("%s value %r is not a scalar; only P.norm2 / P.dot results compare"
                            % (self.vtype, self.name))
        return self.prog._compare(self, other, cmp)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, ">")

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, "<")

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, ">=")

    def __le__(self, other: Any) -> Any:
        return self._compare(other, "<=")

    # --- affine algebra (field values only) ---
    def _affine(self) -> Any:
        if not self.is_field():
            raise TypeError("%s value %r is not a field; only State/RHS support arithmetic"
                            % (self.vtype, self.name))
        return _to_affine(self)

    # --- scalar arithmetic (scalar values only): build a scalar_op node, NOT a Python float ---
    # A runtime Scalar (a reduction, max_wave_speed, hmin, the dt_bound's cfl) composes into a new
    # Scalar via + - * / so a dt_bound can express e.g. cfl * P.hmin() / P.max_wave_speed(U) (spec s18);
    # the value is unknown until the step runs, so the arithmetic builds IR, it is never evaluated here.
    def _scalar_op(self, other: Any, fn: Any, swap: Any = False) -> Any:
        if self.vtype != "scalar":
            raise NotImplementedError("scalar arithmetic is only defined for Scalar values")
        a, b = (other, self) if swap else (self, other)
        return self.prog._scalar_binop(a, b, fn)

    def __add__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "add")
        return self._affine() + _to_affine(other)

    def __radd__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "add", swap=True)
        return self._affine() + _to_affine(other)

    def __neg__(self) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(-1, "mul")
        return -self._affine()

    def __sub__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "sub")
        return self._affine() - _to_affine(other)

    def __rsub__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "sub", swap=True)
        return _to_affine(other) - self._affine()

    def __mul__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "mul")
        if self.vtype == "operator":  # a linear-source operator: scalar/dt * L -> an _Operator term
            if not isinstance(other, _Coeff):
                try:
                    other = _Coeff({0: _exact_number(other)})
                except CoefficientLiteralError:
                    raise
                except (TypeError, ValueError):
                    pass
            if isinstance(other, _Coeff):
                return _Operator(_Coeff({}), [(self, other)])
            return NotImplemented
        if isinstance(other, _Coeff):
            return self._affine() * other
        try:
            return self._affine() * _Coeff({0: _exact_number(other)})
        except CoefficientLiteralError:
            raise
        except (TypeError, ValueError):
            pass
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "mul", swap=True)
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "div")
        try:
            exact = _exact_number(other)
        except CoefficientLiteralError:
            raise
        except (TypeError, ValueError):
            return NotImplemented
        return self._affine() * (_Coeff({0: 1}) / exact)

    def __rtruediv__(self, other: Any) -> Any:
        if self.vtype == "scalar":
            return self._scalar_op(other, "div", swap=True)
        return NotImplemented

    # --- operator application (Spec 3 board notation): operator @ state -> apply ---
    def __matmul__(self, other: Any) -> Any:
        """``L @ U`` -- apply a linear-source operator value ``L`` to a state ``U``.

        Returns the RHS-like value ``P.apply(operator=L, state=U)``. For
        ``operator @ unknown(name)`` (a board solve), this returns ``NotImplemented``
        so :meth:`pops.math.Unknown.__rmatmul__` builds the solve left-hand side.
        """
        if self.vtype == "operator" and isinstance(other, ProgramValue) and other.vtype == "state":
            return self.prog.apply(operator=self, state=other)
        return NotImplemented

    def __repr__(self) -> str:
        return "<%s %s #%d>" % (self.vtype, self.name, self.id)


__all__ = ["ProgramValue"]
