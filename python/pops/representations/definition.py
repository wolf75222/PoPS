"""The contract @2 of a representation on a configuration (spec 0.3 D1, D1b, D2, D4; G03 lot R).

``pops.executable-representation-definition@2`` is the input of the user representations of N16
(N16 rev 3, section 5.5). It holds what contract @1 cannot (``Representation`` with whole-value
encode and decode, ``representations/__init__.py``): the carrier entity, the local index set, a
decode of the entities ``coefficients`` and ``point``, an encode of the carrier entity and the
entity ``field``, the geometry reads they name, the measure and orientation convention, the
pairing as a rule or none, and the kind.

Two layers, two identities (N16 5.5):

- ``RepresentationDefinition`` has no space. Its identity is the digest of its normalized form:
  the normal forms of decode and encode, the numbers of its rule, its carrier, size, kind,
  measure and orientation. A Python name never enters it (P10).
- ``ElaboratedRepresentation(space, definition)`` adds the space. Two quantities of one space
  under two spellings of one definition share it; two spaces under one definition do not.

Elaboration derives, never accepts from the user (G03 note, section 4.1):

- ``point_dependent`` and ``linear``, from the normal form of decode;
- ``constant_mode``: c when decode is c * coefficients[0] with c a point-free expression of the
  geometry reads (1 for an average or a sample, 1 / cell.measure for a total);
- ``rate_weights`` m_K = integral of phi_s over K, the load of the constant 1, which exists
  whenever the measure does (N16 5.2): c * mu(K) for one constant mode, the exact integral over the
  reference box for a polynomial decode in ``point.reference``;
- ``load_weights`` w_K, the coefficients of the constant 1 in the basis: 1 / c for one constant
  mode, the exact solution of C^T w = e_0 for a polynomial decode with rational coefficients;
- ``riesz_scale`` 1 / (c mu(K)), the Riesz map 1 / (c^2 mu(K)) composed with the test weight c.

A derived value is either an exact normal form or an ``Unknown`` with its reason. The lowering of
G03 lot R consumes one constant mode with c = 1 or c = 1 / cell.measure; anything else is the Gram
path of N16 lot 2.

Bodies are data, built from the leaves of this module with Python arithmetic; nothing here calls a
user function. The normal form is the one of the surface (``surface/_scheme.py``, CC2): a sum of
monomials with exact rational coefficients over atoms. Division by a product of geometry reads
gives negative exponents, since the plan certifies measures and widths positive; any other
quotient, a call and a power of an expression stay atoms.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

CONTRACT = "pops.executable-representation-definition@2"
SCHEMA_VERSION = 1

#: What a consumer discharges before it reads a coefficient (G03 note, 4.1).
OBLIGATIONS = (
    "coefficients_carry_definition_and_space",
    "inventory_from_rate_weights",
    "loads_weighted_by_load_weights",
    "encode_is_not_inverse_of_decode",
    "accumulation_density_by_the_stated_rule",
    "geometry_reads_from_the_candidate",
)

CARRIERS = ("cell", "node", "face", "edge")
KINDS = ("coefficient", "average", "total", "sample", "face integral", "edge circulation", "load")
MEASURES = ("support", "none")

# Atom tags. Geometry atoms are positive on every cell (certified by the plan), so a monomial of
# them divides exactly.
_GEOMETRY = ("cell.measure", "cell.width", "node.reference")
_POINT = ("point.reference", "point.position")


# --- bodies ---------------------------------------------------------------------------------------

def _exact(value: Any) -> Fraction:
    if type(value) is bool:
        raise TypeError("a representation body takes numbers, not booleans")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("a representation body takes finite numbers")
        return Fraction(value)
    raise TypeError("a representation body takes numbers and the leaves of "
                    "pops.representations.definition, got %r" % (value,))


def _body(value: Any) -> Body:
    return value if isinstance(value, Body) else Const(_exact(value))


class Body:
    """A node of a decode or encode body: immutable data, combined with Python arithmetic."""

    __slots__ = ()

    def __add__(self, other: Any) -> Body:
        return Op("+", (self, _body(other)))

    def __radd__(self, other: Any) -> Body:
        return Op("+", (_body(other), self))

    def __sub__(self, other: Any) -> Body:
        return Op("-", (self, _body(other)))

    def __rsub__(self, other: Any) -> Body:
        return Op("-", (_body(other), self))

    def __mul__(self, other: Any) -> Body:
        return Op("*", (self, _body(other)))

    def __rmul__(self, other: Any) -> Body:
        return Op("*", (_body(other), self))

    def __truediv__(self, other: Any) -> Body:
        return Op("/", (self, _body(other)))

    def __rtruediv__(self, other: Any) -> Body:
        return Op("/", (_body(other), self))

    def __neg__(self) -> Body:
        return Op("neg", (self,))

    def __pow__(self, exponent: Any) -> Body:
        return Op("**", (self, _body(exponent)))

    def __bool__(self) -> bool:
        raise TypeError("a representation body has no truth value")

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class Const(Body):
    value: Fraction


@dataclass(frozen=True, slots=True, eq=False)
class Leaf(Body):
    """An atom: ``coefficients[k]``, ``point.reference[a]``, ``point.position[a]``,
    ``cell.measure``, ``cell.width(a)``, ``node.reference[n][a]`` or ``field[c]``."""

    tag: str
    indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class Op(Body):
    op: str
    args: tuple[Body, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Call(Body):
    """A word of ``pops.math`` applied to bodies, opaque to the normal form."""

    word: str
    args: tuple[Body, ...]


def _index(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError("%s takes a non-negative Python integer (E4), got %r" % (what, value))
    return value


class _Indexed:
    __slots__ = ("_tag",)

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def __getitem__(self, index: Any) -> Leaf:
        return Leaf(self._tag, (_index(index, self._tag),))


class _Point:
    reference = _Indexed("point.reference")
    position = _Indexed("point.position")


class _NodeReference:
    def __getitem__(self, node: Any) -> _Indexed:
        node = _index(node, "node.reference")

        class _Axis:
            def __getitem__(self, axis: Any) -> Leaf:
                return Leaf("node.reference", (node, _index(axis, "node.reference")))

        return _Axis()  # type: ignore[return-value]


class _Cell:
    measure = Leaf("cell.measure")

    @staticmethod
    def width(axis: Any) -> Leaf:
        return Leaf("cell.width", (_index(axis, "cell.width"),))


coefficients = _Indexed("coefficients")
point = _Point()
cell = _Cell()
node_reference = _NodeReference()
field = _Indexed("field")


def call(word: str, *args: Any) -> Call:
    if type(word) is not str or not word:
        raise TypeError("call(word, ...) takes the name of a word of pops.math")
    return Call(word, tuple(_body(arg) for arg in args))


def _to_data(value: Body) -> Any:
    if isinstance(value, Const):
        return {"const": [value.value.numerator, value.value.denominator]}
    if isinstance(value, Leaf):
        return {"leaf": value.tag, "indices": list(value.indices)}
    if isinstance(value, Op):
        return {"op": value.op, "args": [_to_data(arg) for arg in value.args]}
    if isinstance(value, Call):
        return {"call": value.word, "args": [_to_data(arg) for arg in value.args]}
    raise TypeError("not a representation body: %r" % (value,))


def body_from_data(data: Any) -> Body:
    if "const" in data:
        numerator, denominator = data["const"]
        return Const(Fraction(numerator, denominator))
    if "leaf" in data:
        return Leaf(data["leaf"], tuple(data["indices"]))
    if "op" in data:
        return Op(data["op"], tuple(body_from_data(arg) for arg in data["args"]))
    if "call" in data:
        return Call(data["call"], tuple(body_from_data(arg) for arg in data["args"]))
    raise ValueError("not representation body data: %r" % (data,))


# --- the normal form ------------------------------------------------------------------------------

Monomial = tuple  # sorted ((atom, exponent), ...)
Polynomial = dict  # Monomial -> Fraction


def _times(left: Polynomial, right: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for mono_a, coef_a in left.items():
        for mono_b, coef_b in right.items():
            powers = dict(mono_a)
            for atom, exponent in mono_b:
                powers[atom] = powers.get(atom, 0) + exponent
            mono = tuple(sorted(((atom, e) for atom, e in powers.items() if e != 0), key=repr))
            out[mono] = out.get(mono, 0) + coef_a * coef_b
    return {mono: coef for mono, coef in out.items() if coef != 0}


def _plus(left: Polynomial, right: Polynomial, sign: int = 1) -> Polynomial:
    out = dict(left)
    for mono, coef in right.items():
        out[mono] = out.get(mono, 0) + sign * coef
    return {mono: coef for mono, coef in out.items() if coef != 0}


def _is_geometry_monomial(poly: Polynomial) -> bool:
    if len(poly) != 1:
        return False
    (mono, coef), = poly.items()
    return coef != 0 and all(atom[0] in _GEOMETRY for atom, _ in mono)


def _invert_monomial(poly: Polynomial) -> Polynomial:
    (mono, coef), = poly.items()
    return {tuple(sorted(((atom, -e) for atom, e in mono), key=repr)): 1 / coef}


def _polynomial(value: Body) -> Polynomial:
    if isinstance(value, Const):
        return {(): value.value} if value.value != 0 else {}
    if isinstance(value, Leaf):
        return {(((value.tag, value.indices), 1),): Fraction(1)}
    if isinstance(value, Call):
        return {((("call", value.word, tuple(normal_form(arg) for arg in value.args)), 1),):
                Fraction(1)}
    if not isinstance(value, Op):
        raise TypeError("not a representation body: %r" % (value,))
    if value.op == "neg":
        return {mono: -coef for mono, coef in _polynomial(value.args[0]).items()}
    left, right = value.args
    if value.op in ("+", "-"):
        return _plus(_polynomial(left), _polynomial(right), 1 if value.op == "+" else -1)
    if value.op == "*":
        return _times(_polynomial(left), _polynomial(right))
    if value.op == "/":
        divisor = _polynomial(right)
        if not divisor:
            raise ZeroDivisionError("a representation body divides by zero")
        if len(divisor) == 1 and () in divisor:
            return _times(_polynomial(left), {(): 1 / divisor[()]})
        if _is_geometry_monomial(divisor):
            return _times(_polynomial(left), _invert_monomial(divisor))
        return _times(_polynomial(left), {((("quotient", normal_form(right)), 1),): Fraction(1)})
    if value.op == "**":
        exponent = _polynomial(right)
        base = _polynomial(left)
        if not exponent:
            return {(): Fraction(1)}
        if len(exponent) == 1 and () in exponent and exponent[()].denominator == 1:
            power = int(exponent[()])
            if 0 <= power <= 16:
                out: Polynomial = {(): Fraction(1)}
                for _ in range(power):
                    out = _times(out, base)
                return out
            if -16 <= power < 0 and _is_geometry_monomial(base):
                out = {(): Fraction(1)}
                for _ in range(-power):
                    out = _times(out, _invert_monomial(base))
                return out
        return {((("power", normal_form(left), normal_form(right)), 1),): Fraction(1)}
    raise ValueError("unknown operation %r in a representation body" % (value.op,))


def _sorted(poly: Polynomial) -> tuple:
    return tuple(sorted(poly.items(), key=repr))


def normal_form(value: Body) -> tuple:
    """A sum of monomials with exact rational coefficients over atoms (CC2, as the surface's)."""
    return _sorted(_polynomial(value))


def _atoms(poly: Polynomial):
    for mono in poly:
        for atom, _ in mono:
            yield atom


def _reads(atom: Any, tags: tuple) -> bool:
    if atom[0] in tags:
        return True
    if atom[0] in ("call", "quotient", "power"):
        return any(_form_reads(form, tags) for form in atom[1:]
                   if isinstance(form, tuple))
    return False


def _form_reads(form: Any, tags: tuple) -> bool:
    if not isinstance(form, tuple):
        return False
    if form and isinstance(form[0], tuple) and len(form[0]) == 2 and isinstance(form[0][0], tuple):
        # a normal form: ((monomial, coef), ...)
        return any(_reads(atom, tags) for mono, _ in form for atom, _ in mono)
    return any(_form_reads(item, tags) for item in form)


def _opaque(atom: Any) -> bool:
    return atom[0] in ("call", "quotient", "power")


# --- derived values -------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Unknown:
    """A derived value that elaboration cannot give exactly, with the reason (IR13)."""

    reason: str

    def to_data(self) -> dict[str, Any]:
        return {"unknown": self.reason}


def _poly_data(poly: Polynomial) -> Any:
    def atom_data(atom: Any) -> Any:
        if _opaque(atom):
            return repr(atom)
        tag, indices = atom
        return [tag, list(indices)]

    return [[[[atom_data(atom), exponent] for atom, exponent in mono],
             [coef.numerator, coef.denominator]] for mono, coef in _sorted(poly)]


def _derived_data(value: Any) -> Any:
    if isinstance(value, Unknown):
        return value.to_data()
    if isinstance(value, dict):
        return {"exact": _poly_data(value)}
    if isinstance(value, tuple):
        return {"exact_vector": [_poly_data(item) for item in value]}
    return value


def _constant_poly(value: Fraction) -> Polynomial:
    return {(): value} if value != 0 else {}


_MEASURE: Polynomial = {((("cell.measure", ()), 1),): Fraction(1)}


def _reference_moment(exponents: dict[int, int], rank: int) -> Fraction:
    """The exact integral of prod_a xi_a^k_a over [-1, 1]^rank, divided by 2^rank."""
    out = Fraction(1)
    for axis in range(rank):
        k = exponents.get(axis, 0)
        out *= Fraction(0) if k % 2 else Fraction(1, k + 1)
    return out


def _basis(decode_poly: Polynomial, size: int) -> tuple[Polynomial, ...] | Unknown:
    """phi_s = decode(e_s) for a decode linear in the coefficients, else Unknown."""
    basis: list[Polynomial] = [{} for _ in range(size)]
    for mono, coef in decode_poly.items():
        indices = [(atom, e) for atom, e in mono if atom[0] == "coefficients"]
        if any(_opaque(atom) and _reads(atom, ("coefficients",)) for atom, _ in mono):
            return Unknown("decode reads the coefficients inside a call or quotient")
        if len(indices) != 1 or indices[0][1] != 1:
            return Unknown("decode is not linear in the coefficients")
        (atom, _), = indices
        k = atom[1][0]
        if k >= size:
            return Unknown("decode reads coefficients[%d] outside the local set of size %d"
                           % (k, size))
        rest = tuple(item for item in mono if item[0][0] != "coefficients")
        basis[k][rest] = basis[k].get(rest, 0) + coef
    return tuple(basis)


def _rule(rule: Any) -> tuple:
    """A rule as data: ``pops.Quadrature`` or any value with ``nodes``, ``weights`` and
    ``interval`` (the core names no rule, D0), kept as exact binary64 numbers (the rule as written,
    N16 3.3)."""
    import math
    try:
        nodes, weights, interval = rule.nodes, rule.weights, rule.interval
    except AttributeError:
        raise TypeError("pairing is a pops.Quadrature(nodes=, weights=, interval=) or None, got %r"
                        % (rule,)) from None
    nodes = tuple(float(node) for node in nodes)
    weights = tuple(float(weight) for weight in weights)
    interval = tuple(float(end) for end in interval)
    if not nodes or len(nodes) != len(weights) or len(interval) != 2 \
            or any(not math.isfinite(value) for value in nodes + weights) \
            or not interval[0] < interval[1]:
        raise ValueError("a pairing rule has n >= 1 finite nodes, as many finite weights, and an "
                         "interval lower < upper")
    return nodes, weights, interval


@dataclass(frozen=True, slots=True, eq=False)
class RepresentationDefinition:
    """The definition layer of contract @2: no space (N16 5.5)."""

    carrier: str
    size: int
    decode: Body
    encode: Body | None
    measure: str
    orientation: str
    pairing: Any
    kind: str
    rank: int
    dual_of: RepresentationDefinition | None = None

    def __init__(self, *, decode: Any, rank: int, size: int = 1, carrier: str = "cell",
                 encode: Any = None, measure: str = "support", pairing: Any = None,
                 kind: str = "coefficient", dual_of: Any = None) -> None:
        if carrier not in CARRIERS:
            raise ValueError("the carrier is one of %s, got %r" % (", ".join(CARRIERS), carrier))
        size = _index(size, "size")
        if size < 1:
            raise ValueError("a local index set has at least one coefficient")
        rank = _index(rank, "rank")
        if not 1 <= rank <= 7:
            raise ValueError("a cell of rank %d is outside 1..7" % rank)
        if kind not in KINDS:
            raise ValueError("kind is one of %s, got %r" % (", ".join(KINDS), kind))
        if measure not in MEASURES:
            raise ValueError("measure is 'support' or 'none' in a definition; the elaborated "
                             "representation resolves it against the configuration")
        decode = _body(decode)
        if encode is not None:
            encode = _body(encode)
            if _form_reads(normal_form(encode), _POINT + ("coefficients",)):
                raise ValueError("encode reads the carrier entity and the field, never "
                                 "coefficients or point")
        elif carrier != "cell":
            raise ValueError("encode is required on a %s carrier; the projection under the "
                             "pairing is the default on a cell carrier only" % carrier)
        if _form_reads(normal_form(decode), ("field",)):
            raise ValueError("decode reads coefficients and point, never field")
        if pairing is not None:
            pairing = _rule(pairing)
        if dual_of is not None and not isinstance(dual_of, RepresentationDefinition):
            raise TypeError("dual_of is the definition whose loads these coefficients are")
        orientation = "+axis" if carrier in ("face", "edge") else "none"
        for key, value in dict(carrier=carrier, size=size, decode=decode, encode=encode,
                               measure=measure, orientation=orientation, pairing=pairing,
                               kind=kind, rank=rank, dual_of=dual_of).items():
            object.__setattr__(self, key, value)
        self._check_indices()

    def _check_indices(self) -> None:
        for form in (normal_form(self.decode),) + (
                () if self.encode is None else (normal_form(self.encode),)):
            for mono, _ in form:
                for atom, _ in mono:
                    if _opaque(atom):
                        continue
                    tag, indices = atom
                    if tag in ("point.reference", "point.position", "cell.width") \
                            and indices[0] >= self.rank:
                        raise ValueError("%s[%d] reads an axis outside a cell of rank %d"
                                         % (tag, indices[0], self.rank))
                    if tag == "coefficients" and indices[0] >= self.size:
                        raise ValueError("coefficients[%d] is outside the local set of size %d"
                                         % (indices[0], self.size))
                    if tag == "node.reference" and (indices[0] >= 2 ** self.rank
                                                    or indices[1] >= self.rank):
                        raise ValueError("node.reference[%d][%d] is outside a cell of rank %d"
                                         % (indices[0], indices[1], self.rank))

    # -- data and identity --

    def _pairing_data(self) -> Any:
        if self.pairing is None:
            return None
        nodes, weights, interval = self.pairing
        return {"nodes": [list(value.as_integer_ratio()) for value in nodes],
                "weights": [list(value.as_integer_ratio()) for value in weights],
                "interval": [repr(end) for end in interval]}

    def normalized_data(self) -> dict[str, Any]:
        """What the definition identity digests: normal forms, never a Python name."""
        return {"contract": CONTRACT, "schema_version": SCHEMA_VERSION,
                "carrier": self.carrier, "size": self.size, "rank": self.rank,
                "decode": _poly_data(_polynomial(self.decode)),
                "encode": "projection" if self.encode is None
                else _poly_data(_polynomial(self.encode)),
                "measure": self.measure, "orientation": self.orientation,
                "pairing": self._pairing_data(), "kind": self.kind,
                "dual_of": None if self.dual_of is None else self.dual_of.identity}

    def to_data(self) -> dict[str, Any]:
        data = self.normalized_data()
        data["decode_body"] = _to_data(self.decode)
        data["encode_body"] = None if self.encode is None else _to_data(self.encode)
        data["obligations"] = list(OBLIGATIONS)
        data["identity"] = self.identity
        return data

    @property
    def identity(self) -> str:
        from . import _data_identity
        return "pops.representation-definition:sha256:" + hashlib.sha256(
            bytes.fromhex(_data_identity(self.normalized_data()))).hexdigest()

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)

    # -- derived data --

    @property
    def point_dependent(self) -> bool:
        return _form_reads(normal_form(self.decode), _POINT)

    @property
    def linear(self) -> bool | Unknown:
        basis = _basis(_polynomial(self.decode), self.size)
        if isinstance(basis, Unknown):
            if "not linear" in basis.reason:
                return False
            return basis
        return True

    @property
    def constant_mode(self) -> Polynomial | Unknown:
        """c, when decode is c * coefficients[0] with c point-free and made of geometry reads."""
        if self.carrier != "cell" or self.size != 1 or self.dual_of is not None:
            return Unknown("one constant mode needs a cell carrier of size 1")
        basis = _basis(_polynomial(self.decode), 1)
        if isinstance(basis, Unknown):
            return basis
        c, = basis
        if not c:
            return Unknown("decode is zero")
        if any(_opaque(atom) or atom[0] not in _GEOMETRY for atom in _atoms(c)):
            return Unknown("the factor of coefficients[0] reads the point or an opaque word")
        return c

    def rate_weights(self) -> tuple[Polynomial, ...] | Unknown:
        """m_K[s] = integral of phi_s over K, the load of the constant 1 (N16 5.2)."""
        if self.measure == "none":
            return Unknown("the configuration states no measure")
        if self.carrier != "cell" or self.dual_of is not None:
            return Unknown("rate weights of a %s carrier are G08's" % self.carrier)
        c = self.constant_mode
        if not isinstance(c, Unknown):
            return (_times(c, _MEASURE),)
        basis = _basis(_polynomial(self.decode), self.size)
        if isinstance(basis, Unknown):
            return basis if self.pairing is None else Unknown(
                basis.reason + "; the stated rule takes it in N16 lot 2")
        weights = []
        for phi in basis:
            total: Polynomial = {}
            for mono, coef in phi.items():
                exponents: dict[int, int] = {}
                rest = []
                for atom, e in mono:
                    if _opaque(atom) or atom[0] == "point.position":
                        reason = ("decode is not normalizable in the reference point" if
                                  self.pairing is None else
                                  "decode is not normalizable; the stated rule takes it in "
                                  "N16 lot 2")
                        return Unknown(reason)
                    if atom[0] == "point.reference":
                        exponents[atom[1][0]] = exponents.get(atom[1][0], 0) + e
                    else:
                        rest.append((atom, e))
                moment = _reference_moment(exponents, self.rank)
                if moment:
                    total = _plus(total, _times({tuple(rest): coef * moment}, _MEASURE))
            weights.append(total)
        return tuple(weights)

    def load_weights(self) -> tuple[Polynomial, ...] | Unknown:
        """w with sum_s w_s phi_s = 1, when 1 is in the span (N16 5.2)."""
        c = self.constant_mode
        if not isinstance(c, Unknown):
            if len(c) == 1 and all(atom[0] in _GEOMETRY for atom in _atoms(c)):
                return (_invert_monomial(c),)
            return Unknown("1 / c for a factor c that is a sum")
        basis = _basis(_polynomial(self.decode), self.size)
        if isinstance(basis, Unknown):
            return basis
        monomials = sorted({mono for phi in basis for mono in phi}, key=repr)
        if any(atom[0] != "point.reference" for mono in monomials for atom, _ in mono):
            return Unknown("the basis reads the geometry or the position; N16 lot 2 takes it")
        target = {mono: Fraction(1 if mono == () else 0) for mono in monomials}
        target.setdefault((), Fraction(1))
        rows = sorted(target, key=repr)
        solution = _solve_exact([[phi.get(mono, Fraction(0)) for phi in basis] for mono in rows],
                                [target[mono] for mono in rows])
        if solution is None:
            return Unknown("1 is not in the span of decode")
        return tuple(_constant_poly(value) for value in solution)

    def riesz_scale(self) -> Polynomial | Unknown:
        """1 / (c mu(K)) for one constant mode (N16 5.1)."""
        if self.measure == "none":
            return Unknown("the configuration states no measure")
        c = self.constant_mode
        if isinstance(c, Unknown):
            return Unknown("no constant mode: the Gram path of N16 lot 2")
        cm = _times(c, _MEASURE)
        if not _is_geometry_monomial(cm) and cm != {(): Fraction(1)}:
            if len(cm) == 1 and () in cm:
                return {(): 1 / cm[()]}
            return Unknown("c mu(K) is not a monomial of the geometry reads")
        if () in cm and len(cm) == 1:
            return {(): 1 / cm[()]}
        return _invert_monomial(cm)

    def derived_data(self) -> dict[str, Any]:
        linear = self.linear
        return {"point_dependent": self.point_dependent,
                "linear": linear if isinstance(linear, bool) else linear.to_data(),
                "constant_mode": _derived_data(self.constant_mode),
                "rate_weights": _derived_data(self.rate_weights()),
                "load_weights": _derived_data(self.load_weights()),
                "riesz_scale": _derived_data(self.riesz_scale())}


def _solve_exact(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """A solution of matrix x = rhs over the rationals, or None when there is none."""
    rows = [row[:] + [value] for row, value in zip(matrix, rhs, strict=True)]
    n = len(matrix[0]) if matrix else 0
    pivots: list[int] = []
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, len(rows)) if rows[i][col] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        lead = rows[r][col]
        rows[r] = [value / lead for value in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                factor = rows[i][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r], strict=True)]
        pivots.append(col)
        r += 1
    if any(all(value == 0 for value in row[:-1]) and row[-1] != 0 for row in rows):
        return None
    solution = [Fraction(0)] * n
    for i, col in enumerate(pivots):
        solution[col] = rows[i][-1]
    return solution


def constant_mode_scale(definition: RepresentationDefinition) -> str | None:
    """The residual scale of one constant mode, the compile-time choice of the native residual
    (G03 note 4.3, A6): ``"cell_measure"`` (divide the face sum by mu(K)) for c = 1, as an average
    or a sample; ``"unit"`` (do not divide) for c = 1 / cell.measure, a total; None for anything
    else (the Gram path of N16 lot 2)."""
    c = definition.constant_mode
    if isinstance(c, Unknown):
        return None
    if c == {(): Fraction(1)}:
        return "cell_measure"
    if c == {((("cell.measure", ()), -1),): Fraction(1)}:
        return "unit"
    return None


@dataclass(frozen=True, slots=True, eq=False)
class ElaboratedRepresentation:
    """``Representation(space, definition)``: the definition with the space (N16 5.5)."""

    space: Any
    definition: RepresentationDefinition

    def __init__(self, space: Any, definition: RepresentationDefinition) -> None:
        from pops.model.spaces import Space
        if not isinstance(space, Space):
            raise TypeError("an elaborated representation takes a Space, got %r" % (space,))
        if not isinstance(definition, RepresentationDefinition):
            raise TypeError("an elaborated representation takes a RepresentationDefinition")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "definition", definition)

    def to_data(self) -> dict[str, Any]:
        return {"contract": CONTRACT, "space": self.space.to_data(),
                "definition": self.definition.identity}

    @property
    def identity(self) -> str:
        from . import _data_identity
        return "pops.representation:sha256:" + hashlib.sha256(
            bytes.fromhex(_data_identity(self.to_data()))).hexdigest()

    def require_same(self, other: Any, *, where: str) -> None:
        """K4: an array is read only under the elaborated identity it was written under."""
        if type(other) is not ElaboratedRepresentation or other.identity != self.identity:
            raise TypeError("%s: coefficients written under %s are read under %s; equal shapes do "
                            "not make them interchangeable (support, space, measure or "
                            "representation differ)" % (where, _describe(other), _describe(self)))

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)


def _describe(value: Any) -> str:
    if isinstance(value, ElaboratedRepresentation):
        return "%s of kind %r" % (value.identity[-12:], value.definition.kind)
    return repr(value)


class MeasureMissing(ValueError):
    """D4 (spec 14.1, delta 5): an inventory whose rate weights cannot be formed."""

    rule = "D4"
    phase = "validate"

    def __init__(self, message: str, *, quantity: Any, representation: Any) -> None:
        self.quantity = quantity
        self.representation = representation
        super().__init__("MeasureMissing (D4, validate): %s" % message)


def inventory_weights(quantity: Any, representation: Any) -> tuple[Polynomial, ...]:
    """The rate weights an inventory of ``quantity`` reads, or ``MeasureMissing`` (D4).

    An inventory is sum_K m_K^T c_K, never sum_K c_K. When 1 is not in the span of decode m_K still
    exists; it is a conservation claim, which reads w_K, that is then refused with
    ``ConservationNotStructural`` (N16 5.2)."""
    definition = representation.definition if isinstance(
        representation, ElaboratedRepresentation) else representation
    if not isinstance(definition, RepresentationDefinition):
        raise TypeError("an inventory reads a representation of contract @2")
    weights = definition.rate_weights()
    if isinstance(weights, Unknown):
        raise MeasureMissing("the inventory of %s cannot be formed: %s"
                             % (quantity, weights.reason), quantity=quantity,
                             representation=definition.identity)
    return weights


# --- the four cell words of N16 3.7, as @2 data (lot R's oracles; N16 lot 1 writes the words) ---

def cell_average(rank: int, quadrature: Any = None) -> RepresentationDefinition:
    return RepresentationDefinition(decode=coefficients[0], rank=rank, pairing=quadrature,
                                    kind="average")


def cell_total(rank: int) -> RepresentationDefinition:
    return RepresentationDefinition(decode=coefficients[0] / cell.measure, rank=rank,
                                    kind="total")


def point_value(rank: int) -> RepresentationDefinition:
    return RepresentationDefinition(decode=coefficients[0], rank=rank, encode=field[0],
                                    kind="sample")


__all__ = [
    "CONTRACT", "OBLIGATIONS", "Body", "Call", "Const", "ElaboratedRepresentation", "Leaf",
    "MeasureMissing", "Op", "RepresentationDefinition", "Unknown", "body_from_data", "call", "cell",
    "cell_average", "cell_total", "coefficients", "constant_mode_scale", "field",
    "inventory_weights", "node_reference", "normal_form", "point", "point_value",
]
