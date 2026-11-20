"""The elaboration of a case of the surface into a case of b6 (2.6.6; SU5, SU8, SU9, SU12).

Building ``pops.Case(...)`` traces it (SU5). Its support becomes a b6 ``CartesianDomain`` named
``box`` and its frame; its model a b6 ``pops.Model`` of that frame, each constant a ``ConstParam``,
each parameter a ``RuntimeParam``, each evolved quantity a state of one component, each flux a
flux whose waves are the speed dF/du that R9 derives, each diffusive flux the F^d of R7, the
opposite of the physical flux, and each balance a rate. Its numerics become a
``DiscretizationPlan``: each term of each balance is selected and receives the realization that
its scheme word gives (``SchemeValue.realize``). Its program becomes the traced method of N8,
elaborated for the evolved state with a fixed step, the attempts of ``pops.Advance`` and the
default acceptance ``finite()``; each rate option of the method is the sum of the selections of
the terms that it names, in the order of their balance. Its initial value is a datum supplied at
bind, and its layout a b6 ``Uniform`` layout whose periodic axes are those of the support.

Every name of the elaborated form comes from the case (SU8): the name of the model, of each
declaration and of each named term; an unnamed term takes the name that ``model_names`` derives
from its kind and its quantity. ``pops.inspect(case)`` runs b6's ``validate`` and ``resolve`` on
the elaborated case and reports, row by row, what each construct became, its level, and the
origin of each choice, stated, derived or default (SU12).
"""
from __future__ import annotations

import math as _math
from decimal import Decimal
from fractions import Fraction
from collections.abc import Iterable, Sequence
from typing import Any, cast

from pops import math as _pm
from pops._structural_data import canonical_json
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform as _Uniform
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import ConstParam, RuntimeParam
from pops.params import Positive as _Positive
from pops.physics import Model as _Model
from pops.problem import Case as _Case
from pops.projection import ConservativeCellAverage
from pops.time._program import elaboration as _n8
from pops.time._program import relations as _relations
from pops.time._step.bound import StepBound, step_fraction

from ._expr import Arith, Call, Expr, Predicate, is_number, text, walk
from ._geometry import (
    CHART_GHOSTS, CHART_POINT_READS, CHART_RECONSTRUCTIONS, FRAME_AXES, MAP_WORDS, Box, Charted,
    Coordinate, FrameCoordinate, LogicalCoordinate, MathConstant, Support, chart_expression_data,
)
from ._method import MethodValue, _Trace, specialize, translate
from ._model import (
    Balance, Datum, DiffusiveFlux, Flux, Model, ModelConstant, Parameter, Positive, Quantity,
    Source, Term, model_names,
)
from ._model import _sign as _derived_sign
from ._numerics import CellAverage, Numerics, SchemeValue, TermView
from ._refusal import (
    AmbiguousElaboration, CoverageNotExact, MissingBoundaryRealization, NotYetElaborated,
    SupportMismatch, SurfaceRefusal, UnknownSolvePath,
)
from ._scheme import FaceFlux, StencilRealization, UserScheme

_LATER = "a later lot of N15"
# The words of pops.math that b6's analytic expressions have (pops.analytic), for initial values.
_ANALYTIC_WORDS = ("exp", "log", "sqrt", "sin", "cos", "erf", "erfc", "abs", "where")
# The words of a node function of pops.Graded and of a chart (``MAP_WORDS``): those of the analytic
# programs that generate the static geometry arrays at bind, without erf and erfc.
_NODE_WORDS = MAP_WORDS
# The tolerance of the ends of a graded axis, in ulp of the largest bound (DN3).
_GRADED_END_ULP = 4
_FRAMES = {1: Cartesian1D, 2: Cartesian2D, 3: Cartesian3D}


# --- the report ---------------------------------------------------------------------------------

class Choice:
    """A choice of the elaboration and its origin: ``stated``, ``derived`` or ``default`` (SU9)."""

    __slots__ = ("name", "value", "origin")

    def __init__(self, name: str, value: Any, origin: str) -> None:
        self.name = name
        self.value = value
        self.origin = origin

    def __str__(self) -> str:
        return "%s = %s (%s)" % (self.name, self.value, self.origin)


def written_in(body: Any) -> str:
    """Where the body of a scheme or a method is written, for ``pops.inspect``: ``pops.lib`` and
    its line, or the case file and its line (S1-b of the vertical slice)."""
    from pops._source_origin import code_origin, origin_text

    return origin_text(code_origin(body))


class Row:
    """One construct: its mathematics, its level, its elaborated form and its choices."""

    __slots__ = ("construct", "level", "elaborated", "choices")

    def __init__(self, construct: str, level: str, elaborated: str,
                 choices: Iterable[Any] = ()) -> None:
        self.construct = construct
        self.level = level
        self.elaborated = elaborated
        self.choices = tuple(choices)

    def lines(self) -> list:
        out = ["[%s] %s" % (self.level, self.construct), "    -> %s" % self.elaborated]
        out.extend("    %s" % choice for choice in self.choices)
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


class Report:
    """What ``pops.inspect`` returns: the rows of a value, and for a case its resolved plan."""

    __slots__ = ("subject", "level", "rows", "resolved")

    def __init__(self, subject: str, level: str, rows: Iterable[Any], resolved: Any = None) -> None:
        self.subject = subject
        self.level = level
        self.rows = tuple(rows)
        self.resolved = resolved

    def choices(self) -> tuple:
        return tuple(choice for row in self.rows for choice in row.choices)

    def __str__(self) -> str:
        head = "pops.inspect(%s), level %s" % (self.subject, self.level)
        if self.resolved is not None:
            head += ": validate and resolve passed, nothing compiled"
        return "\n".join([head] + [line for row in self.rows for line in row.lines()])

    __repr__ = __str__


class Elaborated:
    """What a case elaborates into: the b6 case, its layout and block, the rows of its report."""

    __slots__ = ("case", "layout", "block", "model", "plan", "states", "names", "advance", "rows",
                 "params", "outputs", "representations")

    def __init__(self, **fields: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])


# --- the case -----------------------------------------------------------------------------------

def elaborate_case(case: Any) -> Elaborated:
    """The b6 case of a surface case; a refusal names the line of the case as well (SU11)."""
    try:
        return _Elaboration(case).run()
    except SurfaceRefusal as refusal:
        raise refusal.traced_at(case.site) from None


def elaborated_representation(quantity: Any, support: Support, definition: Any = None) -> Any:
    """The elaborated identity of a quantity's representation (G03 lot E, K4).

    b6's state carries neither the support nor the representation of the quantity it elaborates
    (``b6.state(name, components=(name,))``), so two quantities of equal shape on two supports, or
    under two representations, would be interchangeable once elaborated. The surface keeps, beside
    each state, ``Representation(space, definition)`` of contract @2 (G03 lot R): the space is the
    support's coordinates and the scalar value, never the quantity's name (P10), and the definition
    is the one that ``represent=`` states, ``CellAverage()`` when none is (N16 5.5). Nothing of it
    enters b6 beyond the row of ``plan.representations`` that a total, a sample or a stated rule
    adds, so no identity or manifest of a case under ``CellAverage()`` changes. Published arrays
    carry it, and ``bind`` reads it back (``surface/_run.py``): an array of totals is not the
    datum of a quantity represented by averages."""
    from pops.model.spaces import Space
    from pops._ir.quantity import PhysicalSupport
    from pops.representations.definition import ElaboratedRepresentation, cell_average
    coordinates = tuple(
        (name, ("periodic[%r, %r)" if periodic else "[%r, %r]") % (float(lower), float(upper)))
        for name, (lower, upper, periodic) in zip(support._names, support.bounds(), strict=True))
    presentation = None
    if isinstance(support, Charted):
        components = tuple(zip(support._frame_names, support._components, strict=True))
        presentation = coordinates + tuple((name, "chart %s" % text(component))
                                            for name, component in components)
        logical = tuple(
            ("logical_axis_%d" % index, canonical_json((float(lower).hex(), float(upper).hex(),
                                                       periodic)))
            for index, (lower, upper, periodic) in enumerate(support.bounds()))
        image = tuple(
            ("frame_axis_%d" % index, canonical_json(chart_expression_data(
                component, support._box, "the component %s of the chart of %r" % (name, support._box))))
            for index, (name, component) in enumerate(components))
        coordinates = logical + image
    space = Space("scalar", components=("value",), support=PhysicalSupport(coordinates))
    return ElaboratedRepresentation(
        space, cell_average(support.rank) if definition is None else definition,
        support_presentation=presentation)


# The owner of the projection of an initial value by the encode of its representation.
_PROJECTION_BY_ENCODE = ("N16 lot 2 (the projection by encode: cell.integral, the sample, the Riesz "
                         "map of a stated rule); until then the initial value is a datum, the "
                         "coefficients bound at pops.bind")


def _sampled(value: Any) -> Any:
    """``value`` with each ``cell.value(q)`` written as ``q``, for the b6 row of a source."""
    from ._scheme import SchemeLeaf

    if isinstance(value, SchemeLeaf) and value.key == ("sample", 0):
        return value.quantity
    if isinstance(value, Arith):
        return Arith(value.op, tuple(_sampled(arg) for arg in value.args))
    if isinstance(value, Call):
        return Call(value.word, tuple(_sampled(arg) for arg in value.args))
    if isinstance(value, Predicate):
        return Predicate(value.op, *(_sampled(arg) for arg in value.args))
    return value


def _coordinates(value: Any) -> list:
    """The coordinates that ``value`` reads outside its loads, one per axis, by axis."""
    found: dict = {}
    for node in walk(value, descend=lambda part: getattr(part, "key", None) != ("load",)):
        if isinstance(node, Coordinate):
            found.setdefault(node.index, node)
    return [found[index] for index in sorted(found)]


def _lower_predicate(value: Predicate, lower: Any, *, where: str) -> Any:
    """Lower the surface predicate through the comparison algebra of its consumer."""
    # N3 lot 2 (P4): equal(a, b) is b6 ==, all_of and any_of the left folds of & and |,
    # and none_of([p, q, ...]) is ~(p | q | ...), ~p for one predicate.
    if value.op in ("all_of", "any_of", "none_of"):
        items = [lower(arg) for arg in value.args]
        result = items[0]
        for item in items[1:]:
            result = result & item if value.op == "all_of" else result | item
        return ~result if value.op == "none_of" else result
    if value.op not in ("<", "<=", ">", ">=", "equal"):
        raise NotYetElaborated("%s in %s" % (text(value), where), _LATER)
    left, right = (lower(arg) for arg in value.args)
    return {"<": lambda: left < right, "<=": lambda: left <= right,
            ">": lambda: left > right, ">=": lambda: left >= right,
            "equal": lambda: left == right}[value.op]()


def _is_cell_total(definition: Any) -> bool:
    return getattr(definition, "kind", None) == "total"


def _is_cell_average(definition: Any) -> bool:
    """The definition identity of ``CellAverage()`` at its rank, matched whole (N16 5.5)."""
    from ._representation import is_cell_average

    return is_cell_average(definition)


# b6 selects a term of a balance whose weight is 1 or -1 only (the partitions of a physical
# balance).
_WEIGHTED = "a later lot of N15, once a selection of b6 realizes a term of another weight"


def _dissipation(coefficient: Any) -> str:
    """R2, K7: whether the declarations establish that F = -k grad(w) dissipates, k > 0."""
    if _derived_sign(coefficient) == 1:
        return "established: %s > 0" % text(coefficient)
    return "not established: the sign of %s is not derived" % text(coefficient)


def _per_axis(values: tuple | None) -> Any:
    """A value per axis as the surface writes it: the scalar on a support of rank one, else
    ``vector(...)`` of the components (lot 2b)."""
    if values is None:
        return None
    return values[0] if len(values) == 1 else Call("vector", tuple(values))


def _per_axis_text(values: tuple, spell: Any) -> str:
    return spell(values[0]) if len(values) == 1 else \
        "vector(%s)" % ", ".join(spell(value) for value in values)


_G04 = "a later lot of N15 with the boundary relations of G04"
# The reads that a generated ghost of G04-a does not take (D3), with their owners.
_NORMAL_GHOSTS = "a later lot of G04 with the characteristic and reflected ghosts"
_SYMBOLIC_DEPTH = ("G04 phase 2, with the symbolic depth of 2.6.6 (the generated ghosts of G04-a "
                   "read the cells 1..D of the halo of the block)")
_FACE_CENTRE_DATA = "G04, with the boundary data read at the centre of the face (C44)"


_ANALYTIC_LOT = "a later lot of N15 with the generated initial values of G07"
_STEP_REQUEST = "a later lot of N8 (the solve requests of a step rule, 5.5)"


def _tensor(mapped: list) -> list:
    """The tensor nodes of one list of (point, weight) per axis, the first axis slowest."""
    rows: list = [()]
    for nodes in mapped:
        rows = [row + (node,) for row in rows for node in nodes]
    return rows


def _number(value: Any) -> Any:
    """A number of the surface as b6 takes it: an exact fraction or a decimal becomes a float."""
    if isinstance(value, (Fraction, Decimal)):
        return float(value)
    return value


def _logical_text(name: str, low: float, high: float) -> str:
    """The logical coordinate s of an axis on [low, high], as the arrays compute it."""
    shifted = name if low == 0.0 else "%s %s %r" % (name, "-" if low > 0 else "+", abs(low))
    if high - low == 1.0:
        return shifted
    return "%s / %r" % (shifted if low == 0.0 else "(%s)" % shifted, high - low)


class _RateSum:
    """A rate option of a method: the sum of the selections of its terms, in balance order.

    At an unknown of a solve, the rate is read as is, at the solve's operand (§3.4 of the N9 lot
    « La référence sous les solves de relation »), and its claims are those that the schemes of
    its terms state (``linear_claims``, IR16): each term asks its scheme for them when a relation
    reads it, never before, so that an explicit method never needs them. Every term of the rate
    states them: a term that is not linear and homogeneous in the state (a source, a forcing) is
    refused by name, since it would make the operator affine.
    """

    __slots__ = ("views", "linear", "_claims")

    def __init__(self, views: Sequence[Any], linear: Sequence[Any] = ()) -> None:
        self.views = tuple(views)
        self.linear = tuple(linear)
        self._claims: tuple | None = None

    def __call__(self, state: Any) -> Any:
        result = self.views[0](state)
        for view in self.views[1:]:
            result = result + view(state)
        return result

    def claims(self) -> tuple:
        if self._claims is None:
            self._claims = tuple(linear() for linear in self.linear)
        return self._claims

    @property
    def linear_claims(self) -> Any:
        from pops.time._program.relations import LinearClaims

        claims = []
        for stated in self.claims():
            claim = LinearClaims.of(stated)
            if claim is None:
                raise NotYetElaborated(
                    "the rate %r read at the unknown of a solve states no claims (symmetric, "
                    "negative_semidefinite, homogeneous, proof, description; IR16)" % (stated,),
                    "G10, the operator generated from the term, with the Newton solves of N9")
            claims.append(claim)
        return LinearClaims(
            symmetric=all(claim.symmetric for claim in claims),
            negative_semidefinite=all(claim.negative_semidefinite for claim in claims),
            homogeneous=all(claim.homogeneous for claim in claims),
            proof="; ".join(claim.proof for claim in claims),
            description=" + ".join(claim.description for claim in claims))


class _StencilTerm:
    """A diffusive term realized by a ``StencilFlux`` (N3 lot 1, D7): in each evaluation of a rate
    that covers it, ``-div_h`` of its face flux, the evaluation k named ``<name>_<k>``.

    ``stencil_flux_divergence`` gives -div_h(F) of the physical flux F, which is the contribution
    of ``-div(F)`` to the balance; a term written ``+div(F)`` is its opposite. The evaluations are
    counted by name in the Program, wherever they are recorded: a solve's operator evaluates the
    rate inside its declaration and its region, whose values are not among the Program's top-level
    values (§3.4 of the N9 lot « La référence sous les solves de relation »)."""

    __slots__ = ("flux", "name", "weight")

    def __init__(self, flux: Any, name: str, weight: int) -> None:
        self.flux = flux
        self.name = name
        self.weight = weight

    def __call__(self, state: Any) -> Any:
        from pops.time._program.stencil_flux import OPERATION

        program = state.prog
        counts = program.__dict__.setdefault("_stencil_term_evaluations", {})
        evaluation = counts.get(self.name, 0)
        rate = getattr(program, OPERATION)("%s_%d" % (self.name, evaluation), state,
                                           flux=self.flux)
        counts[self.name] = evaluation + 1
        return rate if self.weight == -1 else -rate


class _Elaboration:
    def __init__(self, case: Any) -> None:
        self.case = case
        self.model: Model = case.model
        self.numerics: Numerics = case.numerics
        self.program = case.program
        self.rows: list = []
        self.values: dict = {}
        self.names: dict = {}
        self.linear: dict = {}
        self.layout_facts: dict = {}
        self.params: dict = {}
        self.coordinate_auxiliaries: dict = {}
        self.auxiliaries: list = []
        self.lowered_outputs: tuple = ()
        # N3 lot 3 (P4): the b6 expression of each node of the body of a scheme, by identity, for
        # the inputs of the body lowered last; a shared node is lowered once, as b6 built it.
        self._lowered: tuple = (None, {})
        self.charted = False
        self._flux_coordinate_reads: dict[Coordinate, Any] = {}
        self._flux_coordinate_names: dict[Coordinate, str] = {}
        self._in_flux_law = False

    # The support and its frame.

    def support(self) -> Support:
        model = self.model
        spaces = [q.space for q in model.quantities] + [d.space for d in model.data_fields]
        if not spaces:
            raise AmbiguousElaboration("the model %r has no quantity" % model.name,
                                       write='model.quantity("u", pops.Scalar(Ω))',
                                       site=model.site)
        support = spaces[0].support
        for space in spaces[1:]:
            if space.support is not support:
                raise NotYetElaborated("a model whose quantities live on several supports",
                                       "a later lot of N15 with the product supports of G09",
                                       site=space.site)
        if isinstance(support, Charted):
            # DN28: a degenerate piece needs the identification of the piece with itself.
            if support._degenerate:
                raise NotYetElaborated(
                    "a case on the charted support %r, whose piece %s is degenerate: the chart "
                    "does not depend on the other coordinate there, so its faces have zero "
                    "measure and a read across it needs pole = Ω.identify(...) before the chart"
                    % (support, support._degenerate[0]),
                    "lot L2c of N17 (ROM-110), with identify of G09", site=support.site)
            return support
        names = support._names
        if names != FRAME_AXES[:len(names)]:
            raise NotYetElaborated(
                "a support whose axes are named %s; this lot elaborates the axes x, y and z of the "
                "Cartesian frames, in that order" % ", ".join(names),
                "a later lot of N15 with the supports of G03", site=support.site)
        return support

    def frame(self, support: Support) -> Any:
        """The frame of the box; on a charted support, the frame of its reference box, whose axes
        are its logical axes by position: the domain is the image of the box by the chart, which
        the layout states (A1 of the design note N17, DN26)."""
        lower = tuple(float(lo) for lo, _, _ in support.bounds())
        upper = tuple(float(hi) for _, hi, _ in support.bounds())
        frame = CartesianDomain("box", lower=lower, upper=upper).frame(_FRAMES[support.rank]())
        periodic = [name for name, (_, _, flag)
                    in zip(support._names, support.bounds(), strict=True) if flag]
        choices = [Choice("periodic axes", ", ".join(periodic) or "none", "stated")]
        if isinstance(support, Charted):
            choices += [
                Choice("logical axes", ", ".join(support._names), "stated"),
                Choice("frame axes", "%s, the components of a flux and of a vector"
                       % ", ".join(support._frame_names), "stated")]
        self.rows.append(Row(
            repr(support), "model",
            'CartesianDomain("box", lower=%r, upper=%r).frame(%s())'
            % (lower, upper, _FRAMES[support.rank].__name__), choices))
        return frame

    # The model.

    def lower(self, value: Any, inputs: dict | None = None) -> Any:
        """The b6 expression of a value of the model; ``inputs`` gives the b6 expressions of the
        reads of the entities of a scheme, inside the body of a scheme (lot 2)."""
        if inputs is not None:
            known, lowered = self._lowered
            if known is not inputs:
                known, lowered = inputs, {}
                self._lowered = (known, lowered)
            seen = lowered.get(id(value))
            if seen is not None and seen[0] is value:
                return seen[1]
            result = self._lower(value, inputs)
            lowered[id(value)] = (value, result)
            return result
        return self._lower(value, None)

    def _lower(self, value: Any, inputs: dict | None) -> Any:
        if inputs is not None:
            if isinstance(value, Coordinate) and ("coordinate", value.name) in inputs:
                return inputs[("coordinate", value.name)]
            if ("load", id(value)) in inputs:
                return inputs[("load", id(value))]
            scheme = self._scheme_value(value, inputs)
            if scheme is not None:
                return scheme
        if is_number(value):
            return _number(value)
        if isinstance(value, MathConstant):
            if not _math.isfinite(value.value):
                raise NotYetElaborated("%s in a value of the model" % value.name, _LATER)
            return value.value
        if isinstance(value, (ModelConstant, Parameter, Quantity)):
            return self.values[value]
        if isinstance(value, Datum):
            raise NotYetElaborated("the datum %s read in a value of the model (Q3)" % value,
                                   "a later lot of N15 with the data fields of Q3",
                                   site=value.site)
        if isinstance(value, (Coordinate, FrameCoordinate)):
            if (isinstance(value, Coordinate) and self._in_flux_law
                    and value in self._flux_coordinate_reads):
                return self._flux_coordinate_reads[value]
            if self.charted or isinstance(value.support, Charted):
                raise NotYetElaborated("the coordinate %s of a charted support read in a value "
                                       "of the model" % value, CHART_POINT_READS)
            raise NotYetElaborated("the coordinate %s read in a value of the model" % value,
                                   "a later lot of N15 with the coordinate fields of G03")
        if isinstance(value, Arith):
            args = [self.lower(arg, inputs) for arg in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right,
                    "**": lambda: left ** right}[value.op]()
        if isinstance(value, Call) and value.word == "abs":
            return abs(self.lower(value.args[0], inputs))
        if isinstance(value, Call) and value.word in ("exp", "expm1", "log", "sqrt", "sin", "cos"):
            return getattr(_pm, value.word)(self.lower(value.args[0], inputs))
        if isinstance(value, Term):
            raise NotYetElaborated("the term %s read in the value of another" % value, _LATER,
                                   site=value.site)
        raise NotYetElaborated("%s in a value of the model" % text(value),
                               "N11 (expression closure, C1)")

    def _scheme_value(self, value: Any, inputs: dict) -> Any:
        """The words that only the body of a scheme lowers in this lot: the reads of its entities,
        ``where`` and its comparisons, ``largest`` and ``smallest`` (b6 ``where``, ``maximum`` and
        ``minimum``, whose derivative stays the one of b6). ``None`` for every other value."""
        from ._scheme import SchemeLeaf, is_radius_pair, speed_pair_index

        if isinstance(value, SchemeLeaf):
            if value.key in inputs:
                return inputs[value.key]
            return self.lower(value.expansion, inputs)
        if ("speed",) in inputs and is_radius_pair(value):
            return inputs[("speed",)]
        pair = speed_pair_index(value) if ("speeds", 0) in inputs else None
        if pair is not None:
            return inputs[("speeds", pair)]
        if isinstance(value, Predicate):
            return _lower_predicate(value, lambda arg: self.lower(arg, inputs),
                                    where="the body of a scheme")
        if isinstance(value, Call) and value.word == "where":
            predicate, then, otherwise = value.args
            then_value, otherwise_value = self.lower(then, inputs), self.lower(otherwise, inputs)
            return _pm.where(self.lower(predicate, inputs), lambda: then_value,
                             lambda: otherwise_value)
        if isinstance(value, Call) and value.word in ("largest", "smallest"):
            fold = _pm.maximum if value.word == "largest" else _pm.minimum
            items = [self.lower(arg, inputs) for arg in value.args]
            result = items[0]
            for item in items[1:]:
                result = fold(result, item)
            return result
        return None

    def code(self, value: Any, spelled: dict | None = None) -> str:
        """The b6 spelling of a value, for the report; ``spelled`` gives the spelling of a load
        (by identity) and of a coordinate (by name) read through an auxiliary input."""
        if spelled:
            if id(value) in spelled:
                return spelled[id(value)]
            if isinstance(value, Coordinate) and ("coordinate", value.name) in spelled:
                return spelled[("coordinate", value.name)]
        if isinstance(value, Quantity):
            return "%s[0]" % value.name
        if isinstance(value, Predicate):
            args = [self.code(arg, spelled) for arg in value.args]
            if value.op in ("all_of", "any_of", "none_of"):
                fold = (" & " if value.op == "all_of" else " | ").join(args)
                return ("~" if value.op == "none_of" else "") + "(%s)" % fold
            op = "==" if value.op == "equal" else value.op
            return "(%s %s %s)" % (args[0], op, args[1])
        if isinstance(value, Call):
            word = value.word if value.word == "abs" else "pm." + value.word
            return "%s(%s)" % (word, ", ".join(self.code(arg, spelled) for arg in value.args))
        if isinstance(value, Arith):
            if value.op == "neg":
                return "-(%s)" % self.code(value.args[0], spelled)
            return "(%s %s %s)" % (self.code(value.args[0], spelled), value.op,
                                   self.code(value.args[1], spelled))
        if isinstance(value, MathConstant):
            return repr(value.value)
        return text(value)

    def declarations(self, b6: Any) -> dict:
        model = self.model
        for constant in model.constants:
            param = ConstParam(constant.name, float(constant.value))
            self.values[constant] = b6.value(b6.param(param))
            self.rows.append(Row(
                'model.constant(%r, value=%r)' % (constant.name, constant.value), "model",
                '%s = model.value(model.param(ConstParam(%r, %r)))'
                % (constant.name, constant.name, float(constant.value)),
                [Choice("value", constant.value, "stated"),
                 Choice("binding", "folded at compile time", "derived")]))
        for parameter in model.parameters:
            options: dict = {"default": float(parameter.default)}
            if isinstance(parameter.admissible, Positive):
                options["domain"] = _Positive()
            param = RuntimeParam(parameter.name, **options)
            handle = b6.param(param)
            self.params[parameter] = handle
            self.values[parameter] = b6.value(handle)
            choices = [Choice("default", parameter.default, "stated")]
            if parameter.admissible is not None:
                choices.append(Choice("admissible", parameter.admissible, "stated"))
            self.rows.append(Row(
                "model.parameter(%r, default=%r)" % (parameter.name, parameter.default), "model",
                "%s = model.value(model.param(RuntimeParam(%r, default=%r%s)))"
                % (parameter.name, parameter.name, float(parameter.default),
                   ", domain=Positive()" if "domain" in options else ""), choices))
        evolved = [balance.quantity for balance in model.balances]
        states: dict = {}
        for quantity in model.quantities:
            if not any(quantity is other for other in evolved):
                raise NotYetElaborated(
                    "the quantity %s, which no balance evolves (a constraint unknown or a "
                    "definition)" % quantity, "a later lot of N15 with the constraints of N10",
                    site=quantity.site)
            roles = None
            if isinstance(quantity.space.support, Box) and quantity.space.support._identifications:
                from pops.physics import Scalar as ScalarRole

                roles = {quantity.name: ScalarRole()}
            handle = b6.state(quantity.name, components=(quantity.name,), roles=roles)
            states[quantity] = handle
            self.values[quantity] = handle[0]
            self.rows.append(Row(
                "model.quantity(%r, %r)" % (quantity.name, quantity.space), "model",
                "%s = model.state(%r, components=(%r,))"
                % (quantity.name, quantity.name, quantity.name),
                [Choice("evolved", "by the balance of %s" % quantity, "derived")]))
        return states

    def terms(self, b6: Any, frame: Any, states: dict) -> dict:
        model = self.model
        integrated = [term for balance in model.balances for term in balance.terms]
        for term in model.terms:
            if not any(term is other for other in integrated):
                raise CoverageNotExact(
                    "the term %s of %s is in no balance; every term is integrated (T5)"
                    % (term, term.quantity),
                    write="the term in the balance of its quantity, or no such term",
                    site=term.site)
        self.numerics.check_coverage(model.balances, model.terms)
        self.names = model_names(model)
        elaborated: dict = {}
        for term in model.terms:
            name = self.names[term]
            origin = "stated" if term.name is not None else "derived"
            state = states[term.quantity]
            if isinstance(term, DiffusiveFlux):
                elaborated[term] = self.diffusive(b6, term, name, origin, state)
            elif isinstance(term, Source):
                elaborated[term] = self.source(b6, frame, term, name, origin, state)
            else:
                elaborated[term] = self.flux(b6, frame, term, name, origin, state)
        return elaborated

    def source(self, b6: Any, frame: Any, term: Source, name: str, origin: str,
               state: Any) -> Any:
        """The b6 named source of a source, whose value is the value of the ``CellRate`` that its
        scheme gives, read at the cell (N15 FaceFlux/CellRate FC-1, note §3.4, Q3): for
        ``pointwise()`` it is the law of the source at the cell value. A coordinate of the support
        is read at the centre of the cell, and a load ``cell.load(e)`` is the mean of e under the
        rule of the representation; each is an analytic auxiliary input of b6 (FC-2, §3.6)."""
        from ._builder_value import BuilderValue, classify

        scheme = self.numerics.scheme_of(term)
        builder = scheme.traced(term).builder
        definition = self.numerics.definition_of(term.quantity)
        representation = self.numerics.representation_of(term.quantity)
        typed = classify(builder.value, word="CellRate", definition=definition,
                         site=getattr(scheme, "site", None))
        inputs: dict = {("sample", 0): self.values[term.quantity]}
        spelled: dict = {}
        choices = [Choice("name", name, origin), Choice("scheme", scheme, "stated"),
                   Choice("kind of value", typed.kind, "derived")]
        for index, load in enumerate(typed.loads, start=1):
            aux = "%s_load_%d" % (name, index)
            mean = self.rule_mean(load.value, frame, (definition.pairing,) * len(frame.axes))
            inputs[("load", id(load))] = self.auxiliary(b6, frame, aux, mean)
            spelled[id(load)] = "Var(%r, 'aux')" % aux
            choices.append(Choice(aux, "the mean of %s on the cell under the rule of %r, an "
                                       "analytic auxiliary input" % (text(load.value),
                                                                     representation),
                                  "derived"))
        for coordinate in _coordinates(typed.value):
            aux = "coordinate_%s" % coordinate.name
            if aux not in self.coordinate_auxiliaries:
                from pops.analytic import coordinate as analytic_coordinate

                self.coordinate_auxiliaries[aux] = self.auxiliary(
                    b6, frame, aux, analytic_coordinate(frame, frame.axes[coordinate.index]))
            inputs[("coordinate", coordinate.name)] = self.coordinate_auxiliaries[aux]
            spelled[("coordinate", coordinate.name)] = "Var(%r, 'aux')" % aux
            choices.append(Choice(coordinate.name, "the coordinate %s at the centre of the "
                                                   "cell, an analytic auxiliary input (2.6.4)"
                                  % coordinate.name, "derived"))
        value = self.lower(typed.value, inputs)
        handle = b6.source(name, on=state, value=(value,))
        delivered = ("μ(K) times the value read at the cell, the rate of a total (N16 5.1, A1)"
                     if _is_cell_total(definition) else "the value read at the cell")
        if typed.kind == BuilderValue.LOAD:
            delivered = "the mean of the load on the cell, the load over μ(K)"
        choices.append(Choice("delivered", "the rate of %r: %s" % (representation, delivered),
                              "derived"))
        self.rows.append(Row(
            "model.source(of=%s, value=%s)" % (term.quantity, text(term.value)), "model",
            "%s = model.source(%r, on=%s, value=(%s,))"
            % (name, name, term.quantity.name, self.code(_sampled(typed.value), spelled)),
            choices))
        return handle

    def auxiliary(self, b6: Any, frame: Any, name: str, expression: Any) -> Any:
        """An imposed auxiliary input of b6 named ``name``, fixed on a static layout: its value at
        each cell is ``expression`` (``pops.analytic``), and the b6 expression it returns,
        ``Var(name, 'aux')``, reads it (N15 FaceFlux/CellRate lot FC-2, note §3.6). Its producer
        ``AnalyticAux`` is registered on the final module of the model (``providers``)."""
        self.auxiliaries.append((name, expression, frame))
        return b6.auxiliary(name, frame=frame.canonical_id)

    def providers(self, b6: Any) -> None:
        """Register the producer of each auxiliary input on the final module of the model."""
        from pops.fields import AnalyticAux, AuxiliaryBoundary

        if not self.auxiliaries:
            return
        module = b6.module
        declared = module.aux()
        for name, expression, frame in self.auxiliaries:
            module.aux_provider(AnalyticAux(module.aux_handle(declared[name]), expression,
                                            frame=frame,
                                            boundary=AuxiliaryBoundary(width=0)))

    def declare_flux_coordinates(self, b6: Any, frame: Any, term: Flux) -> None:
        coordinates = tuple(dict.fromkeys(node for node in walk(term.value)
                                          if isinstance(node, Coordinate)))
        if not coordinates or self.charted:
            return
        from ._program import Uniform

        support = self.support()
        scheme = self.numerics.scheme_of(term)
        builder = scheme.traced(term).builder if isinstance(scheme, UserScheme) else None
        representation = self.numerics.representation_of(term.quantity)
        if (type(self.case.layout_of(support)) is not Uniform or not self.layout_facts["periodic"]
                or (representation is not None and type(representation) is not CellAverage)
                or not is_number(self.program.advance.dt)
                or not isinstance(builder, FaceFlux) or builder.reconstruction is not None):
            raise NotYetElaborated(
                "the coordinate %s read in the flux law" % coordinates[0],
                "N15/G03 entity coordinates on a periodic Uniform CellAverage layout, "
                "using constant traces and a fixed step", site=term.site)
        for coordinate in coordinates:
            if coordinate.support is not support:
                raise SupportMismatch("flux coordinate belongs to another support", site=term.site)
            if coordinate not in self._flux_coordinate_reads:
                name = "flux_coordinate_" + coordinate.name
                self._flux_coordinate_names[coordinate] = name
                self._flux_coordinate_reads[coordinate] = b6.auxiliary(name, frame=frame.canonical_id)

    def install_flux_coordinates(self, b6: Any, frame: Any) -> None:
        if not self._flux_coordinate_names:
            return
        from pops.fields import AnalyticAux

        module = b6.module
        for coordinate, name in self._flux_coordinate_names.items():
            target = module.aux_handle(module.aux()[name])
            module.aux_provider(AnalyticAux._entity_coordinate(target, frame, coordinate.index))

    def lower_flux_law(self, value: Any) -> Any:
        self._in_flux_law = True
        try:
            return self.lower(value)
        finally:
            self._in_flux_law = False

    def flux(self, b6: Any, frame: Any, term: Flux, name: str, origin: str, state: Any) -> Any:
        self.declare_flux_coordinates(b6, frame, term)
        axes = frame.axes
        slopes = term.slopes()
        waves = None if slopes is None else {
            axis: (self.lower_flux_law(slope),) for axis, slope in zip(axes, slopes, strict=True)}
        handle = b6.flux(name, state=state, frame=frame,
                         components={axis: (self.lower_flux_law(component),) for axis, component
                                     in zip(axes, term.components, strict=True)}, waves=waves)
        choices = [Choice("name", name, origin)]
        choices.append(Choice("waves", "none: dF/%s is not derived in this lot" % term.quantity
                              if slopes is None else "dF/d%s = %s"
                              % (term.quantity, _per_axis_text(slopes, text)), "derived"))
        signed = ""
        scheme = self.numerics.scheme_of(term)
        if slopes is not None and getattr(scheme, "reads_speeds", lambda key: False)(term):
            # R9: the slowest and the fastest speeds of a scalar flux are both dF/du, the pair
            # that the face input speeds of riemann.User reads (N3 lot 1, hll()).
            b6.wave_speeds(handle, frame=frame, values={
                axis: (self.lower_flux_law(slope), self.lower_flux_law(slope))
                for axis, slope in zip(axes, slopes, strict=True)})
            signed = ", and model.wave_speeds(%s, frame=frame, values=the pair (dF/du, dF/du))" \
                % name
            choices.append(Choice("signed waves", "(dF/d%s, dF/d%s), read by %r"
                                  % (term.quantity, term.quantity, scheme), "derived"))
        names = FRAME_AXES[:len(axes)]
        self.rows.append(Row(
            "model.flux(of=%s, value=%s)" % (term.quantity, text(term.value)), "model",
            "%s = model.flux(%r, state=%s, frame=frame, components={%s}, waves=%s)"
            % (name, name, term.quantity.name,
               ", ".join("%s: (%s,)" % (axis, self.code(component))
                         for axis, component in zip(names, term.components, strict=True)),
               "None" if slopes is None else "{%s}" % ", ".join(
                   "%s: (%s,)" % (axis, self.code(slope))
                   for axis, slope in zip(names, slopes, strict=True))) + signed, choices))
        return handle

    def diffusive(self, b6: Any, term: DiffusiveFlux, name: str, origin: str, state: Any) -> Any:
        (variable,) = term.variables
        value = self.lower(term.coefficient) * _pm.grad(self.values[variable])
        handle = b6.diffusive_flux(name, state=state, value=value)
        self.rows.append(Row(
            "model.diffusive_flux(of=%s, value=%s)" % (term.quantity, text(term.value)),
            "model",
            "%s = model.diffusive_flux(%r, state=%s, value=%s * pm.grad(%s[0])), the F^d of R7, "
            "opposite to the physical flux" % (name, name, term.quantity.name,
                                              self.code(term.coefficient), variable.name),
            [Choice("name", name, origin),
             Choice("differentiated variables", ", ".join(map(str, term.variables)), term.origin),
             Choice("k", text(term.coefficient), "derived"),
             Choice("dissipation", _dissipation(term.coefficient), "derived")]))
        return handle

    def rate(self, b6: Any, balance: Balance, terms: dict, states: dict) -> Any:
        """The b6 rate of the transport terms of a balance; a diffusive term is a stencil of the
        program (``_StencilTerm``), so it is not a term of the b6 rate (N3 lot 1)."""
        rhs = None
        parts = []
        for term, coefficient in balance.occurrences:
            if coefficient not in (1, -1):
                raise NotYetElaborated(
                    "the weight %s of %s in the balance of %s, a weight other than 1 or -1"
                    % (coefficient, term if isinstance(term, Source) else "div(%s)" % term,
                       balance.quantity), _WEIGHTED, site=balance.site)
        stencils = []
        for term, coefficient in balance.occurrences:
            if isinstance(term, DiffusiveFlux):
                stencils.append("%sthe StencilFlux of %s"
                                % ("" if coefficient == -1 else "-", self.names[term]))
                continue
            if isinstance(term, Source):
                # A source enters the rate as itself, with the sign of its occurrence.
                piece = terms[term]
                spelled = self.names[term]
            else:
                piece = _pm.div(terms[term])
                spelled = "div(%s)" % self.names[term]
            if rhs is None:
                rhs = piece if coefficient == 1 else -piece
                parts.append(spelled if coefficient == 1 else "-" + spelled)
            elif coefficient == 1:
                rhs = rhs + piece
                parts.append("+ " + spelled)
            else:
                rhs = rhs - piece
                parts.append("- " + spelled)
        name = balance.name if balance.name is not None else balance.quantity.name
        program = (", and in the program %s" % " + ".join(stencils)) if stencils else ""
        origin = Choice("name", name, "stated" if balance.name is not None else "derived")
        if rhs is None:
            self.rows.append(Row("model.balance(%s)" % balance, "model",
                                 "no rate of b6: in the program %s" % " + ".join(stencils),
                                 [origin]))
            return None
        handle = b6.rate(name, equation=_pm.ddt(states[balance.quantity]) == rhs)
        self.rows.append(Row(
            "model.balance(%s)" % balance, "model",
            "model.rate(%r, equation=pm.ddt(%s) == %s)%s"
            % (name, balance.quantity.name, " ".join(parts), program), [origin]))
        return handle

    # The numerics.

    def plan(self, rates: dict, terms: dict, states: dict) -> tuple:
        plan = DiscretizationPlan()
        selections: dict = {}
        for balance in self.model.balances:
            weights = {id(term): coefficient for term, coefficient in balance.occurrences}
            if rates[balance] is None:
                # Only stencils: the state has no rate of b6 to carry its storage, so it is stated,
                # with the one ghost layer of the two-point StencilFlux (D7).
                plan.states.add(states[balance.quantity], StateStorage(ghost_depth=1))
            for term in balance.terms:
                scheme = self.numerics.scheme_of(term)
                slope = _per_axis(term.slopes()) if isinstance(term, Flux) else None
                speed = _per_axis(term.unit_fluxes()) if isinstance(term, Flux) else None
                realization = scheme.realize(TermView(term, terms[term], states[term.quantity],
                                                      slope=slope, unit_flux=speed,
                                                      lower=self.lower))
                if self.charted and not isinstance(term, DiffusiveFlux):
                    order = (getattr(getattr(realization, "reconstruction", None), "options",
                                     None) or {}).get("formal_order", 1)
                    if order != 1:
                        raise NotYetElaborated(
                            "%r of %s on the charted support %r, whose traces are reconstructed "
                            "to the formal order %s from the logical stencil of a cell"
                            % (scheme, term, term.quantity.space.support, order),
                            CHART_RECONSTRUCTIONS,
                            site=getattr(scheme, "site", None) or self.numerics.site)
                if isinstance(term, DiffusiveFlux):
                    if not isinstance(realization, StencilRealization):
                        raise TypeError("the scheme %r of the diffusive flux %s gives %r; a "
                                        "diffusive key is realized by a face scheme"
                                        % (scheme, term, realization))
                    selections[term] = _StencilTerm(realization.flux, self.names[term],
                                                    weights[id(term)])
                else:
                    view = rates[balance].select(terms[term])
                    selections[term] = view
                    plan.rates.add(view, realization)
                self.linear[term] = self._linear(scheme, TermView(
                    term, terms[term], states[term.quantity], slope=slope, unit_flux=speed,
                    lower=self.lower), balance)
                choices = [Choice("scheme", scheme, "stated")]
                if hasattr(scheme, "scheme"):       # a scheme written with @pops.scheme
                    choices.append(Choice("written in", written_in(scheme.scheme.body),
                                          "stated"))
                choices.extend(Choice("option %s" % name, value, origin)
                               for name, value, origin in getattr(scheme, "option_choices",
                                                                  list)())
                if speed is not None:
                    choices.append(Choice("speed", "s = F(1)·n = %s" % text(speed), "derived"))
                if isinstance(term, DiffusiveFlux) and self.charted:
                    # DN36: K-orthogonality is measured, not required; the inspection states it.
                    choices.append(Choice(
                        "faces", "a face of the chart takes the line of its two centroids for its "
                        "normal (K-orthogonality, DN36); the arrays bound at pops.bind report the "
                        "largest sine of the two as orthogonality", "derived"))
                reads_radius = getattr(scheme, "reads_radius", None)
                if self.charted and reads_radius is not None and reads_radius(term):
                    # DN31: a transitional route, until the exact radius on a physical normal.
                    choices.append(Choice(
                        "speed on the chart", "the face input s, spectral_radius(term, u, n) at "
                        "the two traces, is the bound sum_k |n_k| rho(A_k) of b6 on the physical "
                        "normal of a face, not the exact radius rho(sum_k n_k A_k): equal on an "
                        "axis normal, larger off it and without bound near a face tangent to the "
                        "flow; a transitional route until lot L2d of N17 (DN31)", "derived"))
                self.rows.append(Row(
                    "schemes={%s: %r}" % (term, scheme), "numerics",
                    ("%s, %s" % (self.names[term], scheme.describe()))
                    if isinstance(term, DiffusiveFlux) else
                    "plan.rates.add(%s.select(%s), %s)"
                    % (balance.name or balance.quantity.name, self.names[term],
                       scheme.describe()), choices))
        for quantity in states:
            representation = self.numerics.representation_of(quantity)
            definition = self.numerics.definition_of(quantity)
            if _is_cell_average(definition):
                lowered = ("the cell averages of %s, projected by ConservativeCellAverage()"
                           % quantity)
            else:
                # A total, a sample or an average under a stated rule: the row of b6 that serves
                # it (G03 lot R-b; N16 5.1, A1), matched by the whole identity (N16 5.5).
                plan.representations.add(states[quantity], definition)
                lowered = ("plan.representations.add(%s, the definition of kind %r%s, %s)"
                           % (quantity.name, definition.kind,
                              "" if definition.pairing is None else " with a stated rule",
                              definition.identity[-12:]))
            self.rows.append(Row("represent={%s: %r}" % (quantity, representation),
                                 "numerics", lowered,
                                 [Choice("representation", repr(representation), "stated")]))
        return plan, selections

    def _linear(self, scheme: Any, view: TermView, balance: Balance) -> Any:
        """The claims of the rate of a term read at the unknown of a solve, asked when a solve
        reads it, never before (IR16): the scheme states them, by the normal form of its face, so
        that a copy of ``two_point()`` written in a case is claimed as the word (N3 lot 1, D7). No
        solver states them: the operator of a relation solve is its components' rates at the
        State, and each solve runs the solver that ``solvers=`` keys by its own site (O1 of the
        delta review of #778, the N9 lot « La référence sous les solves de relation »)."""
        from pops.lib.solve import linear_claims

        weight = next(-coefficient if isinstance(term, DiffusiveFlux) else coefficient
                      for term, coefficient in balance.occurrences if term is view.term)

        if isinstance(view.term, Source):
            def refused() -> Any:
                raise NotYetElaborated(
                    "the linear map of the source %s, which a solve reads" % view.term,
                    "a later lot of N15 with the implicit sources (the linear map of a "
                    "CellRate)", site=view.term.site)
            return refused

        def linear() -> Any:
            return linear_claims(view, scheme=scheme, weight=weight, layout=self.layout_facts)
        return linear

    # The program.

    def method_components(self, value: MethodValue, selections: dict) -> dict:
        model = self.model
        components: dict = {}
        covered: list = []
        for name in value.rate_options():
            option = value.options[name]
            if isinstance(option, Model):
                if option is not model:
                    raise ValueError("%s(%s=%r): the model is not the model of the case"
                                     % (value.name, name, option))
                chosen = [term for balance in model.balances for term in balance.terms]
            else:
                listed = option if isinstance(option, list) else [option]
                for term in listed:
                    if term.model is not model:
                        raise ValueError("%s(%s=...): the term %s belongs to another model"
                                         % (value.name, name, term))
                chosen = [term for balance in model.balances for term in balance.terms
                          if any(term is item for item in listed)]
            for term in chosen:
                if any(term is other for other in covered):
                    raise CoverageNotExact(
                        "the term %s is integrated twice by the method %s" % (term, value.name),
                        write="each term in one rate option of %s" % value.name, site=value.site)
                covered.append(term)
            components[name] = _RateSum([selections[term] for term in chosen],
                                        [self.linear[term] for term in chosen])
        for balance in model.balances:
            for term in balance.terms:
                if not any(term is other for other in covered):
                    raise CoverageNotExact(
                        "the term %s of the balance of %s is integrated by no rate option of %s"
                        % (term, balance.quantity, value.name),
                        write="%s(rate=model), or a rate option that names it" % value.name,
                        site=value.site)
        return components

    def step_bound(self, block: Any, states: dict) -> tuple:
        """``dt=f`` with f a function of ``cell``: the bound that f writes and its b6 expression
        over the cell reads of the block state and the widths of the cell (N8 « dt= as a function
        of cell », D1, D3 (a))."""
        from pops.mesh._amr.refinement_criterion import CellRead, CellWidth

        from ._step_cell import trace, value_key, width_key

        rule = self.program.advance.dt
        support = self.support()
        bound = trace(rule, support, tuple(states))
        inputs: dict = {width_key(index): CellWidth(index) for index in range(support.rank)}
        for quantity, name in states.items():
            handle = block[name]
            (component,) = handle.space.components
            inputs[value_key(quantity)] = CellRead(handle, component, ())
        return bound, self.lower(bound, inputs)

    def advance(self, case6: Any, block: Any, states: dict, selections: dict) -> Any:
        advance = self.program.advance
        value = advance.method
        components = self.method_components(value, selections)
        trace = _Trace(value.site)
        specialized = specialize(value, components, trace)
        retry = _n8.Retry() if advance.attempts is None else _n8.Retry(
            max_attempts=advance.attempts)
        (quantity,) = states
        request = None if advance.tolerance is None else advance.tolerance.request()
        realization = self.realization(value)
        outputs = self.outputs(block)
        rule: Any = advance.dt
        cells = callable(rule) and not is_number(rule)
        if cells:
            # dt= as a function of cell (N8, D1): the bound that it writes, lowered over the cell
            # reads of the block state and the widths of the cell, and the Courant budget that it
            # declares (C2): cfl_limit's cfl, or 1 for a function written by hand.
            name = getattr(rule, "__name__", rule)
            if request is not None:
                raise NotYetElaborated(
                    "pops.Advance(dt=%s, tolerance=...), the request of the solves of a step that "
                    "a function of cell bounds" % name, _STEP_REQUEST, site=advance.site)
            bound, lowered = self.step_bound(block, states)
            try:
                fraction, origin = step_fraction(rule)
            except (TypeError, ValueError) as error:
                raise type(error)("pops.Advance(dt=%s): %s" % (name, error)) from None
            proposal: Any = StepBound(lowered, self.support().rank, fraction, origin)
        else:
            proposal = _n8.Proposal(float(cast(float, rule)), request=request)
        try:
            elaboration = _n8.elaborate(_n8.Advance(specialized, retry=retry), case=case6,
                                        state=block[states[quantity]], proposal=proposal,
                                        realization=realization, outputs=outputs)
        except Exception as error:
            translated = translate(error, trace)
            if translated is error:
                raise
            if isinstance(translated, SurfaceRefusal) and isinstance(
                    error, (_relations.UnknownSolvePath, _relations.MissingRequest)):
                translated.site = self.program.site
                translated.args = (translated._message(),)
            raise translated.with_traceback(None) from None
        if cells:
            control = elaboration.control
            stated = (Choice("dt", "%s, the smallest over the cells of %s at the accepted state "
                                   "(PF8)" % (getattr(rule, "__name__", rule), text(bound)),
                             "stated"),
                      Choice("fraction", "%r, the budget of the face guards of each attempt "
                                         "(C2)%s" % (
                                             control.step_bound[0],
                                             ", declared by the rule (step_fraction)"
                                             if proposal.origin == "stated"
                                             else ", that of a fixed step"),
                             proposal.origin))
            shown = "StepBound(%s, fraction=%r)" % (text(bound), control.step_bound[0])
        else:
            stated = (Choice("dt", advance.dt, "stated"),)
            shown = "Proposal(%r%s)" % (float(advance.dt), "" if request is None
                                        else ", request=%r" % (request,))
        self.rows.append(Row(
            repr(advance), "program",
            "elaborate(Advance(%s(%s), retry=Retry(max_attempts=%d)), case=case, state=block[%s], "
            "proposal=%s%s), the traced method of N8"
            % (value.name, ", ".join(components), retry.max_attempts, quantity.name, shown,
               "" if not realization else ", realization={%s}" % ", ".join(
                   "%r: %r" % item for item in realization.items())),
            [Choice("method", "%s, %d evaluations of its rates"
                    % (value.name, len(elaboration.trace.evaluations)), "stated"),
             Choice("written in", written_in(value.method.body), "stated"),
             *stated,
             Choice("attempts", retry.max_attempts,
                    "default" if advance.attempts is None else "stated"),
             Choice("accept", "finite()", "default")]
            + ([Choice("tolerance", advance.tolerance, "stated")]
               if advance.tolerance is not None else [])))
        for site in elaboration.trace.solves:
            self.rows.append(self.solve_row(value, site))
        return elaboration

    def realization(self, value: MethodValue) -> dict:
        """The realization of N8 that ``solvers=`` states: a method keys every solve of its
        attempt, ``method.name`` the solve named ``name`` (2.6.2, 5.5)."""
        realization: dict = {}
        for key, solver in self.program.solvers:
            method = key if isinstance(key, MethodValue) else key.method
            if method is not value:
                raise UnknownSolvePath(
                    "the key %r of solvers= names no solve: the advance runs %r" % (key, value),
                    write="solvers={%s: solver}, keyed by the method of the advance"
                          % value.name, site=self.program.site)
            path = "%s/*" % value.name if isinstance(key, MethodValue) \
                else "%s/%s" % (value.name, key.name)
            realization[path] = solver
        return realization

    def solve_row(self, value: MethodValue, site: Any) -> Row:
        """The row of ``pops.inspect`` of a solve of the method (5.5, K5)."""
        request = site.request
        claims = "; ".join("%s: %s" % (name, claim.description) for name, claim in site.claims)
        proofs = "; ".join(claim.proof for _, claim in site.claims)
        return Row(
            "pops.solve(...) in %s" % value.name, "program",
            "solve(Problem(unknowns=unknowns(U=B.state.space), equations=(rel,)), request=%r%s), "
            "then require_success: LinearProblem(matrix_free_operator a I + sum_k b_k L_k, "
            "initial_guess=B.state), solver %s"
            % (request, "" if site.name is None else ", name=%r" % site.name,
               site.solver.describe()),
            [Choice("path", site.path, "derived" if site.name is None else "stated"),
             Choice("request", request, "stated"),
             Choice("guess", "the state at the start of the attempt (SV3)", "derived"),
             Choice("solver", site.solver, "stated"),
             Choice("implicit operator", claims, "derived"),
             Choice("SPD certificate", "derived (IR16) from the claims of the rates of the "
                    "components, which the operator evaluates at the State, until IR16 derives "
                    "it from the face blocks (§17.1): %s" % proofs if site.spd
                    else "not derived", "derived"),
             Choice("failure", "SolveFailed fails the attempt (T38)", "derived")])

    def outputs(self, block: Any) -> tuple:
        """The Outputs of N8 of the case's outputs, in their order (CC14; lot G)."""
        lowered = []
        for output in self.case.outputs:
            for quantity in output.quantities():
                if quantity.model is not self.model:
                    raise ValueError("pops.Output(%r) publishes %s, a quantity of another model"
                                     % (output.record, quantity))
            value = output.lowered(block)
            lowered.append(value)
            schedule = ("every(%d)" % output.every_steps if output.every_steps is not None
                        else "at=every_dt(%r)" % output.every_time if output.every_time is not None
                        else "at=%r" % (output.times,))
            if output.every_steps is not None:
                schedule += (", after every k-th accepted step of each run, counted from its "
                             "start t0 (N8)")
            elif output.every_time is not None:
                schedule += ", at t0 + k h, t0 the start of each run (N8)"
            self.rows.append(Row(
                repr(output), "program",
                "Output(lambda B, interval: Product(%s), %s), evaluated by the compiled run (CC14)"
                % (", ".join("%s=B.state" % name for name, _ in output.record.items()), schedule),
                [Choice("when", output.when, "stated"),
                 Choice("write", "result.outputs" if output.write is None else output.write,
                        "default" if output.write is None else "stated")]))
        self.lowered_outputs = tuple(lowered)
        return self.lowered_outputs

    def initial(self, case6: Any, block: Any, states: dict, frame: Any) -> None:
        """The initial value of each evolved quantity (CN12, Q3): a datum, supplied at bind; a
        value that reads no coordinate, whose cell average is exact; ``pops.project(e, ...)``,
        lowered on the analytic route of b6 (``Analytic`` under ``ConservativeCellAverage``)
        with the exact cell integral that its rule or its integral gives (lot D-E)."""
        from pops.lib.initial import BindArray   # the datum supplied at bind (Q3), a b6 value

        from ._project import Projection

        program = self.program
        for key, value in program.initial:
            if not any(key is quantity for quantity in states):
                raise ValueError("pops.Program(initial={%s: ...}): %s is not a quantity that the "
                                 "model evolves" % (key, key))
            if isinstance(value, Datum) and value.model is not self.model:
                raise ValueError("pops.Program(initial={%s: %s}): the datum belongs to another "
                                 "model" % (key, value))
        for quantity, state in states.items():
            value = program.initial_of(quantity)
            if value is None:
                raise AmbiguousElaboration(
                    "the evolved quantity %s has no initial value" % quantity,
                    write='pops.Program(initial={%s: d}), d = model.data("d", %r)'
                          % (quantity, quantity.space), site=program.site)
            definition = self.numerics.definition_of(quantity)
            if not isinstance(value, Datum) and not _is_cell_average(definition):
                # The analytic route of b6 projects cell averages; the projection by the encode
                # of a total, a sample or a stated rule is N16 lot 2's, so their initial value
                # is a datum, the coefficients supplied at bind (N16, the lot that lifts the
                # table for the three words; review of #803, R1).
                raise NotYetElaborated(
                    "pops.Program(initial={%s: %s}) under represent={%s: %r}, the projection of "
                    "an initial value by the encode of a representation other than CellAverage() "
                    "(write pops.Program(initial={%s: d}), d = model.data(...))"
                    % (quantity, text(value) if not isinstance(value, Projection) else repr(value),
                       quantity, self.numerics.representation_of(quantity), quantity),
                    _PROJECTION_BY_ENCODE, site=program.site)
            if isinstance(value, Datum):
                provider, spelling = BindArray(), "BindArray()"
                choices = [Choice("value", "the datum %s, supplied at bind" % value, "stated")]
                stated = str(value)
            elif isinstance(value, Projection):
                if self.charted:
                    raise NotYetElaborated(
                        "pops.project(...) on the charted support %r, the average over a mapped "
                        "cell" % (quantity.space.support,),
                        "lot L2b of N17 (ROM-110), with the reads at the points of a chart (DN25)",
                        site=value.site)
                provider, spelling, choices = self.projection(value, quantity, frame)
                stated = repr(value)
            else:
                provider, spelling, choices = self.exact_value(value, frame)
                stated = text(value)
            case6.initials.add(InitialCondition(state=block[state], value=provider,
                                                projection=ConservativeCellAverage()))
            self.rows.append(Row(
                "initial={%s: %s}" % (quantity, stated), "program",
                "case.initials.add(InitialCondition(state=block[%s], value=%s, "
                "projection=ConservativeCellAverage()))" % (quantity.name, spelling),
                choices + [Choice("projection", "ConservativeCellAverage()", "derived")]))

    def exact_value(self, value: Any, frame: Any) -> tuple:
        """A value that reads no coordinate: its cell average is the value itself, which b6's
        analytic route gives exactly (one literal, or equal samples)."""
        from pops.lib.initial import Analytic

        lowered = self.analytic(value, frame, ())
        return (Analytic(frame=frame, components=(lowered,)),
                "Analytic(frame=frame, components=(%s,))" % text(value),
                [Choice("value", text(value), "stated"),
                 Choice("cell average", "exact, the value reads no coordinate", "derived")])

    def projection(self, value: Any, quantity: Any, frame: Any) -> tuple:
        """``pops.project(e, quadrature=Q)``: the cell integral measure x the sum over the tensor
        nodes of the weights (each divided by the length of its interval) times e at the nodes
        mapped onto the cell; ``integral=I``: I run on the bounds of the cell (CN12)."""
        from pops.analytic import CellBounds, coordinate
        from pops.lib.initial import Analytic

        from ._project import ProjectionCell, reads_coordinate

        support = quantity.space.support
        bounds = CellBounds(frame)
        axes = frame.axes
        point = tuple(coordinate(frame, axis) for axis in axes)
        component = self.analytic(value.expression, frame, point)
        if value.integral is not None:
            from ._expr import Expr

            result = value.integral(ProjectionCell(support))
            if not (is_number(result) or isinstance(result, Expr)):
                raise TypeError("pops.project(e, integral=I): I(cell) returns the integral of e "
                                "over the cell, an expression, got %r" % (result,))
            if reads_coordinate(result):
                raise TypeError("pops.project(e, integral=I): I(cell) reads the bounds of the "
                                "cell, cell.lower(axis) and cell.upper(axis), not a coordinate")
            integral = self.analytic(result, frame, (), bounds)
            spelling, rule = text(result), Choice("integral", text(result), "stated")
        else:
            rules = tuple((rule.nodes, rule.weights, rule.interval)
                          for rule in value.rules(len(axes)))
            integral = bounds.measure * self.rule_mean(value.expression, frame, rules)
            spelling = "the cell measure x the rule of %r" % (value.quadrature,)
            rule = Choice("quadrature", value.quadrature, "stated")
        provider = Analytic(frame=frame, components=(component,), cell_integrals=(integral,))
        return (provider,
                "Analytic(frame=frame, components=(%s,), cell_integrals=(%s,))"
                % (text(value.expression), spelling),
                [Choice("value", text(value.expression), "stated"), rule,
                 Choice("cell average", "the cell integral divided by the cell measure, "
                                        "native", "derived")])

    def rule_mean(self, value: Any, frame: Any, rules: tuple) -> Any:
        """The mean of ``value`` on the cell under a rule per axis, ``(nodes, weights, interval)``:
        the sum over the tensor nodes of the weights (each divided by the length of its interval)
        times the value at the nodes mapped onto the cell, an analytic expression over the bounds
        of the cell. ``projection`` multiplies it by the measure; the rate of ``cell.load(e)``
        under one constant mode is the mean itself (N15 FaceFlux/CellRate lot FC-2, note §3.6)."""
        from pops.analytic import CellBounds

        bounds = CellBounds(frame)
        mapped = []
        for axis, (rule_nodes, rule_weights, interval) in zip(frame.axes, rules, strict=True):
            low, high = (float(end) for end in interval)
            lower, upper = bounds.lower(axis), bounds.upper(axis)
            width = upper - lower
            nodes = []
            for node, weight in zip(rule_nodes, rule_weights, strict=True):
                fraction = (node - low) / (high - low)
                where = lower if fraction == 0.0 else upper if fraction == 1.0 \
                    else lower + width * fraction
                nodes.append((where, weight / (high - low)))
            mapped.append(nodes)
        terms = []
        for choice in _tensor(mapped):
            weight = 1.0
            for _, factor in choice:
                weight *= factor
            sample = self.analytic(value, frame, tuple(where for where, _ in choice))
            terms.append(sample if weight == 1.0 else weight * sample)
        average = terms[0]
        for term in terms[1:]:
            average = average + term
        return average

    def analytic(self, value: Any, frame: Any, point: tuple, bounds: Any = None) -> Any:
        """The analytic expression of b6 (``pops.analytic``) of a value of an initial value: the
        coordinates read ``point``, the bounds of the cell ``bounds``; a constant of the model is
        its number and a parameter its handle, bound at bind."""
        from pops import analytic as an

        from ._project import CellBound

        if is_number(value):
            return an.constant(float(_number(value)))
        if isinstance(value, MathConstant):
            if not _math.isfinite(value.value):
                raise NotYetElaborated("%s in an initial value" % value.name, _LATER)
            return an.constant(value.value)
        if isinstance(value, ModelConstant):
            return an.constant(float(value.value))
        if isinstance(value, Parameter):
            return an.param(self.params[value])
        if isinstance(value, Coordinate):
            return point[value.index]
        if isinstance(value, CellBound):
            axes = frame.axes
            if value.index is None:
                return bounds.measure
            return getattr(bounds, value.side)(axes[value.index])
        if isinstance(value, Arith):
            args = [self.analytic(arg, frame, point, bounds) for arg in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            if value.op == "**":
                exponent = value.args[1]
                if not is_number(exponent):
                    raise NotYetElaborated("%s, a power whose exponent is not a number, in an "
                                           "initial value" % text(value), _LATER)
                return left ** float(_number(exponent))
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right}[value.op]()
        if isinstance(value, Predicate):
            return _lower_predicate(value, lambda arg: self.analytic(arg, frame, point, bounds),
                                    where="an analytic expression")
        if isinstance(value, Call) and value.word in _ANALYTIC_WORDS:
            args = [self.analytic(arg, frame, point, bounds) for arg in value.args]
            return getattr(an, value.word)(*args)
        if isinstance(value, Call):
            raise NotYetElaborated("%s in an initial value, a word that b6's analytic "
                                   "expressions (pops.analytic) do not have" % text(value),
                                   _ANALYTIC_LOT)
        raise NotYetElaborated("%s in an initial value" % text(value),
                               "N11 (expression closure, C1)")

    def layout(self, support: Support, frame: Any) -> Any:
        """The b6 layout of the support: ``Uniform`` of a grid of its box; graded nodes and a chart
        are the clock-free coordinate map of ``MovingControlVolumes``, which the static geometry
        arrays serve (N17, C1), a chart with the rule that ``geometry=`` states (D0b, DN29)."""
        from ._program import Graded, Uniform

        case = self.case
        stated = case.layout_of(support)
        if stated is None:
            raise AmbiguousElaboration("the support %r has no layout" % (support,),
                                       write="layout={Ω: pops.Uniform(cells=c)}", site=case.site)
        for key, _ in case.layout:
            if key is not support:
                raise ValueError("pops.Case(layout=...) names the support %r, on which no "
                                 "quantity of the model lives" % (key,))
        uniform = stated.base if isinstance(stated, Graded) else stated
        rule = uniform.geometry
        charted = isinstance(support, Charted)
        if charted and rule is None:
            raise AmbiguousElaboration(
                "the charted support %r states no quadrature rule for its measures, centroids "
                "and normals, which the geometry arrays generate with it (D0b, DN29)" % (support,),
                write="layout={Ω: pops.Uniform(cells=%r, geometry=GaussLegendre(n))}, "
                      "GaussLegendre of pops.lib.quadrature" % (uniform.cells,), site=uniform.site)
        if not charted and rule is not None:
            raise SupportMismatch(
                "pops.Uniform(geometry=...) states a quadrature rule for %r, whose geometry is "
                "exact (D0b, DN29): %s" % (support, (
                    "its graded nodes are node positions, its measures the node differences"
                    if isinstance(stated, Graded) else
                    "the measures of a box are the products of its spacings")),
                write="pops.Uniform(cells=%r) without geometry=, which a charted support states"
                      % (uniform.cells,), site=uniform.site)
        periodic = tuple(axis for axis, (_, _, flag)
                         in zip(frame.axes, support.bounds(), strict=True) if flag)
        identifications = support._identifications if isinstance(support, Box) else ()
        if identifications and type(stated) is not Uniform:
            raise NotYetElaborated("a non-Uniform layout of an identified box",
                                   "G09 I2", site=stated.site)
        topology = self.identifications(support, identifications) if identifications else None
        try:
            layout = _Uniform(CartesianGrid(frame=frame, cells=uniform.cells,
                                            periodic=PeriodicAxes(periodic) if periodic else None,
                                            identifications=topology))
        except ValueError as error:
            if not identifications:
                raise
            raise SupportMismatch(str(error), site=stated.site) from error
        elaborated = ("Uniform(CartesianGrid(frame=frame, cells=%r, periodic=%s))"
                      % (uniform.cells, "PeriodicAxes((%s))" % "".join(
                          "%s," % axis for axis in support._names if support.periodic(axis))
                         if periodic else None))
        choices = [Choice("cells", uniform.cells, "stated")]
        if topology is not None:
            choices.append(Choice("identifications", layout.mesh.native_spatial_data()["identifications"],
                                  "derived"))
        if charted:
            layout, elaborated, charts = self.chart(stated, support, frame, layout, elaborated)
            choices += charts
        elif isinstance(stated, Graded):
            components, spelled, graded = self.graded(stated, support, frame)
            layout = MovingControlVolumes(layout, evolution=GeometryEvolution(components))
            elaborated = ("MovingControlVolumes(%s, evolution=GeometryEvolution((%s,)))"
                          % (elaborated, ", ".join(spelled)))
            choices += graded
        self.rows.append(Row("layout={%r: %r}" % (support, stated), "case", elaborated, choices))
        return layout

    def identifications(self, support: Support, rows: tuple) -> Any:
        from pops.mesh.boundaries import (
            BoundaryHandle, BoundaryOrientation, BoundarySide, BoundaryTopology,
            PeriodicIdentification, PeriodicOrientation,
        )
        from pops.model import OwnerPath

        owner = OwnerPath.case(self.model.name)
        faces = {
            axis + "_" + name: BoundaryHandle(
                axis + "_" + name, owner=owner, orientation=BoundaryOrientation(index, side))
            for index, axis in enumerate(support._names)
            for name, side in (("min", BoundarySide.LOWER), ("max", BoundarySide.UPPER))
        }
        identity = PeriodicOrientation(tuple(range(support.rank)), (1,) * support.rank)
        periodic = [
            PeriodicIdentification(faces[axis + "_min"], faces[axis + "_max"], identity)
            for axis in support._names if support.periodic(axis)]
        periodic.extend(PeriodicIdentification(faces[source], faces[target],
                                                PeriodicOrientation(permutation, signs), shifts)
                        for source, target, permutation, signs, shifts in rows)
        coupled = {face for row in periodic for face in (row.source, row.target)}
        return BoundaryTopology(owner, tuple(faces.values()), tuple(periodic),
                                tuple(face for face in faces.values() if face not in coupled))

    def chart(self, stated: Any, support: Charted, frame: Any, layout: Any,
              elaborated: str) -> tuple:
        """The chart of a charted support as the coordinate map of ``MovingControlVolumes`` with
        its rule (D0b, DN25, DN29): component a is the chart's component a, its coordinates read
        as the logical coordinates of the reference box by position, a graded logical coordinate
        as its node map, X(f(s), g(t)). The map reads no clock, so the static geometry arrays
        serve it as a chart of rank two (N17, C1, A1)."""
        from pops.analytic import coordinate

        from ._program import Graded

        rule = stated.base.geometry if isinstance(stated, Graded) else stated.geometry
        assert rule is not None  # the layout refuses a chart without its rule (DN29)
        values = tuple(coordinate(frame, axis) for axis in frame.axes)
        choices: list = []
        where = ""
        if isinstance(stated, Graded):
            values, spelled, choices = self.graded(stated, support, frame, charted=True)
            where = " where %s" % "; ".join(
                "%s = %s" % (name, spelled[index]) for index, name in enumerate(support._names)
                if stated.function_of(support._coordinates[index]) is not None)
        components = tuple(self._node_map(component, lambda node: values[node.index])
                           for component in support._components)
        layout = MovingControlVolumes(layout, evolution=GeometryEvolution(components),
                                      geometry=rule)
        elaborated = ("MovingControlVolumes(%s, evolution=GeometryEvolution((%s)), geometry=rule)%s"
                      % (elaborated, ", ".join(text(component) for component in support._components),
                         where))
        choices += [
            Choice("chart", ", ".join(
                "%s = %s" % (name, text(component))
                for name, component in zip(support._frame_names, support._components, strict=True)),
                "stated"),
            Choice("geometry", "a rule of %d nodes on %r, mapped onto each logical axis"
                   % (rule.n, rule.interval), "stated"),
            Choice("degenerate pieces", "none", "derived"),
            Choice("geometry arrays", "a chart of rank 2: measures and centroids by the rule, "
                   "chord area vectors and normals (N17, C1)", "derived")]
        return layout, elaborated, choices

    def graded(self, graded: Any, support: Support, frame: Any, charted: bool = False) -> tuple:
        """The coordinate map of ``pops.Graded`` (D0b, DN3): component a is f(s) on a graded axis,
        s = (ξ - lower) / (upper - lower) the logical coordinate of the axis in [0, 1], and ξ on
        any other axis, ξ the logical coordinate of the box. The map reads no clock, and component
        a reads coordinate a alone, so the static geometry arrays serve it as node positions
        (N17, C1). Each function is called once (SU5); its ends and its increase are checked here
        at the nodes s = k / c that the arrays generate, s the correctly rounded quotient. On a
        charted support the components are the logical coordinates that the chart reads (DN29)."""
        from pops.analytic import coordinate

        components, spelled, choices = [], [], []
        for index, (name, (lower, upper, _)) in enumerate(
                zip(support._names, support.bounds(), strict=True)):
            xi = coordinate(frame, frame.axes[index])
            axis = support._logical_axes[index]
            function = graded.function_of(axis)
            if function is None:
                components.append(xi)
                spelled.append(name)
                continue
            low, high = float(lower), float(upper)
            where = "the node function %s of the axis %s" % (
                getattr(function, "__name__", function), name)
            expression = function(LogicalCoordinate(axis))
            if not (is_number(expression) or isinstance(expression, Expr)):
                raise SupportMismatch(
                    "%s returns %r; it returns an expression of the logical coordinate s that "
                    "PoPS passes" % (where, expression),
                    write="def f(s): an expression of s, for nodes={Ω.%s: f}" % name,
                    site=graded.site)
            self._node_reads(expression, axis, where, graded.site)
            cells = graded.cells[index]
            nodes = self._graded_nodes(expression, low, high, cells, where, graded.site)
            s = xi if low == 0.0 else xi - low
            s = s if high - low == 1.0 else s / (high - low)
            components.append(self._node_map(expression, lambda _, s=s: s))
            spelled.append("%s with s = %s" % (text(expression), _logical_text(name, low, high)))
            choices += [
                Choice("nodes of %s" % name, "%s(s) = %s" % (
                    getattr(function, "__name__", "f"), text(expression)), "stated"),
                Choice("ends of %s" % name, "f(0) = %r, f(1) = %r, set to the bounds %r and %r"
                       % (nodes[0], nodes[-1], low, high), "derived")]
            if not charted:
                choices.append(Choice("geometry of %s" % name, "static arrays of node positions, "
                                      "the widths being the node differences (N17, C1)",
                                      "derived"))
        return tuple(components), spelled, choices

    def _node_reads(self, value: Any, axis: Any, where: str, site: Any) -> None:
        """A node function reads its logical coordinate, numbers, pi and the constants of the
        model (SU5), which enter the identity of the artifact (design note N17, 3.8)."""
        if isinstance(value, ModelConstant):
            self.model._own(value, where)
            return
        if is_number(value):
            return
        if isinstance(value, MathConstant):
            if not _math.isfinite(value.value):
                raise SupportMismatch("%s reads %s" % (where, value.name), site=site)
            return
        if isinstance(value, LogicalCoordinate):
            if value.axis is not axis:
                raise SupportMismatch("%s reads the logical coordinate of the axis %s"
                                      % (where, value.axis.name), site=site)
            return
        if isinstance(value, Arith):
            if value.op == "**" and not is_number(value.args[1]):
                raise NotYetElaborated("%s, a power whose exponent is not a number, in %s"
                                       % (text(value), where), "N11 (expression closure, C1)",
                                       site=site)
            for argument in value.args:
                self._node_reads(argument, axis, where, site)
            return
        if isinstance(value, Call):
            if value.word not in _NODE_WORDS:
                raise NotYetElaborated("%s in %s" % (text(value), where),
                                       "N11 (expression closure, C1)", site=site)
            for argument in value.args:
                self._node_reads(argument, axis, where, site)
            return
        raise SupportMismatch(
            "%s reads %s; a node function returns an expression of the logical coordinate s that "
            "PoPS passes, of numbers, pi and the constants of the model, which enter the identity "
            "of the artifact (design note N17, 3.8)" % (where, text(value)),
            write="def f(s): an expression of s, for nodes={Ω.%s: f}" % axis.name, site=site)

    def _node_value(self, value: Any, s: float) -> float:
        """The binary64 value of a node function at s, operation by operation as the analytic
        program of bind computes it."""
        if is_number(value):
            return float(_number(value))
        if isinstance(value, (MathConstant, ModelConstant)):
            return float(value.value)
        if isinstance(value, LogicalCoordinate):
            return s
        if isinstance(value, Arith):
            args = [self._node_value(argument, s) for argument in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            if value.op == "+":
                return left + right
            if value.op == "-":
                return left - right
            if value.op == "*":
                return left * right
            if value.op == "/":
                return left / right
            return _math.pow(left, right)
        return float(_NODE_WORDS[value.word](*(self._node_value(argument, s)
                                               for argument in value.args)))

    def _graded_nodes(self, expression: Any, low: float, high: float, cells: int, where: str,
                      site: Any) -> list:
        """The nodes f(s_k), s_k from the logical node k of the arrays: low + (high - low) (k /
        cells), the last one high itself; the ends within 4 ulp of the bounds, then set to them;
        strictly increasing (DN3, SupportMismatch naming the first pair that does not increase)."""
        nodes = []
        for node in range(cells + 1):
            xi = high if node == cells else low + (high - low) * (node / cells)
            s = (xi - low) / (high - low)
            try:
                value = self._node_value(expression, s)
            except (ArithmeticError, ValueError):
                value = _math.nan
            if not _math.isfinite(value):
                raise SupportMismatch("%s is not finite at the node %d of %d (s = %r)"
                                      % (where, node, cells, s), site=site)
            nodes.append(value)
        tolerance = _GRADED_END_ULP * _math.ulp(max(abs(low), abs(high)))
        if not (abs(nodes[0] - low) <= tolerance and abs(nodes[-1] - high) <= tolerance):
            raise SupportMismatch(
                "%s sends s = 0 and s = 1 to %r and %r, not to the bounds %r and %r of the axis "
                "within %d ulp: graded nodes move inside the box, and the box stays the support"
                % (where, nodes[0], nodes[-1], low, high, _GRADED_END_ULP),
                write="a function with f(0) = %r and f(1) = %r" % (low, high), site=site)
        ends = list(nodes)
        ends[0], ends[-1] = low, high
        for node in range(cells):
            if not ends[node + 1] > ends[node]:
                raise SupportMismatch(
                    "%s does not increase strictly at the nodes of %d cells: node %d is at %r and "
                    "node %d at %r, the first pair that does not increase"
                    % (where, cells, node, ends[node], node + 1, ends[node + 1]),
                    write="a function of s that increases strictly on [0, 1]", site=site)
        return nodes

    def _node_map(self, value: Any, leaf: Any) -> Any:
        """The analytic expression of a coordinate map of the layout, ``leaf`` giving the analytic
        value of each coordinate that it reads: the logical coordinate s of a node function, a
        logical coordinate of the box of a chart."""
        from pops import analytic as an

        if is_number(value):
            return an.constant(float(_number(value)))
        if isinstance(value, (MathConstant, ModelConstant)):
            return an.constant(float(value.value))
        if isinstance(value, (LogicalCoordinate, Coordinate)):
            return leaf(value)
        if isinstance(value, Arith):
            args = [self._node_map(argument, leaf) for argument in value.args]
            if value.op == "neg":
                return -args[0]
            left, right = args
            if value.op == "**":
                return left ** float(_number(value.args[1]))
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right}[value.op]()
        return getattr(an, value.word)(*(self._node_map(argument, leaf)
                                         for argument in value.args))

    def boundaries(self, support: Support) -> list:
        """The relations of ``model.on`` that close each boundary piece of the support (2.6.2,
        D9): each piece that a flux crosses takes one relation of the quantity, and the relations
        take the boundary policy of the numerics (``MissingBoundaryRealization`` otherwise)."""
        pieces = support.piece_names()
        diffusive = [term for term in self.model.terms if isinstance(term, DiffusiveFlux)]
        if pieces and diffusive:
            raise MissingBoundaryRealization(
                "the diffusive flux %s crosses the boundary pieces %s of %r, which no boundary "
                "relation closes; its face is realized by b6's StencilFlux, which is periodic"
                % (diffusive[0], ", ".join(pieces), support),
                write="pops.Periodic(lower, upper) for an axis whose ends are identified; the "
                      "boundary policies of a diffusive face are elaborated by G04",
                site=support.site)
        relations = list(self.model.boundary_relations)
        if not pieces:
            return relations
        if not self.model.terms:
            if relations:
                raise NotYetElaborated("a boundary relation of a balance with no flux",
                                       _G04, site=relations[0].site)
            return relations
        (balance,) = self.model.balances
        quantity = balance.quantity
        missing = [name for name in pieces
                   if not any(relation.quantity is quantity
                              and any(piece.name == name for piece in relation.pieces)
                              for relation in relations)]
        if missing:
            raise MissingBoundaryRealization(
                "the fluxes of %s cross the boundary pieces %s of %r, which no boundary relation "
                "closes" % (quantity, ", ".join(missing), support),
                write="pops.Periodic(lower, upper) for an axis whose ends are identified, or a "
                      "relation of each piece, model.on(Ω.boundary.%s, %s == g) or "
                      "model.on(..., outflow(%s))" % (missing[0], quantity, quantity),
                site=support.site)
        if self.numerics.boundary is None:
            raise MissingBoundaryRealization(
                "the relation %r has no realization: the numerics state no boundary policy"
                % relations[0], write="pops.Numerics(..., boundary=ghosts())",
                site=self.numerics.site)
        if self.charted:
            # DN30: outflow(u) extends the cell; a value reflects it across a face of the chart.
            for relation in relations:
                if relation.form == "Value":
                    raise NotYetElaborated(
                        "the value relation %s of the charted support %r, whose ghost 2 g - u_k "
                        "is the reflection of a cell across a face of the chart"
                        % (relation, support), CHART_GHOSTS, site=relation.site)
        return relations

    def boundary_data(self, relations: list) -> dict:
        """The data g of each value relation, lowered on the model of b6 before its block is
        made, which registers the parameters that g reads."""
        return {relation: _number(relation.data) if is_number(relation.data)
                else self.lower(relation.data)
                for relation in relations if relation.form == "Value"}

    def ghost_depth(self, plan: Any) -> int:
        """The halo depth D of the block, derived from the plan (G04-a, R1): the largest
        ``ghost_depth`` of the methods of its rates and states, and of the storage that a rate
        asks. The composition of the halos of resolved operations is known only at compilation,
        where the package refuses a block that stores more ghost cells than D."""
        depth = 1
        for _subject, method in (*plan.rates.items(), *plan.states.items()):
            depth = max(depth, int(getattr(method, "ghost_depth", 1)))
            requirements: Any = getattr(method, "runtime_storage_requirements", None)
            if requirements is not None:
                storage: Any = requirements()
                if storage is not None:
                    depth = max(depth, int(storage["ghost_depth"]))
        return depth

    def ghost_reads(self, relation: Any, policy: Any, k: int, value: Any, depth: int) -> None:
        """Refuse a read that a generated ghost of this lot does not take (D3), each by its
        name and its owner."""
        from ._expr import walk
        from ._scheme import SchemeLeaf

        for node in walk(value):
            if isinstance(node, SchemeLeaf) and node.key[0] == "normal":
                raise NotYetElaborated(
                    "face.normal in the ghost that %r gives under %r at depth %d"
                    % (policy, relation, k), _NORMAL_GHOSTS, site=relation.site)
            if isinstance(node, SchemeLeaf) and node.key[0] == "inner" and node.key[1] > depth:
                raise NotYetElaborated(
                    "%s in the ghost that %r gives under %r at depth %d, deeper than the %d "
                    "ghost cell%s of the block" % (text(node), policy, relation, k, depth,
                                                   "" if depth == 1 else "s"),
                    _SYMBOLIC_DEPTH, site=relation.site)
            if isinstance(node, Coordinate):
                raise NotYetElaborated(
                    "the coordinate %s in the ghost that %r gives under %r" % (node, policy,
                                                                               relation),
                    _FACE_CENTRE_DATA, site=relation.site)

    def boundary_conditions(self, support: Support, frame: Any, block: Any, plan: Any,
                            states: dict, relations: list) -> None:
        """Run the boundary policy under each relation at the depths 1..D of the block, and add
        to the plan (T5) the b6 ``SourceGhost`` of each piece, the ghost cells generated from the
        body of the policy (G04-a): the model package compiles them, the prepared boundary fills
        them on the device."""
        if not relations:
            return
        from pops.boundary import TransportBoundarySet
        from pops.boundary.transport import SourceGhost

        policy = self.numerics.boundary
        depth = self.ghost_depth(plan)
        depths = tuple(range(1, depth + 1))
        laws: dict = {}
        for relation in relations:
            quantity = relation.quantity
            state = block[states[quantity]]
            values: dict = {}
            for piece, k, value in policy.ghosts(relation, depths):
                self.ghost_reads(relation, policy, k, value, depth)
                values.setdefault(piece.name, {})[k] = value
            for piece in relation.pieces:
                bodies = values[piece.name]

                def body(inner: Any, k: int, bodies: dict = bodies) -> tuple:
                    inputs = {("inner", j): inner(j)[0] for j in depths}
                    return (self.lower(bodies[k], inputs),)

                axis = frame.axes[support._names.index(piece.axis)]
                boundary_piece = getattr(frame.boundaries, "%s_%s" % (axis.name, piece.side))
                laws[boundary_piece] = SourceGhost(
                    state=state, body=body, depth=depth)
            first = values[relation.pieces[0].name]
            self.rows.append(Row(
                repr(relation), "numerics",
                "SourceGhost(state=block[%s], body=the Ghost of %r, depth=%d) on %s"
                % (quantity.name, policy, depth,
                   ", ".join(piece.name for piece in relation.pieces)),
                [Choice("form", relation.form, "derived"),
                 Choice("ghost", "%s, by %r" % ("; ".join("k=%d: %s" % (k, text(first[k]))
                                                         for k in depths), policy),
                        "derived")]))
        periodic = tuple(axis for axis, (_, _, flag)
                         in zip(frame.axes, support.bounds(), strict=True) if flag)
        plan.boundaries.add(TransportBoundarySet(
            laws, periodic=PeriodicAxes(periodic) if periodic else None))

    def run(self) -> Elaborated:
        model = self.model
        if not model.balances:
            raise NotYetElaborated("a model that evolves no quantity",
                                   "a later lot of N15 with the constraints of N10",
                                   site=model.site)
        if len(model.balances) != 1:
            raise NotYetElaborated(
                "a model of several evolved quantities, whose state is a record (SD42)",
                "a later lot of N15 with the record states of N8", site=model.balances[1].site)
        support = self.support()
        self.charted = isinstance(support, Charted)
        relations = self.boundaries(support)
        frame = self.frame(support)
        self.layout_facts = {"uniform": not self.charted, "cartesian": not self.charted,
                             "periodic": all(support.periodic(axis) for axis in support._names)}
        b6 = _Model(model.name, frame=frame)
        self.rows.insert(0, Row("pops.Model(%r)" % model.name, "model",
                                "model = pops.Model(%r, frame=frame)" % model.name))
        states = self.declarations(b6)
        terms = self.terms(b6, frame, states)
        rates = {balance: self.rate(b6, balance, terms, states) for balance in model.balances}
        plan, selections = self.plan(rates, terms, states)
        self.boundary_data(relations)
        self.install_flux_coordinates(b6, frame)
        case6 = _Case(model.name)
        block = case6.block(model.name, b6)
        self.boundary_conditions(support, frame, block, plan, states, relations)
        case6.numerics(plan, block=block)
        self.initial(case6, block, states, frame)
        advance = self.advance(case6, block, states, selections)
        layout = self.layout(support, frame)
        self.providers(b6)
        return Elaborated(case=case6, layout=layout, block=block, model=b6, plan=plan,
                          states=states, names=self.names, advance=advance,
                          rows=tuple(self.rows), params=dict(self.params),
                          outputs=self.lowered_outputs,
                          representations={quantity: elaborated_representation(
                              quantity, support, self.numerics.definition_of(quantity))
                              for quantity in states})


def inspect_case(case: Any) -> Report:
    """``pops.inspect(case)``: ``resolve(validate(case))`` of the elaborated case, and its rows."""
    from pops._api import resolve, validate

    elaborated = case.elaborated
    resolved = resolve(validate(elaborated.case), layout=elaborated.layout)
    return Report("case %r" % case.model.name, "1 (model, numerics, program)", elaborated.rows,
                  resolved=resolved)


def inspect_value(value: Any) -> Report:
    """The report of a value of the surface: its mathematics and its level (SU12)."""
    from ._method import MethodValue, SurfaceMethod
    from ._program import Advance, Program

    if isinstance(value, Model):
        rows = [Row("pops.Model(%r)" % value.name, "model",
                    "quantities %s; terms %s; balances %s" % (
                        ", ".join(map(str, value.quantities)) or "none",
                        ", ".join(map(str, value.terms)) or "none",
                        "; ".join(map(str, value.balances)) or "none"))]
        return Report("model %r" % value.name, "model", rows)
    if isinstance(value, Balance):
        return Report(text(value.relation), "model",
                      [Row(text(value.relation), "model", "the balance of %s" % value.quantity)])
    if isinstance(value, Term):
        kind = "diffusive flux" if isinstance(value, DiffusiveFlux) else "flux"
        choices = []
        if isinstance(value, DiffusiveFlux):
            choices = [Choice("differentiated variables", ", ".join(map(str, value.variables)),
                              value.origin), Choice("k", text(value.coefficient), "derived")]
        return Report(str(value), "model", [Row("%s of %s: %s" % (kind, value.quantity,
                                                                    text(value.value)),
                                                "model", "a term of %r" % value.model, choices)])
    if isinstance(value, (Quantity, ModelConstant, Parameter, Datum)):
        return Report(str(value), "model", [Row(str(value), "model", type(value).__name__)])
    if isinstance(value, Support):
        return Report(repr(value), "model", [Row(repr(value), "model", "a %ssupport of rank %d"
                                                 % ("charted " if isinstance(value, Charted)
                                                    else "", value.rank))])
    if isinstance(value, (Numerics, SchemeValue)):
        return Report(repr(value), "numerics", [Row(repr(value), "numerics",
                                                    value.describe() if isinstance(
                                                        value, SchemeValue) else "a numerics")])
    if isinstance(value, (MethodValue, SurfaceMethod, Advance, Program)):
        return Report(repr(value), "program", [Row(repr(value), "program", type(value).__name__)])
    raise TypeError("pops.inspect takes a case or a value of the surface, got %r" % (value,))
