"""The program, the case and their inspection (spec 0.3, 2.6.2, 2.6.6; §5.2, Q3, 14.1).

``pops.Advance(method, dt=1e-3, attempts=2)`` is one attempt loop of a method: a number ``dt`` is
a fixed proposal, ``attempts=`` the number of attempts of a step (T38), and an omitted ``accept=``
is ``finite()``. ``pops.Program(advance=A, initial={u: u0})`` gives the initial value of each
evolved quantity, a datum being supplied at bind under the representation of its arrays (Q3).
``pops.Case(model=, numerics=, program=, layout={Ω: pops.Uniform(cells=256)})`` holds them, and is
traced when it is built: its model, numerics and program are elaborated into a b6 case, and its
methods are traced once (SU5). ``pops.inspect(case)`` runs ``validate`` and ``resolve`` and
reports without compiling (SU12).

``tolerance=pops.Tolerance(rtol=, atol=)`` is the request of the solves of the method (SV3), and
``pops.Program(solvers={imex: cg()})`` their solvers, keyed by the method or ``method.name``.

The step rule that is a function of ``cell``, the acceptances and the claims have no elaborated
form yet; each refuses with ``NotYetElaborated`` and the lot that elaborates it. The outputs,
compile, bind and run are lot G (``_run``).
"""
from __future__ import annotations

import math as _math
from typing import Any

from ._expr import is_number
from ._geometry import Box
from ._method import MethodValue, SurfaceMethod
from ._model import Datum, Model, Quantity
from ._numerics import Numerics, SchemeValue
from ._quadrature import Quadrature
from ._refusal import (
    AmbiguousElaboration, LevelMismatch, NotYetElaborated, SpellingRefusal, surface_entry,
    user_site,
)

_N8_ACCEPT = "N8 phase 1 (the assessments of T28)"


class Tolerance:
    """``pops.Tolerance(rtol=None, atol=None)``: the request of a solve (SV3, 2.6.2).

    Given to ``pops.Advance(tolerance=...)`` it is the request of every solve of the method, the
    root request of the step; given to ``pops.solve(..., tolerance=...)`` it is that solve's.
    """

    __slots__ = ("rtol", "atol")

    def __init__(self, *, rtol: Any = None, atol: Any = None, observe: Any = None,
                 norm: Any = None) -> None:
        if observe is not None or norm is not None:
            raise NotYetElaborated("pops.Tolerance(observe=..., norm=...), what a tolerance "
                                   "observes and its norm (SV3, D16, D17)",
                                   "a later lot of N15 with the requests of N9")
        for name, value in (("rtol", rtol), ("atol", atol)):
            if value is not None and not _positive_number(value):
                raise TypeError("pops.Tolerance(%s=...) takes a positive number, got %r"
                                % (name, value))
        if rtol is None and atol is None:
            raise TypeError("pops.Tolerance states rtol=, atol= or both (SV3)")
        self.rtol = rtol
        self.atol = atol

    def request(self) -> Any:
        """The request of N8 that this tolerance states, ``Request(rtol=, atol=)``."""
        from pops.time._program.relations import Request

        return Request(rtol=self.rtol, atol=self.atol)

    def __repr__(self) -> str:
        return "pops.Tolerance(%s)" % ", ".join(
            "%s=%r" % (name, value) for name, value in (("rtol", self.rtol), ("atol", self.atol))
            if value is not None)


def _solvers(solvers: Any, advance: Any) -> tuple:
    """``solvers={key: solver}``: each key a method value or a solve path ``method.name`` of the
    advance, each solver a value of ``pops.lib.solve`` (2.6.2, SU8)."""
    from ._solver import SolverValue

    from ._method import SolvePath

    if solvers is None:
        return ()
    if not isinstance(solvers, dict):
        raise TypeError("pops.Program(solvers={key: solver}) takes a dictionary keyed by "
                        "handles, got %r" % (solvers,))
    rows = []
    for key, solver in solvers.items():
        if isinstance(key, str):
            raise TypeError("pops.Program(solvers=...) is keyed by handles, not by the string %r "
                            "(SU8)" % key)
        if not isinstance(key, (MethodValue, SolvePath)):
            raise NotYetElaborated("pops.Program(solvers={%r: ...}), a key other than a method or "
                                   "a solve of a method" % (key,),
                                   "a later lot of N15 with the field, relation and term solves "
                                   "of N9")
        if not isinstance(solver, SolverValue):
            raise TypeError("pops.Program(solvers={%r: ...}) takes a solver of pops.lib.solve "
                            "or of @pops.solver, such as cg(), got %r" % (key, solver))
        rows.append((key, solver))
    return tuple(rows)


class Advance:
    """``pops.Advance(method, dt=rule, attempts=n)``: one attempt loop of a method (5.2)."""

    def __init__(self, method: Any = None, *, dt: Any = None, accept: Any = None,
                 attempts: Any = None, tolerance: Any = None, first_dt: Any = None,
                 choices: Any = None, **earlier: Any) -> None:
        self.site = user_site(2)
        if "retry" in earlier:
            raise SpellingRefusal("pops.Advance(method, attempts=n)").refuse(
                "pops.Advance(retry=Retry(max_attempts=n))", "pops.Advance")
        if earlier:
            raise TypeError("pops.Advance has no option %s" % ", ".join(sorted(earlier)))
        if isinstance(method, SurfaceMethod):
            raise TypeError("pops.Advance(%s): a method is called with its options, %s(rate=...) "
                            "(SU5)" % (method.name, method.name))
        if isinstance(method, (SchemeValue, Numerics)):
            raise LevelMismatch("pops.Advance takes a time method, and %r is a choice of the "
                                "numerics" % (method,), write="pops.Advance(ssprk2(rate=model), "
                                                               "dt=...)")
        if not isinstance(method, MethodValue):
            raise TypeError("pops.Advance(method, ...) takes a method called with its options, "
                            "such as ssprk2(rate=model), got %r" % (method,))
        if dt is None:
            raise AmbiguousElaboration(
                "the advance of %s has no step rule; the step size is a choice that PoPS never "
                "makes by a heuristic" % method.name,
                write="pops.Advance(%r, dt=...) with a number or a function of cell" % (method,))
        if not (callable(dt) and not is_number(dt)) and not _positive_number(dt):
            raise TypeError("pops.Advance(dt=...) takes a positive number, a function of cell or "
                            "a controller, got %r" % (dt,))
        if accept is not None:
            raise NotYetElaborated("pops.Advance(accept=...), checks other than the default "
                                   "finite()", _N8_ACCEPT)
        if tolerance is not None and not isinstance(tolerance, Tolerance):
            raise TypeError("pops.Advance(tolerance=...) takes a pops.Tolerance(rtol=..., "
                            "atol=...), got %r" % (tolerance,))
        if first_dt is not None or choices is not None:
            raise NotYetElaborated("pops.Advance(first_dt=..., choices=...), a controller",
                                   "N8 phase 2 (controllers)")
        if attempts is not None and (isinstance(attempts, bool) or not isinstance(attempts, int)
                                     or attempts < 1):
            raise TypeError("pops.Advance(attempts=n) takes an integer n >= 1, got %r"
                            % (attempts,))
        self.method = method
        self.dt = dt
        self.attempts = attempts
        self.tolerance = tolerance

    def __repr__(self) -> str:
        text = "pops.Advance(%r, dt=%r" % (self.method, self.dt)
        if self.attempts is not None:
            text += ", attempts=%d" % self.attempts
        if self.tolerance is not None:
            text += ", tolerance=%r" % (self.tolerance,)
        return text + ")"


def _initial_value(quantity: Quantity, value: Any, site: Any) -> Any:
    """The initial value of ``quantity`` (CN12, Q3, lot D-E): a datum, ``pops.project(e, ...)``,
    or a value that reads no coordinate, whose cell average is exact."""
    from ._expr import Arith, Call, Expr, is_number, text, walk
    from ._geometry import Coordinate, MathConstant
    from ._model import ModelConstant, Parameter
    from ._project import Projection
    from ._refusal import AmbiguousElaboration, SupportMismatch

    if isinstance(value, Datum):
        if value.space.support is not quantity.space.support:
            raise SupportMismatch("the datum %s lives on another support than %s"
                                  % (value, quantity), write="a datum of the support of %s"
                                                             % quantity)
        return value
    expression = value.expression if isinstance(value, Projection) else value
    if not (is_number(expression) or isinstance(expression, Expr)):
        raise TypeError("pops.Program(initial={%s: ...}) takes a datum, pops.project(e, ...) or "
                        "a value, got %r" % (quantity, value))
    for node in walk(expression):
        if isinstance(node, Coordinate):
            if node.support is not quantity.space.support:
                raise SupportMismatch(
                    "the initial value of %s reads the coordinate %s of another support"
                    % (quantity, node.name), write="the coordinates of the support of %s"
                                                   % quantity)
            if not isinstance(value, Projection):
                raise AmbiguousElaboration(
                    "the initial value %s of %s reads a coordinate, and its cell average is a "
                    "choice: the rule or the exact integral of the projection" % (text(value),
                                                                                  quantity),
                    write="initial={%s: pops.project(e, quadrature=GaussLegendre(n))}, or "
                          "integral=I" % quantity, site=site)
        elif isinstance(node, Datum):
            raise NotYetElaborated("an initial value that reads the datum %s" % node,
                                   "a later lot of N15 with the data fields of Q3")
        elif isinstance(node, Expr) and not isinstance(
                node, (ModelConstant, Parameter, MathConstant, Arith, Call)):
            raise NotYetElaborated("an initial value that reads %s" % text(node),
                                   "a later lot of N15 with the definitions of N10")
    return value


class Program:
    """``pops.Program(advance=A, initial={u: u0})`` (5.2, CN12, Q3)."""

    def __init__(self, *, advance: Any = None, initial: Any = None, solvers: Any = None,
                 regrid: Any = None, equivalence: Any = None, **earlier: Any) -> None:
        self.site = user_site(2)
        for name in ("initialize", "realization"):
            if name in earlier:
                raise SpellingRefusal("pops.Program(advance=A, initial={u: u0}, "
                                      "solvers={key: solver})").refuse(
                    "pops.Program(%s=...)" % name, "pops.Program")
        if earlier:
            raise TypeError("pops.Program has no option %s" % ", ".join(sorted(earlier)))
        if not isinstance(advance, Advance):
            raise TypeError("pops.Program(advance=...) takes a pops.Advance, got %r" % (advance,))
        self.solvers = _solvers(solvers, advance)
        if regrid is not None:
            raise NotYetElaborated("pops.Program(regrid=...)", "a later lot of N15 with N12")
        if equivalence is not None:
            raise NotYetElaborated("pops.Program(equivalence=...)", "a later lot of N15 with G07")
        if not isinstance(initial, dict):
            raise TypeError("pops.Program(initial={u: value}) takes a dictionary keyed by "
                            "quantities, got %r" % (initial,))
        rows = []
        for quantity, value in initial.items():
            if isinstance(quantity, str):
                raise TypeError("pops.Program(initial=...) is keyed by handles, not by the string "
                                "%r (SU8)" % quantity)
            if not isinstance(quantity, Quantity):
                raise TypeError("pops.Program(initial=...) is keyed by quantities, got %r"
                                % (quantity,))
            rows.append((quantity, _initial_value(quantity, value, self.site)))
        self.advance = advance
        self.initial = tuple(rows)

    def initial_of(self, quantity: Quantity) -> Any:
        for key, value in self.initial:
            if key is quantity:
                return value
        return None


class Uniform:
    """``pops.Uniform(cells=c)``: a uniform layout of a support, one size per axis."""

    __slots__ = ("cells", "site")

    def __init__(self, *, cells: Any) -> None:
        cells = (cells,) if isinstance(cells, int) and not isinstance(cells, bool) else cells
        if isinstance(cells, list) or not isinstance(cells, tuple) or not cells or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in cells):
            raise TypeError("pops.Uniform(cells=c) takes a number of cells, or one per axis "
                            "(64, 64), got %r" % (cells,))
        self.cells = tuple(cells)
        self.site = user_site(2)

    def __repr__(self) -> str:
        cells = self.cells[0] if len(self.cells) == 1 else self.cells
        return "pops.Uniform(cells=%r)" % (cells,)


def _later_layout(name: str, owner: str) -> Any:
    def layout(*args: Any, **kwargs: Any) -> Any:
        raise NotYetElaborated("pops.%s, a layout" % name, owner).with_traceback(None)

    layout.__name__ = layout.__qualname__ = name
    return layout


AMR = _later_layout("AMR", "a later lot of N15 with the hierarchies of N12 and T08")


def _positive_number(value: Any) -> bool:
    """Whether ``value`` is a finite positive Python number."""
    if not is_number(value):
        return False
    number = float(value)
    return _math.isfinite(number) and number > 0


class Case:
    """``pops.Case(model=, numerics=, program=, layout={Ω: pops.Uniform(cells=c)})``.

    Building the case traces it: its model, numerics and program are elaborated (``_elaborate``),
    and ``elaborated`` holds the b6 case, its layout and the report of ``pops.inspect``.
    """

    @surface_entry
    def __init__(self, *, model: Any = None, numerics: Any = None, program: Any = None,
                 layout: Any = None, outputs: Any = None, **earlier: Any) -> None:
        self.site = user_site(2)
        for name in ("data",):
            if name in earlier:
                raise SpellingRefusal("model.data(\"d\", space) and pops.bind(artifact, "
                                      "data={d: array})").refuse("pops.Case(data=...)",
                                                                 "pops.Case")
        if earlier:
            raise TypeError("pops.Case has no option %s" % ", ".join(sorted(earlier)))
        if not isinstance(model, Model):
            raise TypeError("pops.Case(model=...) takes a pops.Model, got %r" % (model,))
        if not isinstance(numerics, Numerics):
            raise TypeError("pops.Case(numerics=...) takes a pops.Numerics, got %r" % (numerics,))
        if not isinstance(program, Program):
            raise TypeError("pops.Case(program=...) takes a pops.Program, got %r" % (program,))
        from ._run import Output

        if outputs is None:
            outputs = []
        if isinstance(outputs, Output):
            raise TypeError("pops.Case(outputs=...) takes a list of outputs, outputs=[output] "
                            "(SU14)")
        if not isinstance(outputs, list) or not all(isinstance(item, Output) for item in outputs):
            raise TypeError("pops.Case(outputs=...) takes a list of pops.Output, got %r"
                            % (outputs,))
        if len({id(item) for item in outputs}) != len(outputs):
            raise ValueError("pops.Case(outputs=[...]) names each output once")
        if not isinstance(layout, dict) or not layout:
            raise TypeError("pops.Case(layout=...) takes a dictionary keyed by supports, "
                            "layout={Ω: pops.Uniform(cells=c)}, got %r" % (layout,))
        rows = []
        for support, value in layout.items():
            if not isinstance(support, Box):
                raise TypeError("pops.Case(layout=...) is keyed by supports, got %r" % (support,))
            if isinstance(value, Quadrature):
                raise NotYetElaborated("pops.Case(layout={%r: pops.Quadrature(...)}), a quadrature "
                                       "rule as the layout of a support (D0)" % (support,),
                                       "a later lot of N15 with the supports of G03")
            if not isinstance(value, Uniform):
                raise TypeError("pops.Case(layout={%r: ...}) takes pops.Uniform(cells=c), got %r"
                                % (support, value))
            if len(value.cells) != support.rank:
                raise ValueError("pops.Uniform(cells=%r) gives %d sizes to a support of rank %d"
                                 % (value.cells, len(value.cells), support.rank))
            rows.append((support, value))
        self.model = model
        self.numerics = numerics
        self.program = program
        self.layout = tuple(rows)
        self.outputs = tuple(outputs)
        from ._elaborate import elaborate_case

        self.elaborated = elaborate_case(self)

    def layout_of(self, support: Box) -> Any:
        for key, value in self.layout:
            if key is support:
                return value
        return None


@surface_entry
def inspect(value: Any) -> Any:
    """``pops.inspect(case)``: ``resolve(validate(case))`` and its report, without compiling.

    For another value of the surface, the report of that value: its level and its mathematics.
    """
    from ._elaborate import inspect_case, inspect_value

    if isinstance(value, Case):
        return inspect_case(value)
    return inspect_value(value)


from ._run import CUDA, OpenMP, Resources, bind, compile  # noqa: E402,F401  (lot G)


def claim(*args: Any, **kwargs: Any) -> Any:
    """``pops.claim(subject, proposition, assumptions=[...], evidence=...)`` (K1)."""
    raise NotYetElaborated("pops.claim, a claim of §13 (K1)",
                           "a later lot of N15 with the claims of §13").with_traceback(None)
