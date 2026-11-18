"""A moving configuration on the explicit route, below the surface (G03 lot M1a, ROM-21).

Design note G03 §7.2: a moving configuration adds two members to the candidate, X, the corner
copies of each cell (x_left and x_right in rank 1), and V, the cell volume. They are the state of a
block of their own, so the step's own method advances them like any state, a rejected attempt
restores them, and the codec saves them; no new mechanism carries them. Their rates, at each stage,
are a pointwise source:

    x_left' = w(x_left, t),   x_right' = w(x_right, t),   V' = w(x_right, t) - w(x_left, t),

the last being σ, the swept rate of rank 1. The stage time reaches the source through an analytic
auxiliary that reads the Program's clock, ``AnalyticAux(..., time(clock))``, which the native
provider evaluates at the physical time of each stage. The route needs no native change.

The node rate w takes two forms (§7.4):

1. an analytic expression (``pops.analytic``) of the frame's coordinate and of ``time(clock)``,
   evaluated at each copy's own position;
2. a coordinate map of b6 (``GeometryEvolution``): w = ∂ₜΦ at the reference coordinate of the
   node, derived symbolically here and evaluated by two analytic auxiliaries at the lower and upper
   face coordinates of the base grid. The initial positions come from the base. The method
   integrates X(ξ, t) = X_base(ξ) + Φ(ξ, t) − Φ(ξ, 0) to its own order.

Why the copies of one node stay equal: both copies of a node start from one base node and receive
the same rate, computed by the same expression from the same inputs, so they stay bitwise equal.

A node rate whose normal form is zero normalizes the configuration to its base before any member is
created (K13). A node rate that reads a field is refused (R26), as is a parameter, until a later lot
gives the quantities of a ``Finite`` support. Rank 2 is lot M2.
"""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from pops.analytic import CellBounds, PredicateExpr, ScalarExpr
from pops.analytic import cos as _cos, exp as _exp, log as _log, sin as _sin, sqrt as _sqrt
from pops.analytic import constant as _constant, where as _where
from pops.analytic._model import _time_clocks
from .moving import GeometryEvolution

MEMBER_COMPONENTS = ("x_left", "x_right", "volume")
MOTION_SOURCE = "motion"
_R26 = ("a free-boundary or Lagrangian method, whose law the case must supply "
        "(R26; a later lot of G03)")


def _refuse(what: str, lot: str) -> None:
    from pops.surface._refusal import NotYetElaborated
    raise NotYetElaborated(what, lot)


# --- the analytic node rate ----------------------------------------------------------------------

def _nonnegative(expr: ScalarExpr) -> bool:
    if expr._op == "constant":
        return cast(float, expr._literal) >= 0
    args = cast(tuple[ScalarExpr, ...], expr._arguments)
    if expr._op == "mul":
        left, right = args
        return left.same_as(right) and _defined(left)
    if expr._op == "pow":
        base, exponent = args
        return exponent._literal == 2 and _defined(base)
    if expr._op == "add":
        return all(_nonnegative(item) for item in args)
    return False


def _positive(expr: ScalarExpr) -> bool:
    if expr._op == "constant":
        return cast(float, expr._literal) > 0
    if expr._op == "add":
        left, right = cast(tuple[ScalarExpr, ScalarExpr], expr._arguments)
        return ((_positive(left) and _nonnegative(right))
                or (_nonnegative(left) and _positive(right)))
    return False


def _defined(expr: ScalarExpr | PredicateExpr) -> bool:
    """A conservative proof that a subtree is defined everywhere over real inputs."""
    op = expr._op
    args = cast(tuple[ScalarExpr, ...], expr._arguments)
    if op == "div":
        denominator = args[1]
        return _defined(args[0]) and _known_nonzero(denominator)
    if op == "log":
        return _positive(args[0])
    if op == "sqrt":
        return _nonnegative(args[0])
    if op == "pow":
        base, exponent = args
        n = cast(float, exponent._literal)
        return _positive(base) or (n >= 0 and n.is_integer() and _defined(base))
    return op in {"constant", "coordinate", "time", "abs", "neg", "add", "sub",
                  "mul", "sin", "cos", "exp", "minimum", "maximum", "hypot", "atan2",
                  "erf", "erfc", "where", "eq", "ne", "le", "ge", "lt", "gt",
                  "between", "not", "and", "or"} and all(
        _defined(item) for item in expr._arguments)


def _known_nonzero(expr: ScalarExpr) -> bool:
    return (expr._op == "constant" and expr._literal != 0.0) or _positive(expr)


def _is_zero(expr: Any) -> bool:
    """Whether ``expr`` is an exact zero of the analytic algebra (the normal form of K13)."""
    if expr is None:
        return True
    op, arguments = expr._op, expr._arguments
    if op == "constant":
        return expr._literal == 0.0
    if op == "mul":
        return all(_defined(item) for item in arguments) and any(_is_zero(item) for item in arguments)
    if op == "sub" and arguments[0].same_as(arguments[1]):
        return _defined(arguments[0])
    if op in ("add", "sub"):
        return all(_is_zero(item) for item in arguments)
    if op == "neg":
        return _is_zero(arguments[0])
    if op == "div":
        return _known_nonzero(arguments[1]) and _is_zero(arguments[0])
    return False


def _add(left: Any, right: Any) -> Any:
    if _is_zero(left):
        return None if _is_zero(right) else right
    return left if _is_zero(right) else left + right


def _sub(left: Any, right: Any) -> Any:
    if _is_zero(right):
        return None if _is_zero(left) else left
    return -right if _is_zero(left) else left - right


def _scale(factor: Any, derivative: Any) -> Any:
    if _is_zero(derivative):
        return None
    if derivative._op == "constant" and derivative._literal == 1.0:
        return factor
    return factor*derivative


def time_derivative(expr: Any) -> Any:
    """∂ₜ of an analytic expression, symbolically, or ``None`` when it is exactly zero.

    The elementary operations of the analytic algebra are differentiated; a subtree that reads no
    clock has a zero derivative whatever its operation. Any other operation that reads the clock is
    refused, naming the node rate of form 1 as what to write instead.
    """
    if not expr.time_clocks():
        return None
    op, arguments = expr._op, expr._arguments
    if op == "time":
        return _constant(1.0)
    if op in ("add", "sub"):
        left, right = (time_derivative(item) for item in arguments)
        return _add(left, right) if op == "add" else _sub(left, right)
    if op == "neg":
        derivative = time_derivative(arguments[0])
        return None if derivative is None else -derivative
    if op == "mul":
        a, b = arguments
        return _add(_scale(b, time_derivative(a)), _scale(a, time_derivative(b)))
    if op == "div":
        if _is_zero(expr):
            return None
        a, b = arguments
        da, db = time_derivative(a), time_derivative(b)
        numerator = da if db is None else _sub(_scale(b, da), _scale(a, db))
        if numerator is None:
            return None if _defined(expr) else _constant(0.0)/b
        return numerator/(b if db is None else b*b)
    if op == "pow":
        a, b = arguments
        exponent = b._literal
        if exponent == 0.0:
            return None
        derivative = time_derivative(a)
        if exponent == 1.0:
            return derivative
        return _scale(exponent*a**(exponent - 1.0), derivative)
    if op == "where":
        predicate, left, right = arguments
        if _time_clocks(predicate):
            _refuse("the time derivative of a coordinate map whose selection reads the clock",
                    "a later lot of G03; state the node rate, node_rate=w(x, t)")
        left_rate, right_rate = time_derivative(left), time_derivative(right)
        if left_rate is None and right_rate is None:
            return None
        zero = _constant(0.0)
        return _where(predicate, zero if left_rate is None else left_rate,
                      zero if right_rate is None else right_rate)
    derivative = time_derivative(arguments[0]) if len(arguments) == 1 else None
    if op == "sin":
        return _scale(_cos(arguments[0]), derivative)
    if op == "cos":
        return _scale(-_sin(arguments[0]), derivative)
    if op == "exp":
        return _scale(_exp(arguments[0]), derivative)
    if op == "log":
        return None if derivative is None else derivative/arguments[0]
    if op == "sqrt":
        return None if derivative is None else derivative/(2*_sqrt(arguments[0]))
    _refuse("the time derivative of the analytic operation %r of a coordinate map" % op,
            "a later lot of G03; state the node rate, node_rate=w(x, t)")
    return None  # unreachable


def _rebuild(expr: Any, leaf: Any) -> Any:
    """The same tree with each coordinate, time and input leaf replaced by ``leaf(node)``."""
    if expr._op in ("coordinate", "time", "input"):
        return leaf(expr)
    if not expr._arguments:
        return expr
    return type(expr)._new(expr._op, tuple(_rebuild(item, leaf) for item in expr._arguments))


_NUMPY = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
          "abs": np.abs, "neg": np.negative, "add": np.add, "sub": np.subtract,
          "mul": np.multiply, "div": np.divide, "pow": np.power, "minimum": np.fmin,
          "maximum": np.fmax, "atan2": np.arctan2, "hypot": np.hypot,
          "erf": np.vectorize(math.erf, otypes=[float]),
          "erfc": np.vectorize(math.erfc, otypes=[float]),
          "eq": np.equal, "ne": np.not_equal, "lt": np.less,
          "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
          "not": np.logical_not,
          "between": lambda value, lower, upper: np.logical_and(lower <= value, value <= upper)}


def _evaluate(expr: ScalarExpr | PredicateExpr, coordinates: Any, t: float) -> Any:
    def choose(condition: Any, yes: Any, no: Any, axes: Any, *, dtype: Any) -> Any:
        if np.ndim(condition) == 0:
            return (yes if condition else no)(axes)
        mask = np.broadcast_to(condition, axes[0].shape)
        result = np.empty(mask.shape, dtype=dtype)
        for selected, branch in ((mask, yes), (~mask, no)):
            if selected.any():
                result[selected] = branch(tuple(axis[selected] for axis in axes))
        return result

    def value(node: Any, axes: Any) -> Any:
        op = node._op
        if op == "constant":
            return node._literal
        if op == "coordinate":
            return axes[node._coordinate.axis.index]
        if op == "time":
            return float(t)
        args = node._arguments
        if op == "where":
            return choose(value(args[0], axes), lambda scoped: value(args[1], scoped),
                          lambda scoped: value(args[2], scoped), axes, dtype=float)
        if op == "and":
            return choose(value(args[0], axes), lambda scoped: value(args[1], scoped),
                          lambda _: False, axes, dtype=bool)
        if op == "or":
            return choose(value(args[0], axes), lambda _: True,
                          lambda scoped: value(args[1], scoped), axes, dtype=bool)
        if op not in _NUMPY:
            raise NotImplementedError("no NumPy reading of the analytic operation %r" % op)
        return _NUMPY[op](*(value(item, axes) for item in args))
    return value(expr, np.broadcast_arrays(*coordinates))


def evaluate(expr: ScalarExpr, x: Any, t: float = 0.0) -> np.ndarray:
    """A NumPy reading of a node rate or a map at coordinates ``x`` and time ``t`` (oracles only)."""
    return np.broadcast_to(np.asarray(_evaluate(expr, (np.asarray(x, dtype=float),), t),
                                      dtype=float), np.shape(x)).copy()


def _check_leaves(expr: ScalarExpr, frame: Any, what: str) -> None:
    if expr.input_references():
        _refuse("a %s that reads the discrete input(s) %s, a field on the moving support"
                % (what, ", ".join(component for _, component in expr.input_references())), _R26)
    if expr.has_parameters:
        _refuse("a %s that reads a parameter" % what,
                "a later lot of G03, with the quantities of a Finite support (§7.4)")
    if expr.frame_id not in (None, frame.canonical_id):
        raise ValueError("the %s reads the coordinates of another frame" % what)
    if len(expr.time_clocks()) > 1:
        raise ValueError("the %s reads more than one clock" % what)


def _finite(expr: ScalarExpr) -> PredicateExpr:
    # IEEE self-subtraction is exactly zero for every finite value and NaN for a nonfinite
    # operand. This uses the prepared Real precision, including builds with binary32 Real.
    return expr - expr == 0


def _domain_predicate(law: ScalarExpr, *, reads_clock: bool) -> PredicateExpr | None:
    """A typed check of the reached partial expressions and comparison operands of a map."""
    zero, one = _constant(0.0), _constant(1.0)

    def combine(checks: Any) -> PredicateExpr | None:
        result = None
        for check in checks:
            if check is not None:
                result = check if result is None else result & check
        return result

    def relevant(node: ScalarExpr | PredicateExpr) -> bool:
        return bool(_time_clocks(node)) == reads_clock

    def condition_value(condition: PredicateExpr | None) -> ScalarExpr:
        return one if condition is None else _where(condition, one, zero)

    def selected_checks(condition: PredicateExpr, left: Any, right: Any) -> PredicateExpr | None:
        checked_condition = visit(condition)
        if not reads_clock and _time_clocks(condition):
            # A static pass cannot know which path this condition will select.
            return checked_condition
        left_check = None if left is None else visit(left)
        right_check = None if right is None else visit(right)
        if left_check is None and right_check is None:
            return checked_condition
        selected = _where(condition, condition_value(left_check), condition_value(right_check))
        return combine((checked_condition, selected > 0))

    def visit(node: ScalarExpr | PredicateExpr) -> PredicateExpr | None:
        op, args = node._op, node._arguments
        if op == "where":
            condition, left, right = cast(tuple[PredicateExpr, ScalarExpr, ScalarExpr], args)
            return selected_checks(condition, left, right)
        if op in {"and", "or"}:
            condition, tail = cast(tuple[PredicateExpr, PredicateExpr], args)
            return selected_checks(condition, tail if op == "and" else None,
                                   tail if op == "or" else None)
        checks = [visit(item) for item in args]
        if op == "div" and _is_zero(args[1]):
            checks.append(zero > one)
        elif isinstance(node, ScalarExpr) and relevant(node) \
                and op in {"div", "sqrt", "log", "pow"} and not _defined(node):
            operands = cast(tuple[ScalarExpr, ...], args)
            checks.extend(_finite(operand) for operand in operands)
            if op == "div":
                checks.append(operands[1] != 0)
            elif op == "sqrt":
                checks.append(operands[0] >= 0)
            elif op == "log":
                checks.append(operands[0] > 0)
            elif op == "pow":
                base, exponent = operands
                n = cast(float, exponent._literal)
                if n < 0:
                    checks.append(base != 0 if n.is_integer() else base > 0)
                elif not n.is_integer():
                    checks.append(base >= 0)
            checks.append(_finite(node))
        elif op in {"eq", "ne", "lt", "le", "gt", "ge", "between"}:
            checks.extend(_finite(operand) for operand in cast(tuple[ScalarExpr, ...], args)
                          if operand._arguments and relevant(operand))
        return combine(checks)

    return visit(law)


def _selects_branch(expr: ScalarExpr | PredicateExpr) -> bool:
    return expr._op == "where" or any(_selects_branch(item) for item in expr._arguments)


def _validate_partial_map(law: ScalarExpr, frame: Any, cells: tuple[int, ...]) -> None:
    """Check the reached static domain restrictions at the exact reference nodes."""
    predicate = _domain_predicate(law, reads_clock=False)
    if predicate is None:
        return

    def coordinate(axis: int) -> Any:
        shape = [1]*len(cells)
        shape[axis] = cells[axis] + 1
        lower, upper = float(frame.lower[axis]), float(frame.upper[axis])
        spacing = (upper - lower)*(1.0/float(cells[axis]))
        return lower + np.arange(cells[axis] + 1, dtype=float).reshape(shape)*spacing

    with np.errstate(all="ignore"):
        nodes = tuple(coordinate(axis) for axis in range(len(cells)))
        if not np.asarray(_evaluate(predicate, nodes, 0.0), dtype=bool).all():
            raise ValueError("coordinate map is undefined at a reference node")


def _at_rest_value(node_rate: Any) -> bool:
    """A node rate whose normal form is zero on a base of any rank: 0, an analytic zero, or a map
    that reads no clock (K13)."""
    if isinstance(node_rate, (int, float)) and not isinstance(node_rate, bool):
        return float(node_rate) == 0.0
    if type(node_rate) is ScalarExpr:
        return _is_zero(node_rate)
    if type(node_rate) is GeometryEvolution:
        return not any(item.time_clocks() for item in node_rate.coordinate_map)
    return False


def _rate_form(
    node_rate: Any, frame: Any,
) -> tuple[str, ScalarExpr | None, PredicateExpr | None]:
    """The rate form, its expression and the remaining stage-domain obligation."""
    from pops._ir.expr import Expr
    if isinstance(node_rate, (int, float)) and not isinstance(node_rate, bool):
        node_rate = _constant(float(node_rate))
    if type(node_rate) is GeometryEvolution:
        if len(node_rate.coordinate_map) != 1:
            _refuse("a coordinate map of rank %d" % len(node_rate.coordinate_map), "G03 lot M2")
        (law,) = node_rate.coordinate_map
        derivative = time_derivative(law)
        prior_domain = _domain_predicate(law, reads_clock=True)
        zero = _is_zero(derivative)
        if zero and prior_domain is None:
            return "map", None, None
        rate = _constant(0.0) if zero else derivative
        rate_domain = _domain_predicate(rate, reads_clock=True)
        domain = (prior_domain & rate_domain if prior_domain is not None and rate_domain is not None
                  else prior_domain if rate_domain is None else rate_domain)
        return "map", rate, domain
    if type(node_rate) is ScalarExpr:
        _check_leaves(node_rate, frame, "node_rate")
        return "expression", None if _is_zero(node_rate) else node_rate, None
    if isinstance(node_rate, Expr):
        from pops._ir.visitors import _children
        from pops._ir.expr import Var
        pending, read = [node_rate], []
        while pending:
            node = pending.pop()
            if isinstance(node, Var):
                read.append(node.name)
            pending.extend(_children(node))
        _refuse("a node rate that reads %s on the moving support"
                % (", ".join(sorted(set(read))) or "a symbol of a model"), _R26)
    raise TypeError("MovingConfiguration(node_rate=) takes a pops.analytic expression of the "
                    "coordinate and of time(clock), or a GeometryEvolution")


# --- the configuration --------------------------------------------------------------------------

def _base_frame_and_cells(base: Any) -> tuple[Any, tuple[int, ...], Any, Any]:
    from pops.layouts import Uniform
    layout = base.layout if hasattr(base, "evolution") else base
    if type(layout) is not Uniform:
        raise TypeError("MovingConfiguration(base=) takes a Uniform layout, or a "
                        "MovingControlVolumes of a Uniform layout whose map reads no clock")
    evolution = getattr(base, "evolution", None)
    if evolution is not None and any(item.time_clocks() for item in evolution.coordinate_map):
        raise ValueError("MovingConfiguration(base=) is the configuration at rest; a map that "
                         "reads the clock is the node rate, node_rate=GeometryEvolution(...)")
    return layout.mesh.frame, tuple(layout.mesh.cells), evolution, layout


def _require_motion_served(grid: Any, cells: tuple[int, ...]) -> None:
    """The bases that lot M1 moves: rank 1, no periodic axis."""
    if len(cells) != 1:
        _refuse("a moving configuration of rank %d" % len(cells), "G03 lot M2")
    periodic = getattr(grid.mesh, "periodic", None)
    if periodic is not None and tuple(getattr(periodic, "axes", ()) or ()):
        _refuse("a moving configuration on a periodic axis, whose seam copies are certified by "
                "note 7.5", "G03 lot M1c")


class MovingConfiguration:
    """A base configuration κ moved by the rate w of its nodes, rank 1 (note §7.2, §7.4).

    ``base`` is a ``Uniform`` layout, or the ``MovingControlVolumes`` of one whose map reads no
    clock (a graded segment, N17). ``node_rate`` is a ``pops.analytic`` expression of the frame's
    coordinate and of ``time(program.clock)`` (form 1), or a ``GeometryEvolution`` (form 2).

    ``layout`` is what ``pops.resolve`` takes. In motion it is the logical grid of the base: the
    geometry is the members, and the static arrays of a graded base, which describe the base at
    rest, serve no consumer; the base gives the members their initial positions only. At rest,
    when the normal form of w is zero, ``layout`` is the base itself and ``members`` creates
    nothing, so the case is the base's own (K13).
    """

    def __init__(self, base: Any, *, node_rate: Any) -> None:
        self.base = base
        self.declared: MovingMembers | None = None
        self._rate_domain: PredicateExpr | None = None
        self.frame, cells, self._map, self._grid = _base_frame_and_cells(base)
        if type(node_rate) is GeometryEvolution:
            for law in node_rate.coordinate_map:
                _check_leaves(law, self.frame, "coordinate map")
                _validate_partial_map(law, self.frame, cells)
        elif type(node_rate) is ScalarExpr:
            _check_leaves(node_rate, self.frame, "node_rate")
        if _at_rest_value(node_rate):
            self.form, self.node_rate, self.cells = "rest", None, cells[0]
            return
        _require_motion_served(self._grid, cells)
        self.cells = cells[0]
        self.form, self.node_rate, self._rate_domain = _rate_form(node_rate, self.frame)

    @property
    def at_rest(self) -> bool:
        return self.node_rate is None

    @property
    def layout(self) -> Any:
        return self.base if self.at_rest else self._grid

    def reference_nodes(self) -> np.ndarray:
        """ξ_i, the face coordinates of the base grid, with the native formula
        lower + i ((upper - lower) (1/N))."""
        lower, upper = float(self.frame.lower[0]), float(self.frame.upper[0])
        spacing = (upper - lower)*(1.0/float(self.cells))
        return lower + np.arange(self.cells + 1, dtype=float)*spacing

    def base_nodes(self) -> np.ndarray:
        """The node positions of the base configuration at rest, in binary64."""
        reference = self.reference_nodes()
        if self._map is None:
            return reference
        return evaluate(self._map.coordinate_map[0], reference)

    def initial_members(self) -> np.ndarray:
        """The members at rest, (x_left, x_right, V) by cell, with V = x_right - x_left."""
        nodes = self.base_nodes()
        return np.ascontiguousarray(np.stack((nodes[:-1], nodes[1:], nodes[1:] - nodes[:-1])))

    def node_rate_at(self, positions: Any, reference: Any, t: float) -> np.ndarray:
        """w at the nodes, by the NumPy reading of the node rate (oracles only)."""
        if self.node_rate is None:
            return np.zeros(np.shape(positions))
        if self._rate_domain is not None:
            nodes = np.asarray(reference, dtype=float)
            valid = np.broadcast_to(_evaluate(self._rate_domain, (nodes,), t), nodes.shape)
            result = np.full(nodes.shape, np.nan)
            if np.any(valid):
                result[valid] = evaluate(self.node_rate, nodes[valid], t)
            return result
        return evaluate(self.node_rate, positions if self.form == "expression" else reference, t)

    def members(self, case: Any, name: str = "configuration") -> MovingMembers | None:
        """Declare the members X and V as the state of a block ``name`` of ``case``.

        Returns ``None`` at rest (K13): the configuration is its base, and nothing is declared.
        """
        if self.at_rest:
            return None
        self.declared = MovingMembers(self, case, name)
        return self.declared


class MovingMembers:
    """The block whose state is (x_left, x_right, V), with its rate ``motion`` and its initial
    value. ``rate(u)`` is the Program value of the rates at a stage state u."""

    def __init__(self, configuration: MovingConfiguration, case: Any, name: str) -> None:
        import pops
        from pops.analytic import time as analytic_time
        from pops.fields import AnalyticAux
        from pops.initial import InitialCondition
        from pops.lib.initial import BindArray
        from pops.projection import ConservativeCellAverage

        frame = configuration.frame
        model = pops.Model("%s_moving_members" % name, frame=frame)
        state = model.state(name, components=MEMBER_COMPONENTS)
        node_rate = configuration.node_rate
        if node_rate is None:
            raise ValueError("a configuration at rest has no members (K13)")
        providers = []
        self.rate_domain = None
        if configuration.form == "expression":
            clocks = node_rate.time_clocks()
            clock = (model.auxiliary("stage_clock", frame=frame.canonical_id, unit="1")
                     if clocks else None)
            if clocks:
                providers.append(("stage_clock", analytic_time(clocks[0])))

            def rate_at(position: Any) -> Any:
                return _board(node_rate, position, clock)
            left, right = rate_at(state[0]), rate_at(state[1])
        else:
            bounds = CellBounds(frame)
            axis = frame.axes[0]
            domain = configuration._rate_domain
            faces = (bounds.lower(axis), bounds.upper(axis))
            if domain is None and not _selects_branch(node_rate):
                left = model.auxiliary("rate_left", frame=frame.canonical_id, unit="1")
                right = model.auxiliary("rate_right", frame=frame.canonical_id, unit="1")
                for label, bound in zip(("rate_left", "rate_right"), faces, strict=True):
                    def at_face(node: Any, bound: Any = bound) -> Any:
                        return bound if node._op == "coordinate" else node

                    providers.append((label, _rebuild(node_rate, at_face)))
            else:
                # The providers supply finite input data. The common lazy expression
                # emitter evaluates the selected path of each predicate and rate.
                from pops._ir.control_expr import Where
                from pops._ir.expr import BooleanAnd

                references = tuple(model.auxiliary(label, frame=frame.canonical_id, unit="1")
                                   for label in ("reference_left", "reference_right"))
                providers.extend(zip(("reference_left", "reference_right"), faces, strict=True))
                clocks = node_rate.time_clocks() or (() if domain is None else _time_clocks(domain))
                clock = None
                if clocks:
                    clock = model.auxiliary("stage_clock", frame=frame.canonical_id, unit="1")
                    providers.append(("stage_clock", analytic_time(clocks[0])))
                left, right = (_board(node_rate, reference, clock, map_domain=True)
                               for reference in references)
                if domain is not None:
                    valid = BooleanAnd(*(_board(domain, reference, clock, map_domain=True)
                                         for reference in references))
                    self.rate_domain = model.local_transform(
                        "motion_domain", (Where(valid, 1, 0), 0, 0), on=state)
        model.source(MOTION_SOURCE, on=state, value=(left, right, right - left))
        for label, expression in providers:
            model.module.aux_provider(AnalyticAux(
                model.module.aux_handle(model.module.aux()[label]), expression, frame=frame))
        self.configuration = configuration
        self.model = model
        self.state = state
        self.block = case.block(name, model)
        self.rate = model.module.operator_handle(MOTION_SOURCE)
        case.initials.add(InitialCondition(state=self.block[state], value=BindArray(),
                                           projection=ConservativeCellAverage()))
        _register(model, self)

    def initial_values(self) -> dict[Any, np.ndarray]:
        """``initial_values=`` of ``pops.bind`` for the members, keyed by the block's state."""
        return {self.block[self.state]: self.configuration.initial_members()}


def _board(expr: Any, position: Any, clock: Any, *, map_domain: bool = False) -> Any:
    """The board expression of an analytic node rate, its coordinate read at ``position`` and its
    clock at the stage-clock auxiliary."""
    from pops import math as pm
    from pops._ir.control_expr import Where
    from pops._ir.expr import BooleanAnd, BooleanNot, BooleanOr, Compare, Const

    unary = {"sin": pm.sin, "cos": pm.cos, "exp": pm.exp, "log": pm.log, "sqrt": pm.sqrt}

    def lower(node: Any) -> Any:
        op, arguments = node._op, node._arguments
        if op == "constant":
            return Const(node._literal)
        if op == "coordinate":
            return position
        if op == "time":
            return clock
        if map_domain:
            if op == "where":
                return Where(*(lower(item) for item in arguments))
            if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
                return Compare(op, *(lower(item) for item in arguments))
            if op in {"and", "or", "not"}:
                return {"and": BooleanAnd, "or": BooleanOr, "not": BooleanNot}[op](
                    *(lower(item) for item in arguments))
            if op == "between":
                value, low, high = (lower(item) for item in arguments)
                return BooleanAnd(Compare("le", low, value), Compare("le", value, high))
            if op == "abs":
                return abs(lower(arguments[0]))
        if op in unary:
            return unary[op](lower(arguments[0]))
        if op == "neg":
            return -lower(arguments[0])
        if op in ("add", "sub", "mul", "div", "pow", "minimum", "maximum"):
            a, b = (lower(item) for item in arguments)
            return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a*b,
                    "div": lambda: a/b, "pow": lambda: a**b,
                    "minimum": lambda: pm.minimum(a, b),
                    "maximum": lambda: pm.maximum(a, b)}[op]()
        _refuse("a node rate with the analytic operation %r" % op, "a later lot of G03")
        return None  # unreachable
    return lower(expr)


# The mark that a members model carries, so that resolution finds the blocks of members: an
# attribute of the model itself, with no global registry.
MEMBERS_MARK = "_pops_moving_members"


def _register(model: Any, members: MovingMembers) -> None:
    object.__setattr__(model, MEMBERS_MARK, members)


def members_of(model: Any) -> MovingMembers | None:
    """The members whose model is ``model``, or ``None``."""
    members = getattr(model, MEMBERS_MARK, None)
    return members if isinstance(members, MovingMembers) else None


__all__ = ["MEMBER_COMPONENTS", "MOTION_SOURCE", "MovingConfiguration", "MovingMembers",
           "evaluate", "members_of", "time_derivative"]
