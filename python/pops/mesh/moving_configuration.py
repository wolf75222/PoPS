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
   face coordinates of the base grid.

Why the copies of one node stay equal: both copies of a node start from one base node and receive
the same rate, computed by the same expression from the same inputs, so they stay bitwise equal.

A node rate whose normal form is zero normalizes the configuration to its base before any member is
created (K13). A node rate that reads a field is refused (R26), as is a parameter, until a later lot
gives the quantities of a ``Finite`` support. Rank 2 is lot M2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pops.analytic import CellBounds, ScalarExpr
from pops.analytic import cos as _cos, exp as _exp, log as _log, sin as _sin, sqrt as _sqrt
from pops.analytic import constant as _constant
from .moving import GeometryEvolution

MEMBER_COMPONENTS = ("x_left", "x_right", "volume")
MOTION_SOURCE = "motion"
_R26 = ("a free-boundary or Lagrangian method, whose law the case must supply "
        "(R26; a later lot of G03)")


def _refuse(what: str, lot: str) -> None:
    from pops.surface._refusal import NotYetElaborated
    raise NotYetElaborated(what, lot)


# --- the analytic node rate ----------------------------------------------------------------------

def _is_zero(expr: Any) -> bool:
    """Whether ``expr`` is an exact zero of the analytic algebra (the normal form of K13)."""
    if expr is None:
        return True
    op, arguments = expr._op, expr._arguments
    if op == "constant":
        return expr._literal == 0.0
    if op == "mul":
        return any(_is_zero(item) for item in arguments)
    if op in ("add", "sub"):
        return all(_is_zero(item) for item in arguments)
    if op in ("neg", "div"):
        return _is_zero(arguments[0])
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
        a, b = arguments
        da, db = time_derivative(a), time_derivative(b)
        if db is None:
            return None if da is None else da/b
        return _sub(_scale(b, da), _scale(a, db))/(b*b)
    if op == "pow":
        a, b = arguments
        if b.time_clocks():
            _refuse("the time derivative of a power whose exponent reads the clock",
                    "a later lot of G03; state the node rate, node_rate=w(x, t)")
        return _scale(b*a**(b - 1), time_derivative(a))
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
          "mul": np.multiply, "div": np.divide, "pow": np.power, "minimum": np.minimum,
          "maximum": np.maximum}


def evaluate(expr: ScalarExpr, x: Any, t: float = 0.0) -> np.ndarray:
    """A NumPy reading of a node rate or a map at coordinates ``x`` and time ``t`` (oracles only)."""
    def value(node: Any) -> Any:
        op = node._op
        if op == "constant":
            return node._literal
        if op == "coordinate":
            return np.asarray(x, dtype=float)
        if op == "time":
            return float(t)
        if op not in _NUMPY:
            raise NotImplementedError("no NumPy reading of the analytic operation %r" % op)
        return _NUMPY[op](*(value(item) for item in node._arguments))
    return np.broadcast_to(np.asarray(value(expr), dtype=float), np.shape(x)).copy()


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


def _rate_form(node_rate: Any, frame: Any) -> tuple[str, ScalarExpr | None]:
    """(form, w): ``"expression"`` with w(x, t), or ``"map"`` with ∂ₜΦ(ξ, t); w is ``None`` at rest."""
    from pops._ir.expr import Expr
    if isinstance(node_rate, (int, float)) and not isinstance(node_rate, bool):
        node_rate = _constant(float(node_rate))
    if type(node_rate) is GeometryEvolution:
        if len(node_rate.coordinate_map) != 1:
            _refuse("a coordinate map of rank %d" % len(node_rate.coordinate_map), "G03 lot M2")
        (law,) = node_rate.coordinate_map
        _check_leaves(law, frame, "coordinate map")
        derivative = time_derivative(law)
        return "map", None if _is_zero(derivative) else derivative
    if type(node_rate) is ScalarExpr:
        _check_leaves(node_rate, frame, "node_rate")
        return "expression", None if _is_zero(node_rate) else node_rate
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
        self.frame, cells, self._map, self._grid = _base_frame_and_cells(base)
        if _at_rest_value(node_rate):
            self.form, self.node_rate, self.cells = "rest", None, cells[0]
            return
        _require_motion_served(self._grid, cells)
        self.cells = cells[0]
        self.form, self.node_rate = _rate_form(node_rate, self.frame)

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
            left = model.auxiliary("rate_left", frame=frame.canonical_id, unit="1")
            right = model.auxiliary("rate_right", frame=frame.canonical_id, unit="1")
            for label, bound in (("rate_left", bounds.lower(axis)),
                                 ("rate_right", bounds.upper(axis))):
                providers.append((label, _rebuild(
                    node_rate, lambda node, bound=bound: bound if node._op == "coordinate" else node)))
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


def _board(expr: Any, position: Any, clock: Any) -> Any:
    """The board expression of an analytic node rate, its coordinate read at ``position`` and its
    clock at the stage-clock auxiliary."""
    from pops import math as pm
    from pops._ir.expr import Const

    unary = {"sin": pm.sin, "cos": pm.cos, "exp": pm.exp, "log": pm.log, "sqrt": pm.sqrt}

    def lower(node: Any) -> Any:
        op, arguments = node._op, node._arguments
        if op == "constant":
            return Const(node._literal)
        if op == "coordinate":
            return position
        if op == "time":
            return clock
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
