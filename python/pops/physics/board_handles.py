"""Blackboard-style physics model authoring (Spec 3, layer 1).

``pops.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as
``ddt(U) == -div(F) + S`` and ``-laplacian(phi) == rho`` -- and lowers it to the
Spec 2 operator-first IR (:class:`pops.model.Module`) and the :mod:`pops.dsl`
codegen engine. It is a thin TRANSLATION layer: it owns no numerics and no
codegen of its own. ``pops.dsl.Model`` (the PDE facade) remains valid; the board
API is sugar that produces the same typed operators.

The board notation lives in :mod:`pops.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`; the codegen model through
:pyattr:`Model.dsl`.

Multi-species board authoring (``m.species`` for N >= 2, ``m.coupled_rate``,
``m.solve_fields_from_species``) LOWERS to the existing operator-first multi-block
IR (an :class:`pops.model.Module` with N :class:`pops.model.StateSpace`, a
``coupled_rate`` operator over a :class:`pops.model.RateBundle`, and a multi-input
field operator), not a second runtime: the board surface produces the SAME typed
operators a hand-written ``pops.model.Module`` registers (ADC-457). The single-species
path is byte-identical to the single-state board model. The compiled multi-block
``.so`` run is validated on ROMEO (Kokkos-only AOT).
"""
from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any

from .. import math as _bm
from ..ir import Expr
from ..model.handles import Handle, OperatorHandle
from ._board_contract import (normalize_components, normalize_roles, normalize_sequence,
                              normalize_string_mapping, require_bool, require_name)

__all__ = ["Invariant", "FluxHandle", "SourceHandle", "FieldsHandle", "FieldOutputs", "FieldHandle",
           "LocalLinearOperatorExpr", "StateHandle", "VectorHandle",
           "_safe_name", "_canon_role", "_roles_for", "_BOARD_ROLE"]


def _safe_name(name: Any) -> str:
    """A C-identifier-safe operator name derived from a strict display name."""
    display = require_name(name, "operator name")
    s = re.sub(r"[^0-9a-zA-Z_]", "_", display).strip("_")
    if not s:
        raise ValueError("operator name %r has no identifier characters" % (name,))
    if s[0].isdigit():
        s = "_" + s
    return s


# Board role vocabulary -> dsl canonical role (pops::VariableRole). The dsl roles_for() uses an
# explicit role override verbatim, so a board role must already be canonical for the native HLLC/Roe
# role lookup (which indexes "Density"/"MomentumX"/"MomentumY"/"Energy") to find it.
_BOARD_ROLE = {
    "density": "Density",
    "momentum_x": "MomentumX", "momentum_y": "MomentumY", "momentum_z": "MomentumZ",
    "energy": "Energy", "pressure": "Pressure", "temperature": "Temperature",
}


def _canon_role(role: Any) -> Any:
    """Canonicalize a board role string; ``None`` remains an unspecified role."""
    if role is None:
        return None
    value = require_name(role, "state role")
    return _BOARD_ROLE.get(value.lower(), value)


def _roles_for(hyp: Any) -> Any:
    """The canonical roles of a HyperbolicModel's conservative state."""
    from .aux import roles_for
    return roles_for(hyp.cons_names, hyp.cons_roles)


class StateHandle(Handle):
    """A declared state: a name plus the ordered :mod:`pops.dsl` component vars.

    Unpacks into its components (``rho, mx, my = U``), indexes them by position
    (``U[0]``) or by component name (``e["ne"]`` -- the board access of Spec 3
    section 12.3/16), and remembers its name and roles for the typed
    :class:`pops.model.StateSpace`. The string index returns the conservative
    :class:`pops.dsl.Var` of that component, so a board coupled-rate formula
    written as ``e["ni"] - e["ne"]`` is the same IR as the hand-written
    operator-first ``dsl.Var("ni", "cons") - dsl.Var("ne", "cons")``.
    """

    __slots__ = ("components", "vars", "roles", "space")

    def __init__(self, name: Any, components: Any, vars_: Any, roles: Any, *, owner: Any,
                 space: Any = None) -> None:
        name = require_name(name, "state name")
        components = normalize_components(components, "state")
        vars_ = normalize_sequence(vars_, "state variables")
        if len(vars_) != len(components):
            raise ValueError(
                "state %r has %d component(s) but %d variable(s)"
                % (name, len(components), len(vars_)))
        roles = normalize_roles(roles, components, "state")
        super().__init__(name, kind="state", owner=owner)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "vars", vars_)
        object.__setattr__(self, "roles", MappingProxyType(roles))
        # The typed pops.model.StateSpace this species instantiates (multi-species
        # mode); None for the single-state dsl-backed path, where the space is
        # derived on demand from the dsl model.
        object.__setattr__(self, "space", space)

    def __iter__(self) -> Any:
        return iter(self.vars)

    def __len__(self) -> int:
        return len(self.vars)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self.vars[self.components.index(key)]
            except ValueError:
                raise KeyError(
                    "state %r has no component %r (have: %s)"
                    % (self.name, key, ", ".join(self.components))) from None
        return self.vars[key]

    def __repr__(self) -> str:
        return "StateHandle(%r, %r)" % (self.name, list(self.components))


class FieldHandle(Handle):
    """A solved/auxiliary scalar field (e.g. the potential ``phi``)."""

    __slots__ = ()

    def __init__(self, name: Any, *, owner: Any) -> None:
        super().__init__(require_name(name, "field name"), kind="field", owner=owner)

    def __repr__(self) -> str:
        return "FieldHandle(%r)" % (self.name,)


class VectorHandle(Handle):
    """A named vector field indexed by the typed axes of its frame."""

    __slots__ = ("frame", "components")

    def __init__(self, name: Any, *, frame: Any, components: Any, owner: Any) -> None:
        if not callable(getattr(frame, "to_dict", None)) or not hasattr(frame, "axes"):
            raise TypeError("VectorHandle frame must expose typed axes and to_dict()")
        values = dict(components)
        if set(values) != set(frame.axes):
            raise ValueError("VectorHandle components must name every frame axis exactly once")
        super().__init__(require_name(name, "vector field name"), kind="vector", owner=owner)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "components", MappingProxyType(values))

    def __getitem__(self, axis: Any) -> Any:
        try:
            return self.components[axis]
        except KeyError:
            raise KeyError("axis does not belong to vector frame") from None

    @property
    def x(self) -> Any:
        return next(self.components[axis] for axis in self.frame.axes if axis.name == "x")

    @property
    def y(self) -> Any:
        return next(self.components[axis] for axis in self.frame.axes if axis.name == "y")

    def __repr__(self) -> str:
        return "VectorHandle(%r)" % (self.name,)


class FluxHandle(Handle):
    """A declared physical flux (the default hyperbolic flux of a model)."""

    __slots__ = ("is_default",)

    def __init__(self, name: Any, is_default: bool = True, *, owner: Any) -> None:
        super().__init__(require_name(name, "flux name"), kind="flux", owner=owner)
        object.__setattr__(self, "is_default", require_bool(is_default, "flux is_default"))

    def __repr__(self) -> str:
        return "FluxHandle(%r)" % (self.name,)


class SourceHandle(Handle):
    """Identity of a declared local source term.

    It deliberately is not an :class:`Expr`.  Rate algebra creates a
    :class:`SourceTermExpr` wrapper, keeping handle equality Boolean while
    preserving the blackboard spelling ``-div(F) + S``.
    """

    __slots__ = ("reg_name",)

    def __init__(self, display_name: Any, reg_name: Any, *, owner: Any) -> None:
        display_name = require_name(display_name, "source name")
        reg_name = require_name(reg_name, "source registry name")
        Handle.__init__(self, display_name, kind="source", owner=owner)
        object.__setattr__(self, "reg_name", reg_name)

    def __pops_rate_term__(self) -> "SourceTermExpr":
        return SourceTermExpr(self)

    def __neg__(self) -> Any:
        return -self.__pops_rate_term__()

    def __add__(self, other: Any) -> Any:
        return self.__pops_rate_term__() + other

    def __radd__(self, other: Any) -> Any:
        return _bm._as_rate(other) + self.__pops_rate_term__()

    def __sub__(self, other: Any) -> Any:
        return self.__pops_rate_term__() - other

    def __rsub__(self, other: Any) -> Any:
        return _bm._as_rate(other) - self.__pops_rate_term__()

    def __repr__(self) -> str:
        return "SourceHandle(%r)" % (self.name,)


class SourceTermExpr(_bm.RateTerm):
    """Symbolic rate contribution referring to one :class:`SourceHandle`."""

    def __init__(self, handle: Any) -> None:
        if not isinstance(handle, SourceHandle):
            raise TypeError("SourceTermExpr requires a SourceHandle")
        self.handle = handle

    def _rate_terms(self) -> Any:
        return [("source", self.handle, 1)]

    def __repr__(self) -> str:
        return "source_term(%r)" % (self.handle,)


class LocalLinearOperatorExpr(Expr):
    """A LOCAL linear operator object ``L: U -> U`` -- a MATH object, not a callable operator.

    ``m.local_linear_operator(...)`` returns this; it carries the matrix but is NOT yet a
    typed registry operator. Register it with ``m.operator(name, returns=...)`` (or
    ``@module.operator``) to obtain a callable operator. Calling the math object directly
    is an error -- it cannot resolve its field inputs without a registration.
    """

    def __init__(self, display_name: Any, matrix: Any, on: Any = None) -> None:
        self.name = require_name(display_name, "local linear operator name")
        rows = normalize_sequence(matrix, "local linear operator matrix", nonempty=True)
        self.matrix = tuple(
            normalize_sequence(row, "local linear operator matrix row", nonempty=True)
            for row in rows)
        self.on = on

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "local_linear_operator object %r is not a callable operator. Register it with "
            "m.operator(%r, returns=...) or @module.operator(...) first." % (self.name, self.name))

    def __repr__(self) -> str:
        return "LocalLinearOperatorExpr(%r)" % (self.name,)


class FieldOutputs:
    """Structured, typed access to a field solve's produced outputs (ADC-556).

    Replaces free-string output lookup: a field solve's outputs are reachable BOTH as attributes
    (``fields.outputs.E``) and by item (``fields.outputs["E"]``), and iterate like the underlying
    mapping (``.items()`` / ``in`` / ``len`` still work, so existing dict readers are unaffected).
    An unknown output raises a structured error NAMING the known handles, never a silent miss.
    """

    __slots__ = ("_m",)

    def __init__(self, mapping: Any) -> None:
        object.__setattr__(self, "_m", MappingProxyType(
            normalize_string_mapping(mapping, "field outputs")))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FieldOutputs is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FieldOutputs is immutable")

    def __getattr__(self, key: Any) -> Any:
        # __getattr__ runs only for names not found normally, so it never shadows _m / methods.
        try:
            return self._m[key]
        except KeyError:
            raise AttributeError(
                "unknown field output %r; known outputs: %s" % (key, sorted(self._m))) from None

    def __getitem__(self, key: Any) -> Any:
        try:
            return self._m[key]
        except KeyError:
            raise KeyError(
                "unknown field output %r; known outputs: %s" % (key, sorted(self._m))) from None

    def __contains__(self, key: Any) -> bool:
        return key in self._m

    def __iter__(self) -> Any:
        return iter(self._m)

    def __len__(self) -> int:
        return len(self._m)

    def keys(self) -> Any:
        return self._m.keys()

    def values(self) -> Any:
        return self._m.values()

    def items(self) -> Any:
        return self._m.items()

    def __repr__(self) -> str:
        return "FieldOutputs(%r)" % (sorted(self._m),)


class FieldsHandle(OperatorHandle):
    """The result of a field-solve operator: a typed ``OperatorHandle`` over a bundle of solved
    fields (ADC-556).

    A ``FieldsHandle`` IS an :class:`pops.model.OperatorHandle` of kind ``"field_operator"`` (so it
    resolves through the one public ``P.call`` path like any operator), enriched with the field
    solve's structured :class:`FieldOutputs` and the required elliptic ``solver``. Calling it with a
    Program State value lowers through the same owner-checked ``P.call`` path as every operator and
    returns the FieldContext-tagged value. The Program must first bind the declaring model/module;
    a bare call without a Program value is refused.
    """

    __slots__ = ("outputs", "solver")

    def __init__(self, name: Any, outputs: Any = None, solver: Any = None, *, owner: Any,
                 registered_operator_name: Any = None) -> None:
        name = require_name(name, "field operator name")
        output_handles = FieldOutputs(outputs)
        if registered_operator_name is not None:
            registered_operator_name = require_name(
                registered_operator_name, "field operator registry name")
        super().__init__(
            name, kind="field_operator", owner=owner,
            registered_operator_name=registered_operator_name)
        object.__setattr__(self, "outputs", output_handles)
        object.__setattr__(self, "solver", solver)

    def __call__(self, state: Any, name: Any = None) -> Any:
        prog = getattr(state, "prog", None)
        if prog is None:
            raise ValueError(
                "field operator %r must be called with a time-Program State value "
                "(inside a Program); got %r" % (self.name, state))
        return prog.call(self, state, name=self.name if name is None else name)

    def __repr__(self) -> str:
        return "FieldsHandle(%r)" % (self.name,)


class Invariant:
    """A generic invariant: a typed function ``StateSet -> Scalar``.

    Carries a board ``integral(...)`` value expression and the states it ranges
    over. Nothing about mass / charge / momentum / energy is built in: the value
    is whatever the user writes. Used for diagnostics and conservation checks.
    """

    def __init__(self, name: Any, value: Any, over: Any = None) -> None:
        self.name = require_name(name, "invariant name")
        self.value = value
        self.over = () if over is None else normalize_sequence(over, "invariant states")

    def __repr__(self) -> str:
        return "Invariant(%r)" % (self.name,)
