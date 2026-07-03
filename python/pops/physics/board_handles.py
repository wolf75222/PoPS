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
from typing import Any

from .. import math as _bm
from ..model.handles import OperatorHandle

__all__ = ["Invariant", "FluxHandle", "SourceHandle", "FieldsHandle", "FieldOutputs", "FieldHandle",
           "LocalLinearOperatorExpr", "CallableOperator", "StateHandle", "VectorHandle",
           "_safe_name", "_canon_role", "_roles_for", "_BOARD_ROLE"]


def _safe_name(name: Any) -> str:
    """A C-identifier-safe operator name derived from a board display name."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
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
    """Canonicalize a board role string to a dsl role; pass through None and unknown roles."""
    if role is None:
        return None
    return _BOARD_ROLE.get(str(role).lower(), role)


def _roles_for(hyp: Any) -> Any:
    """The canonical roles of a HyperbolicModel's conservative state."""
    from .aux import roles_for
    return roles_for(hyp.cons_names, hyp.cons_roles)


class StateHandle:
    """A declared state: a name plus the ordered :mod:`pops.dsl` component vars.

    Unpacks into its components (``rho, mx, my = U``), indexes them by position
    (``U[0]``) or by component name (``e["ne"]`` -- the board access of Spec 3
    section 12.3/16), and remembers its name and roles for the typed
    :class:`pops.model.StateSpace`. The string index returns the conservative
    :class:`pops.dsl.Var` of that component, so a board coupled-rate formula
    written as ``e["ni"] - e["ne"]`` is the same IR as the hand-written
    operator-first ``dsl.Var("ni", "cons") - dsl.Var("ne", "cons")``.
    """

    def __init__(self, name: Any, components: Any, vars_: Any, roles: Any, space: Any = None) -> None:
        self.name = str(name)
        self.components = tuple(components)
        self.vars = tuple(vars_)
        self.roles = dict(roles or {})
        # The typed pops.model.StateSpace this species instantiates (multi-species
        # mode); None for the single-state dsl-backed path, where the space is
        # derived on demand from the dsl model.
        self.space = space

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


class FieldHandle:
    """A solved/auxiliary scalar field (e.g. the potential ``phi``)."""

    def __init__(self, name: Any) -> None:
        self.name = str(name)

    def __repr__(self) -> str:
        return "FieldHandle(%r)" % (self.name,)


class VectorHandle:
    """A named vector field with ``.x`` / ``.y`` expression components."""

    def __init__(self, name: Any, x: Any, y: Any) -> None:
        self.name = str(name)
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return "VectorHandle(%r)" % (self.name,)


class FluxHandle:
    """A declared physical flux (the default hyperbolic flux of a model)."""

    def __init__(self, name: Any, is_default: bool = True) -> None:
        self.name = str(name)
        self.is_default = bool(is_default)

    def __repr__(self) -> str:
        return "FluxHandle(%r)" % (self.name,)


class SourceHandle(_bm.RateTerm):
    """A declared local source term -- a summand of a rate equation."""

    def __init__(self, display_name: Any, reg_name: Any) -> None:
        self.name = str(display_name)
        self.reg_name = str(reg_name)

    def _rate_terms(self) -> Any:
        return [("source", self, 1.0)]

    def __repr__(self) -> str:
        return "SourceHandle(%r)" % (self.name,)


class LocalLinearOperatorExpr:
    """A LOCAL linear operator object ``L: U -> U`` -- a MATH object, not a callable operator.

    ``m.local_linear_operator(...)`` returns this; it carries the matrix but is NOT yet a
    typed registry operator. Register it with ``m.operator(name, returns=...)`` (or
    ``@module.operator``) to obtain a callable operator. Calling the math object directly
    is an error -- it cannot resolve its field inputs without a registration.
    """

    def __init__(self, display_name: Any, matrix: Any, on: Any = None) -> None:
        self.name = str(display_name)
        self.matrix = matrix
        self.on = on

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "local_linear_operator object %r is not a callable operator. Register it with "
            "m.operator(%r, returns=...) or @module.operator(...) first." % (self.name, self.name))

    def __repr__(self) -> str:
        return "LocalLinearOperatorExpr(%r)" % (self.name,)


class CallableOperator(OperatorHandle):
    """A board-authored, self-binding :class:`pops.model.OperatorHandle` (ADC-560 fold).

    Returned by ``m.rate`` / ``m.operator``. Since ADC-560 the ONE typed handle is
    ``OperatorHandle`` (callable via ``handle(...)``); ``CallableOperator`` is now that handle
    SUBTYPE, kept for one release for the board path, which needs an extra self-binding step: a board
    program may register operators in any order, so a call binds (or rebinds) the model's FRESH module
    when the Program has no registry yet or the bound one predates this operator. It then delegates to
    the SAME ``P._call(name, ...)`` lowering the base handle uses, so ``explicit_rate(U_n, fields_n)``
    builds the byte-identical IR as the public typed ``P.call(rate_handle, U_n, fields_n)``. Its kind /
    signature are resolved lazily from the model's module so ``inspect()`` reads the math object it
    names.
    """

    __slots__ = ("reg_name", "_model")

    def __init__(self, name: Any, model: Any) -> None:
        super().__init__(str(name))
        object.__setattr__(self, "reg_name", str(name))
        object.__setattr__(self, "_model", model)  # bound to its FRESH module at call time
        self._resolve_kind_signature()

    def _resolve_kind_signature(self) -> None:
        """Stamp the kind / signature / category from the model's module (best effort, never raises).

        The operator was registered by the board declarer before this handle was built, so it is
        usually resolvable now; a not-yet-registered name (unusual ordering) leaves the fields ``None``
        and they stay resolvable through the registry at call time."""
        model = self._model
        if model is None:
            return
        try:
            op = model.module.operator_registry().get(self.reg_name)
        except Exception:  # registry unavailable / name not registered yet -> leave metadata None
            return
        object.__setattr__(self, "kind", op.kind)
        object.__setattr__(self, "signature", op.signature)
        from pops.model.operators import operator_family
        object.__setattr__(self, "category", operator_family(op.kind))

    def __call__(self, *args: Any, name: Any = None) -> Any:
        prog = self._program_from_args(args)
        reg = getattr(prog, "_registry", None)
        # Bind (or rebind) the model's FRESH module if the program has no registry yet or the bound
        # one predates this operator -- so operators registered in any order all resolve, not just
        # those present when the program was first bound.
        if self._model is not None and (reg is None or self.name not in reg):
            prog.bind_operators(self._model.module)
        return prog._call(self.name, *args, name=name)

    def __repr__(self) -> str:
        return "CallableOperator(%r)" % (self.name,)


class FieldOutputs:
    """Structured, typed access to a field solve's produced outputs (ADC-556).

    Replaces free-string output lookup: a field solve's outputs are reachable BOTH as attributes
    (``fields.outputs.E``) and by item (``fields.outputs["E"]``), and iterate like the underlying
    mapping (``.items()`` / ``in`` / ``len`` still work, so existing dict readers are unaffected).
    An unknown output raises a structured error NAMING the known handles, never a silent miss.
    """

    __slots__ = ("_m",)

    def __init__(self, mapping: Any) -> None:
        object.__setattr__(self, "_m", dict(mapping or {}))

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
    Program State value lowers to that Program's per-stage field solve
    (``P.solve_fields(name, state)``), returning the FieldContext-tagged value; a bare call without a
    Program value is refused. ``__call__`` is defined ONLY here, not on the base ``OperatorHandle``.
    """

    __slots__ = ("outputs", "solver")

    def __init__(self, name: Any, outputs: Any = None, solver: Any = None) -> None:
        super().__init__(str(name), kind="field_operator")
        object.__setattr__(self, "outputs", FieldOutputs(outputs))
        object.__setattr__(self, "solver", solver)

    def __call__(self, state: Any, name: Any = None) -> Any:
        prog = getattr(state, "prog", None)
        if prog is None:
            raise ValueError(
                "field operator %r must be called with a time-Program State value "
                "(inside a Program); got %r" % (self.name, state))
        return prog.solve_fields(name=name or self.name, state=state)

    def __repr__(self) -> str:
        return "FieldsHandle(%r)" % (self.name,)


class Invariant:
    """A generic invariant: a typed function ``StateSet -> Scalar``.

    Carries a board ``integral(...)`` value expression and the states it ranges
    over. Nothing about mass / charge / momentum / energy is built in: the value
    is whatever the user writes. Used for diagnostics and conservation checks.
    """

    def __init__(self, name: Any, value: Any, over: Any = None) -> None:
        self.name = str(name)
        self.value = value
        self.over = tuple(over) if over else ()

    def __repr__(self) -> str:
        return "Invariant(%r)" % (self.name,)

