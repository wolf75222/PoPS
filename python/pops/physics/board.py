"""Blackboard-style physics model authoring (Spec 3, layer 1).

``pops.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as ``ddt(U) == -div(F) + S``
and ``-laplacian(phi) == rho`` -- and lowers it to the Spec 2 operator-first IR
(:class:`pops.model.Module`) and the :mod:`pops.dsl` codegen engine. It is a thin
TRANSLATION layer: it owns no numerics and no codegen of its own. ``pops.dsl.Model``
(the PDE facade) remains valid; the board API is sugar that produces the same typed
operators.

The board notation lives in :mod:`pops.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`; the codegen model through
:pyattr:`Model.dsl`.

The handle classes and the multi-species / inspection half live in
``board_handles`` and ``_board_multispecies`` so no file exceeds the Spec-4
500-line bound. Import-graph rule: only :mod:`pops.math` / :mod:`pops.model` /
:mod:`pops.dsl` (the last two LAZILY inside methods); codegen-free, ``_pops``-free.
"""
from __future__ import annotations

from typing import Any

from .. import math as _bm
from ..ir import _wrap
from .board_handles import (FieldHandle, FluxHandle,
                            Invariant, LocalLinearOperatorExpr, SourceHandle, StateHandle,
                            VectorHandle, _canon_role, _safe_name)
from ._board_contract import (atomic_attrs, normalize_components, normalize_roles,
                              normalize_sequence, normalize_string_mapping, require_name)
from ._board_elliptic import _EllipticAuthoringMixin
from ._board_multispecies import _MultiSpeciesMixin
from ._board_rate import _RateAuthoringMixin
from ._board_riemann import _RiemannAuthoringMixin
from ._freeze import PhysicsFreezable


class Model(PhysicsFreezable, _RateAuthoringMixin, _RiemannAuthoringMixin,
            _EllipticAuthoringMixin, _MultiSpeciesMixin):
    """A blackboard-style physical model that lowers to the operator-first IR."""

    _physics_mutators = frozenset({
        "state", "species", "primitive", "scalar", "param", "aux", "field",
        "vector_field", "flux", "source", "local_linear_operator", "solve_field",
        "field_problem", "operator", "riemann", "invariant", "rate",
        "finite_volume_rate", "coupled_rate", "solve_fields_from_species",
    })

    def __init__(self, name: Any) -> None:
        self._init_physics_freeze()
        from .facade import Model as _PdeModel  # lazy: the facade pulls numpy
        self._dsl = _PdeModel(name)
        self.name = self._dsl.name
        self._states = {}
        self._fields = {}
        self._fluxes = {}
        self._sources = {}
        self._operators = {}
        self._operator_inputs = {}  # registered op name -> declared field-input names
        self._aliases = {}          # board operator name -> registered op name
        self._invariants = {}
        self._field_problems = {}   # name -> inert pops.fields field problem (Spec 5 sec.5.1/9.6)
        self._riemann = None        # selected Riemann descriptor (board surface)
        self._reconstruction = None
        self._field_solvers = {}    # field-operator name -> solver descriptor
        # Multi-species mode (Spec 3 sections 12, 16): once a SECOND species is declared the model owns
        # a multi-block pops.model.Module directly (N StateSpaces + a coupled_rate + a multi-block field
        # operator). The single-species path stays byte-identical (keeps the dsl.Model and exposes
        # dsl.Model.module); _multi_module is None until N > 1.
        self._multi_module = None
        self._species = {}          # species name -> StateHandle (multi-species mode)
        self._module_cache = None

    def _invalidate_authoring_views(self) -> None:
        self._module_cache = None

    @property
    def owner_path(self) -> Any:
        """Read-only owner anchor delegated to the underlying typed model."""
        return self._dsl._m.owner_path

    # --- escape hatches ---
    @property
    def dsl(self) -> Any:
        """The underlying :class:`pops.dsl.Model` (the codegen engine)."""
        return self._dsl

    @property
    def module(self) -> Any:
        """The typed :class:`pops.model.Module` view (operator-first IR). Single-species: the
        dsl-derived Module (one StateSpace). Multi-species: the multi-block Module this model assembled
        directly (N StateSpaces, a ``coupled_rate`` operator, a multi-block field operator) -- the SAME
        operator-first IR a hand-written :class:`pops.model.Module` would build.
        """
        if self._module_cache is not None:
            return self._module_cache
        module = self._multi_module if self._multi_module is not None else self._dsl.module
        registry = module.operator_registry()
        for handle in self._fields.values():
            target = getattr(handle, "registered_operator_name", None)
            if target is not None and target != handle.name:
                registry.register_alias(handle.name, target)
        self._module_cache = module
        return module

    # --- state / species ---
    def state(self, name: Any = "U", components: Any = (), roles: Any = None) -> Any:
        """Declare the conservative state. Returns an unpackable :class:`StateHandle`. Board role
        strings (``density`` / ``momentum_x`` / ``momentum_y`` / ``energy`` / ...) are canonicalized to
        the dsl roles (``Density`` / ``MomentumX`` / ...) so the native Riemann capabilities (HLLC/Roe
        role lookup) recognize them.
        """
        name = require_name(name, "state name")
        components = normalize_components(components, "state")
        role_map = normalize_roles(roles, components, "state")
        if self._states:
            raise ValueError(
                "state %r cannot be declared: this physics model already owns state %r; "
                "use species(...) for multiple state blocks" % (name, next(iter(self._states))))
        role_list = None if roles is None else [_canon_role(role_map.get(c)) for c in components]
        hyp = self._dsl._m
        with atomic_attrs((hyp, "cons_names"), (hyp, "cons_roles"), (self, "_states")):
            vars_ = self._dsl.conservative_vars(*components, roles=role_list)
            handle = StateHandle(name, components, vars_, role_map, owner=self.owner_path)
            self._states[handle.name] = handle
        return handle

    def species(self, name: Any, state: Any = (), roles: Any = None) -> Any:
        """Declare a named species: a named block instance of its own StateSpace. Each species lowers
        to one :class:`pops.model.StateSpace` and a named block (Spec 3 sections 12, 16). The returned
        :class:`StateHandle` unpacks into its component vars and indexes them by name (``e["ne"]``) for
        a coupled-rate formula. Arbitrary arity: declare 2, 3, 4, ... species. The single-species case
        is byte-identical to :meth:`state` (no multi-block Module is created); the multi-block path
        engages only from the SECOND species, lowering to the existing operator-first multi-block IR
        (``pops.model.Module`` with N spaces + ``coupled_rate`` + ``solve_fields_from_blocks``), never a
        parallel runtime.
        """
        name = require_name(name, "species name")
        components = normalize_components(state, "species %s state" % name)
        role_map = normalize_roles(roles, components, "species %s" % name)
        if name in self._species:
            raise ValueError(
                "species %r is already declared; each species needs a distinct name "
                "(a reused name would silently alias the StateSpace)" % name)
        if not self._species and self._multi_module is None:
            # First species: keep the single-state dsl-backed path byte-identical to state().
            handle = self.state(
                name, components=components, roles=None if roles is None else role_map)
            self._species[handle.name] = handle
            return handle
        if self._multi_module is None:
            return self._promote_to_multispecies(
                extra=(name, components, role_map))
        return self._add_species(name, components=components, roles=role_map)

    def primitive(self, name: Any, expr: Any) -> Any:
        """Define a primitive quantity by its formula; returns a usable expression."""
        return self._dsl.primitive(require_name(name, "primitive name"), expr)

    def scalar(self, name: Any, expr: Any) -> Any:
        """Define a named derived scalar (e.g. pressure, sound speed)."""
        return self._dsl.primitive(require_name(name, "scalar name"), expr)

    def param(self, declaration: Any) -> Any:
        """Register a typed parameter declaration and return its ParamHandle."""
        return self._dsl.param(declaration)

    def value(self, parameter: Any) -> Any:
        """Return the symbolic Expr read of a registered ParamHandle."""
        return self._dsl.value(parameter)

    def aux(self, name: Any) -> Any:
        """Declare an auxiliary field read by the model (e.g. an imposed ``B_z``)."""
        name = require_name(name, "aux field name")
        canonical = {"phi", "grad_x", "grad_y", "B_z", "T_e"}
        if name in canonical:
            return self._dsl.aux(name)
        return self._dsl.aux_field(name)

    def field(self, name: Any) -> Any:
        """Declare a solved scalar field (e.g. the potential ``phi``)."""
        name = require_name(name, "field name")
        if name in self._fields:
            raise ValueError("field %r is already declared" % name)
        h = FieldHandle(name, owner=self.owner_path)
        self._fields[h.name] = h
        return h

    def vector_field(self, name: Any, x: Any, y: Any) -> Any:
        """Define a named vector field with ``.x`` / ``.y`` expression components."""
        name = require_name(name, "vector field name")
        if name in self._fields:
            raise ValueError("field %r is already declared" % name)
        hyp = self._dsl._m
        with atomic_attrs((hyp, "aux_names"), (hyp, "aux_extra_names"), (self, "_fields")):
            h = VectorHandle(
                name, _wrap(self._to_expr(x)), _wrap(self._to_expr(y)), owner=self.owner_path)
            self._fields[name] = h
        return h

    # --- operators (board equations) ---
    def flux(self, name: Any, on: Any = None, x: Any = None, y: Any = None, waves: Any = None) -> Any:
        """Declare the physical flux and (optionally) its characteristic speeds.

        ``x`` / ``y`` are the per-component flux expressions; ``waves`` gives the
        per-direction eigenvalues. Lowers to the model's default flux.
        """
        name = require_name(name, "flux name")
        self._require_state_handle(on, "flux", optional=True)
        if self._fluxes:
            raise ValueError("flux %r cannot replace already declared physical flux %r"
                             % (name, next(iter(self._fluxes))))
        if x is None or y is None:
            raise ValueError("flux(%r) requires per-component x= and y= expressions" % (name,))
        h = FluxHandle(name, is_default=True, owner=self.owner_path)
        x_values = normalize_sequence(x, "flux x expressions", nonempty=True)
        y_values = normalize_sequence(y, "flux y expressions", nonempty=True)
        expected = self._dsl._m.n_vars
        if len(x_values) != expected or len(y_values) != expected:
            raise ValueError("flux(%r) needs %d expression(s) per direction; got %d/%d"
                             % (name, expected, len(x_values), len(y_values)))
        wave_values = None
        if waves is not None:
            wave_map = normalize_string_mapping(waves, "flux waves")
            if set(wave_map) != {"x", "y"}:
                raise ValueError("flux waves must define exactly the 'x' and 'y' directions")
            wave_values = (
                normalize_sequence(wave_map["x"], "flux x waves", nonempty=True),
                normalize_sequence(wave_map["y"], "flux y waves", nonempty=True))
            if len(wave_values[0]) != expected or len(wave_values[1]) != expected:
                raise ValueError("flux(%r) needs %d wave(s) per direction; got %d/%d"
                                 % (name, expected, len(wave_values[0]), len(wave_values[1])))
        hyp = self._dsl._m
        with atomic_attrs((hyp, "aux_names"), (hyp, "aux_extra_names"), (hyp, "_flux"),
                          (hyp, "_eig"), (self, "_fluxes")):
            x_exprs = [_wrap(self._to_expr(value)) for value in x_values]
            y_exprs = [_wrap(self._to_expr(value)) for value in y_values]
            self._dsl.flux(x_exprs, y_exprs)
            if wave_values is not None:
                self._dsl.eigenvalues(
                    [_wrap(self._to_expr(value)) for value in wave_values[0]],
                    [_wrap(self._to_expr(value)) for value in wave_values[1]])
            self._fluxes[name] = h
        return h

    def source(self, name: Any, on: Any = None, value: Any = None) -> Any:
        """Declare a named local source term; returns a :class:`SourceHandle`."""
        name = require_name(name, "source name")
        self._require_state_handle(on, "source", optional=True)
        if value is None:
            raise ValueError("source(%r) requires value= (one expression per component)" % (name,))
        reg = _safe_name(name)
        if reg in self._sources:
            raise ValueError("source %r collides with already declared source %r"
                             % (name, self._sources[reg].name))
        values = normalize_sequence(value, "source expressions", nonempty=True)
        if len(values) != self._dsl._m.n_vars:
            raise ValueError("source(%r) needs %d expression(s); got %d"
                             % (name, self._dsl._m.n_vars, len(values)))
        h = SourceHandle(name, reg, owner=self.owner_path)
        hyp = self._dsl._m
        with atomic_attrs((hyp, "aux_names"), (hyp, "aux_extra_names"),
                          (hyp, "_source_terms"), (hyp, "_source"), (self, "_sources")):
            self._dsl.source_term(
                reg, [_wrap(self._to_expr(expression)) for expression in values])
            self._sources[reg] = h
        return h

    def local_linear_operator(self, name: Any, on: Any = None, matrix: Any = None) -> Any:
        """Build a local linear operator ``L: U -> U`` as a MATH object (not a callable
        operator). It carries the matrix; register it with :meth:`operator` (or
        ``@module.operator``) to obtain a callable operator. Calling the math object
        directly raises a clear error -- see :class:`LocalLinearOperatorExpr`."""
        if matrix is None:
            raise ValueError("local_linear_operator(%r) requires matrix=" % (name,))
        self._require_state_handle(on, "local_linear_operator", optional=True)
        obj = LocalLinearOperatorExpr(name, matrix, on=on)
        expected = self._dsl._m.n_vars
        if len(obj.matrix) != expected or any(len(row) != expected for row in obj.matrix):
            raise ValueError("local_linear_operator(%r) needs a %dx%d matrix"
                             % (obj.name, expected, expected))
        return obj

    def inspect(self) -> Any:
        """A plain-dict, inert view of the model's authoring state (Spec 5 sec.12.1). Reports the
        declared state / field / flux / source / operator names and the inspectable field problems
        authored via :meth:`field_problem` (each as its descriptor's
        :meth:`~pops.fields.FieldProblem.inspect` dict). Read-only: it touches no numerics, codegen or
        runtime.
        """
        return {
            "name": self.name,
            "states": sorted(self._states),
            "fields": sorted(self._fields),
            "fluxes": sorted(self._fluxes),
            "sources": sorted(self._sources),
            "operators": sorted(self._operators),
            "field_problems": {nm: prob.inspect()
                               for nm, prob in self._field_problems.items()},
        }

    def operator(self, name: Any, handle: Any = None, *, inputs: Any = None,
                 returns: Any = None) -> Any:
        """Register a typed, callable operator under ``name`` from a math object.

        ``returns`` (or the positional ``handle``) is the operator body; ``inputs`` names
        its field dependencies (metadata for requirements). A
        :class:`LocalLinearOperatorExpr` registers as a ``local_linear_operator``
        ``Fields -> LocalLinearOperator(U, U)``. Returns an immutable
        :class:`pops.model.OperatorHandle`.
        """
        obj = returns if returns is not None else handle
        if obj is None:
            raise TypeError("operator(%r) requires returns= (or a positional handle)" % (name,))
        reg = _safe_name(name)
        if isinstance(obj, LocalLinearOperatorExpr):
            self._require_state_handle(obj.on, "operator", optional=True)
            input_names = () if inputs is None else normalize_sequence(inputs, "operator inputs")
            for input_name in input_names:
                require_name(input_name, "operator input")
            hyp = self._dsl._m
            with atomic_attrs(
                    (hyp, "aux_names"), (hyp, "aux_extra_names"), (hyp, "_linear_sources"),
                    (self, "_operators"), (self, "_operator_inputs")):
                self._dsl.linear_source(
                    reg, [[_wrap(self._to_expr(e)) for e in row] for row in obj.matrix])
                self._operators[reg] = obj
                self._operator_inputs[reg] = input_names
                result = self._registered_operator_handle(reg)
            return result
        from pops.model import OperatorHandle
        if isinstance(obj, OperatorHandle):
            # aliasing an already-registered operator under a new role name
            if obj.owner_path != self.owner_path:
                raise ValueError(
                    "operator(%r): the operator handle %r belongs to another physics model"
                    % (name, obj.name))
            target = obj.registered_operator_name
            try:
                self.module.operator_registry().get(target)
            except KeyError:
                raise ValueError(
                    "operator(%r): operator handle %r is not registered by this physics model"
                    % (name, obj.name)) from None
            if name in self._aliases:
                raise ValueError("operator alias %r is already declared" % name)
            self._aliases[name] = target
            return obj
        raise TypeError(
            "operator(%r): returns= must be a local_linear_operator object or a "
            "registered operator; got %r" % (name, obj))

    def _registered_operator_handle(self, name: Any) -> Any:
        """Return the one immutable handle for an operator already in this model's registry."""
        from pops.model import OperatorHandle
        name = require_name(name, "registered operator name")
        module = self._multi_module if self._multi_module is not None else self._dsl.module
        op = module.operator_registry().get(name)
        return OperatorHandle(
            op.name,
            kind=op.kind,
            owner=self.owner_path,
            signature=op.signature,
        )

    def declaration_index(self) -> Any:
        """Read-only membership index spanning the model's small family registries."""
        from pops.model import DeclarationIndex

        records = {}
        for family in (self._states, self._fields, self._fluxes, self._sources):
            for handle in family.values():
                if not hasattr(handle, "kind"):
                    continue
                key = (handle.kind, handle.local_id)
                previous = records.get(key)
                if previous is not None and previous != handle:
                    raise ValueError(
                        "model %r exposes conflicting %s declaration %r"
                        % (self.name, handle.kind, handle.local_id))
                records[key] = handle
        for handle in self.module.declaration_index().records():
            key = (handle.kind, handle.local_id)
            previous = records.get(key)
            if previous is not None and previous != handle:
                raise ValueError(
                    "model %r exposes conflicting %s declaration %r"
                    % (self.name, handle.kind, handle.local_id))
            records[key] = handle
        return DeclarationIndex(owner=self.owner_path, handles=records.values())

    def invariant(self, name: Any, expression: Any = None, over: Any = None) -> Any:
        """Declare a generic invariant ``StateSet -> Scalar`` from an ``integral(...)``."""
        inv = Invariant(name, expression, over=over)
        if inv.name in self._invariants:
            raise ValueError("invariant %r is already declared" % inv.name)
        self._invariants[inv.name] = inv
        return inv

    def invariants(self) -> Any:
        """The declared invariants, by name."""
        return dict(self._invariants)

    # --- validation / compile ---
    def check(self) -> Any:
        """Validate that every referenced quantity is declared (single-species path).

        Multi-species models compose their blocks in a time Program and validate at emit
        (``P.emit_cpp_program`` / ``P._check_lowerable``), so a model-level ``check`` is a
        single-species notion; it is a no-op for a multi-species model."""
        if self._multi_module is not None:
            return None
        return self._dsl.check()

    def lower(self) -> Any:
        """Lower this writing facade to its :class:`pops.model.Module` (ADVANCED / inspection).

        ``pops.physics.Model`` is an AUTHORING facade: it writes the physics (state, primitives,
        flux, sources, field solves) and lowers to the operator-first IR. It does NOT compile.
        The STANDARD flow needs NO manual lower (ADC-557) -- add the model and compile::

            physics_model = pops.physics.Model(...)
            problem.add_block("blk", model=physics_model)
            compiled = pops.compile(problem, layout=..., backend=pops.codegen.Production())

        ``pops.compile`` captures the operator-first Module and validates ONCE internally; ``lower``
        (and its ``to_module`` alias) stay ADVANCED / inspection-only. Identical to :pyattr:`module`."""
        return self.module

    # Spec 5 sec.11 alias: physics.Model.to_module() == physics.Model.lower(). ADVANCED / inspection only
    # (ADC-557): the standard problem.add_block(model=m) -> pops.compile flow captures the Module itself;
    # neither is REQUIRED (pops.compile does the lowering once, internally).
    to_module = lower

    # --- introspection ---

    # --- internals ---
    def _to_expr(self, node: Any) -> Any:
        """Resolve a board node to an :mod:`pops.dsl` expression in this model's context."""
        if isinstance(node, _bm.Partial):
            field = node.field
            if not isinstance(field, FieldHandle):
                raise TypeError("gradient requires a declared FieldHandle; got %r" % (field,))
            if (field.owner_path != self.owner_path
                    or self._fields.get(field.name) != field):
                raise ValueError(
                    "gradient field handle %r belongs to another physics model"
                    % (field.name,))
            aux_name = self._gradient_aux(field.name, node.axis)
            expr = self._dsl.aux(aux_name)
            if node.scale != 1.0:
                expr = node.scale * expr
            return expr
        if isinstance(node, _bm.Gradient):
            raise TypeError("a gradient is a vector; use grad(field).x / .y")
        if isinstance(node, _bm.Laplacian):
            raise TypeError("a laplacian only appears as a field-solve operator")
        return node  # already a dsl Expr / Var / number

    @staticmethod
    def _gradient_aux(field_name: Any, axis: Any) -> Any:
        """Canonical gradient aux name of ``field_name`` along ``axis`` (0=x, 1=y)."""
        field_name = require_name(field_name, "gradient field name")
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1):
            raise ValueError("gradient axis must be integer 0 (x) or 1 (y); got %r" % (axis,))
        if field_name == "phi":
            return "grad_x" if axis == 0 else "grad_y"
        # generic fields keep a <field>_grad_x / _grad_y convention
        return "%s_grad_%s" % (field_name, "x" if axis == 0 else "y")

    def _require_state_handle(self, handle: Any, where: str, *, optional: bool = False) -> Any:
        """Validate an ``on=`` state without accepting a same-named foreign handle."""
        if handle is None and optional:
            return None
        if (not isinstance(handle, StateHandle)
                or handle.owner_path != self.owner_path
                or self._states.get(handle.name) != handle):
            raise ValueError(
                "%s on= must be a StateHandle declared by this physics model; got %r"
                % (where, handle))
        return handle

    def __repr__(self) -> str:
        return "physics.Model(%r)" % (self.name,)
