"""Blackboard-style physics model authoring (Spec 3, layer 1).

``pops.physics.Model`` lets a user write a model the way it appears on a
blackboard -- a state, primitives, a flux, an elliptic field solve, sources and
local linear operators, tied together by equations such as ``ddt(U) == -div(F) + S``
and ``-laplacian(phi) == rho`` -- and lowers it to the operator-first IR
(:class:`pops.model.Module`). It is a thin translation layer: it owns no numerics and exposes no
compiler engine.

The board notation lives in :mod:`pops.math` (``ddt`` / ``div`` / ``grad`` /
``laplacian`` / ``sqrt`` / ``rate`` / ``unknown`` / ``integral``). The typed view
is reachable through :pyattr:`Model.module`. Native lowering is entered only through
``pops.compile(resolved_plan)``.

The handle classes and the multi-species / inspection half live in
``board_handles`` and ``_board_multispecies`` so no file exceeds the Spec-4
500-line bound. The internal formula engine is loaded lazily; import remains codegen-free and
``_pops``-free.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .. import math as _bm
from ..ir import _wrap
from .board_handles import (FieldHandle, FluxHandle,
                            Invariant, LocalLinearOperatorExpr, SourceHandle, StateHandle,
                            VectorHandle, _canon_role, _safe_name)
from ._board_contract import (atomic_attrs, normalize_components, normalize_roles,
                              normalize_sequence, require_name)
from ._board_compile import _BoardCompileMixin
from ._board_elliptic import _EllipticAuthoringMixin
from ._board_multispecies import _MultiSpeciesMixin
from ._board_rate import _RateAuthoringMixin
from ._board_riemann import _RiemannAuthoringMixin
from ._freeze import PhysicsFreezable


class Model(PhysicsFreezable, _BoardCompileMixin, _RateAuthoringMixin, _RiemannAuthoringMixin,
            _EllipticAuthoringMixin, _MultiSpeciesMixin):
    """A blackboard-style physical model that lowers to the operator-first IR."""

    _physics_mutators = frozenset({
        "state", "species", "primitive", "scalar", "param", "aux", "field",
        "vector", "flux", "source", "local_linear_operator", "field_operator",
        "operator", "riemann", "invariant", "rate",
        "finite_volume_rate", "coupled_rate", "solve_fields_from_species",
        "field_provider",
    })

    def __init__(self, name: Any, *, frame: Any = None) -> None:
        self._init_physics_freeze()
        if frame is not None and (
                not callable(getattr(frame, "to_dict", None))
                or not isinstance(getattr(frame, "canonical_id", None), str)
                or not hasattr(frame, "axes")):
            raise TypeError(
                "Model frame must expose typed axes, canonical_id and to_dict()")
        from ._facade import Model as _PdeModel  # lazy: the facade pulls numpy
        self._dsl = _PdeModel(name)
        self.name = self._dsl.name
        self._frame = frame
        self._states = {}
        self._fields = {}
        self._field_operators = {}
        self._fluxes = {}
        self._sources = {}
        self._operators = {}
        # OperatorHandle -> exact physical dependencies retained for numerical-plan validation.
        # DiscretizationPlan can therefore prove that a FiniteVolume(flux=F) discretizes the F
        # referenced by the rate equation, instead of trusting matching display names.
        self._rate_contracts = {}
        self._operator_inputs = {}  # registered op name -> declared field-input names
        self._aliases = {}          # board operator name -> registered op name
        self._invariants = {}
        self._riemann = None        # selected Riemann descriptor (board surface)
        self._reconstruction = None
        # Multi-species mode (Spec 3 sections 12, 16): once a SECOND species is declared the model owns
        # a multi-block pops.model.Module directly (N StateSpaces + a coupled_rate + a multi-block field
        # operator). The single-species path keeps the same dsl.Model backend and numerics;
        # _multi_module is None until N > 1.
        self._multi_module = None
        self._species = {}          # species name -> StateHandle (multi-species mode)
        self._module_cache = None

    def _invalidate_authoring_views(self) -> None:
        self._module_cache = None

    @property
    def owner_path(self) -> Any:
        """Read-only owner anchor delegated to the underlying typed model."""
        return self._dsl._m.owner_path

    @property
    def frame(self) -> Any:
        """The immutable physical frame authored for this model, or ``None``.

        Numerical authorities use this public semantic value to resolve typed geometric
        boundaries; they never inspect the model's private storage or infer axes from strings.
        """
        return self._frame

    @property
    def states(self) -> Mapping[str, StateHandle]:
        """Immutable, live view of the model's declared state handles.

        Library model factories return an ordinary :class:`Model`; callers recover the exact
        registry-issued handles through these family views instead of a preset-specific result
        wrapper.  The mapping cannot be mutated and every value retains this model's OwnerPath.
        """
        return MappingProxyType(self._states)

    @property
    def fields(self) -> Mapping[str, Any]:
        """Immutable, live view of declared scalar/vector field handles."""
        return MappingProxyType(self._fields)

    @property
    def field_operators(self) -> Mapping[str, Any]:
        """Immutable, live view of physics-only field-operator descriptors."""
        return MappingProxyType(self._field_operators)

    @property
    def fluxes(self) -> Mapping[str, FluxHandle]:
        """Immutable, live view of declared physical-flux handles."""
        return MappingProxyType(self._fluxes)

    @property
    def sources(self) -> Mapping[str, SourceHandle]:
        """Immutable, live view of declared local-source handles."""
        return MappingProxyType(self._sources)

    @property
    def operators(self) -> Mapping[str, Any]:
        """Immutable snapshot of registry-authenticated callable operator handles.

        Keys include public aliases as well as their canonical registered operators.  Constructing
        this view never declares an operator and never creates another ownership authority.
        """
        from pops.model import OperatorHandle

        handles = {
            handle.local_id: handle
            for handle in self.module.declaration_index().records()
            if isinstance(handle, OperatorHandle)
        }
        return MappingProxyType(handles)

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
    def state(self, name: Any = "U", components: Any = (), roles: Any = None, *,
              representation: Any = None, space: Any = None, units: Any = None) -> Any:
        """Declare the conservative state and return an unpackable :class:`StateHandle`.

        The final public surface has no partially implemented unit algebra.  Opaque unit strings or
        arbitrary metadata are therefore rejected here until a typed unit protocol can participate in
        validation, semantic identity, lowering and runtime reports end to end.
        """
        name = require_name(name, "state name")
        components = normalize_components(components, "state")
        role_map = normalize_roles(roles, components, "state")
        if self._states:
            raise ValueError(
                "state %r cannot be declared: this physics model already owns state %r; "
                "use species(...) for multiple state blocks" % (name, next(iter(self._states))))
        from pops.representations import Conservative, Representation
        from pops.spaces import CellState, StatePlacement

        selected_representation = Conservative() if representation is None else representation
        if not isinstance(selected_representation, Representation):
            raise TypeError("state representation must be a typed pops.representations value")
        if space is None:
            placement = None if self._frame is None else CellState(frame=self._frame)
        elif isinstance(space, StatePlacement):
            placement = space
        else:
            raise TypeError("state space must be a typed pops.spaces.StatePlacement")
        if placement is not None and self._frame is not None \
                and placement.frame_id != self._frame.canonical_id:
            raise ValueError("state space frame differs from its Model frame")
        if units is not None:
            raise TypeError(
                "Model.state units are unsupported on the final public route; "
                "opaque unit metadata cannot be validated or lowered")
        metadata = {
            "representation": selected_representation.name,
            "centering": "cell" if placement is None else placement.centering,
            "layout": "cell" if placement is None else placement.layout,
            "storage": "multifab" if placement is None else placement.storage,
            "frame": "model" if placement is None else placement.frame_id,
            "clock": "simulation" if placement is None else placement.clock,
            "units": units,
        }
        role_list = None if roles is None else [_canon_role(role_map.get(c)) for c in components]
        hyp = self._dsl._m
        with atomic_attrs(
            (hyp, "cons_names"),
            (hyp, "cons_roles"),
            (hyp, "prim_state"),
            (hyp, "prim_roles"),
            (hyp, "cons_from"),
            (hyp, "_state_space_metadata"),
            (self, "_states"),
        ):
            hyp._state_space_metadata = metadata
            vars_ = self._dsl.conservative_vars(*components, roles=role_list)
            # A blackboard state is a complete coordinate system: use its conservative components
            # as the exact identity primitive layout/inverse, avoiding scalar-law boilerplate.
            self._dsl.primitive_vars(*vars_, roles=role_list)
            self._dsl.conservative_from(list(vars_))
            from pops.model import StateSpace
            from .aux import roles_for
            typed_roles = dict(zip(
                components, roles_for(hyp.cons_names, hyp.cons_roles), strict=True))
            typed_space = StateSpace(
                name,
                components,
                roles=typed_roles,
                layout=metadata["layout"],
                storage=metadata["storage"],
                representation=metadata["representation"],
                centering=metadata["centering"],
                units=metadata["units"],
                frame=metadata["frame"],
                clock=metadata["clock"],
            )
            handle = StateHandle(
                name, components, vars_, role_map, owner=self.owner_path, space=typed_space)
            self._states[handle.name] = handle
        return handle

    def species(self, name: Any, state: Any = (), roles: Any = None) -> Any:
        """Declare a named species: a named block instance of its own StateSpace. Each species lowers
        to one :class:`pops.model.StateSpace` and a named block (Spec 3 sections 12, 16). The returned
        :class:`StateHandle` unpacks into its component vars and indexes them by name (``e["ne"]``) for
        a coupled-rate formula. Arbitrary arity: declare 2, 3, 4, ... species. The single-species case
        uses the same backend as :meth:`state` (no multi-block Module is created); the multi-block path
        engages only from the SECOND species, lowering to the existing operator-first multi-block IR
        (``pops.model.Module`` with N spaces + ``coupled_rate`` + ``solve_fields_from_blocks``), never a
        parallel runtime. Species components are owner-qualified from the first declaration, so an
        existing handle remains exact if a second species later uses the same component names.
        """
        name = require_name(name, "species name")
        components = normalize_components(state, "species %s state" % name)
        role_map = normalize_roles(roles, components, "species %s" % name)
        if name in self._species:
            raise ValueError(
                "species %r is already declared; each species needs a distinct name "
                "(a reused name would silently alias the StateSpace)" % name)
        if not self._species and self._multi_module is None:
            # First species: retain the single-state dsl-backed execution path.
            handle = self.state(
                name, components=components, roles=None if roles is None else role_map)
            # A species coordinate must retain its state owner if this model is later promoted to
            # multiple blocks. The single-state backend rebinds these exact coordinates at its
            # target boundary, so the executable result remains identical without mutating this
            # immutable handle when a second species is declared.
            from pops.ir.expr import Var
            from pops.model.state_symbols import state_component_symbol

            qualified = tuple(
                Var(state_component_symbol(handle.space, component), "cons")
                for component in handle.components
            )
            handle = StateHandle(
                handle.name, handle.components, qualified, dict(handle.roles),
                owner=self.owner_path, space=handle.space)
            self._states[handle.name] = handle
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

    def field(self, name: Any, *, components: Any = None) -> Any:
        """Declare a solved scalar or a multi-component field space.

        Multi-species models use ``components=`` for the potential and its materialized derivatives;
        the returned handle is still the one owner-qualified field declaration consumed by Case.
        """
        name = require_name(name, "field name")
        if name in self._fields:
            raise ValueError("field %r is already declared" % name)
        if components is not None:
            if self._multi_module is None:
                raise ValueError("field components require a multi-species Model")
            values = normalize_components(components, "field %s" % name)
            descriptor = self._multi_module.field_space(name, values)
            h = self._multi_module.field_handle(descriptor)
            self._fields[h.name] = h
            self._invalidate_authoring_views()
            return h
        h = FieldHandle(name, owner=self.owner_path)
        self._fields[h.name] = h
        return h

    def field_spaces(self) -> Any:
        """Return the exact solved-field declarations owned by this blackboard model.

        The generic field compiler consumes this protocol directly.  A scalar
        ``model.field(name)`` is therefore a one-component field space instead
        of being confused with the separate auxiliary ``fields`` context.
        """
        if self._multi_module is not None:
            return self._multi_module.field_spaces()
        from pops.model import FieldSpace

        return {
            name: FieldSpace(name=name, components=(name,), layout="cell")
            for name in self._fields
        }

    def vector(self, name: Any, *, frame: Any, components: Any) -> Any:
        """Define a physical vector by the typed axes of its frame."""
        name = require_name(name, "vector field name")
        if name in self._fields:
            raise ValueError("field %r is already declared" % name)
        if self._frame is not None and frame != self._frame:
            raise ValueError("vector frame differs from its Model frame")
        if not hasattr(frame, "axes") or not isinstance(components, Mapping):
            raise TypeError("vector requires a typed frame and an axis-to-expression mapping")
        if set(components) != set(frame.axes):
            raise ValueError("vector components must name every typed frame axis exactly once")
        hyp = self._dsl._m
        with atomic_attrs((hyp, "aux_names"), (hyp, "aux_extra_names"), (self, "_fields")):
            h = VectorHandle(
                name,
                frame=frame,
                components={axis: _wrap(self._to_expr(components[axis])) for axis in frame.axes},
                owner=self.owner_path,
            )
            self._fields[name] = h
        return h

    # --- operators (board equations) ---
    def flux(self, name: Any, *, frame: Any, state: Any, components: Any,
             waves: Any = None) -> Any:
        """Declare the physical flux and (optionally) its characteristic speeds.

        ``components`` and ``waves`` are keyed by typed axes, never direction strings.  The current
        native route supports Cartesian2D; the public mapping contract extends without another
        method when more dimensions are installed.
        """
        name = require_name(name, "flux name")
        self._require_state_handle(state, "flux", optional=False)
        if self._multi_module is not None:
            state = self._species_handle("flux", name, state)
        if self._frame is not None and frame != self._frame:
            raise ValueError("flux frame differs from its Model frame")
        if not hasattr(frame, "axes") or not isinstance(components, Mapping):
            raise TypeError("flux requires a typed frame and an axis-to-expression mapping")
        if set(components) != set(frame.axes):
            raise ValueError("flux components must name every typed frame axis exactly once")
        if self._multi_module is None and self._fluxes:
            raise ValueError("flux %r cannot replace already declared physical flux %r"
                             % (name, next(iter(self._fluxes))))
        axes = {axis.name: axis for axis in frame.axes}
        if set(axes) != {"x", "y"}:
            raise ValueError("the installed native flux route requires an exact Cartesian2D frame")
        h = FluxHandle(name, is_default=True, owner=self.owner_path)
        x_values = normalize_sequence(
            components[axes["x"]], "flux x expressions", nonempty=True)
        y_values = normalize_sequence(
            components[axes["y"]], "flux y expressions", nonempty=True)
        expected = len(state.components) if self._multi_module is not None else self._dsl._m.n_vars
        if len(x_values) != expected or len(y_values) != expected:
            raise ValueError("flux(%r) needs %d expression(s) per direction; got %d/%d"
                             % (name, expected, len(x_values), len(y_values)))
        wave_values = None
        if waves is not None:
            if not isinstance(waves, Mapping) or set(waves) != set(frame.axes):
                raise TypeError("flux waves must map every typed frame axis exactly once")
            wave_values = (
                normalize_sequence(waves[axes["x"]], "flux x waves", nonempty=True),
                normalize_sequence(waves[axes["y"]], "flux y waves", nonempty=True))
            if len(wave_values[0]) != expected or len(wave_values[1]) != expected:
                raise ValueError("flux(%r) needs %d wave(s) per direction; got %d/%d"
                                 % (name, expected, len(wave_values[0]), len(wave_values[1])))
        if self._multi_module is not None:
            from pops.model import Rate, Signature

            if name in self._fluxes:
                raise ValueError("flux %r is already declared" % name)
            self._multi_module.operator(
                name=name,
                kind="grid_operator",
                signature=Signature((state.space,), Rate(state.space)),
                expr={
                    "x": [self._to_expr(value) for value in x_values],
                    "y": [self._to_expr(value) for value in y_values],
                },
            )
            if wave_values is not None:
                proposed = {
                    "x": tuple(self._to_expr(value) for value in wave_values[0]),
                    "y": tuple(self._to_expr(value) for value in wave_values[1]),
                }
                existing = self._multi_module._eigenvalues
                if existing is not None and repr(existing) != repr(proposed):
                    raise ValueError(
                        "multi-species flux wave declarations must share one exact eigenvalue law"
                    )
                self._multi_module.eigenvalues(**proposed)
            self._fluxes[name] = h
            self._invalidate_authoring_views()
            return h

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

    def source(self, name: Any, on: Any = None, value: Any = None, *, fields: Any = None) -> Any:
        """Declare a named local source term; returns a :class:`SourceHandle`."""
        name = require_name(name, "source name")
        self._require_state_handle(on, "source", optional=True)
        if self._multi_module is not None:
            on = self._species_handle("source", name, on)
        if value is None:
            raise ValueError("source(%r) requires value= (one expression per component)" % (name,))
        reg = _safe_name(name)
        if reg in self._sources:
            raise ValueError("source %r collides with already declared source %r"
                             % (name, self._sources[reg].name))
        values = normalize_sequence(value, "source expressions", nonempty=True)
        expected = len(on.components) if self._multi_module is not None else self._dsl._m.n_vars
        if len(values) != expected:
            raise ValueError("source(%r) needs %d expression(s); got %d"
                             % (name, expected, len(values)))
        h = SourceHandle(name, reg, owner=self.owner_path)
        if self._multi_module is not None:
            from pops.model import Handle, Rate, Signature

            inputs = [on.space]
            if fields is not None:
                if (not isinstance(fields, Handle) or fields.owner_path != self.owner_path
                        or self._fields.get(fields.local_id) != fields):
                    raise ValueError("source fields must be declared by this physics model")
                inputs.append(self._multi_module.field_spaces()[fields.local_id])
            self._multi_module.operator(
                name=reg,
                kind="local_source",
                signature=Signature(tuple(inputs), Rate(on.space)),
                expr=[self._to_expr(expression) for expression in values],
            )
            self._sources[reg] = h
            self._invalidate_authoring_views()
            return h
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
        declared state / field / flux / source / operator names. Physical field operators are
        descriptors returned to the caller and become authoritative only when paired with a
        ``FieldDiscretization`` on a Problem.
        """
        return {
            "name": self.name,
            "states": sorted(self._states),
            "fields": sorted(self._fields),
            "fluxes": sorted(self._fluxes),
            "sources": sorted(self._sources),
            "field_operators": sorted(self._field_operators),
            "operators": sorted(self._operators),
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
                registry = (self._multi_module.operator_registry()
                            if self._multi_module is not None
                            else self._dsl._m.operator_registry())
                registry.get(target)
            except KeyError:
                raise ValueError(
                    "operator(%r): operator handle %r is not registered by this physics model"
                    % (name, obj.name)) from None
            if name in self._aliases:
                raise ValueError("operator alias %r is already declared" % name)
            self._aliases[name] = target
            # Aliases are part of the blackboard Module projection even though they do not mutate
            # the underlying PDE registry. Rebuild that view on its next access so its manifest and
            # declaration index observe the complete authored surface.
            self._invalidate_authoring_views()
            return obj
        raise TypeError(
            "operator(%r): returns= must be a local_linear_operator object or a "
            "registered operator; got %r" % (name, obj))

    def _registered_operator_handle(self, name: Any) -> Any:
        """Return the one immutable handle for an operator already in this model's registry."""
        from pops.model import OperatorHandle
        name = require_name(name, "registered operator name")
        # Facade operators are authored incrementally. Invalidate any previously requested Module
        # projection, but authenticate directly against the authoritative registry: materializing a
        # transient Module for every operator would bind several live fingerprint providers to one
        # model owner and make its identity depend on garbage-collection timing.
        self._dsl._invalidate_authoring_views()
        self._invalidate_authoring_views()
        registry = (self._multi_module.operator_registry()
                    if self._multi_module is not None
                    else self._dsl._m.operator_registry())
        op = registry.get(name)
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
            problem.block("blk", model=physics_model)
            validated = pops.validate(problem)
            resolved = pops.resolve(validated, layout=..., backend=pops.codegen.Production())
            compiled = pops.compile(resolved)

        ``pops.compile`` captures the operator-first Module and validates ONCE internally; ``lower``
        (and its ``to_module`` alias) stay ADVANCED / inspection-only. Identical to :pyattr:`module`."""
        return self.module

    # Spec 5 sec.11 alias: physics.Model.to_module() == physics.Model.lower(). ADVANCED / inspection only
    # (ADC-557): the standard case.block(model=m) -> pops.compile flow captures the Module itself;
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
