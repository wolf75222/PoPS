"""Blackboard physics authoring.

``pops.physics.Model`` writes states, fluxes, sources and field solves, then lowers to the
operator-first :class:`pops.model.Module`. It owns no numerics and exposes no codegen engine;
``lower()`` / ``to_module()`` return the actual Module populated by this facade.
"""
from pops.descriptors import reject_string_selector

from .. import math as _bm
from .. import model as _model
from ..ir.expr import Var
from .board_handles import (CallableOperator, FieldHandle, FieldsHandle, FluxHandle,
                            Invariant, LocalLinearOperatorExpr, SourceHandle, StateHandle,
                            VectorHandle, _canon_role, _safe_name)
from ._board_internals import _BoardInternalsMixin
from ._board_multispecies import _MultiSpeciesMixin
from .model import _NO_KIND, _coerce_param



class Model(_BoardInternalsMixin, _MultiSpeciesMixin):
    """A blackboard-style physical model that lowers to the operator-first IR."""

    def __init__(self, name):
        self.name = str(name)
        self._module = _model.Module(self.name)
        self._state_space = None
        self._default_field_space = None
        self._primitive_defs = {}
        self.params = {}
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
        self._riemann_hooks = {}    # capability formulas for the native-hook codegen (ADC-456)
        self._field_solvers = {}    # field-operator name -> solver descriptor
        # Multi-species mode (Spec 3 sections 12, 16): once a SECOND species is declared the model
        # swaps to a multi-block pops.model.Module directly (N StateSpaces + a coupled_rate + a
        # multi-block field operator). The single-species path is already Module-native.
        self._multi_module = None
        self._species = {}          # species name -> StateHandle (multi-species mode)

    # --- escape hatches ---
    @property
    def dsl(self):
        """No public escape hatch to the legacy codegen engine."""
        raise AttributeError(
            "pops.physics.Model.dsl is not public; call lower()/to_module(), compile with "
            "pops.compile_problem(...), then install with sim.install(...).")

    @property
    def module(self):
        """The typed :class:`pops.model.Module` view (operator-first IR).

        Single-species: the Module authored directly by this facade (one StateSpace). Multi-species: the
        multi-block Module this model assembled directly (N StateSpaces, a
        ``coupled_rate`` operator, a multi-block field operator) -- the SAME
        operator-first IR a hand-written :class:`pops.model.Module` would build.
        """
        if self._multi_module is not None:
            return self._multi_module
        return self._module

    # --- state / species ---
    def state(self, name="U", components=(), roles=None):
        """Declare the conservative state. Returns an unpackable :class:`StateHandle`.

        Board role strings (``density`` / ``momentum_x`` / ``momentum_y`` / ``energy`` / ...)
        are canonicalized to the native C++ roles (``Density`` / ``MomentumX`` / ...) so the native
        Riemann capabilities (HLLC/Roe role lookup) recognize them.
        """
        role_list = None
        if roles:
            role_list = [_canon_role(roles.get(c)) for c in components]
        canon = {c: role_list[i] for i, c in enumerate(components)} if role_list else {}
        space = self._module.state_space(name, components, roles=canon)
        self._state_space = space
        vars_ = tuple(Var(c, "cons") for c in components)
        handle = StateHandle(name, components, vars_, roles, space=space)
        self._states[handle.name] = handle
        return handle

    def species(self, name, state=(), roles=None):
        """Declare a named species: a named block instance of its own StateSpace.

        Each species lowers to one :class:`pops.model.StateSpace` and a named block
        (Spec 3 sections 12, 16). The returned :class:`StateHandle` unpacks into its
        component vars and indexes them by name (``e["ne"]``) for a coupled-rate
        formula. Arbitrary arity: declare 2, 3, 4, ... species. The single-species
        case is byte-identical to :meth:`state` (no multi-block Module is created);
        the multi-block path engages only from the SECOND species, lowering to the
        existing operator-first multi-block IR (``pops.model.Module`` with N spaces +
        ``coupled_rate`` + ``solve_fields_from_blocks``), never a parallel runtime.
        """
        if name in self._species:
            raise ValueError(
                "species %r is already declared; each species needs a distinct name "
                "(a reused name would silently alias the StateSpace)" % name)
        if not self._species and not self._multi_module:
            # First species: use the same Module-native single-state path as state().
            handle = self.state(name, components=state, roles=roles)
            self._species[handle.name] = handle
            return handle
        # Second (or later) species: promote to multi-species mode. Re-realize the first species
        # as a typed StateSpace on the multi-block Module so all species live in the SAME
        # operator-first IR.
        self._promote_to_multispecies()
        return self._add_species(name, components=state, roles=roles)

    def primitive(self, name, expr):
        """Define a primitive quantity by its formula; returns a usable expression."""
        expr = self._to_expr(expr)
        self._primitive_defs[str(name)] = expr
        self._module.primitive(name, expr)
        return Var(str(name), "prim")

    def scalar(self, name, expr):
        """Define a named derived scalar (e.g. pressure, sound speed)."""
        return self.primitive(name, expr)

    def param(self, name, value=None, *, kind=_NO_KIND):
        """Declare a named scalar parameter; returns a usable expression.

        The kind is a TYPED param object (Spec 5 sec.7), not a ``kind=`` string:
        ``param(pops.physics.RuntimeParam("cs2", 1.0))`` / ``param(ConstParam("g", 9.8))`` /
        ``param("g", 9.8)`` (const shorthand). A bare ``kind=`` keyword is REJECTED.
        """
        p = _coerce_param(name, value, kind=kind, who="physics.Model.param")
        self.params[p.name] = p
        self._module.param(p.name, p.value, _kind=p.kind)
        return p

    def aux(self, name):
        """Declare an auxiliary field read by the model (e.g. an imposed ``B_z``)."""
        self._module.aux_field(name)
        return Var(name, "aux")

    def field(self, name):
        """Declare a solved scalar field (e.g. the potential ``phi``)."""
        h = FieldHandle(name)
        self._fields[h.name] = h
        return h

    def vector_field(self, name, x, y):
        """Define a named vector field with ``.x`` / ``.y`` expression components."""
        h = VectorHandle(name, self._to_expr(x), self._to_expr(y))
        self._fields[name] = h
        return h

    # --- operators (board equations) ---
    def flux(self, name, on=None, x=None, y=None, waves=None):
        """Declare the physical flux and (optionally) its characteristic speeds.

        ``x`` / ``y`` are the per-component flux expressions; ``waves`` gives the
        per-direction eigenvalues. Lowers to the model's default flux.
        """
        if x is None or y is None:
            raise ValueError("flux(%r) requires per-component x= and y= expressions" % (name,))
        state = self._require_state("flux")
        x_exprs = [self._to_expr(e) for e in x]
        y_exprs = [self._to_expr(e) for e in y]
        self._module.operator(
            name="flux", signature=(state,) >> _model.Rate(state),
            kind="grid_operator", expr={"x": x_exprs, "y": y_exprs})
        if waves is not None:
            self._module.eigenvalues(
                x=[self._to_expr(e) for e in waves["x"]],
                y=[self._to_expr(e) for e in waves["y"]])
        h = FluxHandle(name, is_default=True)
        self._fluxes[name] = h
        return h

    def source(self, name, on=None, value=None):
        """Declare a named local source term; returns a :class:`SourceHandle`."""
        if value is None:
            raise ValueError("source(%r) requires value= (one expression per component)" % (name,))
        reg = _safe_name(name)
        state = self._require_state("source")
        fields = self._ensure_default_fields()
        op = self._module.operator(
            name=reg, signature=(state, fields) >> _model.Rate(state),
            kind="local_source", expr=[self._to_expr(e) for e in value])
        h = SourceHandle(name, reg, operator=op)
        self._sources[reg] = h
        return h

    def local_linear_operator(self, name, on=None, matrix=None):
        """Build a local linear operator ``L: U -> U`` as a MATH object (not a callable
        operator). It carries the matrix; register it with :meth:`operator` (or
        ``@module.operator``) to obtain a callable operator. Calling the math object
        directly raises a clear error -- see :class:`LocalLinearOperatorExpr`."""
        if matrix is None:
            raise ValueError("local_linear_operator(%r) requires matrix=" % (name,))
        return LocalLinearOperatorExpr(name, matrix, on=on)

    def solve_field(self, name, equation=None, outputs=None, solver=None):
        """Declare an elliptic field solve ``-laplacian(phi) == rhs``.

        Lowers to the model's Poisson coupling; ``outputs`` names the produced
        fields, ``solver`` records the required elliptic solver.
        """
        if isinstance(solver, str):
            raise TypeError(
                "solve_field(solver=%r): solver must be a typed pops.solvers.elliptic descriptor "
                "such as GeometricMG() or FFT(), not a string" % solver)
        if not isinstance(equation, _bm.Equation):
            raise TypeError("solve_field expects an equation '-laplacian(phi) == rhs'")
        lhs = equation.lhs
        if not isinstance(lhs, _bm.Laplacian):
            raise ValueError(
                "solve_field left-hand side must be (-)laplacian(field); got %r" % (lhs,))
        rhs = self._to_expr(equation.rhs)
        # -laplacian(phi) == rhs  ->  -Delta phi = rhs (the dsl Poisson convention).
        # laplacian(phi) == rhs  ->  -Delta phi = -rhs.
        if lhs.scale > 0:
            rhs = -rhs
        state = self._require_state("solve_field")
        out_components = tuple((outputs or {"phi": name, "grad_x": None, "grad_y": None}).keys())
        fields = self._module.field_space("fields", out_components)
        self._default_field_space = fields
        self._module.operator(
            name=_safe_name(name), signature=(state,) >> fields,
            kind="field_operator",
            capabilities={"default": _safe_name(name) == "fields_from_state"},
            expr=rhs)
        h = FieldsHandle(name, outputs, solver)
        self._fields[name] = h
        if solver is not None:
            self._field_solvers[name] = solver
        return h

    def field_problem(self, name, equation, outputs=None, solver=None, bcs=None,
                      coefficients=None):
        """Author an inspectable elliptic field problem (Spec 5 sec.5.1 / sec.9.6).

        The typed-object ergonomic shortcut: it CONSTRUCTS and RETURNS an inert
        :class:`pops.fields.PoissonProblem` (or a :class:`pops.fields.FieldProblem` when
        ``coefficients`` are present, e.g. a screened / anisotropic operator) describing the
        solve ``-laplacian(phi) == rhs`` directly from a :class:`pops.math.Equation`, and
        records it on the model's authoring state so :meth:`inspect` surfaces it.

        Unlike :meth:`solve_field`, this method is INERT: it lowers ONLY to an inspectable
        field-problem descriptor; it does NOT touch any legacy model, the elliptic right-hand
        side, the operator graph, codegen or the runtime. Wiring the problem into the operator
        graph (a second elliptic operator + aux channel) is the deeper lowering and stays
        DEFERRED (see :meth:`solve_field` / the multi-elliptic runtime); this entry point only
        produces the typed descriptor a user can ``validate()`` / ``inspect()`` before any run.

        Args:
            name: the field-problem name (also the unknown's display name when not derivable).
            equation: a :class:`pops.math.Equation` of the form ``-laplacian(phi) == rhs``.
            outputs: the produced fields (passed through to the descriptor's ``outputs``).
            solver: the elliptic solver descriptor (carried; ``None`` leaves it unset so the
                descriptor's own ``available`` / ``validate`` flags the missing solver).
            bcs: an iterable of field boundary-condition descriptors (``pops.fields.bcs``).
            coefficients: an optional operator coefficient; when present the descriptor is a
                general :class:`pops.fields.FieldProblem` rather than a ``PoissonProblem``.
        """
        from pops import fields as _fields  # lazy: keep the module import-graph numpy-free.

        if not isinstance(equation, _bm.Equation):
            raise TypeError(
                "field_problem(%r) expects a pops.math.Equation '-laplacian(phi) == rhs'; got %r"
                % (name, type(equation).__name__))
        unknown = equation.lhs.field if isinstance(equation.lhs, _bm.Laplacian) else name
        cls = _fields.FieldProblem if coefficients is not None else _fields.PoissonProblem
        problem = cls(name=name, unknown=unknown, equation=equation,
                      coefficients=coefficients, bcs=tuple(bcs or ()), outputs=outputs,
                      solver=solver)
        self._field_problems[name] = problem
        return problem

    def inspect(self):
        """A plain-dict, inert view of the model's authoring state (Spec 5 sec.12.1).

        Reports the declared state / field / flux / source / operator names and the inspectable
        field problems authored via :meth:`field_problem` (each as its descriptor's
        :meth:`~pops.fields.FieldProblem.inspect` dict). Read-only: it touches no numerics,
        codegen or runtime.
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

    def rate(self, name, equation):
        """Declare a rate operator from ``ddt(U) == -div(F) + sources``."""
        if not isinstance(equation, _bm.Equation):
            raise TypeError("rate expects an equation 'ddt(U) == -div(F) + sources'")
        if not isinstance(equation.lhs, _bm.TimeDerivative):
            raise ValueError("rate left-hand side must be ddt(U) / rate(U)")
        flux, sources = self._destructure_rate(equation.rhs)
        state = self._require_state("rate")
        typed_sources = []
        for src in sources:
            handle = self._sources.get(src)
            typed_sources.append(handle.operator if handle is not None else src)
        fields = self._default_field_space
        inputs = (state, fields) if fields is not None else (state,)
        src_names = [s.name if hasattr(s, "name") else s for s in typed_sources]
        self._module.operator(
            name=_safe_name(name),
            signature=_model.Signature(inputs, _model.Rate(state)),
            kind="local_rate",
            lowering={"flux": flux, "sources": src_names, "fluxes": None},
            expr={})
        return CallableOperator(_safe_name(name), self)

    def finite_volume_rate(self, name, flux=None, riemann=None, reconstruction=None,
                           sources=()):
        """Declare a finite-volume rate from typed Riemann/reconstruction descriptors."""
        self._reconstruction = reconstruction
        # Selecting a Riemann solver validates the model's capabilities for it and enables
        # the role-derived hooks (criterion 10). Spec 5: behaviour choices must be typed.
        if riemann is not None:
            if isinstance(riemann, str):
                reject_string_selector(riemann, "riemann", suggestion="HLL() / HLLC() / Roe()")
            scheme = getattr(riemann, "scheme", None) or getattr(riemann, "name", None)
            if scheme is None:
                raise TypeError("finite_volume_rate(riemann=) expects a typed Riemann descriptor; "
                                "got %r" % (type(riemann).__name__,))
            self.riemann(riemann)
        src_names = [s.reg_name if isinstance(s, SourceHandle) else _safe_name(s)
                     for s in sources]
        # A finite-volume rate always assembles -div F; the flux selection is recorded
        # for the native bricks (riemann/reconstruction), not toggled off here.
        state = self._require_state("finite_volume_rate")
        typed_sources = []
        for src in sources:
            if isinstance(src, SourceHandle):
                typed_sources.append(src.operator)
            else:
                handle = self._sources.get(_safe_name(src))
                typed_sources.append(handle.operator if handle is not None else src)
        fields = self._default_field_space
        inputs = (state, fields) if fields is not None else (state,)
        src_names = [s.name if hasattr(s, "name") else s for s in typed_sources]
        self._module.operator(
            name=_safe_name(name),
            signature=_model.Signature(inputs, _model.Rate(state)),
            kind="local_rate",
            lowering={"flux": True, "sources": src_names, "fluxes": None},
            expr={})
        return name

    def operator(self, name, handle=None, *, inputs=None, returns=None):
        """Register a typed, callable operator under ``name`` from a math object.

        ``returns`` (or the positional ``handle``) is the operator body; ``inputs`` names
        its field dependencies (metadata for requirements). A
        :class:`LocalLinearOperatorExpr` registers as a ``local_linear_operator``
        ``Fields -> LocalLinearOperator(U, U)``. Returns a :class:`CallableOperator`.
        """
        obj = returns if returns is not None else handle
        if obj is None:
            raise TypeError("operator(%r) requires returns= (or a positional handle)" % (name,))
        reg = _safe_name(name)
        if isinstance(obj, LocalLinearOperatorExpr):
            state = self._require_state("operator")
            fields = self._ensure_default_fields()
            self._module.operator(
                name=reg, signature=(fields,) >> _model.LocalLinearOperator(state, state),
                kind="local_linear_operator",
                expr=[[self._to_expr(e) for e in row] for row in obj.matrix])
            self._operators[reg] = obj
            self._operator_inputs[reg] = tuple(inputs) if inputs else ()
            return CallableOperator(reg, self)
        if isinstance(obj, CallableOperator):
            # aliasing an already-registered operator under a new role name
            self._aliases[name] = obj.reg_name
            return obj
        raise TypeError(
            "operator(%r): returns= must be a local_linear_operator object or a "
            "registered operator; got %r" % (name, obj))

    def riemann(self, name, flux=None, pressure=None, velocity=None, sound_speed=None,
                wave_speeds=None, contact_speed=None, star_state=None):
        """Select a typed Riemann solver and validate required model capabilities."""
        if isinstance(name, str):
            reject_string_selector(name, "riemann", suggestion="HLL() / HLLC() / Roe()")
        scheme = getattr(name, "scheme", None) or getattr(name, "name", None)
        if scheme is None:
            raise TypeError("riemann(...) expects a typed Riemann descriptor; got %r"
                            % (type(name).__name__,))
        self._riemann = name
        hooks = {}
        if pressure is not None:
            hooks["pressure"] = self._to_expr(pressure)
        for hook_name, hook_value in (("contact_speed", contact_speed), ("star_state", star_state)):
            if hook_value is not None and hasattr(hook_value, "deps"):
                raise NotImplementedError(
                    "riemann(%s=...): arbitrary formulas for two-state HLLC hooks are not "
                    "defined in the Module-native API; pass a typed capability descriptor instead"
                    % hook_name)
        self._riemann_hooks = {
            "flux": flux, "pressure": pressure, "velocity": velocity,
            "sound_speed": sound_speed, "wave_speeds": wave_speeds,
            "contact_speed": contact_speed, "star_state": star_state,
        }
        kind = str(scheme).lower()
        self._validate_riemann_capabilities(kind, pressure, wave_speeds)
        self._module.riemann_metadata(
            hllc=(kind == "hllc"),
            roe=(kind == "roe"),
            hooks=hooks,
            wave_speeds=wave_speeds,
        )
        return name

    def invariant(self, name, expression=None, over=None):
        """Declare a generic invariant ``StateSet -> Scalar`` from an ``integral(...)``."""
        inv = Invariant(name, expression, over=over)
        self._invariants[inv.name] = inv
        return inv

    def invariants(self):
        """The declared invariants, by name."""
        return dict(self._invariants)

    # --- validation / compile ---
    def check(self):
        """Validate that every referenced quantity is declared (single-species path).

        Multi-species models compose their blocks in a time Program and validate at emit
        (``P.emit_cpp_program`` / ``P._check_lowerable``), so a model-level ``check`` is a
        single-species notion; it is a no-op for a multi-species model."""
        if self._multi_module is not None:
            return None
        return None

    def lower(self):
        """Lower this writing facade to its :class:`pops.model.Module` (Spec 5 sec.11).

        ``pops.physics.Model`` is an AUTHORING facade: it writes the physics (state, primitives,
        flux, sources, field solves) and lowers to the operator-first IR. It does NOT compile.
        The documented flow is::

            physics_model = pops.physics.Model(...)
            model = physics_model.lower()              # -> pops.model.Module
            compiled = pops.compile_problem(model=model, time=program, backend=pops.codegen.Production())

        Single-species: the Module authored directly by this facade (one StateSpace). Multi-species: the multi-block
        Module this model assembled directly (N StateSpaces + ``coupled_rate`` + a multi-block
        field operator). Identical to :pyattr:`module`; ``lower`` is the Spec 5 sec.11 name."""
        return self.module

    # Spec 5 sec.11 alias: physics.Model.to_module() == physics.Model.lower(). Both return the
    # pops.model.Module that pops.compile / pops.compile_problem(model=...) accepts.
    to_module = lower

    def __repr__(self):
        return "physics.Model(%r)" % (self.name,)
