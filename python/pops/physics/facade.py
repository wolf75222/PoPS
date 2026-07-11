"""Stable PDE facade composing a private :class:`HyperbolicModel`.

It delegates existing numerics and imports codegen lazily to preserve the Spec-4 import graph.
"""
from __future__ import annotations

from typing import Any

from pops.ir import Var  # noqa: F401  -- primitive_vars self-reference check
from pops.ir.ops import left, right  # noqa: F401  -- Model.left / Model.right sugar

from ._modelpkg import model as _model
from .aux import aux_total_n_aux, roles_for  # noqa: F401  -- used in Model.compile
from .model import HyperbolicModel
from ._facade_compile import _FacadeCompileMixin
from ._freeze import PhysicsFreezable


class Model(PhysicsFreezable, _FacadeCompileMixin):
    """Stable facade delegating authoring to a composed ``HyperbolicModel``.

    It separates symbolic declarations (for example ``flux``) from host evaluators such as
    ``eval_flux`` and exposes the compiled model through the lazy codegen mixin.
    """

    _physics_mutators = frozenset({
        "conservative_vars", "primitive", "primitive_vars", "aux", "aux_field",
        "conservative_from", "flux", "flux_term", "eigenvalues", "wave_speeds",
        "wave_speeds_from_jacobian", "stability_speed", "stability_dt", "source",
        "source_term", "linear_source", "rate_operator", "rate", "field_solve",
        "local_linear_map", "source_frequency", "source_jacobian", "projection",
        "implicit_source", "enable_hllc", "set_riemann_hooks", "enable_roe",
        "roe_dissipation", "roe_from_jacobian", "elliptic_rhs", "elliptic_field",
        "gamma", "param",
    })

    def __init__(self, name: Any) -> None:
        self._init_physics_freeze()
        self._m = HyperbolicModel(name)
        self._param_registry = _model.ParamRegistry(
            owner=self._m.owner_path, mutation_guard=self._guard_mutable)
        self._module_cache = None

    def _invalidate_authoring_views(self) -> None: self._module_cache = None
    @property
    def name(self) -> Any: return self._m.name
    @property
    def owner_path(self) -> Any: return self._m.owner_path
    def declaration_index(self) -> Any: return self.module.declaration_index()

    @property
    def capabilities(self) -> Any:
        """Typed capability handles; missing capabilities fail explicitly on access."""
        from pops.numerics.riemann.waves import _CapabilityHandles  # lazy: keep facade lean
        return _CapabilityHandles(self)

    # --- variable declaration (direct delegation to HyperbolicModel) ---
    def conservative_vars(self, *names: Any, roles: Any = None) -> Any:
        """Declares the conservative variables. @p roles: same convention as HyperbolicModel."""
        return self._m.conservative_vars(*names, roles=roles)

    def primitive(self, name: Any, expr: Any) -> Any:
        """Defines a primitive by its formula (as a function of the cons / preceding primitives)."""
        return self._m.primitive(name, expr)

    def primitive_vars(self, *vars: Any, roles: Any = None, **named: Any) -> Any:
        """Declare primitives and their order, from expressions in kwargs or positional names.

        Keyword and positional forms are exclusive; keyword insertion order defines the layout.
        """
        if named and vars:
            raise ValueError("primitive_vars: mixing positional form and named kwargs "
                             "(choose one; kwargs define AND order the primitives)")
        if named:
            # kwargs: define each primitive, then fix the layout in insertion order.
            # A primitive is NOT (re)defined if the kwarg is the Var of the SAME name -- otherwise the codegen
            # would emit `const Real x = x;` (auto-init -> NaN). Two cases of self-reference, both
            # left to JOIN the layout without redefinition (target style primitive_vars(rho=rho, u=u, ...)):
            #  - name ALREADY CONSERVATIVE (e.g. rho=rho: the density, primitive == conservative);
            #  - PRIMITIVE Var ALREADY DEFINED of the same name (e.g. u=u when u comes from m.primitive('u', ...)).
            # Otherwise (kwarg = expression, e.g. p=cs2*rho or u=mx/rho) the primitive is defined normally.
            ordered = list(named.keys())
            for nm in ordered:
                val = named[nm]
                self_ref = isinstance(val, Var) and getattr(val, "name", None) == nm
                if nm in self._m.cons_names or self_ref:
                    continue
                self._m.primitive(nm, val)
            self._m.set_primitive_state(*ordered, roles=roles)
            return tuple(Var(nm, "prim") for nm in ordered)
        # positional form: fixes the layout from already-defined names/Var.
        self._m.set_primitive_state(*vars, roles=roles)
        return None

    def aux(self, name: Any) -> Any:
        """CANONICAL auxiliary field (must be a key of AUX_CANONICAL: phi/grad_x/grad_y/B_z/T_e)."""
        return self._m.aux(name)

    def aux_field(self, name: Any) -> Any:
        """NAMED auxiliary field (ADC-70 phase 1) provided by a block via System.set_aux_field(block, name,
        array). name is ARBITRARY (identifier); the k-th call reserves the aux channel component
        AUX_NAMED_BASE + k (read in C++ via aux.extra_field(k)). At most AUX_NAMED_MAX per model.
        Returns a Var usable in flux / source / eigenvalues. Delegates to HyperbolicModel.aux_field."""
        return self._m.aux_field(name)

    def conservative_from(self, exprs: Any) -> None:
        """Inverse prim -> cons (the DSL cannot invert symbolically)."""
        self._m.set_conservative_from(exprs)

    # --- flux: symbolic DECLARATOR vs numpy EVALUATOR (DISTINCT names, settled decision) ---
    def flux(self, x: Any, y: Any) -> None:
        """Symbolic DECLARATOR of the physical flux (delegates to set_flux). x/y: lists of Expr, one
        per conservative component. DO NOT confuse with the numpy evaluator eval_flux."""
        self._m.set_flux(x, y)

    def flux_term(self, name: Any, x: Any, y: Any) -> None:
        """NAMED physical flux F_name(U, primitives, aux, params): exactly n_cons expressions per
        direction (delegates to HyperbolicModel.flux_term). Opt-in -- emitted only when a compiled time
        Program selects it (ctx.rhs(..., fluxes=[name, ...])), never folded into the historical -div F.
        name='default' is the backward-compatible alias of m.flux(...): ctx.rhs(fluxes=['default']) is
        byte-identical to the historical flux-only RHS. A Program requesting several named fluxes
        assembles -div of their SUM."""
        self._m.flux_term(name, x, y)

    def eval_flux(self, U: Any, aux: Any, dir: Any) -> Any:
        """numpy EVALUATOR of the physical flux (debug / host proto; delegates to HyperbolicModel.flux).
        U: numpy (n_vars, ...); aux: dict name -> array; dir: 0=x, 1=y."""
        return self._m.flux(U, aux, dir)

    def eval_source(self, U: Any, aux: Any) -> Any:
        """numpy EVALUATOR of the source term (debug / host proto; delegates to
        HyperbolicModel.source_value). U: numpy (n_vars, ...); aux: dict name -> array. Returns
        zeros when no source was declared. Lets a host test check the emitted source (e.g. a BGK
        collision) against an oracle without compiling."""
        return self._m.source_value(U, aux)

    def eigenvalues(self, x: Any, y: Any) -> None:
        """Eigenvalues (characteristic speeds) per direction (delegates to set_eigenvalues)."""
        self._m.set_eigenvalues(x, y)

    def wave_speeds(self, x: Any, y: Any) -> None:
        """Explicit SIGNED wave speeds per direction: x = (smin_x, smax_x), y = (smin_y,
        smax_y). Emits ``wave_speeds(U, aux, dir, smin, smax)`` on the brick WITHOUT requiring a
        primitive 'p': riemann='hll' becomes available for a model without pressure (moment
        system, isothermal...). Takes priority over the historical path (eigenvalues + 'p'); if
        eigenvalues is not declared, max_wave_speed (Rusanov / CFL) derives from ``max(|smin|, |smax|)``.
        Delegates to set_wave_speeds; cf. HyperbolicModel.set_wave_speeds."""
        self._m.set_wave_speeds(x, y)

    def wave_speeds_from_jacobian(self, x: Any = None, y: Any = None, eig: str = "numeric",
                                  blocks: Any = None, fd_eps: Any = None,
                                  eig_max_iter: Any = None, im_tol: Any = None) -> None:
        """EXACT signed wave speeds from the eigenvalues of the flux jacobian (delegates to
        set_wave_speeds_from_jacobian, see its full contract): x/y = dF/dU as Expr (None =
        AUTODIFF of the declared flux via flux_jacobian); eig = 'numeric' | 'fd' (finite differences
        of the compiled flux, debug); blocks = lists of indices of the diagonal sub-blocks (None = full
        block, the only unconditionally correct mode -- the blocks ASSERT a
        block-triangular structure); fd_eps = the eig='fd' relative FD step (ADC-617; None = 1e-6, and
        it participates in the compile cache key); eig_max_iter / im_tol (ADC-645) = the Francis-QR
        per-eigenvalue iteration cap of the emitted real_eig_minmax and the imaginary-part tolerance
        of the Roe |A| real-spectrum gate (None = the native defaults, byte-identical emission; both
        participate in the compile cache key when set)."""
        self._m.set_wave_speeds_from_jacobian(x=x, y=y, eig=eig, blocks=blocks, fd_eps=fd_eps,
                                              eig_max_iter=eig_max_iter, im_tol=im_tol)

    def eval_wave_speeds(self, U: Any, aux: Any, dir: Any) -> Any:
        """numpy EVALUATOR of the emitted signed speeds (smin, smax) (delegates to
        HyperbolicModel.wave_speeds_value): explicit pair, jacobian (numpy eig per blocks) or
        min/max of the eigenvalues."""
        return self._m.wave_speeds_value(U, aux, dir)

    def stability_speed(self, expr: Any) -> None:
        """STABILITY speed lambda* driving the block CFL (OPTIONAL; delegates to
        HyperbolicModel.stability_speed). Fallback without a call: max(abs(eigenvalues)), strictly
        the historical behavior. Compiled like flux/source (production GPU/MPI, no per-cell callback)."""
        self._m.stability_speed(expr)

    def stability_dt(self, expr_dt: Any) -> None:
        """Direct ADMISSIBLE step dt(U, aux) local bound of the step (OPTIONAL; delegates to
        HyperbolicModel.stability_dt). The cfl is not applied to this bound. Fallback without a call:
        no additional bound (historical step policy)."""
        self._m.stability_dt(expr_dt)

    def source(self, s: Any) -> None:
        """Source term S(U, aux), one expression per component (optional; delegates to set_source)."""
        self._m.set_source(s)

    def source_term(self, name: Any, exprs: Any) -> Any:
        """NAMED local source S_name(U, primitives, aux, params): exactly n_cons expressions
        (delegates to HyperbolicModel.source_term). Opt-in -- emitted only when a compiled time
        Program requests it (ctx.rhs(..., sources=[name]) / ctx.source(name)), never summed
        implicitly. name='default' is the backward-compatible alias of m.source([...]).

        Returns the declared operator's :class:`pops.model.OperatorHandle` (Spec 5 sec.14.2.3),
        which a Program can pass to ``P.call`` in place of the string name."""
        return self._m.source_term(name, exprs)

    def linear_source(self, name: Any, matrix: Any) -> Any:
        """NAMED local linear operator L_name(aux, params) U, an n_cons x n_cons matrix whose
        coefficients are independent of U / primitives (delegates to HyperbolicModel.linear_source).
        Used explicitly by a Program (ctx.linear_source / ctx.apply / ctx.solve_local_linear);
        never folded into m.source or ctx.rhs.

        Returns the declared operator's :class:`pops.model.OperatorHandle` (Spec 5 sec.14.2.3),
        which a Program can pass to ``P.call`` in place of the string name."""
        return self._m.linear_source(name, matrix)

    def rate_operator(self, name: Any, *, flux: bool = True, sources: Any = ("default",),
                      fluxes: Any = None) -> Any:
        """NAMED composite rate operator R_name = -div F + sum(sources) (Spec 2, operator-first):
        a Program-side alias for ctx.rhs(flux=, sources=, fluxes=) so a model-free Program can call
        P.call(name, U[, fields]) instead of P.rhs(...). Delegates to HyperbolicModel.rate_operator.

        Returns the declared operator's :class:`pops.model.OperatorHandle` (Spec 5 sec.14.2.3),
        which a Program can pass to ``P.call`` in place of the string name."""
        return self._m.rate_operator(name, flux=flux, sources=sources, fluxes=fluxes)

    # --- mathematically explicit operator declarers (ADC-559) ------------------------------
    # `m.rate` / `m.field_solve` / `m.local_linear_map` are thin MATHEMATICAL aliases: each funnels
    # into ONE of the existing typed declarers (rate_operator / elliptic_field / linear_source) so the
    # registry, the module hash and the codegen are UNCHANGED (no parallel facade registry), and each
    # returns the canonical `pops.model.OperatorHandle` -- re-stamped with the operator's derived
    # `Signature` and mathematical `category` so a handle reads as the math object it names:
    #   m.rate(...)             ->  (U, Fields) -> Rate(U)                  category "rate"
    #   m.field_solve(...)      ->  U -> Fields                            category "field_solve"
    #   m.local_linear_map(...) ->  Fields -> LocalLinearOperator(U, U)    category "local_linear_map"
    def _typed_handle(self, name: Any) -> Any:
        """Re-stamp the derived Signature + category onto operator ``name``'s canonical handle.

        Reads the operator's typed :class:`pops.model.Operator` from the DERIVED registry (the single
        source of the signature; see :meth:`operator_registry`) and returns an inspectable
        :class:`pops.model.OperatorHandle` carrying ``(name, kind, signature, category)``. Pure view:
        the operator was already registered by the underlying declarer; this only enriches the handle.
        """
        from pops.model import OperatorHandle
        op = self._m.operator_registry().get(name)
        return OperatorHandle(
            op.name, kind=op.kind, owner=self._m.owner_path, signature=op.signature)

    def rate(self, name: Any, *, flux: bool = True, sources: Any = ("default",),
             fluxes: Any = None) -> Any:
        """Declare a RATE operator ``R: (U, Fields) -> Rate(U)`` (ADC-559): the mathematical spelling of
        :meth:`rate_operator`. ``R = -div F + sum(sources)``; the returned inspectable handle carries
        the derived ``Signature`` and ``category == "rate"``. Funnels into the ONE typed registry (no
        parallel surface); ``P.call(R, U[, fields])`` / ``R(U[, fields])`` (ADC-560) lower identically.
        """
        self._m.rate_operator(name, flux=flux, sources=sources, fluxes=fluxes)
        return self._typed_handle(name)

    def field_solve(self, name: Any, rhs: Any, *, operator: str = "poisson", aux: Any = None) -> Any:
        """Declare a FIELD-SOLVE operator ``F: U -> Fields`` (ADC-559): the mathematical spelling of
        :meth:`elliptic_field`. Solves ``operator(field) = rhs(U)`` populating the named ``aux`` fields
        (default the electrostatic triple); the returned inspectable handle carries the derived
        ``Signature`` and ``category == "field_solve"``. The multi-field RUNTIME stays deferred (see
        :meth:`elliptic_field`); the IR / validation / typed handle land here.
        """
        self._m.elliptic_field(name, rhs, operator=operator, aux=aux)
        return self._typed_handle(name)

    def local_linear_map(self, name: Any, matrix: Any) -> Any:
        """Declare a LOCAL LINEAR MAP ``L: Fields -> LocalLinearOperator(U, U)`` (ADC-559): the
        mathematical spelling of :meth:`linear_source`. ``L`` is an ``n_cons x n_cons`` matrix whose
        coefficients read only aux/params (never U/primitives), so ``S = L U`` is linear in ``U``; the
        returned inspectable handle carries the derived ``Signature`` and ``category ==
        "local_linear_map"``. Used explicitly by a Program (``P.apply`` / ``solve_local_linear``),
        never folded into ``m.source`` / ``P.rhs``.
        """
        self._m.linear_source(name, matrix)
        return self._typed_handle(name)

    def source_frequency(self, expr_mu: Any) -> None:
        """Local frequency mu(U, aux) [1/s] of the source -- the 'source' step bound from the meeting
        (dt <= cfl*substeps/(stride*max mu), without a space step). Emitted on the generated SOURCE
        brick (frequency(U, aux)), forwarded by CompositeModel, aggregated by System/AmrSystem
        step_cfl. REQUIRES m.source([...]). Delegates to HyperbolicModel.source_frequency."""
        self._m.source_frequency(expr_mu)

    def source_jacobian(self, rows: Any) -> None:
        """ANALYTIC Jacobian dS/dU of the source (rows[r][c] = dS_r/dU_c, an n_vars x n_vars
        matrix of expressions): the implicit Newton (IMEX/SourceImplicitBE) uses it instead of
        finite differences. REQUIRES m.source. Delegates to HyperbolicModel.source_jacobian."""
        self._m.source_jacobian(rows)

    def projection(self, exprs: Any) -> None:
        """PROJECTION PONCTUELLE post-pas U <- P(U, aux) (ADC-177, OPTIONNEL) : une expression par
        composante conservative, appliquee par le System a la FIN de chaque macro-pas ENTIER (jamais
        par etage RK) sur les cellules valides. CONTRAT : idempotente et ponctuelle ; clamps en
        max/min via abs_/sign. Backends 'aot'/'production' (System) ; 'prototype' et AMR rejetes.
        Delegue a HyperbolicModel.projection (cf. son contrat complet)."""
        self._m.projection(exprs)

    def projection_value(self, U: Any, aux: Any = None) -> Any:
        """EVALUATEUR numpy de la projection emise (reference de test ; delegue a
        HyperbolicModel.projection_value)."""
        return self._m.projection_value(U, aux)

    def implicit_source(self, jacobian: Any = None) -> None:
        """GROUPED declaration of the local implicit (wave 3 audit, sugar): the RESIDUAL is already
        implied by m.source (backward-Euler: F = W - U^n - dt*S(W)); @p jacobian (optional) =
        analytic dS/dU matrix (cf. source_jacobian). Without jacobian: finite differences."""
        if jacobian is not None:
            self._m.source_jacobian(jacobian)

    def enable_hllc(self) -> None:
        """Emits the HLLC capability (contact_speed + hllc_star_state generated from the ROLES +
        primitive 'p'): riemann='hllc' becomes available for this model EVEN outside 4-variable
        Euler. Delegates to HyperbolicModel.enable_hllc."""
        self._m.enable_hllc()

    def set_riemann_hooks(self, **forms: Any) -> Any:
        """Record ARBITRARY-formula overrides of the role-derived Riemann hooks (ADC-456): e.g.
        ``pressure=<Expr>`` codegen's that formula as the ``pressure(U)`` hook body. Descriptors /
        ``None`` keep the role-derived default. Delegates to HyperbolicModel.set_riemann_hooks."""
        self._m.set_riemann_hooks(**forms)
        return self

    def enable_roe(self) -> None:
        """Emits the ROE capability (roe_dissipation = ``|A_roe| dU`` generated from the ROLES +
        primitive 'p'): riemann='roe' becomes available for this model EVEN outside 4-variable
        Euler (without Energy: c = sqrt(p/rho) averaged Roe-style; components outside the fluid
        roles = passive scalars on the entropy wave). Delegates to HyperbolicModel.enable_roe."""
        self._m.enable_roe()

    def roe_dissipation(self, x: Any, y: Any) -> None:
        """Roe dissipation PROVIDED by the user (outside the fluid roles): n_vars expressions per
        direction (x=, y=), written with m.left(...)/m.right(...) (or dsl.left/right) of the two states,
        emitted as the C++ hook roe_dissipation(UL, AL, UR, AR, dir). During the 'provided' mode of enable_roe
        (a single provider: supplying both together raises). The helper m.flux_jacobian assists the writing.
        Delegates to HyperbolicModel.roe_dissipation (cf. its doc)."""
        self._m.roe_dissipation(x, y)

    def flux_jacobian(self, dir: Any) -> Any:
        """Flux Jacobian A = dF_dir/dU (an n_vars x n_vars matrix of Expr, A[i][j]=d(F_i)/d(U_j)),
        auto-derived from the fluxes declared via dsl.diff (expanded primitives). HELPER for building
        m.roe_dissipation, emits nothing. @p dir: 0/'x' or 1/'y'. Delegates to HyperbolicModel."""
        return self._m.flux_jacobian(dir)

    def roe_from_jacobian(self) -> None:
        """Generic moment Roe: emits roe_dissipation = ``|A| (UR-UL)`` with A the flux Jacobian at
        Uavg = 1/2(UL+UR) and |A| via pops::roe_abs_apply (matrix-sign), spectral-radius Rusanov
        fallback on a complex/singular spectrum. Roles-free (no Density/Momentum, no 'p'): makes
        riemann='roe' available for a moment hierarchy. Exclusive with enable_roe / roe_dissipation.
        Delegates to HyperbolicModel.roe_from_jacobian (cf. its doc)."""
        self._m.roe_from_jacobian()

    def left(self, expr: Any) -> Any:
        """Marks @p expr as evaluated on the LEFT state UL (m.roe_dissipation). Sugar for dsl.left."""
        return left(expr)

    def right(self, expr: Any) -> Any:
        """Marks @p expr as evaluated on the RIGHT state UR (m.roe_dissipation). Sugar for dsl.right."""
        return right(expr)

    def elliptic_rhs(self, e: Any) -> None:
        """Contribution to the elliptic right-hand side (Poisson coupling; delegates to set_elliptic_rhs)."""
        self._m.set_elliptic_rhs(e)

    def elliptic_field(self, name: Any, rhs: Any, operator: str = "poisson", aux: Any = None) -> None:
        """NAMED elliptic field: an elliptic solve operator(field) = rhs(U) populating the named @p aux
        fields (default ['phi', 'grad_x', 'grad_y']); delegates to HyperbolicModel.elliptic_field. The
        IR + validation + hash land; the multi-field RUNTIME (a second elliptic operator + aux channel)
        is DEFERRED -- ctx.solve_fields(field=name) raises NotImplementedError on lowering."""
        self._m.elliptic_field(name, rhs, operator=operator, aux=aux)

    def gamma(self, value: Any) -> None:
        """Adiabatic index (EOS), carried by POPS_EXPORT_BLOCK_GAMMA (delegates to set_gamma)."""
        self._m.set_gamma(value)

    def param(self, declaration: Any) -> Any:
        """Register a canonical parameter declaration and return its handle.

        Only ``pops.params.RuntimeParam`` / ``ConstParam`` / ``DerivedParam``
        are accepted.  The historical ``(name, value)`` shorthand is removed:
        storage lifetime must never be inferred silently.  A handle is identity,
        not an expression; call :meth:`value` when a physics formula reads it.
        """
        from pops.params import ConstParam

        if getattr(declaration, "name", None) == "gamma" and not isinstance(
            declaration, ConstParam
        ):
            raise ValueError(
                "the EOS metadata parameter 'gamma' must be a ConstParam; the native block "
                "exports gamma as compile-time metadata"
            )
        handle = self._param_registry.register(declaration)
        if declaration.name == "gamma":
            self._m.set_gamma(declaration.value)
        self._invalidate_authoring_views()
        return handle

    def value(self, parameter: Any) -> Any:
        """Return the distinct symbolic Expr read of a registered ParamHandle."""
        from pops.ir import parameter_value

        return parameter_value(self._param_registry, parameter)

    @property
    def params(self) -> Any:
        """Canonical declarations keyed by local id; immutable after model freeze."""
        declarations = self._param_registry.declarations()
        if self.frozen:
            from types import MappingProxyType

            return MappingProxyType(declarations)
        return declarations

    def check(self) -> Any:
        """Checks the dependencies (referenced variables are declared). Raises ValueError otherwise."""
        return self._m.check()

    def check_model(self, samples: Any = None, n_samples: int = 64, seed: int = 0, aux: Any = None,
                    rtol: float = 1e-8, atol: float = 1e-10, raise_on_error: bool = True,
                    jac_rtol: float = 1e-3, jac_atol: float = 1e-9) -> Any:
        """Generic NUMERICAL verification of the model (finite flux/source/elliptic, real and finite
        eigenvalues, wave_speeds/max_wave_speed consistency, non-circular bounding of the spectrum by
        the dense Jacobian, cons<->prim round-trip, positivity of Density/'p') on sample states.
        Delegates to HyperbolicModel.check_model (cf. its doc). To be called BEFORE compile(); the
        runtime counterpart of an installed block is System.check_model."""
        return self._m.check_model(samples=samples, n_samples=n_samples, seed=seed, aux=aux,
                                   rtol=rtol, atol=atol, raise_on_error=raise_on_error,
                                   jac_rtol=jac_rtol, jac_atol=jac_atol)

    # --- introspection (read-only, delegated to the backing model) ---
    @property
    def cons_names(self) -> Any: return self._m.cons_names

    @property
    def prim_state(self) -> Any: return self._m.prim_state

    @property
    def n_vars(self) -> Any: return self._m.n_vars

    def state_space(self, name: str = "U") -> Any:
        """Typed pops.model.StateSpace view of the conservative state (Spec 2).
        Delegates to HyperbolicModel.state_space."""
        return self._m.state_space(name)

    def field_space(self, name: str = "fields") -> Any:
        """Typed pops.model.FieldSpace view of the auxiliary surface (Spec 2).
        Delegates to HyperbolicModel.field_space."""
        return self._m.field_space(name)

    def operator_registry(self, state_name: str = "U") -> Any:
        """Typed pops.model.OperatorRegistry derived from this model (Spec 2): the PDE
        shortcuts (source_term / linear_source / elliptic_field / flux) lower into
        typed operators. Pure view -- no hash or codegen impact. Delegates to
        HyperbolicModel.operator_registry."""
        return self._m.operator_registry(state_name)

    @property
    def module(self) -> Any:
        """The pops.model.Module view of this PDE model (Spec 2, operator-first): its typed
        StateSpace / FieldSpace and the OperatorRegistry that source_term / linear_source /
        elliptic_field / flux / rate_operator populate. dsl.Model is the PDE convenience
        facade; the Module is the model-free view a generic Program binds to (P.bind_operators).
        The Module carries no numerics; codegen still reads this Model via compile_problem."""
        if self._module_cache is not None:
            return self._module_cache
        mod = _model.Module(self.name, owner=self._m.owner_path)
        st = self._m.state_space()
        mod.state_space(
            st.name, st.components, roles=st.roles, layout=st.layout, storage=st.storage,
            representation=st.representation, centering=st.centering, units=st.units,
            frame=st.frame, clock=st.clock)
        fs = self._m.field_space()
        mod.field_space(
            fs.name, fs.components, layout=fs.layout, representation=fs.representation,
            centering=fs.centering, units=fs.units, frame=fs.frame, clock=fs.clock)
        # ``module`` is a typed view of this exact model definition, not another
        # declaration owner. Share the one registry so handles never acquire a
        # parallel authority with merely equal-looking IDs.
        mod._param_registry = self._param_registry
        mod.adopt_registry(self._m.operator_registry())
        self._module_cache = mod
        return mod
    # --- operator introspection (Spec 2, S2-5) ---
    def list_operators(self) -> Any:
        """Names of the typed operators this model exposes (registration order)."""
        return self._m.operator_registry().names()

    def list_state_spaces(self) -> Any:
        """Names of the model's state spaces (one, the conservative state)."""
        return [self._m.state_space().name]

    def list_field_spaces(self) -> Any:
        """Names of the model's field spaces (one, the auxiliary surface)."""
        return [self._m.field_space().name]

    def operator_signature(self, name: Any) -> Any:
        """The pops.model.Signature of operator ``name``."""
        return self._m.operator_registry().get(name).signature

    def operator_requirements(self, name: Any) -> Any:
        """The requirements dict of operator ``name``."""
        return dict(self._m.operator_registry().get(name).requirements)

    def operator_capabilities(self, name: Any) -> Any:
        """The capabilities dict of operator ``name``."""
        return dict(self._m.operator_registry().get(name).capabilities)
