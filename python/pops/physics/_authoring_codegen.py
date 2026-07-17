"""Authoring mixin: thin codegen wrappers (lazy, codegen-free at import).

Every method here delegates to a free function in :mod:`pops.codegen`; the
codegen module is imported LAZILY inside each method body so that importing
``pops.physics`` never pulls in :mod:`pops.codegen` or ``_pops`` (Spec-4
import-graph rule). This is the same delegation the historical ``dsl.py`` used.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .aux import roles_for

if TYPE_CHECKING:
    from ._model_contract import _HyperbolicModel
else:
    _HyperbolicModel = object


def _cg_compile() -> Any:
    """The :mod:`pops.codegen._compile` module (lazy import; keeps physics codegen-free)."""
    from pops.codegen import _compile as _cg
    return _cg


class _CodegenMixin(_HyperbolicModel):
    """C++ emission and compilation wrappers, all delegating lazily to pops.codegen."""

    def _codegen_exprs(self, exprs: Any, cse: Any, real: str = "pops::Real", indent: str = "    ") -> Any:
        from pops.codegen import module_codegen as _cg
        return _cg._codegen_exprs(self, exprs, cse, real=real, indent=indent)

    def _live_prims(self, exprs: Any, seed: Any = ()) -> Any:
        from pops.codegen import module_codegen as _cg
        return _cg._live_prims(self, exprs, seed=seed)

    def _prim_block(self, live: Any = None, hoist: bool = False) -> Any:
        from pops.codegen import module_codegen as _cg
        return _cg._prim_block(self, live=live, hoist=hoist)

    def _jac_entries(self) -> Any:
        from pops.codegen import module_codegen as _cg
        return _cg._jac_entries(self)

    def emit_cpp(self, func: Any = None, cse: bool = True) -> Any:
        """Generates a compilable C++ function computing the physical flux from the symbolic
        tree (each Expr node knows how to write itself in C++ via to_cpp).

        Produced signature : template <class Real> void <func>_flux(const Real* U, Real* F, int dir).
        Constants inlined ; each primitive becomes a local variable. cse=True (default) factors
        the common subexpressions (H, c...) into ``cseK_`` locals ; cse=False recomputes them inline.

        Step (2) of the DSL (see docs/ARCHITECTURE_CIBLE.md sect. 3) : HOST C++ (templatable on Real)."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp(self, func=func, cse=cse)

    def emit_cpp_brick(self, name: Any = None, namespace: str = "pops_generated", cse: bool = True,
                       hoist_reciprocals: bool = False) -> Any:
        """Generates a C++ BRICK satisfying the pops::HyperbolicModel concept (wrapping : step
        2bis). The produced struct uses StateVec / Aux / POPS_HD / Variables and exposes flux,
        max_wave_speed, to_primitive, to_conservative, conservative_vars, primitive_vars : it can
        therefore enter a CompositeModel and run in the compiled solver.

        Requires set_primitive_state(...) (Prim layout) and set_conservative_from([...]) (to_conservative,
        which the DSL cannot invert on its own). cse=True (default) factors the common
        subexpressions (H, c...) into ``cseK_`` locals. The emitted brick is consumed by the native
        production package compiler."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_brick(self, name=name, namespace=namespace, cse=cse,
                                  hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_source(self, name: Any = None, namespace: str = "pops_generated", cse: bool = True,
                        hoist_reciprocals: bool = False) -> Any:
        """Generate a composable C++ SOURCE BRICK (in the pops sense) from self._source.

        The produced struct exposes apply(U, a) returning the source term S(U, aux), with one line per
        conservative component (S[i] = self._source[i].to_cpp()). It has the same form as the source
        bricks written by hand (NoSource, PotentialForce in pops/model/bricks.hpp) and can therefore
        enter as the Source parameter of a CompositeModel.

        CONVENTION: the auxiliary names (set via aux(...)) must be FIELDS of pops::Aux,
        because they are read directly as a.<name> (e.g. aux('grad_x') -> a.grad_x, aux('grad_y') ->
        a.grad_y). This convention is the same as that of the manual bricks, where the source reads
        the outer state only through the pops::Aux channel (potential and its gradient).

        Style identical to emit_cpp_brick (inlined constants, cons -> locals, primitives -> locals;
        plus, aux -> locals); cse=True factors the common sub-expressions. Raises ValueError if
        set_source(...) has not been called."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_source(self, name=name, namespace=namespace, cse=cse,
                                   hoist_reciprocals=hoist_reciprocals)

    def _emit_bricks(self, name: Any = None, hoist_reciprocals: bool = False) -> Any:
        """Generate the bricks (hyperbolic + source + elliptic) and the CompositeModel<...> type
        consumed by the native package emitter. Source / elliptic OPTIONAL: without
        set_source -> pops::NoSource; without set_elliptic_rhs -> zero rhs (no Poisson coupling).
        @p hoist_reciprocals: codegen option propagated to the bricks (cf. emit_cpp_brick).
        Returns (nv, bricks_code, composite_type)."""
        from pops.codegen import module_codegen as _cg
        return _cg._emit_bricks(self, name=name, hoist_reciprocals=hoist_reciprocals)

    def _elliptic_field_registrations(self, nm: Any) -> Any:
        """Per named elliptic field (ADC-428): (field, brick_struct, phi_comp, gx_comp, gy_comp) for the
        native loader. The aux component of each output name is its channel index: a CANONICAL name
        (phi/grad_x/...) maps via AUX_CANONICAL; a model-named aux (aux_field) maps to
        AUX_NAMED_BASE + its position in aux_extra_names. A name the model never declared as an aux is
        rejected (the solve would write a component no source can read). gx/gy default to -1 (phi only)
        when the field lists fewer than 3 aux names."""
        from pops.codegen import module_codegen as _cg
        return _cg._elliptic_field_registrations(self, nm)

    def _emit_metadata(self, model_alias: Any) -> Any:
        """Metadata symbols of the native package, read before installation. Names and roles
        are always emitted (POPS_EXPORT_BLOCK_METADATA):
        they come from the model's VariableSet (single source of truth), the System reads them instead of
        the u0.. fallback / no roles. The GAMMA is emitted (POPS_EXPORT_BLOCK_GAMMA) only if set_gamma(...)
        has been called; otherwise no gamma symbol -> the System keeps its default 1.4 (backward-compat).

        @p model_alias must be an alias WITHOUT a top-level comma (the preprocessor splits
        macro arguments on commas): callers pass a `using ... = CompositeModel<...>`."""
        from pops.codegen import module_codegen as _cg
        return _cg._emit_metadata(self, model_alias)

    def emit_cpp_native_loader(self, name: Any = None, target: str = "system",
                               hoist_reciprocals: bool = False) -> Any:
        """Thin wrapper: delegates to pops.codegen._compile.emit_cpp_native_loader."""
        return _cg_compile().emit_cpp_native_loader(self, name=name, target=target,
                                          hoist_reciprocals=hoist_reciprocals)

    def compile_native(self, so_path: Any, include: Any = None, name: Any = None, cxx: Any = None,
                       std: str = "c++23", target: str = "system",
                       hoist_reciprocals: bool = False) -> Any:
        """Thin wrapper: delegates to pops.codegen._compile.compile_native."""
        return _cg_compile().compile_native(self, so_path, include=include, name=name, cxx=cxx, std=std,
                                  target=target, hoist_reciprocals=hoist_reciprocals)

    def _model_hash(self, params: Any = None) -> Any:
        """Stable hash of the model; delegates to pops.codegen._compile.model_hash."""
        return _cg_compile().model_hash(self, params=params)

    def _check_require_metadata(self, require_metadata: Any, backend: Any) -> None:
        """require_metadata guard rails (pure-Python, deterministic on the model + backend). Factored out
        to be called BEFORE the cache (in HyperbolicModel AND Model): a cache HIT must never
        mask a metadata requirement. Without require_metadata, no-op."""
        if not require_metadata:
            return
        missing = []
        roles = roles_for(self.cons_names, self.cons_roles)
        if all(r == "Custom" for r in roles):
            missing.append("physical roles (conservative_vars(..., roles=[...]) or canonical names)")
        if self.gamma is None:
            missing.append("gamma (set_gamma(...))")
        if missing:
            raise ValueError(
                "compile(require_metadata=True): model '%s' does not provide %s; the .so "
                "would fall back to the System fallback (roles 'custom' / gamma 1.4)"
                % (self.name, " nor ".join(missing)))

    def compile(self, so_path: Any = None, include: Any = None, backend: str = "production", name: Any = None,
                cxx: Any = None, std: Any = None, require_metadata: bool = False, target: str = "system",
                hoist_reciprocals: bool = False) -> Any:
        """Thin wrapper: delegates to pops.codegen._compile.compile_model."""
        return _cg_compile().compile_model(self, so_path=so_path, include=include, backend=backend,
                                 name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_elliptic(self, name: Any = None, namespace: str = "pops_generated", cse: bool = True,
                          hoist_reciprocals: bool = False) -> Any:
        """Generates a composable elliptic RIGHT-HAND SIDE BRICK from self._elliptic.

        The produced struct exposes rhs(U) -> Real (charge density, background, gravity...), same shape as
        the manual bricks (ChargeDensity, BackgroundDensity in pops/model/bricks.hpp): it enters
        as the Elliptic parameter of a CompositeModel. Inlined constants, cons/primitives -> locals,
        cse=True factors out common sub-expressions. ValueError if set_elliptic_rhs(...) is missing."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_elliptic(self, name=name, namespace=namespace, cse=cse,
                                     hoist_reciprocals=hoist_reciprocals)

    def emit_cpp_elliptic_field(self, field: Any, struct_name: Any, namespace: str = "pops_generated",
                                hoist_reciprocals: bool = False, cse: bool = True) -> Any:
        """Generates a SELF-CONTAINED elliptic RHS brick for the NAMED field @p field (ADC-428).

        Unlike emit_cpp_elliptic (which emits only ``rhs(U)``, consumed by CompositeModel), this brick
        is shaped like a minimal Model so the runtime can pair it with pops::make_poisson_rhs directly:
        it declares ``n_vars`` + ``State`` (so load_state<Brick> reads the conservative state) and
        exposes ``elliptic_rhs(State)`` (what detail::PoissonRhs<Brick> calls per cell). The native
        loader builds one std::function per named field via make_poisson_rhs(Brick{}) and attaches it to
        the block (System::set_block_elliptic_field). The RHS reads ONLY the conservative state (+
        primitives), never the aux (enforced at declaration). Reuses _codegen_exprs / _prim_block so the
        formula lowers IDENTICALLY to the default elliptic brick."""
        from pops.codegen import module_codegen as _cg
        return _cg.emit_cpp_elliptic_field(self, field, struct_name, namespace=namespace,
                                           hoist_reciprocals=hoist_reciprocals, cse=cse)
