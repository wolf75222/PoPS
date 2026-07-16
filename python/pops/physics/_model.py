"""HyperbolicModel: the symbolic mini-DSL core.

Python WRITES the formulas (named variables, expressions); the operations build
an expression TREE. :class:`HyperbolicModel` declares the conservative variables,
the primitives (defined by formulas), the flux, the eigenvalues, the source and
the elliptic contribution.

The 1500-line ``HyperbolicModel`` is assembled here from topical authoring mixins
so no file exceeds the Spec-4 500-line bound (see ``physics/_authoring_*``):
variables, flux, sources, Riemann, the operator-first view, the numpy evaluators,
the runtime parameters, and the codegen wrappers. The stable user facade
:class:`pops.physics._facade.Model` COMPOSES a private ``HyperbolicModel``.

Import-graph rule (Spec 4): this module imports only :mod:`pops._ir`; any
:mod:`pops.codegen` / ``_pops`` use is LAZY, inside method bodies (the codegen
wrappers in ``_authoring_codegen``).
"""
from __future__ import annotations

from typing import Any

from pops.model.ownership import OwnerKind, OwnerPath

from ._authoring_vars import _VariablesMixin
from ._authoring_flux import _FluxMixin
from ._authoring_sources import _SourceMixin
from ._authoring_riemann import _RiemannMixin
from ._authoring_view import _OperatorViewMixin
from ._authoring_eval import _EvalMixin
from ._authoring_params import _RuntimeParamsMixin
from ._authoring_codegen import _CodegenMixin
from ._freeze import PhysicsFreezable


class HyperbolicModel(PhysicsFreezable, _VariablesMixin, _FluxMixin, _SourceMixin, _RiemannMixin,
                      _OperatorViewMixin, _EvalMixin, _RuntimeParamsMixin, _CodegenMixin):
    """Hyperbolic model written as FORMULAS: conservative variables, primitives (defined by
    expressions), flux, eigenvalues, source, elliptic contribution. cf. module docstring.

    The behaviour lives in the topical authoring mixins; this concrete class only assembles
    them and owns ``__init__`` (the full instance-attribute layout the mixins operate on)."""

    _physics_mutators = frozenset({
        "cons", "conservative_vars", "primitive", "aux", "aux_field",
        "set_primitive_state", "set_conservative_from", "set_flux", "set_eigenvalues",
        "flux_term", "set_wave_speeds", "set_wave_speeds_from_jacobian", "set_gamma",
        "set_source", "set_elliptic_rhs", "elliptic_field", "source_term", "linear_source",
        "rate_operator", "stability_speed", "stability_dt", "source_frequency", "projection",
        "source_jacobian", "enable_hllc", "set_riemann_hooks", "enable_roe",
        "roe_dissipation", "roe_from_jacobian", "operator_alias",
    })

    def _semantic_data(self) -> dict[str, Any]:
        """Closed low-level semantic protocol for the formula codegen model."""
        return {
            "kind": "hyperbolic-model",
            "owner": self.name,
            "component_digest": self._model_hash(),
        }

    def __pops_artifact_model_metadata__(self) -> dict[str, Any]:
        """Exact low-level report projection used before a formula model is discarded."""
        from pops.physics.aux import aux_total_n_aux

        runtime_params = self.runtime_param_nodes()
        if any(node.handle is None for node in runtime_params):
            raise TypeError(
                "artifact metadata refuses unowned low-level RuntimeParamRef values; "
                "declare typed parameters through Model.param"
            )
        params = {node.name: node.handle for node in runtime_params}
        wave_speed_provider = None
        if self._wave_speeds is not None:
            wave_speed_provider = "explicit_pair"
        elif self._ws_jacobian is not None:
            wave_speed_provider = "jacobian"
        elif "p" in self.prim_defs:
            wave_speed_provider = "pressure_derived"
        return {
            "schema_version": 2,
            "state_spaces": ("U",),
            "cons_names": tuple(self.cons_names),
            "n_vars": self.n_vars,
            "params": params,
            "aux_names": tuple(self.aux_extra_names),
            "n_aux": aux_total_n_aux(self.aux_names, self.aux_extra_names),
            "capabilities": {},
            "wave_speed_provider": wave_speed_provider,
        }

    def __init__(self, name: Any) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("HyperbolicModel: name must be a non-empty string")
        self._init_physics_freeze()
        self.name = name
        self._owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        self._operator_registry_cache = {}
        # Persistent authoring aliases rebuilt into every derived OperatorRegistry. This table is
        # mutated only by operator_alias(); reading Module/operator_registry never repairs state.
        self._aliases = {}
        self._state_space_metadata = {
            "representation": "conservative",
            "centering": "cell",
            "layout": "cell",
            "storage": "multifab",
            "frame": "model",
            "clock": "simulation",
            "units": None,
        }
        self.cons_names = []
        self.prim_defs = {}     # name -> Expr (in terms of the cons / previous prims / aux)
        self.aux_names = []      # CANONICAL aux fields read (phi/grad/B_z/T_e), cf. AUX_CANONICAL
        self.aux_extra_names = []  # NAMED aux fields (aux_field): order = index AUX_NAMED_BASE + k
        self._flux = {}         # "x" / "y" -> list of Expr (one per conservative component)
        self._flux_terms = {}   # NAMED physical fluxes (flux_term, ADC-419): name -> {"x": [Expr],
                                # "y": [Expr]} (n_cons each). The implicit "default" flux lives in
                                # self._flux (m.flux / set_flux), so a model that only ever calls m.flux
                                # keeps this dict EMPTY -- the named keys enter _model_hash ONLY when
                                # non-empty (cache key preserved). A compiled Program selects a SUM of
                                # named fluxes via ctx.rhs(..., fluxes=[name, ...]); fluxes=["default"]
                                # (or no list) keeps the historical -div F (rhs_into), byte-identical.
        self._eig = {}          # "x" / "y" -> list of Expr (eigenvalues)
        self._wave_speeds = None  # {"x"/"y": (smin Expr, smax Expr)}: explicit SIGNED speeds
                                  # (set_wave_speeds); None = derived from the eigenvalues if 'p' (historical)
        self._ws_jacobian = None  # {"x"/"y": [[Expr]]} + meta (eig, blocks): EXACT signed speeds
                                  # from the eigenvalues of the flux jacobian (set_wave_speeds_from_jacobian)
        self._source = None     # list of Expr (one per component) or None
        self._source_terms = {}   # NAMED local sources (source_term): name -> [Expr] (n_cons each).
                                  # The implicit "default" source lives in self._source (m.source), so a
                                  # model that only ever calls m.source keeps this dict EMPTY -- the named
                                  # keys enter _model_hash ONLY when non-empty (cache key preserved).
        self._linear_sources = {}  # NAMED local linear operators (linear_source): name -> [[Expr]]
                                   # (n_cons x n_cons), coefficients linear in U (no cons/prim dependency).
        self._elliptic = None   # Expr (contribution to the elliptic right-hand side) or None
        self._elliptic_fields = {}  # NAMED elliptic fields: name -> {rhs, operator, aux,
                                    # gradient_sign}. The unnamed default
                                    # stays in self._elliptic (m.elliptic_rhs); the named keys enter
                                    # _model_hash ONLY when non-empty (cache key preserved). The runtime
                                    # is qualified by Case.field together with its typed numerical
                                    # discretization and output route.
        self._stab_speed = None  # Expr: STABILITY speed lambda* (None = fallback eigenvalues)
        self._stab_dt = None     # Expr: direct ADMISSIBLE step dt(U, aux) (None = no bound)
        self._src_freq = None    # Expr: frequency mu(U, aux) of the SOURCE (None = no bound)
        self._proj = None        # [Expr]: PROJECTION ponctuelle post-pas U <- P(U, aux) (ADC-177)
        self._src_jac = None     # [[Expr]] n x n: ANALYTIC Jacobian dS/dU (None = finite differences)
        self._hllc = False       # True: emit the HLLC capability (contact_speed + star state)
        self._riemann_hook_forms = {}  # ARBITRARY-formula overrides of the role-derived Riemann hooks
                                   # (ADC-456): name -> Expr. Codegen'd key: 'pressure' (single-state
                                   # signature, overrides the pressure(U) hook body). A descriptor or
                                   # None leaves the role-derived default. Folded into _model_hash.
        self._roe = False        # True: emit the ROE capability (roe_dissipation from the roles)
        self._roe_rows = None    # {"x": [Expr], "y": [Expr]}: roe_dissipation PROVIDED (outside roles)
        self._roe_jacobian = None  # {"x"/"y": [[Expr]]}: roe_dissipation from the FLUX JACOBIAN
                                   # (roe_from_jacobian, generic moment Roe via pops::roe_abs_apply)
        self.prim_state = []    # ordered names of the primitive state (Prim layout); for the codegen
        self.cons_from = None   # list of Expr: conservative in terms of the primitives (to_conservative)
        self.cons_roles = None  # explicit override of the conservative roles (otherwise canonical mapping)
        self.prim_roles = None  # explicit override of the primitive roles (otherwise canonical mapping)
        self.gamma = None       # adiabatic index of the block (EOS), read by the inter-species couplings
                                # on the System side. None -> symbol pops_compiled_gamma not emitted (the System
                                # then falls back to its historical default 1.4, strict backward compatibility).
        self._rate_operators = {}  # NAMED composite rate operators (rate_operator, Spec 2): name ->
                                   # {"flux": bool, "sources": [str], "fluxes": [str] | None}. A pure
                                   # Program-side ALIAS for ctx.rhs(flux=..., sources=..., fluxes=...): a
                                   # typed P.call(name) lowers to the SAME rhs IR, so the alias never enters
                                   # the model hash nor the codegen (its flux/sources are already hashed).
    @property
    def owner_path(self) -> OwnerPath:
        """Immutable qualified identity anchor for every symbol declared by this model."""
        return self._owner_path

    def _invalidate_authoring_views(self) -> None:
        """Discard derived registry views after one successful physics declaration."""
        self._operator_registry_cache = {}
