"""System install mixin (Spec-4 PR-F): equation/coupling installation.

Holds the densest part of :class:`pops.runtime._system.System`: ``add_equation``
(direct native versus compiled production-package installation),
``add_background``, ``add_elliptic_model`` and ``add_coupling``. Mixed into ``System`` via
inheritance; methods operate on ``self._s`` (the compiled facade) and ``self._aux_field_index``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._bootstrap import ModelSpec
from pops.runtime._lifecycle import guard_assembling as _guard_assembling
from pops.runtime._numeric import native_block_scalars, native_real, positive_int
from pops.runtime.defaults import (
    NEWTON_DEFAULT_ABS_TOL,
    NEWTON_DEFAULT_DAMPING,
    NEWTON_DEFAULT_FD_EPS,
    NEWTON_DEFAULT_MAX_ITERS,
    NEWTON_DEFAULT_REL_TOL,
    PHYSICAL_DEFAULT_GAMMA,
    numerical_defaults_report,
)
from pops.runtime._engine_descriptors import (
    Spatial,
    Explicit,
    DivEpsGrad,
    CompositeRhs,
    ChargeDensitySource,
)
from pops.runtime.routes import (
    check_riemann_requirement_contract as _check_riemann_requirement_contract,
    resolve as _resolve_route,
)

# Typed Poisson wall / bc lowerers are split out for the 500-line cap (ADC-550).
from pops.runtime._system_install_lowering import (  # noqa: F401
    _cartesian_cg_kwargs,
    _lower_bc,
    _weno_kwargs,
)

if TYPE_CHECKING:
    from pops.runtime._system_contract import _System
else:
    _System = object


def _reject_unpublished_newton_diagnostics(time: Any, *, where: str) -> None:
    if getattr(time, "newton_diagnostics", False):
        raise ValueError(
            f"{where}: newton_diagnostics=True is unavailable on the Program-only System "
            "runtime because no typed implicit Program consumer publishes that report"
        )


class _SystemInstall(_System):
    """Equation/coupling installation methods of System."""

    def add_equation(
        self,
        name: Any,
        model: Any,
        spatial: Any = None,
        time: Any = None,
        substeps: Any = None,
        names: Any = None,
        evolve: bool = True,
        stride: Any = None,
        _bind_params: Any = None,
    ) -> Any:
        """Install a native model or one compiled production package.

        Sole Python block-installation seam below ``pops.bind``. The documented PUBLIC path is the typed
        ``pops.Case(...).block(...)`` assembly passed through ``pops.resolve`` / ``pops.compile``
        and wired by ``pops.bind``; ``add_equation`` stays private to the native runtime.

        A ``ModelSpec`` uses the direct native brick path. A ``CompiledModel`` must be a
        production package; its complete resolved BindSchema vector is provided privately by
        :meth:`_install_compiled` and becomes immutable when native closures are created.

        @p spatial: private adapter lowered from ``pops.numerics.FiniteVolume(...)``.
        @p time: private engine policy lowered from an explicit ``pops.Program`` or
        ``pops.lib.time`` factory. @p substeps: overrides time.substeps.
        @p stride : overrides time.stride (1 = each macro-step, default bit-identical).
        @p names is accepted only by the direct ``ModelSpec`` path. @p evolve controls whether the
        installed block advances. ``_bind_params`` is an internal complete vector, never a public
        mutable setter.
        """
        _guard_assembling(self, "add_equation")  # frozen once pops.bind completes (ADC-592)
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.abi import check_compiled_matches_module
        from pops.codegen.loader import CompiledModel
        from pops.physics.aux import aux_layout

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        _reject_unpublished_newton_diagnostics(time, where="System.add_equation")
        nsub = positive_int(
            substeps if substeps is not None else getattr(time, "substeps", 1),
            where="System.add_equation.substeps",
        )
        nstride = positive_int(
            stride if stride is not None else getattr(time, "stride", 1),
            where="System.add_equation.stride",
        )

        if isinstance(model, ModelSpec):
            rel_tol, abs_tol, fd_eps, damping, positivity_floor = native_block_scalars(
                time, spatial, where="System.add_equation"
            )
            self._s.add_block(
                name,
                model,
                spatial.limiter,
                spatial.flux,
                spatial.recon,
                time.kind,
                nsub,
                evolve,
                nstride,
                getattr(time, "implicit_vars", []),
                getattr(time, "implicit_roles", []),
                getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                rel_tol,
                abs_tol,
                fd_eps,
                getattr(time, "newton_diagnostics", False),
                damping,
                positivity_floor,
                getattr(spatial, "wave_speed_cache", False),
                **_weno_kwargs(spatial),
            )
            return

        # The compiled-package ABI does not carry a per-block implicit mask. Reject it rather than
        # silently selecting a different treatment.
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "add_equation: implicit_vars / implicit_roles (per-block IMEX mask) are carried "
                "only by a private native ModelSpec, available on the internal native "
                "engine API (not part of the pops.bind surface). The compiled model (.so) does not "
                "carry the mask."
            )
        # Same rules for the Newton options/diagnostics (IMEX): not carried by the .so ABI.
        # Non-default values would be ignored SILENTLY -> explicit rejection.
        if (
            getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS) != NEWTON_DEFAULT_MAX_ITERS
            or getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL) != NEWTON_DEFAULT_REL_TOL
            or getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL) != NEWTON_DEFAULT_ABS_TOL
            or getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS) != NEWTON_DEFAULT_FD_EPS
            or getattr(time, "newton_diagnostics", False)
            or getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING) != NEWTON_DEFAULT_DAMPING
        ):
            raise ValueError(
                "add_equation: the Newton options (newton_max_iters/rel_tol/abs_tol/fd_eps/"
                "diagnostics/damping) are carried only by a composed native model "
                "(ModelSpec), available on the internal native engine API (not part of the "
                "pops.bind surface). The compiled model (.so) ABI does not carry them."
            )

        if not isinstance(model, CompiledModel):
            raise TypeError(
                "add_equation: model must be a private ModelSpec or detached CompiledModel; got %r"
                % type(model).__name__
            )

        compiled = model
        # Names guard: length checked early (the C++ also raises, but we diagnose here).
        if names is not None and len(names) != compiled.n_vars:
            raise ValueError(
                "add_equation: names= has %d names but block '%s' has %d variables"
                % (len(names), name, compiled.n_vars)
            )

        # NAMED aux fields (ADC-70 phase 1): table name -> block component, from the ORDERED names of
        # the compiled model (k-th name = component dsl.AUX_NAMED_BASE + k, mirror of the C++ emission).
        # Consumed by set_aux_field / aux_field; the adders have already widened the aux channel
        # (pops_compiled_naux -> ensure_aux_width), so the component exists.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        named_base = aux_layout(compiled.native_dimension).named_base
        self._aux_field_index[name] = {nm: named_base + k for k, nm in enumerate(extra)}

        backend = compiled.backend
        # Descriptor-owned model predicates are shared verbatim with AMR and availability.
        _check_riemann_requirement_contract(
            spatial.riemann_capability_contract,
            compiled,
            "add_equation",
            flux=spatial.flux,
        )

        if backend != "production":
            raise ValueError(
                "add_equation: compiled packages must use backend='production'; got %r" % backend
            )
        if names is not None:
            raise ValueError(
                "add_equation: names= is not supported for a compiled package; names and roles "
                "are immutable artifact metadata"
            )
        if getattr(spatial, "wave_speed_cache", False):
            raise ValueError(
                "add_equation: wave_speed_cache is not carried by the production package ABI; "
                "use a direct native ModelSpec"
            )
        runtime_names = tuple(getattr(compiled, "runtime_param_names", ()) or ())
        if _bind_params is None:
            if runtime_names:
                raise ValueError(
                    "add_equation: compiled package declares runtime parameters; install it through "
                    "pops.bind so BindSchema resolves one complete vector"
                )
            bind_values = []
        else:
            bind_values = [
                native_real(value, where="System.add_equation.bind_params[%d]" % index)
                for index, value in enumerate(_bind_params)
            ]
            if len(bind_values) != len(runtime_names):
                raise ValueError(
                    "add_equation: bound parameter vector has %d values, expected %d"
                    % (len(bind_values), len(runtime_names))
                )
        check_compiled_matches_module(getattr(compiled, "abi_key", ""))
        gamma = compiled.gamma if compiled.gamma is not None else PHYSICAL_DEFAULT_GAMMA
        gamma = native_real(gamma, where="System.add_equation.gamma")
        positivity_floor = native_real(
            getattr(spatial, "positivity_floor", 0.0),
            where="System.add_equation.positivity_floor",
        )
        weno_epsilon = getattr(spatial, "weno_epsilon", None)
        weno_epsilon = (
            float(numerical_defaults_report()["weno"]["epsilon"])
            if weno_epsilon is None
            else native_real(weno_epsilon, where="System.add_equation.weno_epsilon")
        )
        if spatial.external_flux_id is not None:
            if "uniform" not in spatial.external_flux_supported_layouts:
                raise ValueError(
                    "add_equation: external Riemann brick %r does not support uniform layouts"
                    % spatial.external_flux_id
                )
            if runtime_names:
                raise ValueError(
                    "add_equation: external Riemann ABI v2 does not transport model RuntimeParams"
                )
            if spatial.external_flux_model_identity != compiled.model_hash:
                raise ValueError(
                    "add_equation: external Riemann brick %r targets model %r, not %r"
                    % (
                        spatial.external_flux_id,
                        spatial.external_flux_model_identity,
                        compiled.model_hash,
                    )
                )
            if spatial.external_flux_native_abi_key != compiled.abi_key:
                raise ValueError(
                    "add_equation: external Riemann brick %r was built for a different native ABI"
                    % spatial.external_flux_id
                )
            self._s._install_external_riemann_block(
                name,
                spatial.external_flux_library_path,
                spatial.external_flux_id,
                spatial.external_flux_library_sha256,
                spatial.limiter,
                spatial.recon,
                time.kind,
                gamma,
                nsub,
                evolve,
                nstride,
                compiled.n_vars,
                compiled.n_aux,
                compiled.model_hash,
                positivity_floor,
                weno_epsilon,
            )
        else:
            self._s._install_native_block(
                name,
                compiled.so_path,
                spatial.limiter,
                spatial.flux,
                spatial.recon,
                time.kind,
                gamma,
                nsub,
                evolve,
                nstride,
                bind_values,
                positivity_floor,
            )

    def add_background(self, name: Any, model: Any, density: Any, spatial: Any = None) -> Any:
        """FROZEN species (not advanced): a fixed background that contributes to the system Poisson (and,
        later, to coupled sources). density: n*n array. Uses the same type-dispatched
        ``add_equation(evolve=False)`` installation seam as evolved blocks, then sets density.
        """
        self.add_equation(name, model, spatial=spatial, evolve=False)
        self.set_density(name, density)

    def set_poisson(
        self,
        rhs: Any = "charge_density",
        solver: Any = None,
        bc: Any = None,
        abs_tol: Any = None,
        rel_tol: Any = None,
        max_iterations: Any = None,
    ) -> Any:
        """Configure the uniform constant-coefficient Poisson solve.

        ``solver=None`` keeps the exact-ranked Cartesian runtime's explicit ``cartesian_cg``
        default. ``bc`` accepts a typed native boundary descriptor; omission keeps
        automatic boundary selection. A :class:`pops.solvers.elliptic.CartesianCG` descriptor
        owns its exact tolerance and iteration controls. Variable/tensor coefficients, reaction,
        embedded boundaries and ``GeometricMG`` belong to the AMR field-provider route.
        """
        if solver is None:
            solver = self._s.poisson_solver()
        elif getattr(solver, "scheme", None) == "cartesian_cg":
            authored = solver.cg_options()
            if any(value is not None for value in (abs_tol, rel_tol, max_iterations)):
                raise ValueError(
                    "System.set_poisson CartesianCG owns its controls; do not duplicate them"
                )
            abs_tol = authored["abs_tol"]
            rel_tol = authored["rel_tol"]
            max_iterations = authored["max_iterations"]
            solver = solver.scheme
        bc_token = "auto" if bc is None else _lower_bc(bc)
        self._set_poisson_native(
            rhs=rhs,
            solver=solver,
            bc=bc_token,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            max_iterations=max_iterations,
        )

    def _set_poisson_native(
        self,
        *,
        rhs: Any,
        solver: Any,
        bc: Any,
        abs_tol: Any = None,
        rel_tol: Any = None,
        max_iterations: Any = None,
    ) -> Any:
        """Private token-level seam used only after typed authoring has been lowered."""
        _guard_assembling(self, "set_poisson")
        if not isinstance(bc, str):
            raise TypeError("_set_poisson_native requires one native boundary token")
        rhs = _resolve_route("poisson_rhs", rhs, context="set_poisson")
        solver = _resolve_route("field_solver", solver, context="set_poisson")
        bc = _resolve_route("poisson_bc", bc, context="set_poisson")
        controls = _cartesian_cg_kwargs(rel_tol, max_iterations)
        if abs_tol is not None:
            controls["abs_tol"] = native_real(abs_tol, where="System.set_poisson.abs_tol")
        self._s.set_poisson(rhs=rhs, solver=solver, bc=bc, **controls)

    def add_elliptic_model(self, name: Any, model: Any, solver: Any = None, bc: Any = None) -> Any:
        """EPM: configures the system elliptic model (Poisson is its current instance).
        model = pops.elliptic(operator=pops.div_eps_grad(eps), rhs=pops.composite_rhs(),
        output=pops.electric_field_from_potential()). set_poisson(...) remains the equivalent shortcut.

        The uniform exact-ranked route implements only the unit-coefficient Laplacian. Right-hand
        side: composite_rhs() = GENERIC sum
        of the elliptic bricks carried by the blocks (charge q n, background alpha (n-n0), gravity
        coupling sign 4piG (rho-rho0)); charge_density() is its usual case. Other coefficients and
        operators require an explicitly capable AMR field provider."""
        if not isinstance(
            model.operator, DivEpsGrad
        ):  # freeze ADC-592: the delegated set_poisson guards
            raise NotImplementedError(
                "add_elliptic_model: only the div_eps_grad operator (Poisson) "
                "is supported; diffusion / projection -> refinement (solver)"
            )
        if not isinstance(model.rhs, CompositeRhs):
            raise NotImplementedError(
                "add_elliptic_model: rhs must be composite_rhs() (sum of the "
                "per-block bricks) or charge_density() (its usual case)"
            )
        if native_real(model.operator.epsilon, where="add_elliptic_model.operator.epsilon") != 1.0:
            raise ValueError(
                "add_elliptic_model: uniform CartesianCG requires the unit coefficient; "
                "use the exact AMR field-provider contract for other coefficients"
            )
        # Honest token: "composite" for a generic right-hand side, "charge_density" (alias,
        # bit-identical) when all blocks carry a charge density. Both take the
        # SAME numerical path on the C++ side (sum of each block's elliptic bricks).
        rhs_tok = "charge_density" if type(model.rhs) is ChargeDensitySource else "composite"
        self.set_poisson(rhs=rhs_tok, solver=solver, bc=bc)

    def add_coupling(self, coupling: Any) -> Any:
        """Add an inter-species coupling (operator-split, applied after transport):

        - private Ionization / Collision / ThermalExchange descriptor -> preset lowering to a generic
          coupled source (ADC-595): the fixed formula is emitted as a CoupledSource with a DECLARED
          conservation contract, compiled to bytecode, and registered as a typed coupling operator;
        - private ``CompiledCoupledSource`` -> generic bytecode source
          interpreted on the C++ side (no per-cell Python callback, MPI-safe).

        Both paths register through System.add_coupling_operator, so the coupling is inspectable as a
        typed operator (sim.coupled_operators()) with its declared conservation validated at
        registration. There is no longer a named C++ coupling method per coupling."""
        _guard_assembling(self, "add_coupling")  # frozen once pops.bind completes (ADC-592)
        # Late import (the multispecies module imports this package: avoid the cycle).
        from pops.physics.multispecies import CompiledCoupledSource
        from pops.physics.coupling_presets import lower_named_coupling, coupling_operator_args

        if isinstance(coupling, CompiledCoupledSource):
            args = coupling_operator_args(
                coupling,
                getattr(coupling, "conserved_roles", ()),
                getattr(coupling, "created_roles", ()),
            )
            self._s.add_coupling_operator(*args)
            return
        preset = lower_named_coupling(coupling, self._s.block_gamma)
        if preset is None:
            raise TypeError(
                "add_coupling expects a private named-coupling engine descriptor or "
                "CompiledCoupledSource"
            )
        # Validate the DECLARED contract symbolically (Python); the C++ revalidates at registration. A
        # created role (ionization) may net-source, so compile without verify_conservation.
        preset.source.verify_declared_contract(conserved=preset.conserved, created=preset.created)
        args = coupling_operator_args(
            preset.source.compile(), preset.conserved, preset.created, frequency=preset.frequency
        )
        self._s.add_coupling_operator(*args)
