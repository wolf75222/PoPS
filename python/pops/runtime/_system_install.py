"""System install mixin (Spec-4 PR-F): block/equation/coupling installation.

Holds the densest part of :class:`pops.runtime._system.System`: ``add_block`` /
``add_equation`` (direct native versus compiled production-package installation),
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
    NEWTON_DEFAULT_FAIL_POLICY,
    NEWTON_DEFAULT_FD_EPS,
    NEWTON_DEFAULT_MAX_ITERS,
    NEWTON_DEFAULT_REL_TOL,
    PHYSICAL_DEFAULT_GAMMA,
)
from pops.runtime._engine_descriptors import (
    Spatial, Explicit, DivEpsGrad, CompositeRhs, ChargeDensitySource,
)
from pops.runtime.routes import (
    RIEMANN_HLL, check_riemann_capability as _check_riemann_capability,
    resolve as _resolve_route,
)

# Typed Poisson wall / bc lowerers are split out for the 500-line cap (ADC-550).
from pops.runtime._system_install_lowering import (  # noqa: F401
    _lower_bc, _lower_wall, _mg_kwargs, _weno_kwargs,
)

if TYPE_CHECKING:
    from pops.runtime._system_contract import _System
else:
    _System = object


class _SystemInstall(_System):
    """Block/equation/coupling installation methods of System."""

    def add_block(self, name: Any, model: Any, spatial: Any = None, time: Any = None,
                  evolve: bool = True) -> Any:
        """Installs an evolved block composed of NATIVE BRICKS on the shared system Poisson.

        Low-level runtime seam. The documented PUBLIC path is the typed
        ``pops.Case(...).block(...)`` assembly passed through ``pops.resolve`` / ``pops.compile``
        and wired by ``pops.bind`` (which calls this method internally); ``add_block`` stays for that seam,
        the native/AMR runtime, and the tests.

        Installs a private ``ModelSpec`` composed from native bricks. Public ``pops.Model``
        authoring enters through ``pops.Case`` and the lifecycle. For a compiled production model
        or automatic dispatch on the engine value type, use add_equation. Arguments reach the C++ facade
        (System::add_block), which validates the block (names / roles / implicit mask) against the model.

        @param name unique block name; indexes set_density(name) / mass(name) / density(name).
        @param model private ``ModelSpec`` engine value.
        @param spatial private engine adapter lowered from ``pops.numerics.FiniteVolume(...)``
            (default minmod + rusanov + conservative). Carries the limiter (none / minmod /
            vanleer / weno5 --
            weno5 is exposed ONLY by this native path), the Riemann flux (rusanov / hll / hllc /
            roe) and the reconstructed variables (conservative / primitive). positivity_floor is read
            here (Zhang-Shu positivity limiter).
        @param time private engine policy. Public authoring uses an explicit ``pops.Program`` or a
            ``pops.lib.time`` factory. The lowered policy carries cadence, any implicit mask and
            local Newton options; these values are forwarded as-is to C++.
        @param evolve True (default) = block advances; False = frozen field (background) which still
            contributes to the right-hand side of the system Poisson.
        """
        _guard_assembling(self, "add_block")  # frozen once pops.bind completes (ADC-592)
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # Native ABI conversion happens here; descriptors above this seam stay exact.
        rel_tol, abs_tol, fd_eps, damping, positivity_floor = native_block_scalars(
            time, spatial, where="System.add_block")
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), evolve, getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                          rel_tol, abs_tol, fd_eps,
                          getattr(time, "newton_diagnostics", False),
                          damping,
                          getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY),
                          positivity_floor,
                          getattr(spatial, "wave_speed_cache", False), **_weno_kwargs(spatial))

    def add_equation(self, name: Any, model: Any, spatial: Any = None, time: Any = None,
                     substeps: Any = None, names: Any = None, evolve: bool = True,
                     stride: Any = None, _bind_params: Any = None) -> Any:
        """Install a native model or one compiled production package.

        Low-level runtime seam. The documented PUBLIC path is the typed
        ``pops.Case(...).block(...)`` assembly passed through ``pops.resolve`` / ``pops.compile``
        and wired by ``pops.bind``; ``add_equation`` stays private to the native/AMR runtime.

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
        from pops.physics.aux import AUX_NAMED_BASE

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        nsub = positive_int(substeps if substeps is not None else getattr(time, "substeps", 1), where="System.add_equation.substeps")
        nstride = positive_int(stride if stride is not None else getattr(time, "stride", 1), where="System.add_equation.stride")

        if isinstance(model, ModelSpec):
            rel_tol, abs_tol, fd_eps, damping, positivity_floor = native_block_scalars(
                time, spatial, where="System.add_equation")
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, evolve, nstride,
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                              rel_tol, abs_tol, fd_eps,
                              getattr(time, "newton_diagnostics", False),
                              damping,
                              getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY),
                              positivity_floor,
                              getattr(spatial, "wave_speed_cache", False),
                              **_weno_kwargs(spatial))
            return

        # The compiled-package ABI does not carry a per-block implicit mask. Reject it rather than
        # silently selecting a different treatment.
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "add_equation: implicit_vars / implicit_roles (per-block IMEX mask) are carried "
                "only by a private native ModelSpec, available on the internal native "
                "engine API (not part of the pops.bind surface). The compiled model (.so) does not "
                "carry the mask.")
        # Same rules for the Newton options/diagnostics (IMEX): not carried by the .so ABI.
        # Non-default values would be ignored SILENTLY -> explicit rejection.
        if (getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS)
                != NEWTON_DEFAULT_MAX_ITERS
                or getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL)
                != NEWTON_DEFAULT_REL_TOL
                or getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL)
                != NEWTON_DEFAULT_ABS_TOL
                or getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS)
                != NEWTON_DEFAULT_FD_EPS
                or getattr(time, "newton_diagnostics", False)
                or getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING)
                != NEWTON_DEFAULT_DAMPING
                or getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY)
                != NEWTON_DEFAULT_FAIL_POLICY):
            raise ValueError(
                "add_equation: the Newton options (newton_max_iters/rel_tol/abs_tol/fd_eps/"
                "diagnostics/damping/fail_policy) are carried only by a composed native model "
                "(ModelSpec), available on the internal native engine API (not part of the "
                "pops.bind surface). The compiled model (.so) ABI does not carry them.")

        if not isinstance(model, CompiledModel):
            raise TypeError(
                "add_equation: model must be a private ModelSpec or detached CompiledModel; got %r"
                % type(model).__name__)

        compiled = model
        # Names guard: length checked early (the C++ also raises, but we diagnose here).
        if names is not None and len(names) != compiled.n_vars:
            raise ValueError("add_equation: names= has %d names but block '%s' has %d variables"
                             % (len(names), name, compiled.n_vars))

        # NAMED aux fields (ADC-70 phase 1): table name -> block component, from the ORDERED names of
        # the compiled model (k-th name = component dsl.AUX_NAMED_BASE + k, mirror of the C++ emission).
        # Consumed by set_aux_field / aux_field; the adders have already widened the aux channel
        # (pops_compiled_naux -> ensure_aux_width), so the component exists.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        self._aux_field_index[name] = {nm: AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

        backend = compiled.backend
        # Numerical flux guard (ADC-590, shared with AmrSystem.add_equation): generic hllc/roe are
        # GENERIC-ONLY (require has_hllc/has_roe -- the 'p'-only Euler fallback is removed); the
        # explicit euler_hllc/euler_roe routes serve the canonical 4-var Euler layout.
        _check_riemann_capability(spatial.flux, compiled, "add_equation")
        # ADC-552: cross-check a HLL(waves=<provider>) selection against the model's actual source.
        if spatial.flux == RIEMANN_HLL and getattr(spatial, "waves_provider", None) is not None:
            from pops.numerics.riemann.waves import check_hll_waves
            check_hll_waves(spatial.waves_provider, compiled, "add_equation")
        # HLL emits wave_speeds from the EXPLICIT pair m.wave_speeds(x=, y=) (without primitive 'p':
        # moments / isothermal, cf. has_wave_speeds) OR as soon as a primitive 'p' is declared. EARLY
        # guard (like hllc/roe): the C++ requires-gate of make_block only triggers at first use, so we
        # diagnose at install. getattr default True (CompiledModel ALWAYS sets has_wave_speeds; only a
        # foreign object hits the default and falls back on the C++ gate).
        if spatial.flux == RIEMANN_HLL and not getattr(compiled, "has_wave_speeds", True):
            raise ValueError(
                "add_equation: riemann 'hll' requires signed wave speeds: declare "
                "m.wave_speeds(x=(smin, smax), y=(smin, smax)) (without pressure), or a primitive "
                "'p' (m.primitive('p', ...)); otherwise use riemann='rusanov' "
                "[requested route %s -> %s]"
                % (getattr(RIEMANN_HLL, "id", "riemann.hll"), RIEMANN_HLL.native_entry))

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
        self._s._install_native_block(
            name, compiled.so_path, spatial.limiter, spatial.flux, spatial.recon, time.kind,
            gamma, nsub, evolve, nstride, bind_values,
            native_real(
                getattr(spatial, "positivity_floor", 0.0),
                where="System.add_equation.positivity_floor",
            ),
        )

    def add_background(self, name: Any, model: Any, density: Any, spatial: Any = None) -> Any:
        """FROZEN species (not advanced): a fixed background that contributes to the system Poisson (and,
        later, to coupled sources). density: n*n array. Equivalent to add_block(evolve=False) then
        set_density (freeze ADC-592 enforced by the delegated, guarded add_block)."""
        self.add_block(name, model, spatial=spatial, evolve=False)
        self.set_density(name, density)

    def set_poisson(self, rhs: Any = "charge_density", solver: Any = "geometric_mg",
                    bc: Any = None, wall: Any = None,
                    epsilon: float = 1.0, abs_tol: float = 0.0, rel_tol: Any = None,
                    max_cycles: Any = None, min_coarse: Any = None, pre_smooth: Any = None,
                    post_smooth: Any = None, bottom_sweeps: Any = None,
                    coarse_threshold: Any = None) -> Any:
        """Configure the shared Poisson solve with typed boundary and wall selectors.

        ``bc`` accepts a typed native boundary descriptor; omission keeps automatic boundary
        selection. ``wall`` accepts :class:`pops.mesh.geometry.Disc` or
        :class:`pops.mesh.geometry.NoWall`; omission selects no wall. Strings and a separate
        ``wall_radius`` are deliberately absent: every descriptor owns its complete data.
        """
        bc_token = "auto" if bc is None else _lower_bc(bc)
        wall_token, wall_radius = ("none", 0.0) if wall is None else _lower_wall(wall)
        self._set_poisson_native(
            rhs=rhs, solver=solver, bc=bc_token, wall=wall_token,
            wall_radius=wall_radius, epsilon=epsilon, abs_tol=abs_tol, rel_tol=rel_tol,
            max_cycles=max_cycles, min_coarse=min_coarse, pre_smooth=pre_smooth,
            post_smooth=post_smooth, bottom_sweeps=bottom_sweeps,
            coarse_threshold=coarse_threshold)

    def _set_poisson_native(self, *, rhs: Any, solver: Any, bc: Any, wall: Any,
                            wall_radius: Any = 0.0, epsilon: Any = 1.0,
                            abs_tol: Any = 0.0, rel_tol: Any = None,
                            max_cycles: Any = None, min_coarse: Any = None,
                            pre_smooth: Any = None, post_smooth: Any = None,
                            bottom_sweeps: Any = None, coarse_threshold: Any = None) -> Any:
        """Private token-level seam used only after typed authoring has been lowered."""
        _guard_assembling(self, "set_poisson")
        if not isinstance(bc, str) or not isinstance(wall, str):
            raise TypeError("_set_poisson_native requires native bc and wall tokens")
        rhs = _resolve_route("poisson_rhs", rhs, context="set_poisson")
        solver = _resolve_route("field_solver", solver, context="set_poisson")
        bc = _resolve_route("poisson_bc", bc, context="set_poisson")
        wall = _resolve_route("wall", wall, context="set_poisson")
        self._s.set_poisson(rhs=rhs, solver=solver, bc=bc, wall=wall,
                            wall_radius=native_real(
                                wall_radius, where="System.set_poisson.wall_radius"),
                            epsilon=native_real(epsilon, where="System.set_poisson.epsilon"),
                            abs_tol=native_real(abs_tol, where="System.set_poisson.abs_tol"),
                            **_mg_kwargs(rel_tol, max_cycles, min_coarse, pre_smooth,
                                         post_smooth, bottom_sweeps, coarse_threshold))

    def add_elliptic_model(self, name: Any, model: Any, solver: Any = None, bc: Any = None,
                           wall: Any = None) -> Any:
        """EPM: configures the system elliptic model (Poisson is its current instance).
        model = pops.elliptic(operator=pops.div_eps_grad(eps), rhs=pops.composite_rhs(),
        output=pops.electric_field_from_potential()). set_poisson(...) remains the equivalent shortcut.

        Operator: div(eps grad) with CONSTANT eps (eps != 1 supported: eps lap phi = f); variable
        eps(x) is plugged in via set_epsilon_field. Right-hand side: composite_rhs() = GENERIC sum
        of the elliptic bricks carried by the blocks (charge q n, background alpha (n-n0), gravity
        coupling sign 4piG (rho-rho0)); charge_density() is its usual case. Diffusion / projection (other
        operator) would require a variable-coefficient solver (refinement not available)."""
        if not isinstance(model.operator, DivEpsGrad):  # freeze ADC-592: the delegated set_poisson guards
            raise NotImplementedError("add_elliptic_model: only the div_eps_grad operator (Poisson) "
                                      "is supported; diffusion / projection -> refinement (solver)")
        if not isinstance(model.rhs, CompositeRhs):
            raise NotImplementedError("add_elliptic_model: rhs must be composite_rhs() (sum of the "
                                      "per-block bricks) or charge_density() (its usual case)")
        kind = solver.kind if solver is not None else "geometric_mg"
        # Honest token: "composite" for a generic right-hand side, "charge_density" (alias,
        # bit-identical) when all blocks carry a charge density. Both take the
        # SAME numerical path on the C++ side (sum of each block's elliptic bricks).
        rhs_tok = "charge_density" if type(model.rhs) is ChargeDensitySource else "composite"
        self.set_poisson(rhs=rhs_tok, solver=kind, bc=bc, wall=wall,
                         epsilon=model.operator.epsilon)

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
            args = coupling_operator_args(coupling, getattr(coupling, "conserved_roles", ()),
                                          getattr(coupling, "created_roles", ()))
            self._s.add_coupling_operator(*args)
            return
        preset = lower_named_coupling(coupling, self._s.block_gamma)
        if preset is None:
            raise TypeError(
                "add_coupling expects a private named-coupling engine descriptor or "
                "CompiledCoupledSource")
        # Validate the DECLARED contract symbolically (Python); the C++ revalidates at registration. A
        # created role (ionization) may net-source, so compile without verify_conservation.
        preset.source.verify_declared_contract(conserved=preset.conserved, created=preset.created)
        args = coupling_operator_args(preset.source.compile(), preset.conserved, preset.created,
                                      frequency=preset.frequency)
        self._s.add_coupling_operator(*args)
