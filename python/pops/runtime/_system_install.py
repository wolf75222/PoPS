"""System install mixin (Spec-4 PR-F): block/equation/coupling installation.

Holds the densest part of :class:`pops.runtime.system.System`: ``add_block`` /
``add_equation`` (the backend-adder dispatch + explicit-rejection guards), ``set_source_stage``,
``add_background``, ``add_elliptic_model`` and ``add_coupling``. Mixed into ``System`` via
inheritance; methods operate on ``self._s`` (the compiled facade) and ``self._aux_field_index``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._bootstrap import ModelSpec
from pops.runtime._lifecycle import guard_assembling as _guard_assembling
from pops.runtime._lifecycle import reject_compiled_time_route as _reject_compiled_time_route
from pops.runtime.defaults import (
    NEWTON_DEFAULT_ABS_TOL,
    NEWTON_DEFAULT_DAMPING,
    NEWTON_DEFAULT_FAIL_POLICY,
    NEWTON_DEFAULT_FD_EPS,
    NEWTON_DEFAULT_MAX_ITERS,
    NEWTON_DEFAULT_REL_TOL,
    PHYSICAL_DEFAULT_GAMMA,
)
from pops.runtime.bricks import (
    Spatial, Explicit, Split, DivEpsGrad, CompositeRhs, ChargeDensitySource,
)
from pops.runtime.routes import (
    RIEMANN_HLL, check_riemann_capability as _check_riemann_capability,
    resolve as _resolve_route,
)

# The Poisson wall / bc lowerers are split into ``_system_install_lowering`` for the 500-line cap
# (ADC-550) and re-imported so ``set_poisson`` below and the direct-import tests are unchanged.
from pops.runtime._system_install_lowering import _lower_bc, _lower_wall  # noqa: F401

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
        ``pops.Problem(...).block(...)`` assembly lowered by ``pops.compile`` and wired by
        ``pops.bind`` (which calls this method internally); ``add_block`` stays for that seam,
        the native/AMR runtime, and the tests.

        Installs a model composed in Python from native bricks (pops.Model(...)). For a
        compiled DSL model (.so) or an automatic dispatch on the model type,
        use add_equation. The arguments are marshaled to the C++ facade
        (System::add_block), which validates the block (names / roles / implicit mask) against the model.

        @param name unique block name; indexes set_density(name) / mass(name) / density(name).
        @param model an pops.Model(...) (ModelSpec: state + transport + source + elliptic brick).
        @param spatial spatial discretization, an pops.Spatial(...) / pops.FiniteVolume(...) (default
            minmod + rusanov + conservative). Carries the limiter (none / minmod / vanleer / weno5 --
            weno5 is exposed ONLY by this native path), the Riemann flux (rusanov / hll / hllc /
            roe) and the reconstructed variables (conservative / primitive). positivity_floor is read
            here (Zhang-Shu positivity limiter).
        @param time temporal treatment, an pops.Explicit (default) / pops.IMEX / pops.SourceImplicit.
            Carries substeps (sub-steps per macro-step), stride (multirate hold-then-catch-up cadence),
            the implicit mask (implicit_vars / implicit_roles) and the Newton options (IMEX). All
            these parameters are forwarded as-is to the C++.
        @param evolve True (default) = block advances; False = frozen field (background) which still
            contributes to the right-hand side of the system Poisson.
        @throws TypeError if time is an pops.Split / pops.Strang (Schur-condensed source stage),
            not wired here: go through add_equation(..., time=pops.Split(...)).
        """
        _guard_assembling(self, "add_block")  # frozen once pops.bind completes (ADC-592)
        _reject_compiled_time_route(time, "System.add_block")  # ADC-554: no CompiledTime time= bypass
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # pops.Split (condensed source stage) is only wired by add_equation (which plugs
        # set_source_stage after adding the block): reject it HERE rather than running only the transport
        # silently (the condensed source would be lost).
        if isinstance(time, Split):
            raise TypeError(
                "System.add_block: pops.Split (Schur-condensed source stage) is not wired on this "
                "native seam. Declare the splitting on the pops.Problem time scheme "
                "(time=pops.Split(...)) and lower it with pops.compile(...) + pops.bind(...).")
        # Implicit mask + Newton options carried by the temporal policy (IMEX/SourceImplicit);
        # neutral defaults on the other policies (Explicit). Resolved/validated on the C++ side
        # (System::add_block) against the block's names/roles.
        # ADC-645: the WENO-Z regulariser rides along the Spatial (WENO5(epsilon=...)); None (the
        # default) forwards NOTHING so the native add_block keeps its kWenoEpsilon default.
        weno_kwargs = {}
        weps = getattr(spatial, "weno_epsilon", None)
        if weps is not None:
            weno_kwargs["weno_epsilon"] = float(weps)
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), evolve, getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                          getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL),
                          getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL),
                          getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS),
                          getattr(time, "newton_diagnostics", False),
                          getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING),
                          getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY),
                          getattr(spatial, "positivity_floor", 0.0),
                          getattr(spatial, "wave_speed_cache", False), **weno_kwargs)

    def add_equation(self, name: Any, model: Any, spatial: Any = None, time: Any = None,
                     substeps: Any = None, names: Any = None, evolve: bool = True,
                     stride: Any = None) -> Any:
        """Adds an equation/block by dispatching on the TYPE of @p model (DSL Phase A).

        Low-level runtime seam. The documented PUBLIC path is the typed
        ``pops.Problem(...).block(...)`` assembly lowered by ``pops.compile`` and wired by
        ``pops.bind``; ``add_equation`` stays for that seam, the native/AMR runtime, and the tests.

        Dispatch:

        - a ModelSpec (pops.Model(...)) -> add_block (composed native bricks);
        - a CompiledModel (m.compile(...)) -> the backend adder (add_dynamic_block for prototype,
          add_compiled_block for aot, add_native_block for production), with the names/roles/gamma
          carried by the .so.

        Centralizes the backend <-> adder coupling (an AOT .so must not be plugged into
        add_dynamic_block, and vice versa). cf. docs/DSL_MODEL_DESIGN.md section 3.

        @p spatial : pops.FiniteVolume(...) / pops.Spatial(...) (default minmod+rusanov+conservative).
        @p time : pops.Explicit / IMEX (default Explicit). @p substeps : overrides time.substeps.
        @p stride : overrides time.stride (1 = each macro-step, default bit-identical).
        @p names : component names (length = n_vars of the compiled model). @p evolve : block advances;
        evolve=False (frozen field) is only wired on the native path (ModelSpec -> add_block, backend
        'production' -> add_native_block). On backend 'prototype'/'aot' (the .so ABI does not carry
        evolve) an evolve=False is REJECTED explicitly -> use a native block (add_background).
        """
        _guard_assembling(self, "add_equation")  # frozen once pops.bind completes (ADC-592)
        _reject_compiled_time_route(time, "System.add_equation")  # ADC-554 (see add_block)
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.abi import check_compiled_matches_module
        from pops.codegen.loader import CompiledModel
        from pops.physics.aux import AUX_NAMED_BASE

        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # --- pops.Split (Lie) / pops.Strang (2nd order): EXPLICIT / IMPLICIT splitting, Schur OPT-IN --
        # The block is added with the explicit HYPERBOLIC stage (existing production path, no dispatch
        # duplication), THEN the condensed SOURCE stage is plugged (set_source_stage, C++), run AFTER
        # transport each step; the default (without Split) is unchanged. The splitting POLICY is WIRED
        # to the stepper via set_time_scheme: pops.Split -> "lie" (bit-identical), pops.Strang -> "strang".
        if isinstance(time, Split):
            self.add_equation(name, model, spatial=spatial, time=time.hyperbolic,
                              substeps=substeps, names=names, evolve=evolve, stride=stride)
            src = time.source
            self._s.set_source_stage(name, src.kind, src.theta, src.alpha,
                                     getattr(src, "krylov_tol", 0.0),
                                     getattr(src, "krylov_max_iters", 0),
                                     getattr(src, "density_spec", ""),
                                     getattr(src, "momentum_x_spec", ""),
                                     getattr(src, "momentum_y_spec", ""),
                                     getattr(src, "energy_spec", ""),
                                     getattr(src, "bz_aux_component", -1),
                                     # ADC-645: preconditioner knobs (0/"" = historical defaults).
                                     getattr(src, "n_precond_vcycles", 0),
                                     getattr(src, "polar_precond", ""))
            self._s.set_time_scheme(time.scheme)  # "lie" (Split) or "strang" (Strang)
            return

        nsub = substeps if substeps is not None else getattr(time, "substeps", 1)
        nstride = stride if stride is not None else getattr(time, "stride", 1)

        # --- ModelSpec: composed native bricks -> add_block (existing path) ---
        # NB: we call _s.add_block DIRECTLY with nsub/nstride (not self.add_block, whose
        # signature has no substeps -> it would use time.substeps and IGNORE the overrides).
        if isinstance(model, ModelSpec):
            # ADC-645: the WENO-Z regulariser rides along the Spatial (WENO5(epsilon=...)); None (the
            # default) forwards NOTHING so the native add_block keeps its kWenoEpsilon default.
            weno_kwargs = {}
            weps = getattr(spatial, "weno_epsilon", None)
            if weps is not None:
                weno_kwargs["weno_epsilon"] = float(weps)
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, evolve, nstride,
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                              getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL),
                              getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL),
                              getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS),
                              getattr(time, "newton_diagnostics", False),
                          getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING),
                          getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY),
                          getattr(spatial, "positivity_floor", 0.0),
                          getattr(spatial, "wave_speed_cache", False), **weno_kwargs)
            return

        # Implicit mask (IMEX): only the composed native path (ModelSpec -> add_block) wires it. The .so
        # backends (dynamic/aot/production) lack the argument -> REJECT a non-empty mask rather than
        # ignore it silently (cf. the stride rejection on backend 'aot').
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "add_equation: implicit_vars / implicit_roles (per-block IMEX mask) are carried "
                "only by a composed native model (pops.Model(...)), available on the internal native "
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
                "(pops.Model(...)), available on the internal native engine API (not part of the "
                "pops.bind surface). The compiled model (.so) ABI does not carry them.")

        if not isinstance(model, CompiledModel):
            raise TypeError("add_equation: model must be an pops.Model(...) (ModelSpec) or a "
                            "CompiledModel (m.compile(...)); got %r" % type(model).__name__)

        compiled = model
        # Names guard: length checked early (the C++ also raises, but we diagnose here).
        if names is not None and len(names) != compiled.n_vars:
            raise ValueError("add_equation: names= has %d names but block '%s' has %d variables"
                             % (len(names), name, compiled.n_vars))
        names_arg = list(names) if names is not None else []

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

        # AUTHORITATIVE dispatch by the CompiledModel adder (fixed by the backend, cf. dsl._BACKENDS):
        # prototype->add_dynamic_block, aot->add_compiled_block, production->add_native_block (#85).
        adder = compiled.adder
        if adder == "add_dynamic_block":
            # JIT, HOST Rusanov order-1 residual: takes only the MUSCL LIMITER (none/minmod/vanleer)
            # + substeps; no HLLC/Roe flux, no primitive recon. WENO5 (5-point stencil) is
            # NOT a MUSCL limiter and this path does not run assemble_rhs: we reject it HERE (the
            # aot/production paths accept weno5 -- the .so grid / native block allocate 3 ghosts).
            if spatial.limiter == "weno5":
                raise ValueError(
                    "add_equation: limiter 'weno5' not supported on backend 'prototype' (JIT, host "
                    "Rusanov order-1 residual, without assemble_rhs); use backend='aot'/'production' "
                    "(WENO5 wired end to end), or a composed native model (pops.Model(...)) on the "
                    "internal native engine API.")
            if spatial.flux != "rusanov":
                raise ValueError(
                    "add_equation: backend 'prototype' (JIT, host Rusanov order-1 residual) only exposes "
                    "riemann='rusanov' (got '%s'); use backend='aot'/'production' for "
                    "HLLC/Roe" % spatial.flux)
            # evolve=False (FROZEN block / fixed background) is NOT wired: the add_dynamic_block ABI does
            # not carry evolve (push_dynamic forces it to true on the C++ side) -> the block would be
            # advanced SILENTLY. We REJECT it (rather than silent ignore); for a frozen field, use a
            # native/production block (add_background -> add_block(..., evolve=False)).
            if not evolve:
                raise ValueError(
                    "add_equation: evolve=False not supported on backend 'prototype' (the JIT .so ABI "
                    "does not carry evolve; the block would be advanced silently). A frozen field "
                    "(evolve=False / background) is available only on the internal native engine API "
                    "(a composed native model, pops.Model(...)), not part of the pops.bind surface.")
            # positivity_floor (ADC-76) is NOT wired on the host JIT path (no assemble_rhs, dedicated
            # Rusanov order-1 residual): reject rather than silently ignore the floor.
            if getattr(spatial, "positivity_floor", 0.0) > 0.0:
                raise ValueError(
                    "add_equation: positivity_floor not supported on backend 'prototype' (dedicated "
                    "host residual, without high-order reconstruction); use backend='aot'/'production' "
                    "or a composed native model (pops.Model(...)) on the internal native engine API.")
            # NB wave_speed_cache (ADC-199): no dedicated guard here -- the cache requires riemann='hll',
            # already rejected above on 'prototype' (rusanov order 1 only) -> never silently ignored here.
            self._s.add_dynamic_block(name, compiled.so_path, nsub, names_arg, spatial.limiter)
            return
        if adder == "add_compiled_block":
            # AOT host-marshaled: limiter x riemann x recon, single-rank (without MPI/AMR). The extern "C"
            # ABI of the AOT .so (add_compiled_block) does NOT carry a cadence: the block would run at stride=1
            # SILENTLY. We therefore REJECT stride > 1 on this backend (explicit route) rather than
            # ignore it. The per-block stride is wired on add_block (composed native) and add_native_block
            # (backend='production'). We read time.stride AND the stride= override (nstride covers both).
            if nstride != 1:
                raise ValueError(
                    "add_equation: stride=%d not supported on backend 'aot' (the AOT .so ABI does not "
                    "carry the cadence; the block would run at stride=1 silently). Use "
                    "backend='production' (native path, cadence wired) or a composed native model "
                    "(pops.Model(...)) on the internal native engine API." % nstride)
            # evolve=False (FROZEN block / fixed background) is NOT wired: the add_compiled_block ABI does
            # not carry evolve (add_compiled_block forces it to true on the C++ side) -> the block would be
            # advanced SILENTLY. We REJECT it (rejection rather than silent ignore). For a frozen field,
            # use backend='production' (add_native_block carries evolve) or a composed native model
            # pops.Model(...) -> add_block(..., evolve=False) (or add_background).
            if not evolve:
                raise ValueError(
                    "add_equation: evolve=False not supported on backend 'aot' (the AOT .so ABI does not "
                    "carry evolve; the block would be advanced silently). Use "
                    "backend='production' (native path, evolve wired) or a composed native model "
                    "(pops.Model(...)) on the internal native engine API for a frozen field.")
            # wave_speed_cache (ADC-199): the AOT .so ABI does not carry the wave speed cache -> it would
            # be silently ignored. Reject it (the cache is only wired on the composed native add_block).
            if getattr(spatial, "wave_speed_cache", False):
                raise ValueError(
                    "add_equation: wave_speed_cache not supported on backend 'aot' (the AOT .so ABI does "
                    "not carry the HLL wave speed cache; it would be silently ignored). It is available "
                    "only on the internal native engine API (a composed native model, pops.Model(...)).")
            self._s.add_compiled_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                       spatial.recon, time.kind, nsub, names_arg,
                                       getattr(spatial, "positivity_floor", 0.0))
            return
        if adder == "add_native_block":
            # NATIVE zero-copy (#85): block installed on the REAL System CONTEXT (same path as
            # add_block). Takes a gamma, NO names= (the names/roles come from the .so metadata).
            # End-to-end device/MPI validation from Python is a later dedicated PR.
            if names is not None:
                raise ValueError(
                    "add_equation: names= not supported on the native path (production); the names and "
                    "roles are carried by the compiled model metadata (.so)")
            # PRE-DLOPEN guard at plug time: ALSO covers the cache HIT (where compile_native does not
            # run) -- a stale _pops module would otherwise give a cryptic dlopen 'symbol not found'.
            # wave_speed_cache (ADC-199): the add_native_block ABI does not (yet) carry the wave speed
            # cache -> it would be silently ignored. Reject BEFORE the C++ boundary (and before the ABI
            # check: a clear message rather than a dlopen error); the cache is wired on add_block.
            if getattr(spatial, "wave_speed_cache", False):
                raise ValueError(
                    "add_equation: wave_speed_cache not supported on backend 'production' (the "
                    "add_native_block ABI does not carry the HLL wave speed cache; it would be silently "
                    "ignored). It is available only on the internal native engine API (a composed "
                    "native model, pops.Model(...)).")
            check_compiled_matches_module(getattr(compiled, "abi_key", ""))
            gamma = compiled.gamma if compiled.gamma is not None else PHYSICAL_DEFAULT_GAMMA
            self._s.add_native_block(name, compiled.so_path, spatial.limiter, spatial.flux,
                                     spatial.recon, time.kind, gamma, nsub, evolve, nstride,
                                     getattr(spatial, "positivity_floor", 0.0))
            return
        raise ValueError("add_equation: adder %r unknown (backend %r)" % (adder, backend))

    def set_source_stage(self, name: Any, kind: Any, theta: Any, alpha: Any,
                         krylov_tol: float = 0.0, krylov_max_iters: int = 0,
                         density: str = "", momentum_x: str = "", momentum_y: str = "",
                         energy: str = "", bz_aux_component: int = -1,
                         n_precond_vcycles: int = 0, polar_precond: str = "") -> Any:
        """Attach a Schur-condensed source stage to an already-added block (ADC-308).

        Thin public pass-through to the C++ binding (_pops.System.set_source_stage): same flat
        signature and defaults. add_equation(time=pops.Split(source=pops.CondensedSchur(...))) wires
        this internally; this method exposes the same control for a block added with a plain
        transport time scheme, so cases configure the stage without reaching into the private _s.
        @p name: block; @p kind: 'electrostatic_lorentz'; @p theta in (0, 1]; @p alpha: stage
        coupling. The krylov_* / field descriptors / bz_aux_component defaults reproduce the historical
        bit-identical behavior. ADC-645 adds @p n_precond_vcycles (cartesian stage, 1|2; 0 = the
        historical ONE MG V-cycle per preconditioner application) and @p polar_precond (polar stage,
        'radial_line'|'jacobi'; '' = the historical RadialLine); cross-geometry misuse refuses at the
        native seam. Prerequisite: B_z set via set_magnetic_field beforehand.
        """
        _guard_assembling(self, "set_source_stage")  # frozen once pops.bind completes (ADC-592)
        self._s.set_source_stage(name, kind, theta, alpha, krylov_tol, krylov_max_iters,
                                 density, momentum_x, momentum_y, energy, bz_aux_component,
                                 n_precond_vcycles, polar_precond)

    def add_background(self, name: Any, model: Any, density: Any, spatial: Any = None) -> Any:
        """FROZEN species (not advanced): a fixed background that contributes to the system Poisson (and,
        later, to coupled sources). density: n*n array. Equivalent to add_block(evolve=False) then
        set_density (freeze ADC-592 enforced by the delegated, guarded add_block)."""
        self.add_block(name, model, spatial=spatial, evolve=False)
        self.set_density(name, density)

    def set_poisson(self, rhs: Any = "charge_density", solver: Any = "geometric_mg",
                    bc: Any = "auto", wall: Any = "none", wall_radius: float = 0.0,
                    epsilon: float = 1.0, abs_tol: float = 0.0, rel_tol: Any = None,
                    max_cycles: Any = None, min_coarse: Any = None, pre_smooth: Any = None,
                    post_smooth: Any = None, bottom_sweeps: Any = None,
                    coarse_threshold: Any = None) -> Any:
        """Configure the shared system Poisson solve (thin wrapper over the native binding).

        Low-level runtime seam. The documented PUBLIC elliptic surface is the typed
        ``pops.fields.PoissonProblem(unknown="phi", equation=(-laplacian(phi) == rhs),
        solver=GeometricMG(), bcs=[Dirichlet()])`` attached with ``case.field(...)`` and lowered
        by ``pops.compile`` / ``pops.bind`` (which call this method internally); ``set_poisson``
        stays for that seam, the native/AMR runtime, and the tests.

        Spec 5 sec.8.16 / sec.14.2.6 let ``bc`` and ``wall`` be TYPED objects in addition to the
        legacy strings::

            from pops import Dirichlet, Neumann, Periodic
            from pops.mesh.geometry import Disc, NoWall
            sim.set_poisson(bc=Dirichlet(), wall=Disc(radius=0.4))   # == bc="dirichlet", wall="circle"
            sim.set_poisson(bc="dirichlet", wall=NoWall())           # == bc="dirichlet", wall="none"

        A typed boundary brick (:class:`pops.Dirichlet` / :class:`pops.Neumann` /
        :class:`pops.Periodic`) lowers to its ``bc`` token. A typed
        :class:`pops.mesh.geometry.Disc` lowers to ``wall="circle"`` + its radius (the
        ``wall_radius=`` argument is then ignored in favour of the disc's radius); a
        :class:`pops.mesh.geometry.NoWall` lowers to ``wall="none"``. The legacy string forms are
        passed through unchanged (byte-identical native call). All the other arguments mirror the
        native ``set_poisson`` defaults verbatim.

        ADC-613 adds the GeometricMG V-cycle knobs ``rel_tol`` / ``max_cycles`` / ``min_coarse`` /
        ``pre_smooth`` / ``post_smooth`` / ``bottom_sweeps``; ADC-644 adds ``coarse_threshold`` (a
        total-cell coarsening ceiling, distinct from the per-axis ``min_coarse``). Left ``None`` they
        are NOT forwarded, so the native solver keeps its ``kMG*`` defaults (bit-identical historical
        V-cycle); the typed :class:`pops.solvers.elliptic.GeometricMG` descriptor lowers its resolved
        scalars through here. They are inert for the FFT solver (direct, no iterative tolerance).
        """
        _guard_assembling(self, "set_poisson")  # frozen once pops.bind completes (ADC-592)
        bc = _lower_bc(bc)
        lowered = _lower_wall(wall)
        if lowered is not None:
            wall, wall_radius = lowered  # typed wall overrides the wall string + radius
        # PRE-BIND route validation (ADC-584): unknown tokens are refused here (family +
        # requested token + valid set), never defaulted; a valid Route IS its wire token
        # (str subclass), so the native call below stays byte-identical.
        rhs = _resolve_route("poisson_rhs", rhs, context="set_poisson")
        solver = _resolve_route("field_solver", solver, context="set_poisson")
        bc = _resolve_route("poisson_bc", bc, context="set_poisson")
        wall = _resolve_route("wall", wall, context="set_poisson")
        # ADC-613: forward the GeometricMG V-cycle knobs ONLY when the caller (or the lowered typed
        # descriptor) set them, so the native set_poisson keeps its kMG*-sourced defaults otherwise
        # (bit-identical). A None means "unspecified" -> not passed -> native default.
        mg_kwargs = {}
        if rel_tol is not None:
            mg_kwargs["rel_tol"] = float(rel_tol)
        if max_cycles is not None:
            mg_kwargs["max_cycles"] = int(max_cycles)
        if min_coarse is not None:
            mg_kwargs["min_coarse"] = int(min_coarse)
        if pre_smooth is not None:
            mg_kwargs["pre_smooth"] = int(pre_smooth)
        if post_smooth is not None:
            mg_kwargs["post_smooth"] = int(post_smooth)
        if bottom_sweeps is not None:
            mg_kwargs["bottom_sweeps"] = int(bottom_sweeps)
        if coarse_threshold is not None:  # ADC-644: total-cell coarsening ceiling (0 = disabled).
            mg_kwargs["coarse_threshold"] = int(coarse_threshold)
        self._s.set_poisson(rhs=rhs, solver=solver, bc=bc, wall=wall,
                            wall_radius=wall_radius, epsilon=epsilon, abs_tol=abs_tol, **mg_kwargs)

    def add_elliptic_model(self, name: Any, model: Any, solver: Any = None, bc: Any = "auto",
                           wall: Any = "none", wall_radius: float = 0.0) -> Any:
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
        self.set_poisson(rhs=rhs_tok, solver=kind, bc=bc, wall=wall, wall_radius=wall_radius,
                         epsilon=model.operator.epsilon)

    def add_coupling(self, coupling: Any) -> Any:
        """Add an inter-species coupling (operator-split, applied after transport):

        - NAMED object pops.Ionization / Collision / ThermalExchange -> a PRESET lowering to the generic
          coupled source (ADC-595): the fixed formula is emitted as a CoupledSource with a DECLARED
          conservation contract, compiled to bytecode, and registered as a typed coupling operator;
        - CompiledCoupledSource (pops.dsl.CoupledSource(...).compile(...)) -> GENERIC bytecode source
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
            raise TypeError("add_coupling expects pops.Ionization / Collision / ThermalExchange or a "
                            "CompiledCoupledSource (pops.dsl.CoupledSource(...).compile(...))")
        # Validate the DECLARED contract symbolically (Python); the C++ revalidates at registration. A
        # created role (ionization) may net-source, so compile without verify_conservation.
        preset.source.verify_declared_contract(conserved=preset.conserved, created=preset.created)
        args = coupling_operator_args(preset.source.compile(), preset.conserved, preset.created,
                                      frequency=preset.frequency)
        self._s.add_coupling_operator(*args)
