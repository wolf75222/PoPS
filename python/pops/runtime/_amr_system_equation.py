"""AmrSystem equation/aux mixin (Spec-4 PR-F).

``add_equation`` (the AMR backend dispatcher) + the named-aux resolution / set of
:class:`pops.runtime._amr_system.AmrSystem`, plus the module-level guard
``_reject_newton_amr_compiled`` used only by this path. Mixed in via inheritance; operates on
``self._s`` and ``self._aux_field_index``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._bootstrap import ModelSpec
from pops.runtime._numeric import native_real, positive_int
from pops.runtime._engine_descriptors import Spatial, Explicit
from pops.runtime.routes import (
    check_riemann_requirement_contract as _check_riemann_requirement_contract,
)
from pops.runtime.defaults import (
    NEWTON_DEFAULT_ABS_TOL,
    NEWTON_DEFAULT_DAMPING,
    NEWTON_DEFAULT_FAIL_POLICY,
    NEWTON_DEFAULT_FD_EPS,
    NEWTON_DEFAULT_MAX_ITERS,
    NEWTON_DEFAULT_REL_TOL,
    PHYSICAL_DEFAULT_GAMMA,
    numerical_defaults_report,
)

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


def _reject_newton_amr_compiled(label: Any, time: Any) -> Any:
    """Reject Newton options absent from the compiled AMR package ABI. On the native side, the
    Newton OPTIONS and newton_diagnostics REPORT use the unified runtime at every block count; the
    flat ABI of the .so loader transports NEITHER
    the options (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy) NOR the report. Passed
    via the loader, they would be taken at their defaults SILENTLY (iters=2, no report). We
    REJECT them explicitly (same spirit as the stride/mask rejection of the AMR production path). For these
    parameters : AmrSystem.add_block (native model) or add_compiled_model(AmrSystem&) directly (C++)."""
    if (getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS)
            != NEWTON_DEFAULT_MAX_ITERS
            or getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL)
            != NEWTON_DEFAULT_REL_TOL
            or getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL)
            != NEWTON_DEFAULT_ABS_TOL
            or getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS)
            != NEWTON_DEFAULT_FD_EPS
            or getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING)
            != NEWTON_DEFAULT_DAMPING
            or getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY)
            != NEWTON_DEFAULT_FAIL_POLICY
            or getattr(time, "newton_diagnostics", False)):
        raise ValueError(
            "%s : the Newton options/diagnostics (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/"
            "fail_policy/diagnostics) are not transported by the AMR production package; "
            "They are available only on the internal native engine API (a private ModelSpec on "
            "the AMR layout)." % label)


class _AmrSystemEquation(_AmrSystem):
    """add_equation + named-aux methods of AmrSystem."""

    def _lower_spatial(self, spatial: Any) -> Spatial:
        """Return the exact runtime Spatial consumed by AMR install and bound snapshots."""
        if spatial is None:
            return Spatial()
        if type(spatial) is Spatial:
            return spatial
        runtime_spatial = getattr(spatial, "runtime_spatial", None)
        if not callable(runtime_spatial):
            raise TypeError(
                "AMR spatial selection must implement the pops.numerics finite-volume lowering "
                "protocol or be an exact private Spatial value; got %r"
                % type(spatial).__name__)
        first, second = runtime_spatial(), runtime_spatial()
        if type(first) is not Spatial or type(second) is not Spatial:
            raise TypeError("runtime_spatial() must return an exact private Spatial value")
        if first != second:
            raise ValueError("runtime_spatial() must be deterministic")
        return first

    def add_equation(self, name: Any, model: Any, spatial: Any = None, time: Any = None,
                     substeps: Any = None, _bind_params: Any = None) -> Any:
        """Add the SINGLE AMR equation/block by dispatching on the TYPE of @p model (DSL Phase D).

        Low-level runtime seam. The documented PUBLIC path is the typed ``pops.Case`` assembly
        resolved with ``pops.resolve(case, layout=...)``, compiled with ``pops.compile(plan)`` and
        wired by ``pops.bind``;
        ``add_equation`` stays for that seam and the tests.

        Dispatch:

        - a private ``ModelSpec`` -> add_block (native bricks composed on the hierarchy);
        - a CompiledModel(backend='production', target='amr_system') installs a package whose loader
          inlines add_compiled_model(AmrSystem&), so the block runs
          the SAME AMR hierarchy as add_block (conservative reflux, regrid), ZERO-COPY.

        Time handling is wired to ``Explicit(method="ssprk2"|"ssprk3"|"euler")`` and ``IMEX``.
        SSPRK2/Heun and SSPRK3 evaluate the explicit source at every stage and expose the matching
        stage-weighted effective face flux to conservative reflux. At coarse/fine boundaries, every
        stage samples the authored parent time window at its RK abscissa. IMEX remains the distinct
        forward-Euler transport plus backward-Euler stiff-source split; the cell-local implicit source
        does not enter reflux. ``recon="primitive"`` and fluxes ``roe`` / ``hllc`` use the same
        compiled dispatch as ``add_block``. The low-level dispatch also contains the WENO5-Z
        stencil and its three-cell halo, but the resolved Case route accepts it only when the
        owner-qualified coarse/fine provider certifies order 5 and ghost depth 3. The native
        catalogue resolves that provider from the reconstruction requirements and never lowers
        the coarse/fine interface order silently.

        MULTIRATE CADENCE (stride) and PARTIAL IMEX MASK (implicit_vars / implicit_roles):

        - private ``ModelSpec`` path: FORWARDED to AmrSystem::add_block, which SUPPORTS and
          validates them (parity with the add_block wrapper);
        - CompiledModel production path (.so): explicitly REJECTED (ValueError). The flat ABI of the
          package ABI does not transport them; they would be taken
          at their defaults SILENTLY (stride=1, full backward-Euler). For a multirate .so or one with a
          partial IMEX mask, use AmrSystem.add_block (native) or add_compiled_model(AmrSystem&) directly
          (C++), which expose stride and the mask.

        @p spatial: private adapter lowered from ``pops.numerics.FiniteVolume(...)``.
        @p time: private engine policy lowered from an explicit ``pops.Program`` or a
        ``pops.lib.time`` factory. @p substeps: overrides time.substeps.
        """
        from pops.runtime._lifecycle import guard_assembling
        guard_assembling(self, "add_equation")  # frozen once pops.bind completes (ADC-592)
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.loader import CompiledModel
        from pops.physics.aux import AUX_NAMED_BASE

        spatial = self._lower_spatial(spatial)
        time = time if time is not None else Explicit()

        # positivity_floor (ADC-259) IS wired on the NATIVE AMR transport (Density-role face states +
        # C/F fine ghost means). It is threaded below on the ModelSpec (native) branch and on the
        # amr-schur transport (the recursive add_equation on time.hyperbolic). The COMPILED .so path
        # carries it too: the generated package marshals it through pops_install_native_amr.

        nsub = positive_int(
            substeps if substeps is not None else getattr(time, "substeps", 1),
            where="AmrSystem.add_equation.substeps")

        # --- ModelSpec: native bricks composed -> add_block (existing path) ---
        # We FORWARD stride (multirate, capstone iv) AND the partial IMEX mask implicit_vars /
        # implicit_roles (capstone vii), exactly like the AmrSystem.add_block wrapper above:
        # the C++ AmrSystem::add_block SUPPORTS and validates them (empty -> full backward-Euler; a
        # mask requested in explicit raises a clear error on the C++ side. Do NOT duplicate these
        # guards here.
        if isinstance(model, ModelSpec):
            # Native model: Newton options and diagnostics are wired at every block count. No facade
            # filtering: C++ AmrSystem::add_block validates the complete contract.
            spatial_options: dict[str, bool | float] = {
                "wave_speed_cache": bool(getattr(spatial, "wave_speed_cache", False)),
            }
            if getattr(spatial, "weno_epsilon", None) is not None:
                spatial_options["weno_epsilon"] = native_real(
                    spatial.weno_epsilon, where="AmrSystem.add_equation.weno_epsilon")
            self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                              nsub, getattr(time, "stride", 1),
                              getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                              getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                              native_real(getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL),
                                          where="AmrSystem.add_equation.newton_rel_tol"),
                              native_real(getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL),
                                          where="AmrSystem.add_equation.newton_abs_tol"),
                              native_real(getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS),
                                          where="AmrSystem.add_equation.newton_fd_eps"),
                              native_real(getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING),
                                          where="AmrSystem.add_equation.newton_damping"),
                              getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY),
                              getattr(time, "newton_diagnostics", False),
                              native_real(getattr(spatial, "positivity_floor", 0.0),
                                          where="AmrSystem.add_equation.positivity_floor"),
                              **spatial_options)
            return

        if not isinstance(model, CompiledModel):
            raise TypeError(
                "AmrSystem.add_equation: model must be a private ModelSpec or detached "
                "CompiledModel; received %r" % type(model).__name__)

        compiled = model
        if compiled.backend != "production":
            raise ValueError(
                "AmrSystem.add_equation: compiled packages must use backend='production'; "
                "received backend=%r" % compiled.backend)
        if getattr(compiled, "target", "system") != "amr_system":
            raise ValueError(
                "AmrSystem.add_equation: the CompiledModel was compiled for target='system'; "
                "re-resolve and compile the Case for its AMR layout so that the loader inlines "
                "add_compiled_model(AmrSystem&) (symbol pops_install_native_amr)")

        # Descriptor-owned model predicates are shared verbatim with System and availability.
        _check_riemann_requirement_contract(
            spatial.riemann_capability_contract,
            compiled,
            "AmrSystem.add_equation",
            flux=spatial.flux,
        )

        # The package ABI transports NEITHER the
        # multirate cadence (stride) NOR the partial IMEX mask (implicit_vars / implicit_roles):
        # add_compiled_model(AmrSystem&) exposes them only DIRECTLY (C++ path). Passed through the
        # loader, they would take their defaults (stride=1, empty mask = full backward-Euler) SILENTLY.
        # We REJECT them rather than ignore them (explicit route, same spirit as the rejection
        # of stride/mask on the compiled backends of System.add_equation, cf. ~lines 886-955).
        nstride = getattr(time, "stride", 1)
        if nstride != 1 and spatial.external_flux_id is None:
            raise ValueError(
                "AmrSystem.add_equation: stride=%d not transported by the production AMR path "
                "(the block would otherwise run at stride=1 silently). "
                "The multirate cadence is available only on the internal native engine API "
                "(a private ModelSpec on the AMR layout)." % nstride)
        if getattr(time, "implicit_vars", []) or getattr(time, "implicit_roles", []):
            raise ValueError(
                "AmrSystem.add_equation: implicit_vars / implicit_roles (partial IMEX mask) not "
                "transported by the production AMR package (the "
                "mask would be empty = full backward-Euler silently). The partial IMEX mask is "
                "available only on the internal native engine API (a private ModelSpec on the "
                "AMR layout).")
        # Newton options / diagnostics: same flat ABI -> neither the options nor the report transit
        # through the .so loader. Explicit rejection (otherwise iters=2 / no report silently), parity with
        # the stride/mask rejection above and with System.add_equation (compiled backend).
        _reject_newton_amr_compiled("AmrSystem.add_equation", time)
        # positivity_floor (ADC-322): the regenerated .so loader carries the Zhang-Shu floor now
        # (pops_install_native_amr -> add_compiled_model -> set_compiled_block), so it is threaded
        # through instead of rejected. 0 (default) = inactive, bit-identical. The C++
        # The native package seam validates floor >= 0 and finite (parity with add_block).

        # PRE-DLOPEN guard at attach (covers the cache HIT, cf. System.add_equation): module
        # _pops stale vs .so compiled against the up-to-date headers -> actionable error, not a dlopen
        # 'symbol not found' cryptic message.
        from pops.codegen.abi import check_compiled_matches_module
        check_compiled_matches_module(getattr(compiled, "abi_key", ""))
        gamma = native_real(
            compiled.gamma if compiled.gamma is not None else PHYSICAL_DEFAULT_GAMMA,
            where="AmrSystem.add_equation.gamma")
        runtime_names = tuple(getattr(compiled, "runtime_param_names", ()) or ())
        if _bind_params is None:
            if runtime_names:
                raise ValueError(
                    "AmrSystem.add_equation: compiled package declares runtime parameters; "
                    "install it through pops.bind so BindSchema resolves one complete vector")
            bind_values = []
        else:
            bind_values = [
                native_real(value, where="AmrSystem.add_equation.bind_params[%d]" % index)
                for index, value in enumerate(_bind_params)
            ]
            if len(bind_values) != len(runtime_names):
                raise ValueError(
                    "AmrSystem.add_equation: bound parameter vector has %d values, expected %d"
                    % (len(bind_values), len(runtime_names)))
        spatial_options: dict[str, bool | float] = {
            "wave_speed_cache": bool(getattr(spatial, "wave_speed_cache", False)),
        }
        if getattr(spatial, "weno_epsilon", None) is not None:
            spatial_options["weno_epsilon"] = native_real(
                spatial.weno_epsilon, where="AmrSystem.add_equation.weno_epsilon")
        positivity_floor = native_real(
            getattr(spatial, "positivity_floor", 0.0),
            where="AmrSystem.add_equation.positivity_floor")
        if spatial.external_flux_id is not None:
            if "amr" not in spatial.external_flux_supported_layouts:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r does not support AMR"
                    % spatial.external_flux_id)
            if runtime_names:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann ABI v2 does not transport model "
                    "RuntimeParams")
            if spatial.external_flux_model_identity != compiled.model_hash:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r targets model %r, not %r"
                    % (spatial.external_flux_id, spatial.external_flux_model_identity,
                       compiled.model_hash))
            if spatial.external_flux_native_abi_key != compiled.abi_key:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r was built for a different "
                    "native ABI" % spatial.external_flux_id)
            if spatial_options["wave_speed_cache"]:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann ABI v2 does not transport "
                    "wave_speed_cache")
            self._s._install_external_riemann_block(
                name, spatial.external_flux_library_path, spatial.external_flux_id,
                spatial.external_flux_library_sha256, spatial.limiter, spatial.recon,
                time.kind, gamma, nsub, nstride, compiled.n_vars, compiled.n_aux,
                compiled.model_hash,
                positivity_floor,
                spatial_options.get(
                    "weno_epsilon",
                    float(numerical_defaults_report()["weno"]["epsilon"])),
            )
        else:
            self._s._install_native_block(
                name, compiled.so_path, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                gamma, nsub, bind_values, positivity_floor, **spatial_options,
            )
        # ADC-291: record the named aux fields the block declares (component of the k-th name =
        # AUX_NAMED_BASE + k), so set_aux_field(block, name, array) can resolve name -> component.
        extra = list(getattr(compiled, "aux_extra_names", []) or [])
        if extra:
            self._aux_field_index[name] = {nm: AUX_NAMED_BASE + k for k, nm in enumerate(extra)}

    def _resolve_aux_field(self, block: Any, name: Any) -> Any:
        """Resolve (block, named aux field) -> aux channel component (ADC-291). Mirror of
        System._resolve_aux_field: a canonical name is redirected to its dedicated path; an unknown
        block or an undeclared field raises (no silent component-0 fallback)."""
        from pops.physics.aux import AUX_CANONICAL
        if name in AUX_CANONICAL:
            if name == "B_z":
                raise ValueError(
                    "set_aux_field: 'B_z' (magnetic field) is set via sim.set_magnetic_field(Bz), "
                    "NOT via set_aux_field (B_z is a canonical aux field, not a named field).")
            raise ValueError(
                "set_aux_field: '%s' is a CANONICAL aux field (derived by the solver, not settable); "
                "set_aux_field only carries the NAMED fields declared by m.aux_field(...)." % name)
        table = self._aux_field_index.get(block)
        if table is None:
            raise ValueError(
                "set_aux_field: block '%s' unknown (or bound without a named aux field); declare "
                "m.aux_field('%s') on that block's model in the pops.Case." % (block, name))
        if name not in table:
            raise ValueError(
                "set_aux_field: aux field '%s' not declared by block '%s'; known named fields: %s"
                % (name, block, sorted(table)))
        return table[name]

    def set_aux_field(self, block: Any, name: Any, field: Any, halo: Any = None) -> Any:
        """Set a model-NAMED aux field of @p block (declared via m.aux_field(name)) on the AMR
        hierarchy. AMR counterpart of System.set_aux_field. ``field`` is exactly one 2D
        ``(ny, nx)`` array on the coarse level; it is static (re-applied each step, injected to the
        fine levels, survives a regrid). Call before the first step (like ``set_density``).

        @p halo (ADC-369): an optional ``pops.mesh.AuxHalo`` declaring this field's coarse-level ghost
        boundary policy (foextrap / dirichlet), applied to the non-periodic faces after the shared aux
        fill. Default None inherits the shared aux BC (bit-identical)."""
        import numpy as np
        comp = self._resolve_aux_field(block, name)
        arr = np.asarray(field, dtype=float)
        self._s.set_aux_field_component(comp, arr)
        if halo is not None:
            self._s.set_aux_field_halo_component(comp, halo.bc_type, halo.value)
