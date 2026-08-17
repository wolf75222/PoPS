"""AmrSystem equation mixin (Spec-4 PR-F).

``add_equation`` (the AMR backend dispatcher) plus the module-level guard
``_reject_newton_amr_compiled`` used only by this path. Mixed in via inheritance; operates on
``self._s``.
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
    """Reject Newton metadata absent from the compiled AMR package ABI.

    The AMR spatial runtime never turns this descriptor into an implicit step. A typed Program
    primitive must own the local solve, its options and its report. The flat ``.so`` block-loader ABI
    transports neither these options nor ``newton_diagnostics``; accepting them here would silently
    replace the authored values with defaults before the Program is installed.
    """
    if (
        getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS) != NEWTON_DEFAULT_MAX_ITERS
        or getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL) != NEWTON_DEFAULT_REL_TOL
        or getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL) != NEWTON_DEFAULT_ABS_TOL
        or getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS) != NEWTON_DEFAULT_FD_EPS
        or getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING) != NEWTON_DEFAULT_DAMPING
        or getattr(time, "newton_diagnostics", False)
    ):
        raise ValueError(
            "%s : the Newton options/diagnostics (newton_max_iters/rel_tol/abs_tol/fd_eps/damping/"
            "diagnostics) are not transported by the AMR production package; "
            "the AMR target does not yet provide the typed local nonlinear/Newton Program "
            "primitive that could consume them. Keep the AMR Program explicit, use a typed "
            "LocalLinear solve for a linear implicit operator, or use the uniform System "
            "source-Newton route. The AMR spatial runtime has no private Newton fallback." % label
        )


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
                "protocol or be an exact private Spatial value; got %r" % type(spatial).__name__
            )
        first, second = runtime_spatial(), runtime_spatial()
        if type(first) is not Spatial or type(second) is not Spatial:
            raise TypeError("runtime_spatial() must return an exact private Spatial value")
        if first != second:
            raise ValueError("runtime_spatial() must be deterministic")
        return first

    def add_equation(
        self,
        name: Any,
        model: Any,
        spatial: Any = None,
        time: Any = None,
        substeps: Any = None,
        _bind_params: Any = None,
    ) -> Any:
        """Add the SINGLE AMR equation/block by dispatching on the TYPE of @p model (DSL Phase D).

        Low-level runtime seam. The documented PUBLIC path is the typed ``pops.Case`` assembly
        resolved with ``pops.resolve(case, layout=...)``, compiled with ``pops.compile(plan)`` and
        wired by ``pops.bind``;
        ``add_equation`` stays for that seam and the tests.

        Dispatch:

        - a private ``ModelSpec`` compiles to one production ``Prepared*Block`` package, then
          installs through the CompiledModel path; native ``AmrSystem::add_block(ModelSpec)`` stays
          retired;
        - a CompiledModel(backend='production', target='amr_system') installs one complete inert
          package containing its prepared block and elliptic attachments, so the block runs the
          same AMR hierarchy as the native-brick ABI (conservative reflux, regrid), ZERO-COPY.

        The ``time`` value carried by a block is immutable Program-authoring metadata, not an
        executable method in the AMR spatial runtime. The compiled ``pops.Program`` installed after
        all blocks is the only time authority. Explicit Programs own their RK stages and reflux
        weights. The ``IMEX`` token alone remains authoring metadata; a request for an implicit mask,
        Newton controls, or diagnostics fails closed until a typed implicit Program primitive exists.
        It never reaches a private backward-Euler/Newton engine.
        ``recon="primitive"`` and fluxes ``roe`` / ``hllc`` use the same compiled spatial dispatch as
        the native-brick branch. The low-level dispatch also contains the WENO5-Z stencil and its three-cell
        halo, but the resolved Case route accepts it only when the owner-qualified coarse/fine
        provider certifies order 5 and ghost depth 3. The native catalogue resolves that provider
        from the reconstruction requirements and never lowers the coarse/fine interface order
        silently.

        MULTIRATE CADENCE (stride) and PARTIAL IMEX MASK (implicit_vars / implicit_roles):

        - stride is transported by ``add_native_block`` / ``pops_install_native_amr`` and executed
          by the compiled hold-then-catch-up Program (clocks + subcycle + SampleAndHold);
        - a non-empty implicit mask is consumed by the typed ``Program.implicit_source`` primitive
          after ``add_equation`` records the descriptor; Newton options stay refused.

        @p spatial: private adapter lowered from ``pops.numerics.FiniteVolume(...)``.
        @p time: private engine policy lowered from an explicit ``pops.Program`` or a
        ``pops.lib.time`` factory. @p substeps: overrides time.substeps.
        """
        from pops.runtime._lifecycle import guard_assembling

        guard_assembling(self, "add_equation")  # frozen once pops.bind completes (ADC-592)
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.loader import CompiledModel

        spatial = self._lower_spatial(spatial)
        time = time if time is not None else Explicit()

        # positivity_floor (ADC-259) IS wired on the NATIVE AMR transport (Density-role face states +
        # C/F fine ghost means). It is threaded below on the ModelSpec (native) branch and on the
        # amr-schur transport (the recursive add_equation on time.hyperbolic). The COMPILED .so path
        # carries it too: the generated package marshals it through pops_install_native_amr.

        nsub = positive_int(
            substeps if substeps is not None else getattr(time, "substeps", 1),
            where="AmrSystem.add_equation.substeps",
        )

        # --- ModelSpec: native bricks composed through the sole Python dispatch seam ---
        # Forward the complete authoring request to the native contract. Unsupported masks and
        # Newton controls are rejected there rather than retained by the spatial runtime.
        if isinstance(model, ModelSpec):
            from pops.runtime._modelspec_compile import compile_modelspec_package

            compiled = compile_modelspec_package(model, name=name, target="amr_system")
            elliptic = str(getattr(model, "elliptic", "") or "")
            if elliptic in ("charge", "background", "gravity"):
                slots = list(self._s.field_provider_slots())
                if not slots:
                    self.set_poisson()
            return self.add_equation(
                name,
                compiled,
                spatial=spatial,
                time=time,
                substeps=substeps,
                _bind_params=_bind_params,
            )

        if not isinstance(model, CompiledModel):
            raise TypeError(
                "AmrSystem.add_equation: model must be a private ModelSpec or detached "
                "CompiledModel; received %r" % type(model).__name__
            )

        compiled = model
        if compiled.backend != "production":
            raise ValueError(
                "AmrSystem.add_equation: compiled packages must use backend='production'; "
                "received backend=%r" % compiled.backend
            )
        if getattr(compiled, "target", "system") != "amr_system":
            raise ValueError(
                "AmrSystem.add_equation: the CompiledModel was compiled for target='system'; "
                "re-resolve and compile the Case for its AMR layout so that the loader inlines "
                "the complete prepared AMR package (symbol pops_install_native_amr)"
            )

        # Descriptor-owned model predicates are shared verbatim with System and availability.
        _check_riemann_requirement_contract(
            spatial.riemann_capability_contract,
            compiled,
            "AmrSystem.add_equation",
            flux=spatial.flux,
        )

        nstride = positive_int(
            getattr(time, "stride", 1), where="AmrSystem.add_equation.stride"
        )
        # Newton options / diagnostics: same flat ABI -> neither the options nor the report transit
        # through the .so loader. Explicit rejection prevents silent substitution of the prepared
        # provider defaults, in parity with the stride/mask rejection above and System.add_equation.
        _reject_newton_amr_compiled("AmrSystem.add_equation", time)
        # positivity_floor (ADC-322): the regenerated .so loader carries the Zhang-Shu floor in
        # the complete prepared package, so it is threaded through instead of rejected. 0
        # (default) = inactive, bit-identical. The native package seam validates floor >= 0 and
        # finite (parity with add_block).

        # PRE-DLOPEN guard at attach (covers the cache HIT, cf. System.add_equation): module
        # _pops stale vs .so compiled against the up-to-date headers -> actionable error, not a dlopen
        # 'symbol not found' cryptic message.
        from pops.codegen.abi import check_compiled_matches_module

        check_compiled_matches_module(getattr(compiled, "abi_key", ""))
        from pops.runtime._amr_package_lane import (
            ensure_amr_native_package_lane,
            ensure_native_block_state_route,
        )

        ensure_amr_native_package_lane(self, compiled)
        ensure_native_block_state_route(self, name, compiled)
        gamma = native_real(
            compiled.gamma if compiled.gamma is not None else PHYSICAL_DEFAULT_GAMMA,
            where="AmrSystem.add_equation.gamma",
        )
        runtime_names = tuple(getattr(compiled, "runtime_param_names", ()) or ())
        if _bind_params is None:
            if runtime_names:
                raise ValueError(
                    "AmrSystem.add_equation: compiled package declares runtime parameters; "
                    "install it through pops.bind so BindSchema resolves one complete vector"
                )
            bind_values = []
        else:
            bind_values = [
                native_real(value, where="AmrSystem.add_equation.bind_params[%d]" % index)
                for index, value in enumerate(_bind_params)
            ]
            if len(bind_values) != len(runtime_names):
                raise ValueError(
                    "AmrSystem.add_equation: bound parameter vector has %d values, expected %d"
                    % (len(bind_values), len(runtime_names))
                )
        spatial_options: dict[str, bool | float] = {
            "wave_speed_cache": bool(getattr(spatial, "wave_speed_cache", False)),
        }
        if getattr(spatial, "weno_epsilon", None) is not None:
            spatial_options["weno_epsilon"] = native_real(
                spatial.weno_epsilon, where="AmrSystem.add_equation.weno_epsilon"
            )
        weno_epsilon = spatial_options.get(
            "weno_epsilon", float(numerical_defaults_report()["weno"]["epsilon"])
        )
        positivity_floor = native_real(
            getattr(spatial, "positivity_floor", 0.0),
            where="AmrSystem.add_equation.positivity_floor",
        )
        from pops.runtime._system_install import (
            _candidate_coupling_contracts,
            _model_coupling_contract,
        )

        contract = _model_coupling_contract(
            compiled, dimension=compiled.native_dimension, adiabatic_index=gamma
        )
        contract_candidate = _candidate_coupling_contracts(self, name, contract)
        if spatial.external_flux_id is not None:
            if "amr" not in spatial.external_flux_supported_layouts:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r does not support AMR"
                    % spatial.external_flux_id
                )
            if runtime_names:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann ABI does not transport model "
                    "RuntimeParams"
                )
            expected_provider_count = len(tuple(compiled.provider_components))
            external_shape = (
                spatial.external_flux_dimension,
                spatial.external_flux_n_vars,
                spatial.external_flux_provider_count,
            )
            compiled_shape = (
                compiled.native_dimension,
                compiled.n_vars,
                expected_provider_count,
            )
            if external_shape != compiled_shape:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r carries model shape %r, "
                    "not %r" % (spatial.external_flux_id, external_shape, compiled_shape)
                )
            if spatial.external_flux_model_identity != compiled.model_hash:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r targets model %r, not %r"
                    % (
                        spatial.external_flux_id,
                        spatial.external_flux_model_identity,
                        compiled.model_hash,
                    )
                )
            if spatial.external_flux_native_abi_key != compiled.abi_key:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann brick %r was built for a different "
                    "native ABI" % spatial.external_flux_id
                )
            if spatial_options["wave_speed_cache"]:
                raise ValueError(
                    "AmrSystem.add_equation: external Riemann ABI does not transport "
                    "wave_speed_cache"
                )
            self._s._register_external_riemann_package(
                name,
                spatial.external_flux_library_path,
                spatial.external_flux_id,
                spatial.external_flux_library_sha256,
                spatial.external_flux_n_vars,
                spatial.external_flux_provider_count,
                spatial.external_flux_model_identity,
                name,
                spatial.limiter,
                spatial.recon,
                time.kind,
                gamma,
                nsub,
                nstride,
                positivity_floor,
                weno_epsilon,
            )
        else:
            self._s._install_native_block(
                name,
                compiled.so_path,
                compiled.model_hash,
                str(compiled.binary_identity),
                spatial.limiter,
                spatial.flux,
                spatial.recon,
                time.kind,
                gamma,
                nsub,
                nstride,
                bind_values,
                positivity_floor,
                **spatial_options,
            )
        from pops.runtime._cadence_install import _record_block_time

        _record_block_time(self, name, time)
        self._coupling_block_contracts = contract_candidate
