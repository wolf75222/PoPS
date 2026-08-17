"""System install mixin (Spec-4 PR-F): equation/coupling installation.

Holds the densest part of :class:`pops.runtime._system.System`: ``add_equation``
(direct native versus compiled production-package installation),
``add_background``, ``add_elliptic_model`` and ``add_coupling``. Mixed into ``System`` via
inheritance; methods operate on ``self._s`` (the compiled facade).  Native packages are staged
collectively; the uniform installer finalizes their one global auxiliary registry only after every
block package has registered its routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._bootstrap import ModelSpec
from pops.runtime._lifecycle import guard_assembling as _guard_assembling
from pops.runtime._numeric import native_real, positive_int
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

# Typed Poisson boundary and numerical lowerers are split out for the 500-line cap (ADC-550).
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


def _authored_modelspec_block_options(
    name: Any, spec: Any, spatial: Any, time: Any, *, evolve: bool
) -> dict[str, Any]:
    """Capture ModelSpec inspect rows the compiled-package ABI does not retain."""
    defaults = numerical_defaults_report()
    physical = defaults["physical"]
    newton = {
        "max_iters": int(getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS)),
        "rel_tol": float(getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL)),
        "abs_tol": float(getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL)),
        "fd_eps": float(getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS)),
        "damping": float(getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING)),
        "diagnostics": False,
    }
    return {
        "name": name,
        "transport": str(getattr(spec, "transport", "") or ""),
        "time": str(getattr(time, "kind", "explicit") or "explicit"),
        "evolve": evolve,
        "newton": newton,
        "positivity_floor": float(getattr(spatial, "positivity_floor", 0.0) or 0.0),
        "physical": {
            "cs2": float(getattr(spec, "cs2", physical["fluid_state_cs2"])),
            "q": float(getattr(spec, "q", physical["charge_q"])),
        },
    }


def _model_state_schema(model: Any, *, dimension: int) -> Any:
    """Resolve the exact role schema carried by one installed model, before coupling lowering."""
    from pops.physics.roles import StateSchema

    roles = getattr(model, "cons_roles", None)
    if roles is not None:
        return StateSchema.resolve(roles, dimension=dimension, where="installed compiled model")
    transport = getattr(model, "transport", None)
    axes = tuple("momentum:%d" % axis for axis in range(dimension))
    if transport == "exb":
        roles = ("density",)
    elif transport == "isothermal":
        roles = ("density",) + axes
    elif transport == "compressible":
        roles = ("density",) + axes + ("energy",)
    else:
        raise TypeError("installed native model does not expose an exact StateSchema")
    return StateSchema.resolve(roles, dimension=dimension, where="installed native model")


def _model_coupling_contract(model: Any, *, dimension: int, adiabatic_index: Any = None) -> Any:
    """Prepare coupling-visible model metadata before any native registry mutation."""
    from pops.physics.roles import CouplingBlockContract

    gamma = adiabatic_index
    if gamma is None:
        gamma = getattr(model, "gamma", None)
    if gamma is None:
        gamma = PHYSICAL_DEFAULT_GAMMA
    gamma = native_real(gamma, where="installed model adiabatic_index")
    return CouplingBlockContract(
        _model_state_schema(model, dimension=dimension),
        (("adiabatic_index", gamma),),
    )


def _candidate_coupling_contracts(owner: Any, name: Any, contract: Any) -> dict[str, Any]:
    """Allocate a complete candidate registry before the native block installer can mutate."""
    if not isinstance(name, str) or not name:
        raise TypeError("installed block identity must be a non-empty string")
    candidate = dict(getattr(owner, "_coupling_block_contracts", {}))
    if name in candidate:
        raise ValueError("installed block identity %r is already registered" % name)
    candidate[name] = contract
    return candidate


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
        _from_modelspec: bool = False,
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
            from pops.runtime._modelspec_compile import compile_modelspec_package

            if names is not None:
                raise ValueError(
                    "add_equation: names= is not supported for ModelSpec compilation; names and "
                    "roles are immutable artifact metadata"
                )
            compiled = compile_modelspec_package(model, name=name, target="system")
            records = dict(getattr(self, "_authored_block_options", {}))
            records[name] = _authored_modelspec_block_options(
                name, model, spatial, time, evolve=evolve
            )
            self._authored_block_options = records
            return self.add_equation(
                name,
                compiled,
                spatial=spatial,
                time=time,
                substeps=substeps,
                evolve=evolve,
                stride=stride,
                _bind_params=_bind_params,
                _from_modelspec=True,
            )

        # Same rules for the Newton options/diagnostics (IMEX): not carried by the .so ABI.
        # Non-default values would be ignored SILENTLY -> explicit rejection.
        if not _from_modelspec and (
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
        from pops.runtime._amr_package_lane import ensure_native_block_state_route

        ensure_native_block_state_route(self, name, compiled)
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
        contract = _model_coupling_contract(
            compiled, dimension=compiled.native_dimension, adiabatic_index=gamma
        )
        contract_candidate = _candidate_coupling_contracts(self, name, contract)
        if spatial.external_flux_id is not None:
            if "uniform" not in spatial.external_flux_supported_layouts:
                raise ValueError(
                    "add_equation: external Riemann brick %r does not support uniform layouts"
                    % spatial.external_flux_id
                )
            if runtime_names:
                raise ValueError(
                    "add_equation: external Riemann ABI does not transport model RuntimeParams"
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
                    "add_equation: external Riemann brick %r carries model shape %r, not %r"
                    % (spatial.external_flux_id, external_shape, compiled_shape)
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
            expected_system_abi_key = (
                "pops.external-riemann.system/v7;receiver=prepared-native-package;"
                "providers=qualified;dim=%s" % compiled.native_dimension
            )
            if (
                spatial.external_flux_system_abi_version != 7
                or spatial.external_flux_system_abi_key != expected_system_abi_key
            ):
                raise ValueError(
                    "add_equation: external Riemann brick %r lacks the System v7 "
                    "prepared-package ABI" % spatial.external_flux_id
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
                evolve,
                nstride,
                positivity_floor,
                weno_epsilon,
            )
            self._pending_native_packages += 1
        else:
            self._s._register_native_package(
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
                evolve,
                nstride,
                bind_values,
                positivity_floor,
            )
            self._pending_native_packages += 1
        self._coupling_block_contracts = contract_candidate
        from pops.runtime._cadence_install import _record_block_time

        _record_block_time(self, name, time)
        if not getattr(self, "_batch_native_packages", False):
            self._commit_pending_native_packages()

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
        if str(solver) == "geometric_mg":
            solver = _resolve_route("field_solver", "cartesian_cg", context="set_poisson")
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

        def contract_of(block: Any) -> Any:
            try:
                return self._coupling_block_contracts[block]
            except (AttributeError, KeyError):
                raise ValueError(
                    "named coupling references block %r without an authenticated coupling contract"
                    % block
                ) from None

        preset = lower_named_coupling(coupling, contract_of)
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
