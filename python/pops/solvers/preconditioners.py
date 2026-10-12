"""pops.solvers.preconditioners -- the preconditioner brick catalog (Spec 5 sec.5.7).

Identity lowers to the exact native identity provider. External providers register one typed
compiler contract plus an authenticated header-only native component, then construct their
descriptor with ``preconditioners.Prepared(provider, ...)``. The former callback-based geometric-MG
Program preconditioner is absent; ``TensorDiagonalMG`` prepares a native single V-cycle against
the exact tensor coefficient field captured by its matrix-free operator. This is
the ONE public home formerly
parked under
``pops.lib.solvers.preconditioners`` (that re-export shim is removed; no second public path).

ADC-502 RATIFIES ``pops.solvers.preconditioners`` as that single home: a preconditioner configures
a solver, so it lives with the solver descriptors (not under ``pops.linalg``); no move, no shim. The
invariant is pinned by ``tests/python/architecture/test_spec5_public_api.py`` (``pops.linalg`` has
NO ``preconditioners`` submodule).
"""
from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from pops.descriptors import _native
from pops.solvers._prepared_preconditioner_registry import (
    PreparedPreconditionerIntOption,
    PreparedPreconditionerNativeEmission,
    PreparedPreconditionerNativeContext,
    PreparedPreconditionerProvider,
    PreparedPreconditionerScratchResource,
    PreparedPreconditionerUsePolicy,
    prepared_preconditioner_provider_by_emitter_id,
    prepared_preconditioner_provider_by_id,
    prepared_preconditioner_provider_from_identity,
    register_prepared_preconditioner_provider,
)
from pops.native_components import PreparedNativeComponent

_IDENTITY_PROVIDER = prepared_preconditioner_provider_by_id(
    "pops.preconditioner.identity"
)


def _program_preconditioner_provider(descriptor: Any) -> PreparedPreconditionerProvider:
    """Authenticate a descriptor against the executable provider registry.

    Matching a convenient ``scheme`` string is insufficient: the descriptor must carry the exact
    native provider authority minted by this module.  In particular, an external manifest
    entry cannot impersonate a prepared provider until a real C++ factory ABI and matching emitter
    are implemented.
    """
    capabilities = getattr(descriptor, "capabilities", None)
    authority = (
        capabilities.get("prepared_program_provider")
        if isinstance(capabilities, Mapping)
        else None
    )
    provider = prepared_preconditioner_provider_from_identity(authority)
    if (
        getattr(descriptor, "category", None) != "preconditioner"
        or getattr(descriptor, "brick_type", None) != "native"
        or getattr(descriptor, "name", None) != provider.descriptor_name
        or getattr(descriptor, "native_id", None) != provider.native_id
        or authority != provider.authority()
    ):
        raise ValueError(
            "preconditioner descriptor %r is not authenticated for prepared Program execution"
            % (getattr(descriptor, "name", descriptor),)
        )
    options = getattr(descriptor, "options", None)
    if not isinstance(options, Mapping):
        raise ValueError(
            "preconditioner provider %r options must be an exact mapping" % provider.scheme
        )
    provider.prepare_options(options, where="preconditioner provider %r" % provider.scheme)
    return provider


def _native_program_preconditioner(
    provider: PreparedPreconditionerProvider, **options: Any
) -> Any:
    """Mint the sole descriptor shape accepted by the prepared Program provider registry."""
    return _native(
        provider.descriptor_name,
        provider.native_id,
        provider.scheme,
        category="preconditioner",
        capabilities={"prepared_program_provider": provider.authority()},
        **options,
    )


def _register_prepared_provider(
    provider: PreparedPreconditionerProvider,
) -> PreparedPreconditionerProvider:
    """Register one immutable provider through the public prepared-preconditioner surface."""
    return register_prepared_preconditioner_provider(provider)


def _prepared_provider_descriptor(
    provider: PreparedPreconditionerProvider, **options: Any
) -> Any:
    """Construct a descriptor from an exact registered provider, without name dispatch."""
    if type(provider) is not PreparedPreconditionerProvider:
        raise TypeError("preconditioners.Prepared requires an exact registered Provider")
    registered = prepared_preconditioner_provider_by_emitter_id(provider.emitter_id)
    if registered is not provider:
        raise ValueError("preconditioners.Prepared provider is not the registered authority")
    validated = provider.validate_options(
        options, where="preconditioners.Prepared(%r)" % provider.scheme
    )
    return _native_program_preconditioner(provider, **validated)


def _validate_tensor_diagonal_use(use: Any, where: str) -> None:
    if use.components != 1:
        raise ValueError("%s: TensorDiagonalMG requires one scalar elliptic unknown" % where)
    contract = use.nullspace_contract.get("contract")
    if not isinstance(contract, Mapping) or contract.get("declaration") != "nonsingular":
        raise ValueError("%s: TensorDiagonalMG requires an authenticated nonsingular problem" % where)
    # Red-black smoothing followed by injection is a nonsymmetric fixed linear map.
    # It may serve nonsymmetric Krylov methods, but is not certified as a CG-SPD map.
    if use.method_provider.get("provider_id") == "pops.krylov.cg":
        raise ValueError("%s: TensorDiagonalMG is not certified as a symmetric CG preconditioner" % where)


def _missing_tensor_context(*_args: Any, **_kwargs: Any) -> Any:
    raise ValueError("TensorDiagonalMG requires an authenticated tensor operator context")


def _emit_tensor_diagonal_mg(
    node: Any, prelude: Any, prototype: str, vector_distribution_expr: str,
    provider: PreparedPreconditionerProvider, native_context: PreparedPreconditionerNativeContext,
) -> PreparedPreconditionerNativeEmission:
    prelude.append("static_assert(pops::kNativeDimension == %d, "
        '"prepared tensor coefficient rank differs from native rank");'
        % native_context.coefficient_dimension)
    options = provider.prepare_options(node.attrs.get("preconditioner_options"),
        where="TensorDiagonalMG native options")
    values = provider.resolved_cpp_option_literals(options, where="TensorDiagonalMG native options")
    return PreparedPreconditionerNativeEmission(
        "pops::elliptic::mg::make_tensor_diagonal_mg_sessions<pops::kNativeDimension>("
        "%s, %s, %s, %s, %s, %s)" % (
            prototype, native_context.coefficient_field_expr, native_context.geometry_expr,
            native_context.boundary_expr, vector_distribution_expr, ", ".join(values)))


_TENSOR_DIAGONAL_PROVIDER = register_prepared_preconditioner_provider(
    PreparedPreconditionerProvider(
        provider_id="pops.preconditioner.tensor-diagonal-mg", interface_version=1,
        options_schema="pops.preconditioner.tensor-diagonal-mg.options@1",
        scheme="tensor_diagonal_mg", descriptor_name="tensor_diagonal_mg",
        display_name="preconditioners.TensorDiagonalMG()",
        native_id="pops::elliptic::mg::make_tensor_diagonal_mg_sessions",
        validator_id="pops.prepared-preconditioner.tensor-diagonal-mg.validate@1",
        planner_id="pops.prepared-preconditioner.tensor-diagonal-mg.plan@1",
        emitter_id="pops.prepared-preconditioner.tensor-diagonal-mg@1",
        preconditioned=True, prepared_buffers=1,
        use_policy=PreparedPreconditionerUsePolicy(
            "pops.prepared-preconditioner.tensor-diagonal-mg.use", 1,
            {"components": 1, "nullspace": "nonsingular", "tensor": "row-major",
             "diagonal_faces": "harmonic", "cycles_per_apply": 1,
             "smoother": "red-black-gauss-seidel", "prolongation": "constant-injection",
             "coefficient_publication": "frozen-prepared-operator-snapshot"},
            _validate_tensor_diagonal_use),
        options=(
            PreparedPreconditionerIntOption("minimum_coarse_extent", 2, 2),
            PreparedPreconditionerIntOption("pre_sweeps", 2, 0),
            PreparedPreconditionerIntOption("post_sweeps", 2, 0),
            PreparedPreconditionerIntOption("bottom_sweeps", 50, 1),
        ),
        emitter=_missing_tensor_context, context_emitter=_emit_tensor_diagonal_mg,
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.prepared-preconditioner.tensor-diagonal-mg",
            entry_headers=("pops/numerics/elliptic/mg/tensor_diagonal_preconditioner.hpp",)),
        scratch_resources=(PreparedPreconditionerScratchResource(
            "native-geometric-hierarchy", 0, False,
            "Prepared field-object count is 1+7*number_of_exact_factor_two_levels; topology-dependent, excluding communication metadata."),),
    )
)


def _tensor_diagonal_mg(**options: Any) -> Any:
    return _prepared_provider_descriptor(_TENSOR_DIAGONAL_PROVIDER, **options)


preconditioners = SimpleNamespace(
    Identity=lambda: _native_program_preconditioner(_IDENTITY_PROVIDER),
    TensorDiagonalMG=_tensor_diagonal_mg,
    Prepared=_prepared_provider_descriptor,
    register=_register_prepared_provider,
    Provider=PreparedPreconditionerProvider,
    IntOption=PreparedPreconditionerIntOption,
    NativeEmission=PreparedPreconditionerNativeEmission,
    ScratchResource=PreparedPreconditionerScratchResource,
    UsePolicy=PreparedPreconditionerUsePolicy,
    NativeComponent=PreparedNativeComponent,
    HeaderOnlyComponent=PreparedNativeComponent.header_only,
)

__all__ = ["preconditioners"]
