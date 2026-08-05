"""pops.solvers.preconditioners -- the preconditioner brick catalog (Spec 5 sec.5.7).

Identity lowers to the exact native identity provider. External providers register one typed
compiler contract plus an authenticated header-only native component, then construct their
descriptor with ``preconditioners.Prepared(provider, ...)``. The former callback-based geometric-MG
Program preconditioner is absent; ``GeometricMG<Dim>`` remains an exact-ranked field solver. This is
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

preconditioners = SimpleNamespace(
    Identity=lambda: _native_program_preconditioner(_IDENTITY_PROVIDER),
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
