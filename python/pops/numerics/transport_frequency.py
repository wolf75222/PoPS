"""Authenticated endpoint wave envelope shared by independent and joint partitions."""


def transport_frequency_contract(transport):
    """Sufficient scalar monotonicity and declared-speed restriction for systems.

    This does not prove arbitrary system invariants. Higher-order reconstruction
    and non-endpoint Riemann waves require their own prepared frequency provider.
    """
    from .reconstruction import authenticated_reconstruction_route
    from .riemann._contract import riemann_capability_contract
    from pops.runtime.routes import resolve
    if transport is None:
        raise ValueError("combined diffusion frequency requires a transport selection")
    reconstruction = authenticated_reconstruction_route(transport.reconstruction)
    flux = transport.riemann
    contract = riemann_capability_contract(flux)
    route = resolve("riemann", flux.scheme, context="combined diffusion face envelope")
    if (flux.brick_type != "native" or flux.native_id != route.native_entry
            or not contract.requires("stability_bound")):
        raise ValueError("combined diffusion needs an authenticated native face stability envelope")
    if reconstruction.metadata["formal_order"] != 1:
        raise ValueError("combined diffusion has no prepared stability contract for higher-order reconstruction")
    # These native policies use the endpoint model envelope already exposed
    # by ctx.max_wave_speed. Roe/contact/reconstructed states need a different
    # provider, rather than an unproved multiplier of that cellwise bound.
    if route.native_entry not in ("pops::RusanovFlux", "pops::HLLFlux"):
        raise ValueError("combined diffusion has no prepared frequency provider for this numerical flux")
    if flux.options.get("waves") in ("einfeldt", "davis"):
        raise ValueError("combined diffusion requires the endpoint model envelope; this face wave provider needs a separate frequency realization")
    return {"provider": "native_endpoint_model_wave_envelope",
            "reconstruction": reconstruction.native_entry,
            "numerical_flux": route.native_entry,
            "system_invariants": "not_guaranteed"}
