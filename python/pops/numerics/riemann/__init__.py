"""pops.numerics.riemann -- the Riemann-flux brick catalog (Spec 3 / Spec 5).

Native numerical fluxes (Rusanov/HLL/HLLC/Roe), the closed typed ``Recovery`` policy, plus a
``User`` selector for an external C++ flux brick. The capability-hook selectors (``riemann.speeds`` /
``riemann.hllc``) are attached from :mod:`pops.numerics.riemann.capabilities`.

Spec 5 (sec.4 / sec.5.4) homes the discretisation descriptors in ``pops.numerics``;
the individual fluxes are importable directly (``from pops.numerics.riemann import
HLL``), not only via the ``riemann`` namespace.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pops.descriptors import BrickDescriptor, _native, _external_descriptor
from . import waves
from .waves import (WaveSpeedProvider, ExplicitPair, FromJacobian, FromPressure,
                    Einfeldt, Davis, MaxWaveSpeed, provider_of)


def _riemann(name: Any, native_id: Any, caps: Any, **options: Any) -> Any:
    return _native(name, native_id, name, category="riemann", caps=caps, **options)


def _scalar_upwind(*, velocity: Any) -> Any:
    """Exact scalar upwind route expressed through the native Rusanov flux.

    For a linear scalar flux ``F=aU`` with the declared velocity as exact stability bound, Rusanov
    and upwind are algebraically identical. The public identity records that stronger contract while
    the native route remains the generic ``pops::RusanovFlux`` implementation.
    """
    from pops.model import Handle

    if not isinstance(velocity, Handle) or velocity.kind != "vector":
        raise TypeError("ScalarUpwind(velocity=) requires a typed vector Handle")
    return _native(
        "scalar_upwind", "pops::RusanovFlux", "rusanov", category="riemann",
        caps=["physical_flux", "provider_pack", "stability_bound"],
        velocity=velocity, exact_linear_scalar=True,
    )


def _hll(waves: Any = None) -> Any:
    """The HLL numerical flux descriptor, optionally pinned to a typed wave-speed provider.

    ``HLL()`` is the historical generic signed-wave flux (requires the model's wave_speeds). With
    ``waves=`` it takes a TYPED :class:`~pops.numerics.riemann.waves.WaveSpeedProvider` (e.g.
    ``ExplicitPair()`` / ``FromJacobian()`` / ``F.capabilities.wave_speeds``): a bare string is
    REJECTED (pointing at the typed factories) and a NON-signed provider (``MaxWaveSpeed``) is
    REFUSED with a precise message -- HLL needs a signed pair, ``MaxWaveSpeed`` is the Rusanov
    majorant. The accepted provider enters the descriptor options (``options["waves"]``) and
    requirements so the identity / inspection / install guard reflect it."""
    desc = _riemann("hll", "pops::HLLFlux",
                    ["physical_flux", "provider_pack", "stability_bound", "wave_speeds"])
    if waves is None:
        return desc
    if isinstance(waves, str):
        from pops.descriptors import reject_string_selector
        reject_string_selector(
            waves, "waves",
            "pops.numerics.riemann.waves.ExplicitPair() / FromJacobian() / FromPressure() / "
            "Einfeldt() / Davis(), or F.capabilities.wave_speeds")
    if not isinstance(waves, WaveSpeedProvider):
        raise TypeError(
            "HLL(waves=): expected a typed WaveSpeedProvider (pops.numerics.riemann.waves.*), "
            "got %r." % (type(waves).__name__,))
    if not waves.signed_pair:
        raise ValueError(
            "HLL requires a signed wave-speed provider; %s is the Rusanov majorant "
            "(unsigned) -- use Rusanov() or a signed provider (ExplicitPair() / FromJacobian() / "
            "FromPressure() / Einfeldt() / Davis())." % (waves.describe(),))
    desc.options["waves"] = waves.kind
    # The provider participates in the descriptor requirements (identity / inspection reflect it).
    desc.requirements.setdefault(
        "capabilities", ["physical_flux", "provider_pack", "stability_bound", "wave_speeds"])
    desc.requirements["wave_speed_provider"] = waves.kind
    return desc


_RECOVERY_NATIVE_ID = (
    "pops::PreparedRiemannRecoveryPolicy<pops::RoeFlux,pops::HLLFlux,"
    "pops::RusanovFlux,pops::RejectRiemannRecovery>"
)
_RECOVERY_SEQUENCE = (
    ("roe", "pops::RoeFlux"),
    ("hll", "pops::HLLFlux"),
    ("rusanov", "pops::RusanovFlux"),
)


def _canonical_recovery_candidates() -> tuple[BrickDescriptor, ...]:
    return (
        _riemann(
            "roe",
            "pops::RoeFlux",
            ["physical_flux", "provider_pack", "stability_bound", "roe_dissipation"],
        ),
        _hll(),
        _riemann(
            "rusanov",
            "pops::RusanovFlux",
            ["physical_flux", "provider_pack", "stability_bound"],
        ),
    )


def _recovery(*, primary: Any, fallbacks: Any) -> Any:
    """Fixed fail-closed Roe -> HLL -> Rusanov recovery policy.

    The policy is deliberately a closed typed value, not a general Python list lowered into an
    arbitrary C++ template. Only the one native policy instantiated by PoPS is accepted; every
    mismatch is refused while authoring, before compile or bind.
    """
    if not isinstance(fallbacks, tuple):
        raise TypeError(
            "riemann.Recovery(fallbacks=) requires a tuple of typed built-in descriptors; "
            "use fallbacks=(riemann.HLL(), riemann.Rusanov())"
        )
    authored = (primary, *fallbacks)
    labels = ("primary", *("fallbacks[%d]" % index for index in range(len(fallbacks))))
    actual: list[tuple[str, str]] = []
    for label, candidate in zip(labels, authored, strict=True):
        if not isinstance(candidate, BrickDescriptor) or candidate.category != "riemann":
            raise TypeError(
                "riemann.Recovery(%s) requires a typed built-in Riemann descriptor; got %s"
                % (label, type(candidate).__name__)
            )
        if candidate.brick_type != "native" or candidate.scheme == "user":
            raise ValueError(
                "riemann.Recovery(%s) refuses external/non-native descriptor %r; prepared "
                "recovery candidates must be compiled device-copyable built-ins"
                % (label, candidate.name)
            )
        if candidate.options:
            raise ValueError(
                "riemann.Recovery(%s=%r) carries candidate options that the fixed native policy "
                "does not transport; use the option-free built-in descriptor"
                % (label, candidate.name)
            )
        actual.append((str(candidate.scheme), candidate.native_id))

    schemes = tuple(scheme for scheme, _ in actual)
    duplicates = tuple(sorted({scheme for scheme in schemes if schemes.count(scheme) > 1}))
    if duplicates:
        raise ValueError(
            "riemann.Recovery candidates must be unique; duplicates=%s"
            % ",".join(duplicates)
        )
    if tuple(actual) != _RECOVERY_SEQUENCE:
        raise ValueError(
            "riemann.Recovery supports exactly primary=Roe(), "
            "fallbacks=(HLL(), Rusanov()); requested order=%s"
            % " -> ".join(schemes)
        )
    constructors = ("Roe", "HLL", "Rusanov")
    for label, candidate, canonical, constructor in zip(
        labels, authored, _canonical_recovery_candidates(), constructors, strict=True
    ):
        if candidate != canonical:
            raise ValueError(
                "riemann.Recovery(%s=%r) is not the catalog-authenticated option-free built-in; "
                "construct it with riemann.%s()"
                % (label, candidate.name, constructor)
            )
    return _riemann(
        "roe_hll_rusanov_recovery",
        _RECOVERY_NATIVE_ID,
        ["physical_flux", "provider_pack", "stability_bound", "wave_speeds",
         "roe_dissipation"],
        recovery_order=("roe", "hll", "rusanov", "reject"),
    )


riemann = SimpleNamespace(
    Rusanov=lambda: _riemann(
        "rusanov", "pops::RusanovFlux", ["physical_flux", "provider_pack", "stability_bound"]),
    ScalarUpwind=_scalar_upwind,
    HLL=_hll,
    HLLC=lambda: _riemann("hllc", "pops::HLLCFlux",
                          ["physical_flux", "provider_pack", "stability_bound", "pressure", "wave_speeds",
                           "contact_speed", "hllc_star_state"]),
    Roe=lambda: _riemann(
        "roe", "pops::RoeFlux",
        ["physical_flux", "provider_pack", "stability_bound", "roe_dissipation"]),
    Recovery=_recovery,
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="riemann"),
)

# Attach the capability-hook selectors (riemann.speeds / riemann.hllc) onto the ns.
from .capabilities import _attach_capabilities  # noqa: E402

_attach_capabilities(riemann)

# The typed wave-speed provider layer (ADC-552): reachable as ``riemann.waves.ExplicitPair()``
# (the real submodule exposes the factories) so ``HLL(waves=riemann.waves.ExplicitPair())`` works.
riemann.waves = waves

# Pre-runtime capability refusals (ADC-533): the model-aware available/validate that surface the
# HLL/HLLC/Roe/Euler route refusals through the descriptor surface. They DELEGATE to the exact
# install-time predicates in pops.runtime.routes (single source), so a mismatch is testable before
# any compile. Reachable as riemann.available(HLL(), context) / riemann.validate(...).
from .availability import flux_available as available, flux_validate as validate  # noqa: E402

riemann.available = available
riemann.validate = validate

# Spec 5: expose the fluxes at module scope so ``from pops.numerics.riemann import HLL``
# works (the namespace stays for ``riemann.HLL`` and the attached capability hooks).
Rusanov = riemann.Rusanov
ScalarUpwind = riemann.ScalarUpwind
HLL = riemann.HLL
HLLC = riemann.HLLC
Roe = riemann.Roe
Recovery = riemann.Recovery
User = riemann.User

__all__ = ["riemann", "waves", "Rusanov", "ScalarUpwind", "HLL", "HLLC", "Roe",
           "Recovery", "User", "WaveSpeedProvider", "ExplicitPair", "FromJacobian", "FromPressure",
           "Einfeldt", "Davis", "MaxWaveSpeed", "provider_of", "available", "validate"]
