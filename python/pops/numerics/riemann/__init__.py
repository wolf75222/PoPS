"""pops.numerics.riemann -- the Riemann-flux brick catalog (Spec 3 / Spec 5).

Native numerical fluxes (Rusanov/HLL/HLLC/Roe) plus a ``User`` selector for an
external C++ flux brick. The capability-hook selectors (``riemann.speeds`` /
``riemann.hllc``) are attached from :mod:`pops.numerics.riemann.capabilities`.

Spec 5 (sec.4 / sec.5.4) homes the discretisation descriptors in ``pops.numerics``;
the individual fluxes are importable directly (``from pops.numerics.riemann import
HLL``), not only via the ``riemann`` namespace.
"""
from types import SimpleNamespace

from pops.descriptors import _native, _external_descriptor
from . import waves
from .waves import (WaveSpeedProvider, ExplicitPair, FromJacobian, FromPressure,
                    Einfeldt, Davis, MaxWaveSpeed, provider_of)


def _riemann(name, native_id, caps):
    return _native(name, native_id, name, category="riemann", caps=caps)


def _hll(waves=None):
    """The HLL numerical flux descriptor, optionally pinned to a typed wave-speed provider.

    ``HLL()`` is the historical generic signed-wave flux (requires the model's wave_speeds). With
    ``waves=`` it takes a TYPED :class:`~pops.numerics.riemann.waves.WaveSpeedProvider` (e.g.
    ``ExplicitPair()`` / ``FromJacobian()`` / ``F.capabilities.wave_speeds``): a bare string is
    REJECTED (pointing at the typed factories) and a NON-signed provider (``MaxWaveSpeed``) is
    REFUSED with a precise message -- HLL needs a signed pair, ``MaxWaveSpeed`` is the Rusanov
    majorant. The accepted provider enters the descriptor options (``options["waves"]``) and
    requirements so the identity / inspection / install guard reflect it."""
    desc = _riemann("hll", "pops::HLLFlux", ["physical_flux", "wave_speeds"])
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
    desc.requirements.setdefault("capabilities", ["physical_flux", "wave_speeds"])
    desc.requirements["wave_speed_provider"] = waves.kind
    return desc


riemann = SimpleNamespace(
    Rusanov=lambda: _riemann("rusanov", "pops::RusanovFlux", ["max_wave_speed"]),
    HLL=_hll,
    HLLC=lambda: _riemann("hllc", "pops::HLLCFlux",
                          ["physical_flux", "pressure", "wave_speeds",
                           "contact_speed", "hllc_star_state"]),
    Roe=lambda: _riemann("roe", "pops::RoeFlux", ["physical_flux", "roe_average"]),
    # Explicit canonical Euler 2D routes (ADC-590): force EulerHLLCFlux2D / EulerRoeFlux2D
    # (4-var rho/mx/my/E + pressure), never a fallback. Use HLLC()/Roe() for a generic model
    # that emits the capability (m.enable_hllc()/m.enable_roe()).
    EulerHLLC2D=lambda: _riemann("euler_hllc", "pops::EulerHLLCFlux2D",
                                 ["physical_flux", "pressure", "euler_2d_layout"]),
    EulerRoe2D=lambda: _riemann("euler_roe", "pops::EulerRoeFlux2D",
                                ["physical_flux", "pressure", "euler_2d_layout"]),
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
HLL = riemann.HLL
HLLC = riemann.HLLC
Roe = riemann.Roe
EulerHLLC2D = riemann.EulerHLLC2D
EulerRoe2D = riemann.EulerRoe2D
User = riemann.User

__all__ = ["riemann", "waves", "Rusanov", "HLL", "HLLC", "Roe", "EulerHLLC2D", "EulerRoe2D",
           "User", "WaveSpeedProvider", "ExplicitPair", "FromJacobian", "FromPressure",
           "Einfeldt", "Davis", "MaxWaveSpeed", "provider_of", "available", "validate"]
