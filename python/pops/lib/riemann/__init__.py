"""pops.lib.riemann -- the Riemann-flux brick catalog (Spec 3).

Native numerical fluxes (Rusanov/HLL/HLLC/Roe) plus a ``User`` selector for an
external C++ flux brick. The capability-hook selectors (``riemann.speeds`` /
``riemann.hllc``) are attached from :mod:`pops.lib.riemann.capabilities`.
"""
from types import SimpleNamespace

from ..descriptors import _native, _external_descriptor


def _riemann(name, native_id, caps):
    return _native(name, native_id, name, category="riemann", caps=caps)


riemann = SimpleNamespace(
    Rusanov=lambda: _riemann("rusanov", "pops::RusanovFlux", ["max_wave_speed"]),
    HLL=lambda: _riemann("hll", "pops::HLLFlux", ["physical_flux", "wave_speeds"]),
    HLLC=lambda: _riemann("hllc", "pops::HLLCFlux",
                          ["physical_flux", "pressure", "wave_speeds",
                           "contact_speed", "hllc_star_state"]),
    Roe=lambda: _riemann("roe", "pops::RoeFlux", ["physical_flux", "roe_average"]),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="riemann"),
)

# Attach the capability-hook selectors (riemann.speeds / riemann.hllc) onto the ns.
from .capabilities import _attach_capabilities  # noqa: E402

_attach_capabilities(riemann)

__all__ = ["riemann"]
