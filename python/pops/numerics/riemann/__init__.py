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

# Spec 5: expose the fluxes at module scope so ``from pops.numerics.riemann import HLL``
# works (the namespace stays for ``riemann.HLL`` and the attached capability hooks).
Rusanov = riemann.Rusanov
HLL = riemann.HLL
HLLC = riemann.HLLC
Roe = riemann.Roe
User = riemann.User

__all__ = ["riemann", "Rusanov", "HLL", "HLLC", "Roe", "User"]
