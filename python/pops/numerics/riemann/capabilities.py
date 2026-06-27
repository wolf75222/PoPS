"""pops.numerics.riemann.capabilities -- the Riemann capability-hook selectors (Spec 3).

The canonical capability-hook selectors used by
``m.riemann(..., wave_speeds=, contact_speed=, star_state=)``: ``riemann.speeds``
(Einfeldt / Davis wave-speed estimates) and ``riemann.hllc`` (the Euler contact
speed / star state). Each selector is a macro descriptor naming a canonical model
hook; the hook C++ is generated from the model roles by the dsl backend.
"""
from types import SimpleNamespace

from pops.descriptors import BrickDescriptor


def _hook(name, scheme):
    """A capability-hook selector descriptor: it picks a canonical model hook (e.g. the Euler
    contact speed / star state, the Einfeldt wave speeds) that the native solver consumes. It
    computes nothing; the hook C++ is generated from the model (roles) by the dsl backend."""
    return BrickDescriptor(name, "macro", category="riemann_hook", scheme=scheme)


def _attach_capabilities(riemann):
    """Attach the ``speeds`` / ``hllc`` capability-hook selectors onto the @p riemann ns."""
    riemann.speeds = SimpleNamespace(
        einfeldt=lambda: _hook("einfeldt", "einfeldt"),
        davis=lambda: _hook("davis", "davis"),
    )
    riemann.hllc = SimpleNamespace(
        contact_speed=SimpleNamespace(euler=lambda: _hook("euler_contact", "euler")),
        star_state=SimpleNamespace(euler=lambda: _hook("euler_star", "euler")),
    )
