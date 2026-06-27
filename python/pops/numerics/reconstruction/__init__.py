"""pops.numerics.reconstruction -- the spatial-reconstruction brick catalog (Spec 3 / Spec 5).

FirstOrder / MUSCL / WENO5 / WENO5Z selectors plus a ``User`` selector for an
external C++ reconstruction brick. The slope limiters are catalogued separately
in :mod:`pops.numerics.reconstruction.limiters`.

pops::Weno5 IS the WENO5-Z reconstruction (it wraps weno5z()); WENO5 and WENO5Z both
select it. MUSCL is reconstruction-by-limiter; its native limiter type is pops::Minmod.
"""
from types import SimpleNamespace

from pops.descriptors import _native, _external_descriptor
from .limiters import limiters

reconstruction = SimpleNamespace(
    FirstOrder=lambda: _native("firstorder", "pops::NoSlope", "firstorder",
                               category="reconstruction"),
    MUSCL=lambda limiter="minmod": _native(
        "muscl", "pops::Minmod", limiter, category="reconstruction", limiter=limiter),
    WENO5=lambda: _native("weno5", "pops::Weno5", "weno5", category="reconstruction"),
    WENO5Z=lambda: _native("weno5z", "pops::Weno5", "weno5", category="reconstruction"),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="reconstruction"),
)

# Spec 5: expose the schemes at module scope (``from pops.numerics.reconstruction import MUSCL``).
FirstOrder = reconstruction.FirstOrder
MUSCL = reconstruction.MUSCL
WENO5 = reconstruction.WENO5
WENO5Z = reconstruction.WENO5Z
User = reconstruction.User

__all__ = ["reconstruction", "limiters", "FirstOrder", "MUSCL", "WENO5", "WENO5Z", "User"]
