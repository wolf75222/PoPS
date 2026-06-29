"""pops.solvers.preconditioners -- the preconditioner brick catalog (Spec 5 sec.5.7).

Only descriptors with a real compiled route are public here. ``Identity()`` lowers to the
runtime's empty ``ApplyFn`` preconditioner, and ``GeometricMG()`` lowers to the wired
``pops::GeometricMG`` V-cycle. Jacobi / block-Jacobi are intentionally not exposed until their
native C++ kernels exist: no public descriptor may be decorative or raise a "planned" error.
``User`` surfaces a loaded external preconditioner brick.
"""
from types import SimpleNamespace

from pops.descriptors import BrickDescriptor, _external_descriptor, _native


def _identity(**options):
    """The real unpreconditioned route: available, with no fabricated native symbol."""
    return BrickDescriptor("identity", "macro", category="preconditioner",
                           native_id="", scheme="identity", options=options or None)

preconditioners = SimpleNamespace(
    Identity=_identity,
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="preconditioner", **o),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="preconditioner"),
)

__all__ = ["preconditioners"]
