"""pops.solvers.preconditioners -- compiled preconditioner descriptors.

Only descriptors with a real compiled route are public here. ``Identity()`` lowers to the
runtime's empty ``ApplyFn`` preconditioner, and ``GeometricMG()`` lowers to the wired
``pops::GeometricMG`` V-cycle. Jacobi / block-Jacobi are not exposed until their
native C++ kernels exist: no public descriptor may be decorative or route to Python numerics.
``User`` surfaces a loaded external preconditioner brick.
"""
from pops.descriptors import BrickDescriptor, _external_descriptor, _native


def Identity(**options):
    """The real unpreconditioned route: available, with no fabricated native symbol."""
    return BrickDescriptor("identity", "macro", category="preconditioner",
                           native_id="", scheme="identity", options=options or None)


def GeometricMG(**options):
    """One compiled geometric-multigrid V-cycle used as a Krylov preconditioner."""
    return _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                   category="preconditioner", **options)


def User(brick_id):
    """Reference an external C++ preconditioner brick loaded through the external catalog."""
    return _external_descriptor(brick_id, expect_category="preconditioner")


__all__ = ["Identity", "GeometricMG", "User"]
