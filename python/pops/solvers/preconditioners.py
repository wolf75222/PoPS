"""pops.solvers.preconditioners -- compiled preconditioner descriptors.

Only descriptors with a real compiled route are public here. ``Identity()`` lowers to the
runtime's empty ``ApplyFn`` preconditioner, and ``GeometricMG()`` lowers to the wired
``pops::GeometricMG`` V-cycle. Jacobi, block-Jacobi, and external user preconditioners are
not exposed until their native C++ lowering exists: every public descriptor must route to
compiled numerics.
"""
from pops.descriptors import BrickDescriptor, _native


def Identity(**options):
    """The real unpreconditioned route: available, with no fabricated native symbol."""
    return BrickDescriptor("identity", "macro", category="preconditioner",
                           native_id="", scheme="identity", options=options or None)


def GeometricMG(**options):
    """One compiled geometric-multigrid V-cycle used as a Krylov preconditioner."""
    return _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                   category="preconditioner", **options)


__all__ = ["Identity", "GeometricMG"]
