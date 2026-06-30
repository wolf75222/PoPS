"""pops.solvers.schur -- the Schur-complement solver catalog (Spec 5 sec.5.7).

The Schur-condensation solver eliminates a coupled (e.g. source) block and solves the reduced
system; the native symbol is ``pops::SchurCondensationOperator``. :func:`Schur` returns the
inert :class:`pops.descriptors.BrickDescriptor` naming it. This is the ONE public home of the
``solvers.Schur`` entry.

``CondensedSchur`` names the compiled source-stage route in this solver package only. It is not
re-exported as a top-level ``pops.CondensedSchur`` constructor.
"""
from pops.descriptors import _native
from pops.solvers.requirements import capability_map

# The Schur-condensation operator condenses a coupled (source) block and solves the reduced
# system; it runs on a uniform mesh and on an AMR hierarchy (the amr-schur source stage solves
# it on the coarse grid), under MPI and on the GPU (Kokkos). It declares every route capability
# (Spec 6 sec.4 / sec.9), so a route check sees it is AMR-capable.
_SCHUR_CAPABILITIES = capability_map(uniform=True, amr=True, mpi=True, gpu=True)


def Schur(**options):
    """The Schur-condensation solver descriptor (``pops::SchurCondensationOperator``).

    Scheme token ``"schur"``; inert (the C++ runtime applies the condensation operator).
    """
    return _native("schur", "pops::SchurCondensationOperator", "schur",
                   category="solver", capabilities=_SCHUR_CAPABILITIES, **options)


def CondensedSchur(*, theta=1.0, alpha=1.0, tolerance=1e-10, max_iter=200):
    """Descriptor for the compiled condensed-Schur source-stage solve.

    The C++ route is ``pops::CondensedSchurSourceStepper`` for uniform Cartesian layouts and
    its AMR counterpart where the runtime selects it. This object is inert: it configures the
    route; it does not perform a source step in Python.
    """
    for label, value in (("theta", theta), ("alpha", alpha), ("tolerance", tolerance)):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("CondensedSchur: %s must be a positive number" % label)
    if theta > 1.0:
        raise ValueError("CondensedSchur: theta must be in (0, 1]")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("CondensedSchur: max_iter must be a positive integer")
    return _native(
        "condensed_schur",
        "pops::CondensedSchurSourceStepper",
        "condensed_schur",
        category="schur_solver",
        capabilities=_SCHUR_CAPABILITIES,
        theta=float(theta),
        alpha=float(alpha),
        tolerance=float(tolerance),
        max_iter=int(max_iter),
    )


__all__ = ["Schur", "CondensedSchur"]
