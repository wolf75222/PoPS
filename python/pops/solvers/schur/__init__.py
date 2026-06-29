"""pops.solvers.schur -- the Schur-complement solver catalog (Spec 5 sec.5.7).

The Schur-condensation solver eliminates a coupled (e.g. source) block and solves the reduced
system; the native symbol is ``pops::SchurCondensationOperator``. :func:`Schur` returns the
inert :class:`pops.descriptors.BrickDescriptor` naming it. This is the ONE public home of the
``solvers.Schur`` entry formerly parked under ``pops.lib.solvers`` (that shim is removed; no
second public path).

The time-integration source stage is exposed as ``pops.ElectrostaticLorentzSchur(...)``. Do not
add a ``CondensedSchur`` alias here: one public name per behavior keeps the Spec-5 API clean.
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


__all__ = ["Schur"]
