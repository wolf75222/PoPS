"""pops.solvers.krylov -- the matrix-free Krylov solver catalog (Spec 5 sec.5.7).

The Krylov solvers are FREE FUNCTIONS in the C++ ``namespace pops`` (generic_krylov.hpp);
each factory here returns an inert :class:`pops.descriptors.BrickDescriptor` naming the real
C++ symbol and the runtime ``scheme`` token. They compute nothing; codegen / the runtime
consume the descriptor. This is the ONE public home of the catalog formerly parked under
``pops.lib.solvers`` (that re-export shim is removed; there is no second public path).

* :func:`CG` -- conjugate gradient (SPD systems);
* :func:`BiCGStab` -- stabilised bi-conjugate gradient (nonsymmetric);
* :func:`GMRES` -- generalised minimal residual (nonsymmetric);
* :func:`Richardson` -- preconditioned Richardson iteration.
"""
from pops.descriptors import _native
from pops.solvers.requirements import capability_map

# The Krylov solvers are matrix-free free functions over ``pops::MultiFab`` primitives (dot /
# saxpy / the operator apply): the algebra is layout-agnostic, so it runs on a uniform mesh or
# an AMR level, under MPI (the inner-product reductions are collective) and on the GPU (the
# kernels are Kokkos). They therefore declare every route capability (Spec 6 sec.4 / sec.9), so
# a route check can see they are AMR-capable rather than guessing from an empty capability set.
_KRYLOV_CAPABILITIES = capability_map(uniform=True, amr=True, mpi=True, gpu=True)


def _solver(name, native_id, **options):
    """A native Krylov-solver descriptor in the ``solver`` category (scheme == @p name)."""
    return _native(name, native_id, name, category="solver",
                   capabilities=_KRYLOV_CAPABILITIES, **options)


def CG(**options):
    """The conjugate-gradient Krylov solver (``pops::cg_solve``; scheme ``"cg"``). Inert."""
    return _solver("cg", "pops::cg_solve", **options)


def BiCGStab(**options):
    """The stabilised bi-CG Krylov solver (``pops::bicgstab_solve``; scheme ``"bicgstab"``)."""
    return _solver("bicgstab", "pops::bicgstab_solve", **options)


def GMRES(**options):
    """The GMRES Krylov solver (``pops::gmres_solve``; scheme ``"gmres"``). Inert."""
    return _solver("gmres", "pops::gmres_solve", **options)


def Richardson(**options):
    """The Richardson iteration (``pops::richardson_solve``; scheme ``"richardson"``). Inert."""
    return _solver("richardson", "pops::richardson_solve", **options)


__all__ = ["CG", "BiCGStab", "GMRES", "Richardson"]
