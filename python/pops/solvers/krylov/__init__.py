"""pops.solvers.krylov -- the matrix-free Krylov solver catalog (Spec 5 sec.5.7).

The Krylov solvers are FREE FUNCTIONS in the C++ ``namespace pops`` (generic_krylov.hpp);
each factory here returns an inert :class:`pops.descriptors.BrickDescriptor` naming the real
C++ symbol and the runtime ``scheme`` token. They compute nothing; codegen / the runtime
consume the descriptor. This is the ONE public home of the catalog.

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


def _check_common_options(name, options):
    opts = dict(options)
    if "tol" in opts and "tolerance" in opts:
        raise TypeError("%s: use either tolerance= or tol=, not both" % name)
    if "tol" in opts:
        opts["tolerance"] = opts.pop("tol")
    if "tolerance" in opts:
        tol = opts["tolerance"]
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("%s: tolerance must be a positive number" % name)
        opts["tolerance"] = float(tol)
    if "max_iter" not in opts:
        raise ValueError(
            "%s: max_iter is required; Krylov loops are compiled C++ runtime loops and must "
            "declare an iteration budget" % name)
    max_iter = opts["max_iter"]
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("%s: max_iter must be a positive integer" % name)
    opts["max_iter"] = int(max_iter)
    return opts


def _solver(name, native_id, **options):
    """A native Krylov-solver descriptor in the ``solver`` category (scheme == @p name)."""
    options = _check_common_options(name, options)
    return _native(name, native_id, name, category="solver",
                   capabilities=_KRYLOV_CAPABILITIES, **options)


def CG(**options):
    """The conjugate-gradient Krylov solver (``pops::cg_solve``; scheme ``"cg"``). Inert.

    ``max_iter`` is required so the generated C++ loop has an explicit budget.
    """
    return _solver("cg", "pops::cg_solve", **options)


def BiCGStab(**options):
    """The stabilised bi-CG Krylov solver (``pops::bicgstab_solve``; scheme ``"bicgstab"``)."""
    return _solver("bicgstab", "pops::bicgstab_solve", **options)


def GMRES(**options):
    """The GMRES Krylov solver (``pops::gmres_solve``; scheme ``"gmres"``). Inert."""
    if "restart" in options:
        restart = options["restart"]
        if isinstance(restart, bool) or not isinstance(restart, int) or restart <= 0:
            raise ValueError("gmres: restart must be a positive integer")
    return _solver("gmres", "pops::gmres_solve", **options)


def Richardson(**options):
    """The Richardson iteration (``pops::richardson_solve``; scheme ``"richardson"``). Inert."""
    return _solver("richardson", "pops::richardson_solve", **options)


__all__ = ["CG", "BiCGStab", "GMRES", "Richardson"]
