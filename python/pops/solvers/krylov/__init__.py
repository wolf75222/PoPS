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

Every factory takes a MANDATORY ``max_iter`` (a positive Python int): a dynamic Krylov loop
with no iteration budget is a configuration error, and the native ``pops::*_solve`` loops
themselves throw ``std::invalid_argument("dynamic solver loops require max_iter")`` on a
non-positive budget (generic_krylov.hpp ``require_max_iter``). Refusing a missing / non-positive
budget HERE, at descriptor construction, surfaces the same error before the runtime is ever
touched (Spec 5 sec.6: a route is refused explainably, pre-compile).
"""
from pops.descriptors import _native
from pops.solvers.requirements import capability_map

# The Krylov solvers are matrix-free free functions over ``pops::MultiFab`` primitives (dot /
# saxpy / the operator apply): the algebra is layout-agnostic, so it runs on a uniform mesh or
# an AMR level, under MPI (the inner-product reductions are collective) and on the GPU (the
# kernels are Kokkos). They therefore declare every route capability (Spec 6 sec.4 / sec.9), so
# a route check can see they are AMR-capable rather than guessing from an empty capability set.
_KRYLOV_CAPABILITIES = capability_map(uniform=True, amr=True, mpi=True, gpu=True)


def _check_max_iter(name, max_iter):
    """Refuse a missing / non-positive Krylov iteration budget at descriptor construction.

    ``max_iter`` is MANDATORY (a positive Python int): a dynamic solver loop with no budget is a
    configuration error the native ``pops::*_solve`` loop itself throws on. Raising here (with the
    SAME message shape as the native ``require_max_iter``) refuses the route before the runtime is
    touched, so a test can assert the refusal without compiling. A bool is rejected (``True`` /
    ``False`` are ints in Python but never a valid budget). @p name is the public factory name
    used in the message.
    """
    if max_iter is None:
        raise ValueError(
            "%s: max_iter is required (dynamic solver loops require max_iter); pass a positive "
            "int, e.g. %s(max_iter=200)" % (name, name))
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(
            "%s: max_iter must be a positive int (dynamic solver loops require max_iter); got %r"
            % (name, max_iter))
    return int(max_iter)


def _solver(name, native_id, factory, max_iter, **options):
    """A native Krylov-solver descriptor in the ``solver`` category (scheme == @p name).

    ``max_iter`` is validated (positive int, mandatory) and folded into the descriptor options so
    the budget travels with the route and is inspectable pre-runtime. @p factory is the public
    factory name used in the refusal message.
    """
    options["max_iter"] = _check_max_iter(factory, max_iter)
    return _native(name, native_id, name, category="solver",
                   capabilities=_KRYLOV_CAPABILITIES, **options)


def CG(max_iter=None, **options):
    """The conjugate-gradient Krylov solver (``pops::cg_solve``; scheme ``"cg"``). Inert.

    ``max_iter`` is a MANDATORY positive int (the iteration budget); a missing / non-positive
    budget is refused at construction (see the module docstring).
    """
    return _solver("cg", "pops::cg_solve", "CG", max_iter, **options)


def BiCGStab(max_iter=None, **options):
    """The stabilised bi-CG Krylov solver (``pops::bicgstab_solve``; scheme ``"bicgstab"``).

    ``max_iter`` is a MANDATORY positive int; a missing / non-positive budget is refused.
    """
    return _solver("bicgstab", "pops::bicgstab_solve", "BiCGStab", max_iter, **options)


def GMRES(max_iter=None, **options):
    """The GMRES Krylov solver (``pops::gmres_solve``; scheme ``"gmres"``). Inert.

    ``max_iter`` is a MANDATORY positive int; a missing / non-positive budget is refused.
    """
    return _solver("gmres", "pops::gmres_solve", "GMRES", max_iter, **options)


def Richardson(max_iter=None, **options):
    """The Richardson iteration (``pops::richardson_solve``; scheme ``"richardson"``). Inert.

    ``max_iter`` is a MANDATORY positive int; a missing / non-positive budget is refused.
    """
    return _solver("richardson", "pops::richardson_solve", "Richardson", max_iter, **options)


__all__ = ["CG", "BiCGStab", "GMRES", "Richardson"]
