"""pops.solvers.nonlinear -- nonlinear solver descriptors with real compiled routes."""

from pops.descriptors import BrickDescriptor
from pops.solvers.requirements import capability_map

_NEWTON_CAPABILITIES = capability_map(uniform=True, amr=True, mpi=True, gpu=True)


def Newton(*, tolerance=1e-10, max_iter=25, fd_eps=1e-7, damping=1.0):
    """Per-cell Newton descriptor for the generated C++ local nonlinear solve.

    This is not a Python-runtime solver and not a global assembled Newton route. It names the
    compiled per-cell Newton kernel used by ``Program.solve_local_nonlinear``. Public code may
    inspect it as a solver descriptor; numerical work remains generated C++.
    """
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("Newton: tolerance must be a positive number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Newton: max_iter must be a positive integer")
    if not isinstance(fd_eps, (int, float)) or fd_eps <= 0:
        raise ValueError("Newton: fd_eps must be a positive number")
    if not isinstance(damping, (int, float)) or damping <= 0:
        raise ValueError("Newton: damping must be a positive number")
    return BrickDescriptor(
        "newton",
        "generated",
        category="nonlinear_solver",
        native_id="",
        scheme="newton",
        capabilities=_NEWTON_CAPABILITIES,
        options={
            "tolerance": float(tolerance),
            "max_iter": int(max_iter),
            "fd_eps": float(fd_eps),
            "damping": float(damping),
        },
    )


__all__ = ["Newton"]
