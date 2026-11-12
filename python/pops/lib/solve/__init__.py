"""The solve algorithms of the library, written with ``@pops.solver`` (spec 0.3 2.6.9, SV9).

``cg(preconditioner=None, max_iterations=2000)`` is the reference conjugate gradient of the row
``cg`` of 2.6.9. A case passes the called word as the value of ``solvers={key: cg()}`` (SV4), and
the solve gives it its request (SV3). The native ``solve_cg`` is a bound substitution of this
reference (SV12, SV13) that runs only under ``pops.auto()`` until SV-T2 qualifies it at 64 ranks
(N9 lot 4, Q2).

**The relation solves of a traced method (N15 lot « solves », §5.5) run the bound native.** A
``pops.solve`` of a relation in a method body composes its operator with ``matrix_free_operator``
and solves it with b6's conjugate gradients, ``pops.solvers.CG``, under the request of the solve:
the passes that read an accepted step (the affine decomposition of the accepted transport, the
trial rows and the attempt reuse of N8, the partition and exchange passes) know that node and not
yet the outcome of the reference. ``cg`` therefore stays on the MG9 list of bridges, with this
narrower reason, until the N9 lot « La référence sous les solves de relation » runs the reference
there (N9 lot 4, step 2, ruling of 12/11 00:52). Conjugate gradients need a symmetric positive
definite operator: the relation solve derives it from the claims of its components (IR16) and
refuses it otherwise with ``PropertyNotDerived`` (K4).

Every other word of 2.6.9 is importable and refuses its use with the lot that elaborates it.
"""
from __future__ import annotations

from typing import Any

from .krylov import cg

_LATER = "N9 (the references of SV9 and the selection of SV11)"


def _native_cg(options: dict, request: Any) -> Any:
    """The b6 descriptor of conjugate gradients that a relation solve runs under ``cg(...)``, with
    the request of the solve (SV3, SV4): the bridge of MG9 named above."""
    from pops.solvers import CG

    max_iterations = options["max_iterations"]
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
            or max_iterations < 1:
        raise ValueError("cg(max_iterations=n) takes an integer n >= 1, got %r"
                         % (max_iterations,))
    rtol = 0.0 if request.rtol is None else float(request.rtol)
    atol = None if request.atol is None else float(request.atol)
    return CG(max_iter=max_iterations, rel_tol=rtol, abs_tol=atol)


cg.bind_relation_native(
    _native_cg, requires=("symmetric_positive_definite",),
    description="the native conjugate gradients of b6, pops.solvers.CG, which a relation solve of "
                "a traced method runs (a bridge of MG9 until the N9 lot « La référence sous les "
                "solves de relation » runs the reference there)")


def _later(word: str) -> Any:
    def refused(*args: Any, **kwargs: Any) -> Any:
        from pops.surface import NotYetElaborated

        raise NotYetElaborated("pops.lib.solve.%s" % word, _LATER)
    refused.__name__ = word
    return refused


auto = _later("auto")
bicgstab = _later("bicgstab")
gmres = _later("gmres")
CGS2 = _later("CGS2")
richardson = _later("richardson")
jacobi = _later("jacobi")
newton = _later("newton")
dense_lu = _later("dense_lu")

__all__ = ["CGS2", "auto", "bicgstab", "cg", "dense_lu", "gmres", "jacobi", "newton",
           "richardson"]
