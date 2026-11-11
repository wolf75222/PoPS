"""The solvers of the surface (spec 0.3, 2.6.9 ``pops.lib.solve``; SV4, SV9, SV11; MG9).

``cg(preconditioner=None, max_iterations=2000)`` is the one solver of this lot. It is a bridge of
MG9: it realizes the solve on b6's native conjugate gradients (``pops.solvers.CG``), until N9
writes the reference CG of SV9 in this module, Python over the IR, with the native CG behind it as
a substitution qualified by Trace (§17.1 « Runs », SV13, CF4). The bridge leaves within Phase 1,
whose exit requires the list of bridges of MG9 empty (spec line 3632).

A solver of this module is a reference that the realization of a solve path names
(``solvers={method: cg()}``). Its protocol with the traced methods of N8 is ``linear_solver(
request=, spd=, path=, claims=)``, which returns the b6 descriptor of a linear solve whose operator
the relation composed, or refuses: conjugate gradients need a symmetric positive definite
operator, and a solve whose properties do not derive it is refused with ``PropertyNotDerived``
(K4, IR16).

Every other word of 2.6.9 is importable and refuses its use with the lot that elaborates it.
"""
from __future__ import annotations

from typing import Any

_LATER = "N9 (the references of SV9 and the selection of SV11)"


class SolverValue:
    """A solver of ``pops.lib.solve``, the value of an entry of ``solvers=``."""

    __slots__ = ("word",)

    def __init__(self, word: str) -> None:
        self.word = word


class _CG(SolverValue):
    __slots__ = ("preconditioner", "max_iterations")

    def __init__(self, preconditioner: Any, max_iterations: Any) -> None:
        super().__init__("cg")
        from pops.time._program.traced_method import NotYetElaborated

        if preconditioner is not None:
            raise NotYetElaborated("cg(preconditioner=...): the preconditioners of SV11 are "
                                   "elaborated by %s" % _LATER)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
                or max_iterations < 1:
            raise ValueError("cg(max_iterations=n) needs an integer n >= 1, got %r"
                             % (max_iterations,))
        self.preconditioner = preconditioner
        self.max_iterations = max_iterations

    def linear_solver(self, *, request: Any, spd: bool, path: str, claims: tuple) -> Any:
        """The b6 descriptor of conjugate gradients for the solve ``path`` (SV4, K4)."""
        if not spd:
            from pops._ir.refusals import PropertyNotDerived

            stated = "; ".join("%s: %s" % (name, claim.proof) for name, claim in claims)
            raise PropertyNotDerived(
                "cg() needs a symmetric positive definite operator, which the relation of %s does "
                "not derive (IR16): the operator is a I + sum_k b_k L_k, SPD when a > 0, every b_k "
                "a nonpositive multiple of the step and every L_k symmetric negative "
                "semidefinite; the claims stated are %s" % (path, stated or "none"),
                region=path, property="symmetric_positive_definite")
        from pops.solvers import CG

        rtol = 0.0 if request.rtol is None else float(request.rtol)
        atol = None if request.atol is None else float(request.atol)
        return CG(max_iter=self.max_iterations, rel_tol=rtol, abs_tol=atol)

    def describe(self) -> str:
        return ("the native conjugate gradients of b6, pops.solvers.CG(max_iter=%d), a bridge of "
                "MG9 until the reference CG of N9" % self.max_iterations)

    def __repr__(self) -> str:
        return "cg()" if self.max_iterations == 2000 else "cg(max_iterations=%d)" \
            % self.max_iterations


def cg(*, preconditioner: Any = None, max_iterations: int = 2000) -> SolverValue:
    """``cg(preconditioner=None, max_iterations=2000)``: conjugate gradients, for a symmetric
    positive definite operator (IR16); exhausting the iterations fails the solve (SV10)."""
    return _CG(preconditioner, max_iterations)


def _later(word: str) -> Any:
    def refused(*args: Any, **kwargs: Any) -> Any:
        from pops.time._program.traced_method import NotYetElaborated

        raise NotYetElaborated("pops.lib.solve.%s is elaborated by %s" % (word, _LATER))
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

__all__ = ["CGS2", "SolverValue", "auto", "bicgstab", "cg", "dense_lu", "gmres", "jacobi",
           "newton", "richardson"]
