"""The solve algorithms of the library, written with ``@pops.solver`` (spec 0.3 2.6.9, SV9).

``cg(preconditioner=None, max_iterations=2000)`` is the reference conjugate gradient of the row
``cg`` of 2.6.9. A case passes the called word as the value of ``solvers={key: cg()}`` (SV4), and
the solve gives it its request (SV3). The native ``solve_cg`` is a bound substitution of this
reference (SV12, SV13) that runs only under ``pops.auto()`` until SV-T2 qualifies it at 64 ranks
(N9 lot 4, Q2).

**The relation solves of a traced method (N15 lot « solves », §5.5) run the reference.** A
``pops.solve`` of a relation in a method body records its operator, the rates of its components at
the unknown, on a State operand (IR26), and solves it with the body of ``cg()`` as the region of
its ``solve_linear`` node, under the request of the solve (N9 lot « La référence sous les solves de
relation », §3.4 and §3.8). Conjugate gradients need a symmetric positive definite operator: the
relation solve derives it from the claims of its components (IR16) and refuses it otherwise with
``PropertyNotDerived`` (K4).

The claims of the rate of a term read at the unknown are stated here, where N3 lot 1 put the map of
``two_point()`` beside ``cg()`` (D7), by ``linear_claims``: the two-point diffusion of
``two_point()``, recognized by the normal form of its face, of a coefficient that is a number or a
``model.constant``, on a uniform Cartesian periodic layout, is symmetric, negative semidefinite
when its coefficient is not negative, and linear and homogeneous in the state. Every other term is
refused with the lot that generates its operator (G10).

Every other word of 2.6.9 is importable and refuses its use with the lot that elaborates it.
"""
from .claims import linear_claims
from .krylov import cg
from .refused import CGS2, auto, bicgstab, dense_lu, gmres, jacobi, newton, richardson

__all__ = ["CGS2", "auto", "bicgstab", "cg", "dense_lu", "gmres", "jacobi", "newton",
           "richardson"]
