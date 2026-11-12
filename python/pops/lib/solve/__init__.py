"""The solve algorithms of the library, written with ``@pops.solver`` (spec 0.3 2.6.9, SV9).

``cg(preconditioner=None, max_iterations=2000)`` is the reference conjugate gradient of the row
``cg`` of 2.6.9. A case passes the called word as the value of ``solvers={key: cg()}`` (SV4), and
the solve gives it its request (SV3). The native ``solve_cg`` is a bound substitution of this
reference (SV12, SV13) that runs only under ``pops.auto()`` until SV-T2 qualifies it at 64 ranks
(N9 lot 4, Q2).
"""
from .krylov import cg

__all__ = ["cg"]
