"""pops.linalg -- the abstract algebraic layer (Spec 5 sec.5.6).

``pops.linalg`` NAMES the linear algebra of a solve: the system ``A x = b``
(:class:`LinearProblem`), the operators (:class:`LinearOperator` /
:class:`MatrixFreeOperator`), the residual ``b - A x`` (:class:`Residual`), the typed vector
norms (:class:`L1` / :class:`L2` / :class:`LInf`) and the scalar reductions (:class:`Dot` /
:class:`Norm2`, with the :func:`dot` / :func:`norm2` builders).

It does NOT solve and it does NOT compute in Python -- everything is an inert typed descriptor;
the C++ runtime applies the operators, forms the residual and evaluates the norms/reductions.
The solvers AND the preconditioners live in :mod:`pops.solvers` (ADC-502 ratifies
:mod:`pops.solvers.preconditioners` as their single home); ``pops.linalg`` deliberately exposes
NO ``preconditioners`` submodule -- a preconditioner configures a solver, so it belongs with the
solver descriptors, and the no-retro-compat regime forbids a second public path / shim.
"""
from .operator import LinearOperator, MatrixFreeOperator
from .problem import LinearProblem, Residual
from .norms import L1, L2, LInf
from .reductions import Dot, Norm2, dot, norm2
from . import operator, problem, norms, reductions

__all__ = [
    "LinearOperator", "MatrixFreeOperator",
    "LinearProblem", "Residual",
    "L1", "L2", "LInf",
    "Dot", "Norm2", "dot", "norm2",
    "operator", "problem", "norms", "reductions",
]
