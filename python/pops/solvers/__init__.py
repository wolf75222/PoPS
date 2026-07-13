"""pops.solvers -- the linear / nonlinear / elliptic solver descriptor catalog (Spec 5 sec.5.7).

Spec 5 (sec.4 / 5.7 / 13.11.1) homes the solver catalog in this top-level central package
(alongside :mod:`pops.numerics` / :mod:`pops.linalg` / :mod:`pops.fields`). This is the ONE
public home for the solver descriptors: the transitional ``pops.lib.solvers`` re-export shim is
removed (no second public path; ``pops.lib`` is presets-only). Every entry is an inert
descriptor: codegen / the runtime consume it, nothing here computes in Python (the C++ solvers
execute).

Sub-packages:

* :mod:`pops.solvers.krylov` -- matrix-free Krylov solvers (CG / BiCGStab / GMRES / Richardson);
* :mod:`pops.solvers.nonlinear` -- executable global ``Newton`` and cell-local ``LocalNewton``;
* :mod:`pops.solvers.schur` -- the Schur-condensation solver (``pops::SchurCondensationOperator``);
* :mod:`pops.solvers.elliptic` -- the RICH GeometricMG (typed smoother / coarse / tolerance /
  max_cycles + capabilities) and the planned FFT spectral Poisson solver;
* :mod:`pops.solvers.preconditioners` -- Identity / Jacobi / BlockJacobi / GeometricMG / User;
* :mod:`pops.solvers.options` / :mod:`pops.solvers.tolerances` -- the typed smoother / coarse /
  tolerance sub-descriptors the elliptic solver takes;
* :mod:`pops.solvers.requirements` -- the solver capability vocabulary.

The ``solvers`` :class:`types.SimpleNamespace` gathers the Krylov + nonlinear + Schur factories
under one attribute surface (``solvers.CG()`` / ``solvers.GMRES()`` / ``solvers.Newton()`` /
``solvers.Schur()``); the custom-solver GENERATION DSL (``@solver`` / ``SolverContext`` /
``build_solver_ir`` / ``generate_solver_cpp``) is internal / experimental and lives in
:mod:`pops.codegen.solvers` (Spec 5 criterion 19; it imports the heavy ``pops.time`` lazily); it
is NOT a public attribute of this package. The custom-solver registry hooks (``solvers.custom`` /
``solvers.registered``) are attached onto the ``solvers`` namespace below when
:mod:`pops.codegen.solvers` is imported (so this package stays a pure descriptor catalog).
"""
from types import SimpleNamespace

from . import elliptic, krylov, nonlinear, options, requirements, schur, tolerances
from .elliptic import FFT, GeometricMG
from .krylov import CG, BiCGStab, GMRES, Richardson
from .local import DenseLU
from .nonlinear import LocalNewton, Newton
from .preconditioners import preconditioners
from .schur import CondensedSchur, Schur
from .scopes import Hierarchy, Level, SolveScope
from .providers import CompositeTensorFAC, HierarchySolveProvider

# The flat solver factory surface (``solvers.CG()`` ... ``solvers.Schur()``), the one public
# factory namespace (the legacy ``pops.lib.solvers.solvers`` shim was removed). The custom-solver
# registry hooks
# (``solvers.custom`` / ``solvers.registered``) are attached onto this namespace by
# :mod:`pops.codegen.solvers`, which owns the generation DSL -- keeping ``pops.solvers`` free of
# any DSL / codegen import (no cycle: pops.solvers imports nothing).
solvers = SimpleNamespace(
    CG=CG, BiCGStab=BiCGStab, GMRES=GMRES, Richardson=Richardson,
    Newton=Newton, LocalNewton=LocalNewton, DenseLU=DenseLU, Schur=Schur,
)

__all__ = [
    "elliptic", "krylov", "nonlinear", "schur", "options", "tolerances",
    "preconditioners", "requirements",
    "GeometricMG", "FFT",
    "CG", "BiCGStab", "GMRES", "Richardson",
    "Newton", "LocalNewton", "DenseLU", "Schur", "CondensedSchur",
    "SolveScope", "Level", "Hierarchy", "HierarchySolveProvider", "CompositeTensorFAC",
    "solvers",
]
