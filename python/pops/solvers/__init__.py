"""pops.solvers -- the linear / elliptic solver descriptor catalog (Spec 5 sec.5.7).

Spec 5 (sec.4 / 5.7 / 13.11.1) homes the solver catalog in this top-level central package
(alongside :mod:`pops.numerics` / :mod:`pops.linalg` / :mod:`pops.fields`). This is the ONE
public home for the solver descriptors: the transitional ``pops.lib.solvers`` re-export shim is
removed (no second public path; ``pops.lib`` is presets-only). Every entry is an inert
descriptor: codegen / the runtime consume it, nothing here computes in Python (the C++ solvers
execute).

Sub-packages:

* :mod:`pops.solvers.krylov` -- matrix-free Krylov solvers (CG / BiCGStab / GMRES / Richardson);
* :mod:`pops.solvers.schur` -- Schur and condensed-Schur solver descriptors;
* :mod:`pops.solvers.nonlinear` -- generated nonlinear solver descriptors;
* :mod:`pops.solvers.elliptic` -- GeometricMG (typed smoother / coarse / tolerance /
  max_cycles + capabilities) and FFT spectral Poisson descriptors;
* :mod:`pops.solvers.preconditioners` -- Identity / GeometricMG / User;
* :mod:`pops.solvers.options` / :mod:`pops.solvers.tolerances` -- the typed smoother / coarse /
  tolerance sub-descriptors the elliptic solver takes;
* :mod:`pops.solvers.requirements` -- the solver capability vocabulary.

The ``solvers`` :class:`types.SimpleNamespace` gathers the Krylov + Schur factories under one
attribute surface (``solvers.CG(max_iter=...)`` / ``solvers.GMRES(max_iter=...)`` /
``solvers.Schur()``). Solver-generation helpers live only in
:mod:`pops.codegen.solvers` experimental internals; importing that package must not mutate this
public catalog.
"""
from types import SimpleNamespace

from . import elliptic, krylov, nonlinear, options, preconditioners, requirements, schur, tolerances
from .elliptic import FFT, GeometricMG
from .krylov import CG, BiCGStab, GMRES, Richardson
from .schur import Schur

# The flat solver factory surface (``solvers.CG(max_iter=...)`` ... ``solvers.Schur()``), the one public
# factory namespace (the legacy ``pops.lib.solvers.solvers`` shim was removed). It intentionally
# exposes only compiled solver descriptors, never custom-solver authoring hooks.
solvers = SimpleNamespace(
    CG=CG, BiCGStab=BiCGStab, GMRES=GMRES, Richardson=Richardson,
    Schur=Schur,
)

__all__ = [
    "elliptic", "krylov", "schur", "nonlinear", "options", "tolerances",
    "preconditioners", "requirements",
    "GeometricMG", "FFT",
    "CG", "BiCGStab", "GMRES", "Richardson",
    "Schur",
    "solvers",
]
