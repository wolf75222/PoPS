"""pops.lib.solvers -- BACK-COMPAT shim onto :mod:`pops.solvers` (Spec 5 sec.5.7, criterion 4).

Spec 5 moved the linear / nonlinear / Schur / elliptic / preconditioner solver catalog OUT of
``pops.lib`` into the top-level :mod:`pops.solvers` central package (criterion 4 / sec.5.7 /
13.11.1). This module is now a thin re-export so the in-flight install path and existing code
(``pops.lib.solvers.GMRES()``, ``pops.lib.solvers.solvers``, ``pops.lib.solvers.preconditioners``)
keep resolving. New code should import from :mod:`pops.solvers`.

``pops.lib`` is presets-only (criterion 7): the custom-solver AUTHORING / GENERATION DSL no
longer lives here. It moved to :mod:`pops.codegen.solvers` (Spec 5 criterion 19: a solver-gen
DSL, if any, lives in ``pops.codegen.solvers``, internal / experimental). Importing that package
wires the custom-solver registry hooks (``solvers.custom`` / ``solvers.registered``) onto the
shared :data:`pops.solvers.solvers` namespace re-exported below.
"""
from pops.solvers import (CG, BiCGStab, FixedPoint, GMRES, GeometricMG, Newton,
                          Richardson, Schur, solvers)
from pops.solvers.preconditioners import preconditioners

__all__ = ["solvers", "preconditioners",
           "CG", "BiCGStab", "GMRES", "Richardson", "Newton", "FixedPoint", "Schur",
           "GeometricMG"]
