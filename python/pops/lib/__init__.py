"""pops.lib -- ready-to-use presets (Spec 5 sec.5.15).

Spec 5 moves the generic *building blocks* out of ``pops.lib`` into top-level
central packages and keeps ``pops.lib`` for things that are ready to use:

* :mod:`pops.lib.time` -- provided time-stepping scheme macros (forward_euler /
  ssprk2 / ssprk3 / rk4 / strang / imex / bdf / predictor_corrector ...);
* :mod:`pops.lib.models` -- provided physical models (HyQMOM15 / Gaussian).

The relocated central catalogs now live in:

* numerical fluxes / reconstruction / projections -> :mod:`pops.numerics`
* moments tools -> :mod:`pops.moments`
* diagnostics -> :mod:`pops.diagnostics`
* the brick descriptor -> :mod:`pops.descriptors`

TRANSITIONAL (Spec 5 Phase A2): ``spatial`` / ``fields`` / ``solvers`` /
``preconditioners`` are still re-exported here until they move to
:mod:`pops.numerics` / :mod:`pops.fields` / :mod:`pops.solvers` / :mod:`pops.linalg`
(held back so the in-flight unified-install path is not disturbed). New code should
not rely on these ``pops.lib`` names.
"""
from .spatial import spatial
from .fields import fields
from .solvers import (solvers, solver, SolverContext, SolverIR,
                      build_solver_ir, generate_solver_cpp)
from .solvers.preconditioners import preconditioners
from . import time
from . import models

__all__ = ["spatial", "fields", "solvers", "preconditioners", "time", "models",
           "solver", "build_solver_ir", "generate_solver_cpp",
           "SolverContext", "SolverIR"]
