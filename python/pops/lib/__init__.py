"""pops.lib -- ready-to-use presets (Spec 5 sec.5.15, criterion 7).

Spec 5 moves the generic *building blocks* out of ``pops.lib`` into top-level central
packages and keeps ``pops.lib`` for things that are ready to use (criterion 7: presets only):

* :mod:`pops.lib.time` -- provided time-stepping scheme macros (forward_euler /
  ssprk2 / ssprk3 / rk4 / strang / imex / bdf / predictor_corrector ...);
* :mod:`pops.lib.models` -- provided physical models (HyQMOM15 / Gaussian).

The relocated central catalogs now live in:

* numerical fluxes / reconstruction / projections / spatial -> :mod:`pops.numerics`
* the elliptic-field brick catalog -> :mod:`pops.fields.catalog`
* moments tools -> :mod:`pops.moments`
* diagnostics -> :mod:`pops.diagnostics`
* the brick descriptor -> :mod:`pops.descriptors`
* linear / nonlinear / Schur / elliptic solvers + preconditioners -> :mod:`pops.solvers`
* the custom-solver generation DSL (internal / experimental) -> :mod:`pops.codegen.solvers`

The ``solvers`` / ``preconditioners`` names are re-exported here from :mod:`pops.solvers` via
the :mod:`pops.lib.solvers` back-compat shim (presets only: the authoring DSL no longer lives
in ``pops.lib``; it moved to :mod:`pops.codegen.solvers`, criterion 19).
"""
from .solvers import solvers, preconditioners
from . import time
from . import models

__all__ = ["solvers", "preconditioners", "time", "models"]
