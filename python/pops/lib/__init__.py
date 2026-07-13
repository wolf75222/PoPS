"""pops.lib -- ready-to-use presets (Spec 5 sec.5.15, criterion 7).

Spec 5 moves the generic *building blocks* out of ``pops.lib`` into top-level central
packages and keeps ``pops.lib`` for things that are ready to use (criterion 7: presets only):

* :mod:`pops.lib.time` -- provided time-stepping scheme macros (forward_euler /
  SSPRK2 / ssprk3 / rk4 / strang / imex / bdf / predictor_corrector ...);
* :mod:`pops.lib.models` -- provided physical models (HyQMOM15 / Gaussian);
* :mod:`pops.lib.presets` -- ready-to-run compose-and-go bundles (a provided model paired
  with a provided time scheme; the user still picks the mesh layout at
  ``pops.compile(problem, layout=...)``).

The relocated central catalogs now live in:

* numerical fluxes / reconstruction / projections / spatial -> :mod:`pops.numerics`
* the elliptic-field brick catalog -> :mod:`pops.fields.catalog`
* moments tools -> :mod:`pops.moments`
* diagnostics -> :mod:`pops.diagnostics`
* the brick descriptor -> :mod:`pops.descriptors`
* linear / nonlinear / Schur / elliptic solvers + preconditioners -> :mod:`pops.solvers`
* the custom-solver generation DSL (internal / experimental) -> :mod:`pops.codegen.solvers`

There is exactly ONE public home for the solver descriptors: :mod:`pops.solvers`
(``pops.solvers.CG`` / ``GMRES`` / ``GeometricMG`` / ``Newton`` / ``Schur`` ...). ``pops.lib``
is NOT a second path -- the old ``pops.lib.solvers`` shim was removed (no back-compat alias).
"""
from . import time
from . import models
from . import presets
from . import amr

__all__ = ["amr", "time", "models", "presets"]
