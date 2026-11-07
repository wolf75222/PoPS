"""Ready-to-use implementations built from the ordinary public contracts.

Spec 5 moves the generic *building blocks* out of ``pops.lib`` into top-level central
packages and keeps ``pops.lib`` for things that are ready to use (criterion 7: presets only):

* :mod:`pops.lib.time` -- provided time-stepping Programs;
* :mod:`pops.lib.models` -- provided physical models (HyQMOM15 / Gaussian);
* :mod:`pops.lib.initial` -- native initial profiles;
* :mod:`pops.lib.amr` -- native transfer implementations;
* :mod:`pops.lib.numerics` -- numerical fluxes written in Python (spec 0.3 section 4.4) and
  builders of the discrete algebra (spec 0.3 IR25), from the N3a and N7a pilots.

The relocated central catalogs now live in:

* numerical fluxes / reconstruction / projections / spatial -> :mod:`pops.numerics`
* the elliptic-field brick catalog -> :mod:`pops.fields.catalog`
* moments tools -> :mod:`pops.moments`
* diagnostics -> :mod:`pops.diagnostics`
* the brick descriptor -> :mod:`pops.descriptors`
* executable linear / nonlinear / elliptic solvers + preconditioners -> :mod:`pops.solvers`
* the custom-solver generation DSL (internal / experimental) -> :mod:`pops.codegen.solvers`

There is exactly ONE public home for the solver descriptors: :mod:`pops.solvers`
(``pops.solvers.CG`` / ``GMRES`` / ``GeometricMG`` / ``Newton`` ...). ``pops.lib``
is NOT a second path -- the old ``pops.lib.solvers`` shim was removed (no back-compat alias).
"""
from . import time
from . import models
from . import amr
from . import initial
from . import numerics

__all__ = ["amr", "initial", "models", "numerics", "time"]
