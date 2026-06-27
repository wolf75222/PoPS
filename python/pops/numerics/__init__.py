"""pops.numerics -- discretisation-of-PDE descriptors (Spec 5 sec.5.4).

This package answers "how a PDE is turned into discrete operators":

* :mod:`pops.numerics.riemann` -- numerical fluxes (Rusanov/HLL/HLLC/Roe);
* :mod:`pops.numerics.reconstruction` -- face reconstruction + slope limiters;
* :mod:`pops.numerics.projections` -- per-cell projection bricks (positivity ...);
* :mod:`pops.numerics.spatial` -- the finite-volume residual bundle (Spec 5 Phase A2).

Every entry is an inert :class:`pops.descriptors.BrickDescriptor`; codegen and the
runtime consume them, nothing here computes in Python. ``pops.numerics`` is the Spec 5
home of the catalogs formerly parked under ``pops.lib``.
"""
from . import riemann, reconstruction, projections
from .reconstruction import limiters

__all__ = ["riemann", "reconstruction", "limiters", "projections"]
