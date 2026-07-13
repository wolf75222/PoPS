"""pops.solvers.elliptic -- elliptic field-solver descriptors (Spec 5 sec.5.7).

Re-exports the rich :class:`GeometricMG` and the executable, route-constrained :class:`FFT`
descriptors from :mod:`pops.solvers.elliptic._descriptor`. See that module
for the parameter surface (typed smoother / coarse / tolerance) and the capability declaration.
"""
from ._descriptor import FFT, GeometricMG

__all__ = ["GeometricMG", "FFT"]
