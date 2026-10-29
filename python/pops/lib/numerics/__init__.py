"""pops.lib.numerics -- numerical methods written in Python over the public algebra.

Spec 0.3 (section 4.4, P11, A4, IR25) places the named numerical fluxes here as Python functions:
a library method is the same object a case would write by hand (CF4), and its formula becomes the
generated C++ (G2). :mod:`.fluxes` holds the Rusanov, upwind and HLL face fluxes of the N3a pilot;
the native ``pops.numerics.riemann`` routes stay in place until N3 moves the catalog.
"""
from .fluxes import hll, rusanov, spectral_radius, upwind

__all__ = ["hll", "rusanov", "spectral_radius", "upwind"]
