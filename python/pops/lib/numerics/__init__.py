"""pops.lib.numerics -- numerical methods written in Python over the public algebra.

Spec 0.3 (section 4.4, P11, A4, IR25) places the named numerical fluxes and the discrete-algebra
builders here: a library method is the same object a case would write by hand (CF4), and its
formula becomes the generated C++ (G2).

* :mod:`.fluxes` -- the Rusanov, upwind and HLL face fluxes of the N3a pilot; the native
  ``pops.numerics.riemann`` routes stay in place until N3 moves the catalog.
* :mod:`.face_flux` -- the ``FaceFlux`` builder of the N7a pilot, realized by the b6
  ``StencilFlux`` native (IR24). The spelling is provisional: in spec 0.3 a case imports
  ``FaceFlux`` from ``pops.discrete``, which re-exports the builders of ``pops.lib`` (sec. 2.6.4,
  sec. 4.3, IR25) and does not exist at b6; N15 settles the surface.
"""
from .face_flux import (
    Compose, FaceFlux, Incidence, RealizedFaceFlux, Shift, realize_face_flux,
)
from .fluxes import hll, rusanov, upwind

__all__ = [
    "Compose", "FaceFlux", "Incidence", "RealizedFaceFlux", "Shift", "realize_face_flux",
    "hll", "rusanov", "upwind",
]
