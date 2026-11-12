"""pops.lib.numerics -- numerical methods written in Python over the public algebra.

Spec 0.3 (section 4.4, P11, A4, IR25) places the named numerical fluxes and the discrete-algebra
builders here: a library method is the same object a case would write by hand (CF4), and its
formula becomes the generated C++ (G2).

* :mod:`.fluxes` -- the Rusanov, upwind and HLL face fluxes (N3a pilot, written to the rows of
  2.6.9 by N3); the native ``pops.numerics.riemann`` routes stay in place until N3 retires them
  after parity (MG1).
* :mod:`.reconstructions` -- the per-axis ``Reconstruct`` builder, ``muscl`` and ``weno5`` (N3).
* :mod:`.limiters` -- the slope limiters ``minmod``, ``vanleer``, ``mc`` and ``superbee`` (N3),
  passed uncalled to ``muscl``.
* :mod:`.boundaries` -- ``ghosts()``, the boundary policy of 2.6.9, written with the surface
  (N15 lot B): the ghost 2g - u_k of a value and u_1 of no data. It is imported from its module,
  as the scheme words of :mod:`.schemes` are, so that importing this package does not import the
  surface; ``from pops.lib.numerics import ghosts`` comes with the switch to one API (N3).
* :mod:`.ghost_laws` -- the b6 conditions of those two ghost laws, the bridge of the boundary
  policy until G04 generates the ghost fills.
* :mod:`.face_flux` -- the ``FaceFlux`` builder of the N7a pilot, realized by the b6
  ``StencilFlux`` native (IR24). The spelling is provisional: in spec 0.3 a case imports
  ``FaceFlux`` from ``pops.discrete``, which re-exports the builders of ``pops.lib`` (sec. 2.6.4,
  sec. 4.3, IR25) and does not exist at b6; N15 settles the surface. The typed maps it reads
  through are those of ``pops.regions``, their one home.
"""
from .face_flux import FaceFlux, RealizedFaceFlux, realize_face_flux
from .fluxes import hll, rusanov, upwind
from .limiters import mc, minmod, superbee, vanleer
from .reconstructions import Reconstruct, muscl, weno5

__all__ = [
    "FaceFlux", "RealizedFaceFlux", "realize_face_flux", "hll", "rusanov", "upwind",
    "Reconstruct", "muscl", "weno5", "mc", "minmod", "superbee", "vanleer",
]

