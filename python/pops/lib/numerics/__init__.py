"""pops.lib.numerics -- numerical methods written in Python over the public algebra.

Spec 0.3 (section 4.4, P11, A4, IR25) places the named numerical fluxes and the discrete-algebra
builders here: a library method is the same object a case would write by hand (CF4), and its
formula becomes the generated C++ (G2).

* :mod:`.fluxes` -- the scheme words ``upwind()``, ``rusanov()``, ``hll()`` and ``two_point()``,
  written with ``@pops.scheme`` over ``face`` and ``term`` (2.6.9, N3 lot 1). The b6 bodies that
  they replace are the parity oracles of ``examples/api_v02/b6_numerics.py`` (D5).
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
  ``StencilFlux`` native (IR24). It is imported from its module, not from this package: a case
  writes ``FaceFlux`` from ``pops.surface.discrete`` (2.6.4), so the package does not re-export
  the pilot's spelling (N3 lot 1, O2).
"""
from .fluxes import hll, rusanov, two_point, upwind
from .limiters import mc, minmod, superbee, vanleer
from .reconstructions import Reconstruct, muscl, weno5

__all__ = [
    "hll", "rusanov", "two_point", "upwind",
    "Reconstruct", "muscl", "weno5", "mc", "minmod", "superbee", "vanleer",
]

