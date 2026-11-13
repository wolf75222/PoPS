"""pops.lib.numerics -- numerical methods written in Python over the public algebra.

Spec 0.3 (section 4.4, P11, A4, IR25) places the named numerical fluxes and the discrete-algebra
builders here: a library method is the same object a case would write by hand (CF4), and its
formula becomes the generated C++ (G2).

* :mod:`.fluxes` -- the scheme words ``upwind()``, ``rusanov()``, ``hll()`` and ``two_point()``,
  written with ``@pops.scheme`` over ``face`` and ``term`` (2.6.9, N3 lot 1). The b6 bodies that
  they replace are the parity oracles of ``examples/api_v02/b6_numerics.py`` (D5).
* :mod:`.limiters` -- ``muscl(limiter=)``, a scheme that returns ``Reconstruct``, and the slope
  limiters ``minmod``, ``vanleer``, ``mc`` and ``superbee``, passed uncalled to it, written with the
  surface (2.6.9, N3 lot 2).
* :mod:`.reconstructions` -- ``weno5(epsilon=)``, a scheme that returns ``Reconstruct``, written
  with the surface (2.6.9, N3 lot 3). The b6 ``Reconstruct`` builder is not ported (D4): a case
  writes ``Reconstruct`` from ``pops.surface.discrete``. The b6 bodies of ``muscl``, its limiters
  and ``weno5`` are the parity oracles of ``examples/api_v02/b6_reconstructions.py``.
* :mod:`.boundaries` -- ``ghosts()``, the boundary policy of 2.6.9, written with the surface
  (N15 lot B): the ghost 2g - u_k of a value and u_1 of no data. The package re-exports it, so
  that a case writes ``from pops.lib.numerics import upwind, muscl, vanleer, ghosts`` as the
  gallery does (N3, ahead of the switch to one API: the phase 1 vertical slice reads it).
* :mod:`.ghost_laws` -- the b6 conditions of those two ghost laws, the bridge of the boundary
  policy until G04 generates the ghost fills.
* :mod:`.face_flux` -- the ``FaceFlux`` builder of the N7a pilot, realized by the b6
  ``StencilFlux`` native (IR24). It is imported from its module, not from this package: a case
  writes ``FaceFlux`` from ``pops.surface.discrete`` (2.6.4), so the package does not re-export
  the pilot's spelling (N3 lot 1, O2).
"""
from .boundaries import ghosts
from .fluxes import hll, rusanov, two_point, upwind
from .limiters import mc, minmod, muscl, superbee, vanleer
from .reconstructions import weno5

__all__ = [
    "ghosts", "hll", "rusanov", "two_point", "upwind",
    "muscl", "weno5", "mc", "minmod", "superbee", "vanleer",
]

