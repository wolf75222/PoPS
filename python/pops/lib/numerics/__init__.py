"""Discrete-algebra builders of spec 0.3 (IR25), written over the public surface.

Only the phase 0 pilot of N7a lives here so far: the ``FaceFlux`` builder of a
scheme over a state, realized by the b6 ``StencilFlux`` native (IR24). The
spelling is provisional: in spec 0.3 a case imports ``FaceFlux`` from
``pops.discrete``, which re-exports the builders of ``pops.lib`` (sec. 2.6.4,
sec. 4.3, IR25) and does not exist at b6; N15 settles the surface.
"""
from .face_flux import (
    Compose, FaceFlux, Incidence, RealizedFaceFlux, Shift, realize_face_flux,
)

__all__ = ["Compose", "FaceFlux", "Incidence", "RealizedFaceFlux", "Shift", "realize_face_flux"]
