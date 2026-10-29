"""Discrete-algebra builders of spec 0.3 (IR25), written over the public surface.

Only the phase 0 pilot of N7a lives here so far: the ``FaceFlux`` builder of a
scheme over a state, realized by the b6 ``StencilFlux`` native (IR24).
"""
from .face_flux import (
    Compose, FaceFlux, Incidence, RealizedFaceFlux, Shift, realize_face_flux,
)

__all__ = ["Compose", "FaceFlux", "Incidence", "RealizedFaceFlux", "Shift", "realize_face_flux"]
