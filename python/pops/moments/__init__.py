"""pops.moments -- generic moment-model authoring tools.

This is the central package for generic moments. Ready-to-use models live under
``pops.lib.models.moments``; ``pops.lib`` is not the home for moment authoring tools.
"""
# --- generator surface (the engine) ----------------------------------------
from .model_builder import moment_indices, moment_names, build_moment_model
from .sources import (lorentz_sources, maxwellian_moments, bgk_source, VlasovSources,
                      MomentSource, VlasovElectricSource, MagneticRotationSource,
                      MagneticMomentSource)
from .closures import gaussian_closure, closure, Closure, HyQMOM15Closure

# --- Spec-4 facade API (thin wrappers over the generator) -------------------
from .hierarchy import CartesianVelocityMoments, MomentModel, MomentHierarchy
from .ordering import MomentOrdering
from .basis import MomentBasis
from .transforms import CenteredTransform, StandardizedTransform
from .speeds import ExactSpeeds
from .projection import RealizabilityProjection

__all__ = [
    # generator surface (back-compat with the flat pops.moments)
    "moment_indices",
    "moment_names",
    "gaussian_closure",
    "lorentz_sources",
    "maxwellian_moments",
    "bgk_source",
    "build_moment_model",
    # Spec-4 NEW API
    "CartesianVelocityMoments",
    "MomentModel",
    "MomentHierarchy",
    "MomentOrdering",
    "MomentBasis",
    "CenteredTransform",
    "StandardizedTransform",
    "ExactSpeeds",
    "RealizabilityProjection",
    "VlasovSources",
    "MomentSource",
    "VlasovElectricSource",
    "MagneticRotationSource",
    "MagneticMomentSource",
    "closure",
    "Closure",
    "HyQMOM15Closure",
]
