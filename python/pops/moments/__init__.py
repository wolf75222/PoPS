"""pops.moments -- generic 2D moment-model generator and facade API.

The GENERATOR surface (the systematic binomial algebra for 2D Vlasov/QMOM moment
hierarchies) is re-exported from the sub-modules: index helpers, the Gaussian closure,
the source terms, and the model-builder entry point.

The public construction API is a set of thin facades over that generator: a fluent
:class:`MomentModel` (built by :func:`CartesianVelocityMoments`) that records options and
calls :func:`build_moment_model` only on ``.build()``, plus the inert structural
descriptors (:class:`MomentHierarchy` / :class:`MomentBasis` / ... ) and the closures
surface (:mod:`pops.moments.closures`).
"""
# --- generator surface (the engine) ----------------------------------------
from .model_builder import (
    build_moment_model,
    moment_indices,
    moment_names,
    moment_transport_blocks,
)
from .sources import (lorentz_sources, maxwellian_moments, bgk_source,
                      VlasovSources, MagneticMomentSource)
from .closures import (gaussian_closure, closure, Closure, LocalClosure,
                       apply_local_closure, HyQMOM15Closure)

# --- facade API (thin wrappers over the generator) -------------------------
from .hierarchy import CartesianVelocityMoments, MomentModel, MomentHierarchy
from .ordering import MomentOrdering
from .basis import MomentBasis, RawMomentBasis
from .transforms import CenteredTransform, StandardizedTransform
from .speeds import ExactSpeeds
from .projection import RealizabilityProjection, RealizableSet
from .relaxation import HyQMOM15Relaxation
from .space import VelocitySpace, MomentState
from .transport import MomentTransport

__all__ = [
    # public generator surface
    "moment_indices",
    "moment_names",
    "moment_transport_blocks",
    "gaussian_closure",
    "lorentz_sources",
    "maxwellian_moments",
    "bgk_source",
    "build_moment_model",
    # facade API
    "CartesianVelocityMoments",
    "MomentModel",
    "MomentHierarchy",
    "MomentOrdering",
    "MomentBasis",
    "CenteredTransform",
    "StandardizedTransform",
    "ExactSpeeds",
    "RealizabilityProjection",
    "HyQMOM15Relaxation",
    "VlasovSources",
    "MagneticMomentSource",
    "closure",
    "Closure",
    "LocalClosure",
    "apply_local_closure",
    "HyQMOM15Closure",
    # generic construction vocabulary (ADC-543): inert handles + typed aliases
    "VelocitySpace",
    "MomentState",
    "MomentTransport",
    "RawMomentBasis",
    "RealizableSet",
]
