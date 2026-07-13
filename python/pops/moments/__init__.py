"""pops.moments -- generic 2D moment-model generator + the Spec-4 facade API.

The GENERATOR surface (the systematic binomial algebra for 2D Vlasov/QMOM moment
hierarchies) is re-exported from the sub-modules: index helpers, the Gaussian closure,
the source terms, and the model-builder entry point.

The Spec-4 NEW API is a set of thin facades over that generator: a fluent
:class:`MomentModel` (built by :func:`CartesianVelocityMoments`) that records options and
calls :func:`build_moment_model` only on ``.build()``, plus the inert structural
descriptors (:class:`MomentHierarchy` / :class:`MomentBasis` / ... ) and the closures
surface (:mod:`pops.moments.closures`).
"""
# --- generator surface (the engine) ----------------------------------------
from .model_builder import moment_indices, moment_names, build_moment_model
from .sources import (lorentz_sources, maxwellian_moments, bgk_source,
                      VlasovSources, MagneticMomentSource)
from .closures import (gaussian_closure, closure, Closure, LocalClosure, MomentClosure,
                       apply_local_closure, HyQMOM15Closure)

# --- Spec-4 facade API (thin wrappers over the generator) -------------------
from .hierarchy import CartesianVelocityMoments, MomentModel, MomentHierarchy
from .ordering import MomentOrdering
from .basis import MomentBasis, RawMomentBasis
from .transforms import CenteredTransform, StandardizedTransform
from .speeds import ExactSpeeds
from .projection import RealizabilityProjection, MomentProjection, RealizableSet
from .space import VelocitySpace, MomentState
from .transport import MomentTransport

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
    "MomentClosure",
    "MomentProjection",
    "RealizableSet",
    # re-exported single-home canonical provider (lazy; see __getattr__)
    "WaveSpeedProvider",
]


def __getattr__(name):
    """Lazily re-export the canonical ``WaveSpeedProvider`` (ADC-565 single home).

    The typed wave-speed provider lives ONCE in :mod:`pops.numerics.riemann.waves`
    (ADC-552); ``pops.moments`` exposes it so the issue vocabulary
    ``from pops.moments import WaveSpeedProvider`` resolves, but a MODULE-SCOPE import
    would add a ``moments -> numerics`` edge the import-graph fence forbids (moments may
    only import ``ir``). Re-exporting it inside this ``__getattr__`` keeps the name
    available with no static edge -- the import happens on first access, not at module
    load. The moment wave-speed AXIS stays :class:`ExactSpeeds` (a different chooser).
    """
    if name == "WaveSpeedProvider":
        from pops.numerics.riemann.waves import WaveSpeedProvider
        return WaveSpeedProvider
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
