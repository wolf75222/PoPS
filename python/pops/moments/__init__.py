"""pops.moments -- explicit two-velocity-dimensional moment generator and facade API.

The generator surface (the systematic binomial algebra for 2V Vlasov/QMOM moment
hierarchies) is re-exported from the sub-modules: index helpers, the Gaussian closure,
the source terms, and the model-builder entry point.

The public construction API is a set of thin facades over that generator: a fluent
:class:`MomentModel` (built by :func:`CartesianVelocityMoments`) that records options and
calls :func:`build_moment_model` only on ``.build()``, plus the inert structural
descriptors (:class:`MomentHierarchy` / :class:`MomentBasis` / ... ) and the closures
surface (:mod:`pops.moments.closures`). This package is an advertised 2V/2D physical
specialization, not a fallback used by the generic 1D/2D/3D runtime. One
explicit order-four axial B.1 flux is also exported for a one-velocity case;
no all-order one-velocity recurrence is implied. The mathematical
:func:`affine_push_forward` is independent of these physical closures and accepts
explicit multi-indices in any velocity dimension.
"""
# --- generator surface (the engine) ----------------------------------------
from .model_builder import (
    build_moment_model,
    moment_flux_expressions,
    moment_indices,
    moment_names,
    moment_transport_blocks,
)
from .sources import (MOMENT_VELOCITY_DIMENSION, lorentz_sources, maxwellian_moments, bgk_source,
                      VlasovSources, MagneticMomentSource)
from .closures import (gaussian_closure, closure, Closure, LocalClosure,
                       apply_local_closure, HyQMOM15Closure, hyqmom_b1_axial_flux,
                       DiscreteEntropyQuadrature)

# --- facade API (thin wrappers over the generator) -------------------------
from .hierarchy import CartesianVelocityMoments, CompositeMean, MomentModel, MomentHierarchy
from .ordering import MomentOrdering
from .basis import CartesianMonomialBasis, MomentBasis, RawMomentBasis
from .transforms import CenteredTransform, StandardizedTransform
from .affine import affine_push_forward
from .speeds import ExactSpeeds
from .projection import RealizabilityProjection, RealizableSet
from .relaxation import HyQMOM15Relaxation
from .space import VelocitySpace, MomentState
from .transport import MomentTransport
from .fan_li import (
    FAN_LI15_INDICES,
    FAN_LI15_REGULARIZED_COMPONENTS,
    FanLi15Expressions,
    fan_li15_expressions,
    fan_li15_from_hermite,
)

__all__ = [
    # public generator surface
    "moment_indices",
    "affine_push_forward",
    "moment_names",
    "moment_transport_blocks",
    "MOMENT_VELOCITY_DIMENSION",
    "gaussian_closure",
    "lorentz_sources",
    "maxwellian_moments",
    "bgk_source",
    "build_moment_model",
    "moment_flux_expressions",
    "FAN_LI15_INDICES",
    "FAN_LI15_REGULARIZED_COMPONENTS",
    "FanLi15Expressions",
    "fan_li15_expressions",
    "fan_li15_from_hermite",
    # facade API
    "CartesianMonomialBasis",
    "CartesianVelocityMoments",
    "CompositeMean",
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
    "hyqmom_b1_axial_flux",
    "DiscreteEntropyQuadrature",
    # generic construction vocabulary (ADC-543): inert handles + typed aliases
    "VelocitySpace",
    "MomentState",
    "MomentTransport",
    "RawMomentBasis",
    "RealizableSet",
]

def fan_li15_path(product, *, frame, covectors, basis):
    from .fan_li_path import fan_li15_path as compose
    return compose(product, frame=frame, covectors=covectors, basis=basis)
from .polynomial_path import (NormalizedPathInputs, EndpointPathInputs, normalized_polynomial_path,
    endpoint_polynomial_path, derivative as polynomial_derivative, fma, compensated_sum)

__all__ += ["fan_li15_path", "NormalizedPathInputs", "EndpointPathInputs",
            "normalized_polynomial_path", "endpoint_polynomial_path",
            "polynomial_derivative", "fma", "compensated_sum"]
