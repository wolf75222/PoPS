"""pops.lib.spatial -- the finite-volume spatial-discretisation brick catalog (Spec 3).

The finite-volume residual is assembled by the pops::SpatialDiscretisation<Limiter,
NumericalFlux> tag-type bundle (spatial_discretisation.hpp); there are no separate
residual/divergence/source-assembly types, so these name that bundle.
"""
from types import SimpleNamespace

from ..descriptors import _native

spatial = SimpleNamespace(
    FiniteVolumeResidual=lambda **o: _native(
        "fv_residual", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
    FluxDivergence=lambda **o: _native(
        "flux_divergence", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
    SourceAssembly=lambda **o: _native(
        "source_assembly", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
    # The whole finite-volume spatial brick selected per instance by the unified sim.install (Spec 3
    # section 22): it carries the runtime scheme options (riemann / reconstruction / positivity_floor)
    # that System.install lowers to the existing add_equation spatial args. ``riemann`` names the
    # NUMERICAL Riemann flux (not the model's physical flux); ``reconstruction`` is the limiter
    # (none/minmod/vanleer/weno5).
    FiniteVolume=lambda **o: _native(
        "finite_volume", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
)

__all__ = ["spatial"]
