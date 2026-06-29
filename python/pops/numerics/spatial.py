"""pops.numerics.spatial -- the finite-volume spatial-discretisation brick catalog (Spec 5 sec.5.4).

Spec 5 (criterion 7) homes the generic spatial-discretisation brick catalog in the top-level
:mod:`pops.numerics` package (alongside ``riemann`` / ``reconstruction`` / ``projections``),
moving it out of the transitional ``pops.lib.spatial``. ``pops.lib`` keeps only presets.

The finite-volume residual is assembled by the ``pops::SpatialDiscretisation<Limiter,
NumericalFlux>`` tag-type bundle (spatial_discretisation.hpp); there are no separate
residual/divergence/source-assembly types, so these name that bundle. Every entry is an inert
:class:`pops.descriptors.BrickDescriptor`; the codegen / runtime consume it.
"""
from types import SimpleNamespace

from pops.descriptors import _native, reject_string_selector


_RIEMANN_SCHEMES = {"rusanov", "hll", "hllc", "roe", "user"}
_RECON_SCHEMES = {
    "firstorder": "none",
    "none": "none",
    "minmod": "minmod",
    "vanleer": "vanleer",
    "weno5": "weno5",
    "weno5z": "weno5",
}
_VARIABLE_SCHEMES = {"conservative", "primitive"}


def _typed_scheme(value, *, param, categories, schemes, suggestion):
    if value is None:
        return None
    if isinstance(value, str):
        reject_string_selector(value, param, suggestion)
    category = getattr(value, "category", None)
    scheme = getattr(value, "scheme", None)
    if category not in categories or scheme is None:
        raise TypeError(
            "pops.numerics.spatial.FiniteVolume: %s must be a typed descriptor "
            "(got %r). Use %s." % (param, type(value).__name__, suggestion))
    if isinstance(schemes, dict):
        lowered = schemes.get(scheme)
    else:
        lowered = scheme if scheme in schemes else None
    if lowered is None:
        raise ValueError(
            "pops.numerics.spatial.FiniteVolume: %s descriptor scheme %r is not supported. "
            "Use %s." % (param, scheme, suggestion))
    return lowered


def _finite_volume(**o):
    """Finite-volume spatial descriptor with typed numerical choices.

    The descriptor stores canonical C++ routing tokens in ``options`` after validating the public
    arguments as typed ``pops.numerics`` descriptors. Strings are rejected here rather than later in
    ``System._lower_spatial``.
    """
    reconstruction = o.pop("reconstruction", None)
    limiter = o.pop("limiter", None)
    if reconstruction is not None:
        if limiter is not None:
            raise TypeError("FiniteVolume: pass reconstruction= or limiter=, not both")
        limiter = reconstruction
    riemann = o.pop("riemann", o.pop("flux", None))
    variables = o.pop("variables", o.pop("recon", None))
    if o:
        allowed = {"positivity_floor", "wave_speed_cache"}
        unknown = set(o) - allowed
        if unknown:
            raise TypeError("FiniteVolume: unexpected keyword argument(s): %s"
                            % ", ".join(sorted(unknown)))
    opts = {}
    rec = _typed_scheme(
        limiter, param="reconstruction", categories=("reconstruction", "limiter"),
        schemes=_RECON_SCHEMES,
        suggestion="pops.numerics.reconstruction.FirstOrder()/WENO5()/MUSCL(...)")
    flux = _typed_scheme(
        riemann, param="riemann", categories=("riemann",), schemes=_RIEMANN_SCHEMES,
        suggestion="pops.numerics.riemann.Rusanov()/HLL()/HLLC()/Roe()")
    var = _typed_scheme(
        variables, param="variables", categories=("variables",), schemes=_VARIABLE_SCHEMES,
        suggestion="pops.numerics.variables.Conservative()/Primitive()")
    if rec is not None:
        opts["reconstruction"] = rec
    if flux is not None:
        opts["riemann"] = flux
    if var is not None:
        opts["variables"] = var
    for key in ("positivity_floor", "wave_speed_cache"):
        if key in o:
            opts[key] = o[key]
    return _native("finite_volume", "pops::SpatialDiscretisation", "fv",
                   category="spatial", **opts)

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
    FiniteVolume=_finite_volume,
)

FiniteVolumeResidual = spatial.FiniteVolumeResidual
FluxDivergence = spatial.FluxDivergence
SourceAssembly = spatial.SourceAssembly
FiniteVolume = spatial.FiniteVolume

__all__ = [
    "spatial",
    "FiniteVolumeResidual",
    "FluxDivergence",
    "SourceAssembly",
    "FiniteVolume",
]
