"""pops.numerics.spatial -- the finite-volume spatial-discretisation home (Spec 5 sec.5.4).

Spec 5 (criterion 7) homes the generic spatial-discretisation surface in the top-level
:mod:`pops.numerics` package (alongside ``riemann`` / ``reconstruction`` / ``projections``),
moving it out of the transitional ``pops.lib.spatial``. ``pops.lib`` keeps only presets.

Two things live here:

* the inert brick catalog (:data:`spatial`, a namespace of :class:`pops.descriptors.BrickDescriptor`
  entries) the codegen / runtime consume; and
* the real :func:`FiniteVolume` composite (ADC-533) -- the ``FiniteVolume(riemann=HLL(...),
  reconstruction=MUSCL(...))`` authoring surface used across the runtime and the tests. It is homed
  HERE and re-exported from ``pops.runtime._bricks_scheme`` (its historical site) so every existing
  ``pops.FiniteVolume`` import path keeps working.

The finite-volume residual is assembled by the ``pops::SpatialDiscretisation<Limiter,
NumericalFlux>`` tag-type bundle (spatial_discretisation.hpp); there are no separate
residual/divergence/source-assembly types, so these name that bundle.
"""
from types import SimpleNamespace

from pops.descriptors import _native

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
    # (none/minmod/vanleer/weno5). It stores its scheme choice as STRING options, distinct from the
    # module-level composite ``FiniteVolume`` below (which requires TYPED pops.numerics descriptors).
    FiniteVolume=lambda **o: _native(
        "finite_volume", "pops::SpatialDiscretisation", "fv", category="spatial", **o),
)


# --- The composite finite-volume authoring surface (ADC-533, homed here) ----------------------
# reconstruction / riemann / variables typed-descriptor suggestions (mirror of the runtime tables).
_LIMITER_SUGGEST = ("pops.numerics.reconstruction.limiters.Minmod() / .VanLeer(), "
                    "pops.numerics.reconstruction.FirstOrder() / WENO5() / MUSCL(...)")
_FLUX_SUGGEST = "pops.numerics.riemann.Rusanov() / HLL() / HLLC() / Roe()"
_RECON_SUGGEST = "pops.numerics.variables.Conservative() / Primitive()"


def FiniteVolume(limiter=None, riemann=None, variables=None,
                 positivity_floor=None, wave_speed_cache=False, *, reconstruction=None,
                 none=False, minmod=False, vanleer=False, weno5=False, primitive=False):
    """Finite-volume scheme: a TYPED reconstruction + numerical Riemann flux + variable set.

    Homed in ``pops.numerics.spatial`` (Spec 5 criterion 7 / ADC-533) and re-exported at
    ``pops.runtime._bricks_scheme.FiniteVolume`` and ``pops.FiniteVolume`` so every existing import
    path keeps working. The NUMERICAL Riemann flux is named ``riemann`` (NOT ``flux``, reserved for
    the PHYSICAL flux of the DSL model m.flux) so the two meanings do not collide. Spec 5 sec.7:
    each argument is a TYPED ``pops.numerics`` descriptor (a bare string raises, pointing at the
    typed object). Argument mapping:

    - ``limiter`` (Spec 5 sec.14.1 alias: ``reconstruction``; a reconstruction / limiter descriptor)
      -> Spatial.limiter (``pops.numerics.reconstruction.FirstOrder()`` -> none, ``.limiters.Minmod()``
      / ``.VanLeer()``, ``.WENO5()``, ``.MUSCL(limiter=...)``)
    - ``riemann`` (``pops.numerics.riemann`` descriptor) -> Spatial.flux (Rusanov()/HLL()/HLLC()/Roe());
      HLL() is the generic signed-wave path (requires model.wave_speeds), HLLC()/Roe() run on the
      canonical Euler 2D layout or generically via the model hooks HasHLLCStructure / HasRoeDissipation
    - ``variables`` (``pops.numerics.variables`` descriptor) -> Spatial.recon (Conservative()/Primitive())

    The boolean-flag shortcuts of ``Spatial`` are forwarded identically:
    ``none=/minmod=/vanleer=/weno5=`` select the limiter and ``primitive=`` selects the variable set.
    Returns a ``Spatial`` (consumed as-is by add_block / add_equation). ``positivity_floor`` (ADC-76):
    density floor of the face states (Zhang-Shu limiter), None/0 = inactive. ``wave_speed_cache``:
    HLL wave speed cache (riemann=HLL() + explicit), cf. Spatial.
    """
    from pops.descriptors import reject_string_selector

    # Reject a bare string at THIS boundary so the message names the FiniteVolume parameter
    # (``riemann`` / ``variables``), not the internal Spatial slot (``flux`` / ``recon``).
    if isinstance(riemann, str):
        reject_string_selector(riemann, "riemann", _FLUX_SUGGEST)
    if isinstance(variables, str):
        reject_string_selector(variables, "variables", _RECON_SUGGEST)
    # Lazy import: pops.numerics must not carry a module-scope pops.runtime edge (the acyclic
    # layering test). Spatial genuinely depends on the runtime route registry; the composite is
    # homed here as the authoring entry point and defers to it.
    from pops.runtime._bricks_scheme import Spatial

    return Spatial(limiter=limiter, flux=riemann, recon=variables, reconstruction=reconstruction,
                   none=none, minmod=minmod, vanleer=vanleer, weno5=weno5, primitive=primitive,
                   positivity_floor=positivity_floor, wave_speed_cache=wave_speed_cache)


__all__ = ["spatial", "FiniteVolume"]
