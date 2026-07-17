"""pops.runtime.amr -- the AMR runtime inspection surface (Spec 5 sec.8.12 / sec.8.4).

The runtime-bound counterpart of the inert layout report returned by ``pops.inspect(layout)``.
Where authoring inspection reports a *layout descriptor* (the declared level / ratio / regrid /
refine envelope, before any runtime), this package reports a *live*
:class:`pops.runtime._amr_system.AmrSystem`: the patches that actually exist on the built
hierarchy, the regrid cadence in force, and the ghost / reflux / checkpoint route limitations.

The surface is INERT: every method READS the already-built runtime (the box accessors
``patch_rectangles`` / ``coarse_local_boxes`` / ``coarse_total_boxes`` and the static config the
``AmrSystem`` retained) and the descriptor metadata; it RUNS nothing, ALLOCATES nothing, and never
steps the clock. A measure that the current native build cannot answer (composite multi-level
Poisson, a per-level ghost depth not exposed by C++) is DECLARED unavailable, never fabricated.

Layering: this module sits in the ``runtime`` layer, the only one allowed to reach ``_pops``
(through the bound :class:`AmrSystem`). The report value classes themselves
(:class:`PatchReport`, :class:`RegridReport`, :class:`HierarchySnapshot`,
:class:`RuntimeInspection`, and the ``explain_*`` reports) are plain inert data -- they hold
pre-read numbers and strings and import nothing. Only :class:`AmrRuntimeView` is bound to the
live system. ``sim.amr.inspect()`` returns the unified :class:`RuntimeInspection` (ADC-589/555):
hierarchy + patches + regrid + capability limitations in one call.
"""

from pops.runtime.amr._reports import (
    PatchReport,
    RegridReport,
    GhostReport,
    RefluxReport,
    CheckpointReport,
    HierarchySnapshot,
    RuntimeInspection,
)
from pops.runtime.amr._view import AmrRuntimeView

__all__ = [
    "AmrRuntimeView",
    "PatchReport",
    "RegridReport",
    "GhostReport",
    "RefluxReport",
    "CheckpointReport",
    "HierarchySnapshot",
    "RuntimeInspection",
]
