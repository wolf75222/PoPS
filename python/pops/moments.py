"""pops.moments : TRANSITIONAL re-export shim for the moment-model generator.

The generic 2D moment-model generator (the systematic binomial algebra for Vlasov/QMOM
moment hierarchies) moved to :mod:`pops.lib.moments` (Spec 4). This module is a thin
RE-EXPORT shim kept so the historical surface (``from pops.moments import
build_moment_model`` etc.) keeps working during the migration. It owns nothing.

The flat ``moments.py`` is DELETED in a later Spec-4 step (PR-E), once every consumer
migrates to ``pops.lib.moments``.
"""
from .lib.moments import (moment_indices, moment_names, gaussian_closure,  # noqa: F401
                          lorentz_sources, maxwellian_moments, bgk_source,
                          build_moment_model)

__all__ = [
    "moment_indices",
    "moment_names",
    "gaussian_closure",
    "lorentz_sources",
    "maxwellian_moments",
    "bgk_source",
    "build_moment_model",
]
