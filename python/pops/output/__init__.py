"""pops.output -- output / checkpoint policy descriptors (Spec 5 sec.5.14).

Typed output/checkpoint descriptors:
:class:`OutputPolicy` / :class:`CheckpointPolicy` declare a typed format
(:mod:`pops.output.formats`), a cadence, the fields / diagnostics, and the level selection
(``AllLevels`` / ``CoarseOnly`` / ``SelectedLevels``). Inert descriptors; the runtime does
the I/O. The level policies are the canonical home shared with :mod:`pops.mesh.amr`.
"""
from .policies import (OutputPolicy, CheckpointPolicy,
                       AllLevels, CoarseOnly, SelectedLevels)
from .formats import HDF5, NPZ, VTK
from . import policies, formats

__all__ = [
    "OutputPolicy", "CheckpointPolicy", "AllLevels", "CoarseOnly", "SelectedLevels",
    "NPZ", "VTK", "HDF5", "policies", "formats",
]
