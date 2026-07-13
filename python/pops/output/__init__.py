"""pops.output -- output / checkpoint policy descriptors (Spec 5 sec.5.14).

Typed replacements for ``output(format="hdf5", every=20)`` / ``checkpoint(mode=...)``:
:class:`OutputPolicy` / :class:`CheckpointPolicy` declare a typed format
(:mod:`pops.output.formats`), a cadence, the fields / diagnostics, and the level selection
(``AllLevels`` / ``CoarseOnly`` / ``SelectedLevels``). Inert descriptors; the runtime does
the I/O. The level policies are the canonical home shared with :mod:`pops.mesh.amr`.
"""
from .consumers import Checkpoint, ScientificOutput
from .levels import AllLevels, CoarseOnly, LevelSelection, SelectedLevels
from .policies import OutputPolicy, CheckpointPolicy
from .formats import FormatInterface, HDF5, NPZ, ParaView, Plotfile
from .data import (
    ArrayPiece, DiagnosticKey, DiagnosticPayload, FieldKey, FieldPayload, LevelGeometry, OutputClock,
    OutputProvenance, OutputRequest, OutputSnapshot,
)
from .diagnostics import BalanceTerms, composite_integrals
from .writers import (
    HDF5Writer, NPZWriter, OutputPublicationReceipt, ParaViewWriter,
    PreparedOutputFile, deterministic_target, read_hdf5, read_npz, read_paraview,
)
from .runtime_policies import RuntimePolicies, RuntimePoliciesReport
from . import policies, formats, runtime_policies

__all__ = [
    "Checkpoint", "ScientificOutput",
    "OutputPolicy", "CheckpointPolicy", "AllLevels", "CoarseOnly", "SelectedLevels",
    "LevelSelection",
    "FormatInterface", "HDF5", "NPZ", "ParaView", "Plotfile",
    "ArrayPiece", "DiagnosticKey", "DiagnosticPayload", "FieldKey", "FieldPayload",
    "LevelGeometry", "OutputClock",
    "OutputProvenance", "OutputRequest", "OutputSnapshot", "BalanceTerms",
    "composite_integrals", "HDF5Writer", "NPZWriter", "ParaViewWriter",
    "PreparedOutputFile", "OutputPublicationReceipt", "deterministic_target",
    "read_hdf5", "read_npz", "read_paraview",
    "RuntimePolicies", "RuntimePoliciesReport",
    "policies", "formats", "runtime_policies",
]
