"""Exact accepted-side-effect authoring and scientific-output data protocols."""

from ._consumer_contracts import ConsumerGraph
from .consumers import Checkpoint, ScientificOutput
from .levels import AllLevels, CoarseOnly, LevelSelection, SelectedLevels
from .formats import ExternalWriter, FormatInterface, HDF5, NPZ, ParaView
from .data import (
    ArrayPiece, DiagnosticKey, DiagnosticPayload, FieldKey, FieldPayload, LevelGeometry, OutputClock,
    OutputProvenance, OutputRequest, OutputSnapshot,
)
from .diagnostics import BalanceTerms, composite_integrals
from .writers import (
    HDF5Writer, NPZWriter, OutputPublicationReceipt, ParaViewWriter,
    PreparedOutputFile, deterministic_target, read_hdf5, read_npz, read_paraview,
)
from . import formats

__all__ = [
    "Checkpoint", "ConsumerGraph", "ScientificOutput",
    "AllLevels", "CoarseOnly", "SelectedLevels",
    "LevelSelection",
    "FormatInterface", "ExternalWriter", "HDF5", "NPZ", "ParaView",
    "ArrayPiece", "DiagnosticKey", "DiagnosticPayload", "FieldKey", "FieldPayload",
    "LevelGeometry", "OutputClock",
    "OutputProvenance", "OutputRequest", "OutputSnapshot", "BalanceTerms",
    "composite_integrals", "HDF5Writer", "NPZWriter", "ParaViewWriter",
    "PreparedOutputFile", "OutputPublicationReceipt", "deterministic_target",
    "read_hdf5", "read_npz", "read_paraview", "formats",
]
