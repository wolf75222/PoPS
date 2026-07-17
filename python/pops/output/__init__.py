"""Exact accepted-side-effect authoring and scientific-output data protocols."""

from ._consumer_contracts import ConsumerGraph, ParallelMode
from .consumers import Checkpoint, ScientificOutput
from .levels import AllLevels, CoarseOnly, LevelSelection, SelectedLevels
from .formats import ExternalWriter, FormatInterface, HDF5, NPZ, ParaView
from .data import (
    ArrayPiece, DiagnosticKey, DiagnosticPayload, FieldKey, FieldPayload, LevelGeometry, OutputClock,
    OutputProvenance, OutputRequest, OutputSnapshot,
)
from .diagnostics import BalanceTerms, composite_integrals
from ._writers.common import (
    OutputPublicationReceipt, ScientificWriter, WriterSession,
    deterministic_target, writer_session_authority,
)
from ._writers.hdf5 import HDF5Writer, read_hdf5
from ._writers.npz import NPZWriter, read_npz
from ._writers.paraview import ParaViewWriter, read_paraview
from . import formats

__all__ = [
    "Checkpoint", "ConsumerGraph", "ParallelMode", "ScientificOutput",
    "AllLevels", "CoarseOnly", "SelectedLevels",
    "LevelSelection",
    "FormatInterface", "ExternalWriter", "HDF5", "NPZ", "ParaView",
    "ArrayPiece", "DiagnosticKey", "DiagnosticPayload", "FieldKey", "FieldPayload",
    "LevelGeometry", "OutputClock",
    "OutputProvenance", "OutputRequest", "OutputSnapshot", "BalanceTerms",
    "composite_integrals", "HDF5Writer", "NPZWriter", "ParaViewWriter",
    "ScientificWriter", "WriterSession", "OutputPublicationReceipt",
    "deterministic_target", "writer_session_authority",
    "read_hdf5", "read_npz", "read_paraview", "formats",
]
