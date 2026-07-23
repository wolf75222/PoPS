"""Exact accepted-side-effect authoring and scientific-output data protocols."""

from ._consumer_contracts import (
    ConsumerGraph,
    FailRun,
    ParallelMode,
    Retry,
    SkipSampleReported,
)
from .consumers import Checkpoint, ConsoleMonitor, ScientificOutput
from .levels import AllLevels, CoarseOnly, LevelSelection, SelectedLevels
from .formats import (
    ExternalWriter, FormatInterface, HDF5, MpiRelayToRoot, NPZ, ParaView, ParaViewPreset,
    SharedDirectory,
)
from .paraview_state import (
    MaterializedPVSM,
    PortableState,
    materialize_paraview_state,
    read_portable_paraview_state,
)
from .data import (
    ArrayPiece, DiagnosticKey, DiagnosticPayload, FieldKey, FieldPayload, LevelGeometry, OutputClock,
    OutputProvenance, OutputRequest, OutputSnapshot,
)
from .diagnostics import BalanceTerms, composite_integrals
from .observers import (
    AsyncScientificOutput,
    Catalyst,
    LiveFailurePolicy,
    LiveVisualization,
    RaiseOnFlush,
    ReportOnly,
)
from ._durable_journal import DurableJournal
from ._writers.common import (
    FileSeriesCatalog, OutputPublicationReceipt, ReopenedOutput, ReopenedSeries,
    ScientificSeriesCatalog, ScientificWriter, SeriesSample, WriterSession,
    deterministic_target, output_series_path, writer_session_authority,
)
from ._writers.hdf5 import HDF5Writer, read_hdf5
from ._writers.npz import NPZWriter, read_npz
from ._writers.paraview import (
    ParaViewWriter,
    ReopenedParaViewIndex,
    read_paraview,
    read_paraview_parallel,
    read_paraview_series,
)
from . import formats

__all__ = [
    "AsyncScientificOutput", "Catalyst", "Checkpoint", "ConsoleMonitor", "ConsumerGraph", "FailRun",
    "DurableJournal", "LiveFailurePolicy", "LiveVisualization", "ParallelMode", "RaiseOnFlush",
    "ReportOnly", "Retry",
    "ScientificOutput", "SkipSampleReported",
    "AllLevels", "CoarseOnly", "SelectedLevels",
    "LevelSelection",
    "FormatInterface", "ExternalWriter", "HDF5", "NPZ", "ParaView", "ParaViewPreset",
    "MaterializedPVSM", "MpiRelayToRoot", "PortableState", "SharedDirectory",
    "materialize_paraview_state",
    "read_portable_paraview_state",
    "ArrayPiece", "DiagnosticKey", "DiagnosticPayload", "FieldKey", "FieldPayload",
    "LevelGeometry", "OutputClock",
    "OutputProvenance", "OutputRequest", "OutputSnapshot", "BalanceTerms",
    "composite_integrals", "HDF5Writer", "NPZWriter", "ParaViewWriter",
    "ReopenedParaViewIndex",
    "ScientificWriter", "WriterSession", "OutputPublicationReceipt",
    "ScientificSeriesCatalog", "FileSeriesCatalog",
    "ReopenedOutput", "ReopenedSeries", "SeriesSample",
    "deterministic_target", "output_series_path", "writer_session_authority",
    "read_hdf5", "read_npz", "read_paraview", "read_paraview_parallel",
    "read_paraview_series", "formats",
]
