"""pops.output.formats -- typed output-format descriptors (Spec 5 sec.5.14).

A format is a typed object (``HDF5()`` / ``Plotfile()``), not a string ``format="hdf5"``.
Inert; the runtime writes the actual files.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet


class FormatInterface(Descriptor):
    """Typed scientific-format protocol; concrete descriptors select one exact writer."""

    category = "output_format"
    format_name = ""
    extension = ""

    def writer(self) -> Any:
        raise NotImplementedError("output format does not provide a writer")


class HDF5(FormatInterface):
    """HDF5 output. ``parallel=True`` requests the parallel-HDF5 path (build-dependent)."""

    category = "output_format"
    format_name = "hdf5"
    extension = ".h5"

    def __init__(self, parallel: bool = False) -> None:
        self.parallel = bool(parallel)

    def options(self) -> dict:
        return {"parallel": self.parallel}

    def requirements(self) -> Any:
        return RequirementSet({"parallel_io": True} if self.parallel else {})

    def writer(self) -> Any:
        from .writers import HDF5Writer
        return HDF5Writer()


class NPZ(FormatInterface):
    """Compressed NumPy scientific container, always serial and independently verifiable."""

    format_name = "npz"
    extension = ".npz"

    def writer(self) -> Any:
        from .writers import NPZWriter
        return NPZWriter()


class ParaView(FormatInterface):
    """Single-file VTK UnstructuredGrid output read directly by ParaView."""

    format_name = "paraview-vtu"
    extension = ".vtu"

    def writer(self) -> Any:
        from .writers import ParaViewWriter
        return ParaViewWriter()


class Plotfile(Descriptor):
    """AMReX-style plotfile output (per-level directories)."""

    category = "output_format"

    def capabilities(self) -> Any:
        return CapabilitySet({"per_level": True})


__all__ = ["FormatInterface", "HDF5", "NPZ", "ParaView", "Plotfile"]
