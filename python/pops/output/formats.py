"""Exact output formats backed by independently verifiable writers."""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import RequirementSet


class FormatInterface(Descriptor):
    """Typed scientific-format protocol; concrete descriptors select one exact writer."""

    category = "output_format"
    format_name = ""
    extension = ""

    def writer(self) -> Any:
        raise NotImplementedError("output format does not provide a writer")

    def consumer_data(self) -> dict[str, Any]:
        """Exact inert writer selection consumed by policy-to-graph authoring."""
        if not isinstance(self.format_name, str) or not self.format_name:
            raise ValueError("output format must declare a non-empty format_name")
        return {
            "descriptor": "%s.%s" % (type(self).__module__, type(self).__qualname__),
            "format_name": self.format_name,
            "extension": self.extension,
            "options": self.options(),
            "requirements": self.requirements().to_dict(),
        }


class HDF5(FormatInterface):
    """HDF5 output. ``parallel=True`` requests the parallel-HDF5 path (build-dependent)."""

    category = "output_format"
    format_name = "hdf5"
    extension = ".h5"

    def __init__(self, parallel: bool = False) -> None:
        if type(parallel) is not bool:
            raise TypeError("HDF5.parallel must be an exact bool")
        self.parallel = parallel

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


__all__ = ["FormatInterface", "HDF5", "NPZ", "ParaView"]
