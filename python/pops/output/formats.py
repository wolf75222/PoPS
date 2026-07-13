"""Exact output formats backed by independently verifiable writers."""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import RequirementSet


class FormatInterface(Descriptor):
    """Typed scientific-format protocol; concrete descriptors select one exact writer."""

    category = "output_format"
    __pops_ir_immutable__ = True
    format_name = ""
    extension = ""

    def writer(self) -> Any:
        raise NotImplementedError("output format does not provide a writer")

    def consumer_data(self) -> dict[str, Any]:
        raise NotImplementedError("output format does not provide canonical consumer data")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("scientific output providers are immutable")


class HDF5(FormatInterface):
    """HDF5 output. ``parallel=True`` requests the parallel-HDF5 path (build-dependent)."""

    category = "output_format"
    format_name = "hdf5"
    extension = ".h5"

    def __init__(self, parallel: bool = False) -> None:
        if type(parallel) is not bool:
            raise TypeError("HDF5.parallel must be an exact bool")
        object.__setattr__(self, "parallel", parallel)

    def options(self) -> dict:
        return {"parallel": self.parallel}

    def requirements(self) -> Any:
        return RequirementSet({"parallel_io": True} if self.parallel else {})

    def writer(self) -> Any:
        from .writers import HDF5Writer
        return HDF5Writer()

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.hdf5.v1",
            "extension": self.extension,
            "parallel_mode": "collective" if self.parallel else "serial",
            "options": {"parallel": self.parallel},
        }


class NPZ(FormatInterface):
    """Compressed NumPy scientific container, always serial and independently verifiable."""

    format_name = "npz"
    extension = ".npz"

    def writer(self) -> Any:
        from .writers import NPZWriter
        return NPZWriter()

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.npz.v1",
            "extension": self.extension,
            "parallel_mode": "serial",
            "options": {},
        }


class ParaView(FormatInterface):
    """Single-file VTK UnstructuredGrid output read directly by ParaView."""

    format_name = "paraview-vtu"
    extension = ".vtu"

    def writer(self) -> Any:
        from .writers import ParaViewWriter
        return ParaViewWriter()

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.paraview-vtu.v1",
            "extension": self.extension,
            "parallel_mode": "serial",
            "options": {},
        }


__all__ = ["FormatInterface", "HDF5", "NPZ", "ParaView"]
