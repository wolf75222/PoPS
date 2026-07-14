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


class _InstalledWriterOnly:
    """Structural marker: the native writer is resolved by RuntimeInstance."""

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "an ExternalWriter can publish only through its authenticated RuntimeInstance")


class ExternalWriter(FormatInterface):
    """Select one qualified native Writer component for this scientific output.

    The format carries the component id *and* its immutable manifest identity.  It never
    searches a process-global registry or selects the only installed writer implicitly.
    """

    format_name = "external-writer"

    def __init__(self, component: Any, *, extension: str) -> None:
        from pops.external import CompiledComponentArtifact, ExternalComponent
        from pops import interfaces

        if type(component) is ExternalComponent:
            component_id = component.component_manifest.component_id
            manifest_identity = component.component_manifest.manifest_digest
            interface = component.component_type.interface
        elif type(component) is CompiledComponentArtifact:
            component_id = component.component_id
            manifest_identity = component.component_manifest
            interface = component.interface
        else:
            raise TypeError(
                "ExternalWriter component must be an exact ExternalComponent or "
                "CompiledComponentArtifact")
        if interface != interfaces.Writer:
            raise TypeError("ExternalWriter component must implement the exact Writer interface")
        if not isinstance(extension, str) or not extension.startswith(".") \
                or extension.strip() != extension or "/" in extension or "\\" in extension:
            raise TypeError("ExternalWriter extension must be a canonical file suffix")
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "component_manifest_identity", manifest_identity)
        object.__setattr__(self, "native_interface", interface)
        object.__setattr__(self, "extension", extension)

    def writer(self) -> Any:
        return _InstalledWriterOnly()

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.external-writer.v1",
            "extension": self.extension,
            "parallel_mode": "serial",
            "options": {},
            "component_id": self.component_id,
            "component_manifest_identity": self.component_manifest_identity.token,
            "native_interface": self.native_interface.to_data(),
        }


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
        from ._writers.hdf5 import HDF5Writer
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
        from ._writers.npz import NPZWriter
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
        from ._writers.paraview import ParaViewWriter
        return ParaViewWriter()

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.paraview-vtu.v1",
            "extension": self.extension,
            "parallel_mode": "serial",
            "options": {},
        }


__all__ = ["FormatInterface", "ExternalWriter", "HDF5", "NPZ", "ParaView"]
