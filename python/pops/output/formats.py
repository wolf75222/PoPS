"""Exact output formats backed by independently verifiable writers."""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import RequirementSet

from ._consumer_contracts import ParallelMode


def _mode(value: Any, *, where: str, supported: frozenset[ParallelMode]) -> ParallelMode:
    if type(value) is not ParallelMode:
        raise TypeError("%s must be an exact pops.output.ParallelMode" % where)
    if value not in supported:
        raise ValueError(
            "%s does not support %s publication" % (where.rsplit(".", 1)[0], value.value))
    return value


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

    def __init__(self, mode: ParallelMode, requirement: dict[str, Any]) -> None:
        self._mode = mode
        self._requirement = dict(requirement)

    def preflight(self, execution_context: Any) -> dict[str, Any]:
        from ._writers.common import writer_execution_capability

        return writer_execution_capability(
            execution_context,
            self._mode,
            provider_id="pops.output.external-writer.v1",
        )

    def prepare_session(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "an ExternalWriter can publish only through its authenticated RuntimeInstance")

    def installed_component_requirement(self) -> dict[str, Any]:
        """Return the exact native Writer authority consumed during runtime installation."""
        return dict(self._requirement)


class ExternalWriter(FormatInterface):
    """Select one qualified native Writer component for this scientific output.

    The format carries the component id *and* its immutable manifest identity.  It never
    searches a process-global registry or selects the only installed writer implicitly.  Writer
    ABI v1 consumes one complete snapshot, either directly in SERIAL mode or after the runtime has
    gathered that snapshot to rank zero in ROOT mode.
    """

    format_name = "external-writer"
    component_id: str
    component_manifest_identity: Any
    native_interface: Any
    mode: ParallelMode

    def __init__(
        self,
        component: Any,
        *,
        extension: str,
        mode: ParallelMode = ParallelMode.SERIAL,
    ) -> None:
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
        object.__setattr__(self, "mode", _mode(
            mode,
            where="ExternalWriter.mode",
            supported=frozenset({ParallelMode.SERIAL, ParallelMode.ROOT}),
        ))

    def writer(self) -> Any:
        return _InstalledWriterOnly(self.mode, {
            "component_id": self.component_id,
            "component_manifest_identity": self.component_manifest_identity.token,
            "native_interface": self.native_interface.to_data(),
        })

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.external-writer.v1",
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "options": {},
            "component_id": self.component_id,
            "component_manifest_identity": self.component_manifest_identity.token,
            "native_interface": self.native_interface.to_data(),
        }


class HDF5(FormatInterface):
    """HDF5 output with one explicit publication topology."""

    category = "output_format"
    format_name = "hdf5"
    extension = ".h5"
    mode: ParallelMode

    def __init__(self, mode: ParallelMode = ParallelMode.SERIAL) -> None:
        object.__setattr__(self, "mode", _mode(
            mode,
            where="HDF5.mode",
            supported=frozenset(ParallelMode),
        ))

    def options(self) -> dict:
        return {"mode": self.mode.value}

    def requirements(self) -> Any:
        return RequirementSet(
            {"parallel_io": True}
            if self.mode is ParallelMode.COLLECTIVE else {})

    def writer(self) -> Any:
        from ._writers.hdf5 import HDF5Writer
        return HDF5Writer(self.mode)

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.hdf5.v1",
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "options": {"mode": self.mode.value},
        }


class NPZ(FormatInterface):
    """Verifiable NumPy container for SERIAL, gathered ROOT, or rank-local publication."""

    format_name = "npz"
    extension = ".npz"
    mode: ParallelMode

    def __init__(self, mode: ParallelMode = ParallelMode.SERIAL) -> None:
        object.__setattr__(self, "mode", _mode(
            mode,
            where="NPZ.mode",
            supported=frozenset({
                ParallelMode.SERIAL, ParallelMode.ROOT, ParallelMode.PER_RANK,
            }),
        ))

    def writer(self) -> Any:
        from ._writers.npz import NPZWriter
        return NPZWriter(self.mode)

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.npz.v1",
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "options": {"mode": self.mode.value},
        }


class ParaView(FormatInterface):
    """VTK UnstructuredGrid output, shared or explicitly rank-qualified for ParaView."""

    format_name = "paraview-vtu"
    extension = ".vtu"
    mode: ParallelMode

    def __init__(self, mode: ParallelMode = ParallelMode.SERIAL) -> None:
        object.__setattr__(self, "mode", _mode(
            mode,
            where="ParaView.mode",
            supported=frozenset({
                ParallelMode.SERIAL, ParallelMode.ROOT, ParallelMode.PER_RANK,
            }),
        ))

    def writer(self) -> Any:
        from ._writers.paraview import ParaViewWriter
        return ParaViewWriter(self.mode)

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.paraview-vtu.v1",
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "selection_contract": {
                "schema_version": 1,
                "layout_cardinality": "single",
            },
            "options": {"mode": self.mode.value},
        }


__all__ = ["FormatInterface", "ExternalWriter", "HDF5", "NPZ", "ParaView"]
