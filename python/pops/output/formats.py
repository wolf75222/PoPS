"""Exact output formats backed by independently verifiable writers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import warnings

from pops.descriptors import Descriptor
from pops.descriptors_report import RequirementSet

from ._consumer_contracts import ParallelMode


_DEFAULT_PARAVIEW_STATE = object()
_UNSET_PARAVIEW_OPTION = object()


def _mode(value: Any, *, where: str, supported: frozenset[ParallelMode]) -> ParallelMode:
    if type(value) is not ParallelMode:
        raise TypeError("%s must be an exact pops.output.ParallelMode" % where)
    if value not in supported:
        raise ValueError(
            "%s does not support %s publication" % (where.rsplit(".", 1)[0], value.value))
    return value


def _series(value: Any, *, mode: ParallelMode, where: str) -> bool:
    if value is None:
        return mode is not ParallelMode.PER_RANK
    if type(value) is not bool:
        raise TypeError("%s must be an exact bool or None" % where)
    if value and mode is ParallelMode.PER_RANK:
        raise ValueError(
            "%s requires one shared artifact per sample; select SERIAL, ROOT, or COLLECTIVE"
            % where.rsplit(".", 1)[0])
    return value


class FormatInterface(Descriptor):
    """Typed scientific-format protocol; concrete descriptors select one exact writer."""

    category = "output_format"
    __pops_ir_immutable__ = True
    format_name = ""
    extension = ""
    series = False

    def writer(self) -> Any:
        raise NotImplementedError("output format does not provide a writer")

    def consumer_data(self) -> dict[str, Any]:
        raise NotImplementedError("output format does not provide canonical consumer data")

    def reopen(self, path: Any) -> Any:
        raise NotImplementedError("output format does not provide an authenticated reader")

    def reopen_series(self, path: Any) -> Any:
        raise NotImplementedError("output format does not provide a time-series catalogue")

    def series_catalog(self) -> Any:
        """Optional structural publication capability; formats without one return ``None``."""
        return None

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
            "format_name": self.format_name,
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
    series: bool

    def __init__(
        self,
        mode: ParallelMode = ParallelMode.SERIAL,
        *,
        series: Any = None,
    ) -> None:
        selected_mode = _mode(
            mode,
            where="HDF5.mode",
            supported=frozenset(ParallelMode),
        )
        object.__setattr__(self, "mode", selected_mode)
        object.__setattr__(self, "series", _series(
            series, mode=selected_mode, where="HDF5.series"))

    def options(self) -> dict:
        return {"mode": self.mode.value, "series": self.series}

    def requirements(self) -> Any:
        return RequirementSet(
            {"parallel_io": True}
            if self.mode is ParallelMode.COLLECTIVE else {})

    def writer(self) -> Any:
        from ._writers.hdf5 import HDF5Writer
        return HDF5Writer(self.mode)

    def reopen(self, path: Any) -> Any:
        from ._writers.hdf5 import read_hdf5
        return read_hdf5(path)

    def reopen_series(self, path: Any) -> Any:
        from ._writers.common import FileSeriesCatalog
        return FileSeriesCatalog(
            self.consumer_data(), format_name=self.format_name, reopen=self.reopen
        ).reopen(path)

    def series_catalog(self) -> Any:
        if not self.series:
            return None
        from ._writers.common import FileSeriesCatalog
        return FileSeriesCatalog(
            self.consumer_data(), format_name=self.format_name, reopen=self.reopen)

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.hdf5.v1",
            "format_name": self.format_name,
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "options": {"mode": self.mode.value, "series": self.series},
        }


class NPZ(FormatInterface):
    """Verifiable NumPy container for SERIAL, gathered ROOT, or rank-local publication."""

    format_name = "npz"
    extension = ".npz"
    mode: ParallelMode
    series: bool

    def __init__(
        self,
        mode: ParallelMode = ParallelMode.SERIAL,
        *,
        series: Any = None,
    ) -> None:
        selected_mode = _mode(
            mode,
            where="NPZ.mode",
            supported=frozenset({
                ParallelMode.SERIAL, ParallelMode.ROOT, ParallelMode.PER_RANK,
            }),
        )
        object.__setattr__(self, "mode", selected_mode)
        object.__setattr__(self, "series", _series(
            series, mode=selected_mode, where="NPZ.series"))

    def writer(self) -> Any:
        from ._writers.npz import NPZWriter
        return NPZWriter(self.mode)

    def reopen(self, path: Any) -> Any:
        from ._writers.npz import read_npz
        return read_npz(path)

    def reopen_series(self, path: Any) -> Any:
        from ._writers.common import FileSeriesCatalog
        return FileSeriesCatalog(
            self.consumer_data(), format_name=self.format_name, reopen=self.reopen
        ).reopen(path)

    def series_catalog(self) -> Any:
        if not self.series:
            return None
        from ._writers.common import FileSeriesCatalog
        return FileSeriesCatalog(
            self.consumer_data(), format_name=self.format_name, reopen=self.reopen)

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.npz.v1",
            "format_name": self.format_name,
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "options": {"mode": self.mode.value, "series": self.series},
        }


class ParaViewPreset:
    """Portable presentation intent used when an actual ParaView state is requested.

    The preset is deliberately smaller than ParaView's server-manager state.  PoPS records only
    stable user intent here, then asks the explicitly selected ``pvpython`` installation to create
    the version-specific ``.pvsm`` file.  ``color_by=None`` selects the first emitted scientific
    field after collision-safe display names have been resolved.
    """

    __slots__ = (
        "color_by", "component", "color_map", "representation", "show_scalar_bar",
    )
    __pops_ir_immutable__ = True

    _REPRESENTATIONS = frozenset({
        "Surface", "Surface With Edges", "Wireframe", "Points",
    })

    def __init__(
        self,
        *,
        color_by: str | None = None,
        component: str | None = None,
        color_map: str = "Viridis",
        representation: str = "Surface",
        show_scalar_bar: bool = True,
    ) -> None:
        for name, value in (("color_by", color_by), ("component", component)):
            if value is not None and (
                    not isinstance(value, str) or not value or value.strip() != value):
                raise TypeError("ParaViewPreset.%s must be canonical text or None" % name)
        if not isinstance(color_map, str) or not color_map or color_map.strip() != color_map:
            raise TypeError("ParaViewPreset.color_map must be canonical text")
        if representation not in self._REPRESENTATIONS:
            raise ValueError(
                "ParaViewPreset.representation must be one of %s"
                % sorted(self._REPRESENTATIONS))
        if type(show_scalar_bar) is not bool:
            raise TypeError("ParaViewPreset.show_scalar_bar must be an exact bool")
        object.__setattr__(self, "color_by", color_by)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "color_map", color_map)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "show_scalar_bar", show_scalar_bar)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("ParaViewPreset is immutable")

    def to_data(self) -> dict[str, Any]:
        return {
            "color_by": self.color_by,
            "component": self.component,
            "color_map": self.color_map,
            "representation": self.representation,
            "show_scalar_bar": self.show_scalar_bar,
        }


@dataclass(frozen=True, slots=True)
class SharedDirectory:
    """Publish rank-local VTU leaves directly into one reader-visible shared directory."""

    __pops_ir_immutable__: ClassVar[bool] = True

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "mode": "shared_directory"}


@dataclass(frozen=True, slots=True)
class MpiRelayToRoot:
    """Relay bounded VTU chunks to rank zero before publishing the PVTU bundle."""

    chunk_bytes: int = 8 * 1024 * 1024
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if isinstance(self.chunk_bytes, bool) or type(self.chunk_bytes) is not int \
                or self.chunk_bytes < 4096 or self.chunk_bytes > 256 * 1024 * 1024:
            raise ValueError(
                "MpiRelayToRoot.chunk_bytes must be an integer from 4096 to 268435456")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "mpi_relay_to_root",
            "chunk_bytes": self.chunk_bytes,
        }


class ParaView(FormatInterface):
    """Compact VTK output plus an authenticated temporal/parallel ParaView catalogue.

    ``compression`` is the standard VTK zlib level (``None`` disables compression).
    ``collection=True`` publishes one immutable cumulative ``.pvd`` per accepted sample.  A
    ``PER_RANK`` output additionally publishes the standard ``.pvtu`` index for that sample.
    A relocatable JSON/``pvpython`` recipe is emitted by default and requires no ParaView install
    during the simulation.  ``MaterializedPVSM`` additionally asks a real ``pvpython`` to create
    and reopen version-specific server-manager state; PoPS never fabricates PVSM XML.
    """

    format_name = "paraview-vtu"
    extension = ".vtu"
    mode: ParallelMode

    def __init__(
        self,
        mode: ParallelMode = ParallelMode.SERIAL,
        *,
        compression: int | None = 6,
        collection: Any = _UNSET_PARAVIEW_OPTION,
        preset: ParaViewPreset | None = None,
        placement: Any = None,
        state: Any = _DEFAULT_PARAVIEW_STATE,
        series: Any = _UNSET_PARAVIEW_OPTION,
    ) -> None:
        selected_mode = _mode(
            mode,
            where="ParaView.mode",
            supported=frozenset({
                ParallelMode.SERIAL, ParallelMode.ROOT, ParallelMode.PER_RANK,
            }),
        )
        object.__setattr__(self, "mode", selected_mode)
        if compression is not None and (
                isinstance(compression, bool) or type(compression) is not int
                or compression not in range(10)):
            raise ValueError("ParaView.compression must be None or an integer from 0 to 9")
        if collection is not _UNSET_PARAVIEW_OPTION and type(collection) is not bool:
            raise TypeError("ParaView.collection must be an exact bool")
        if series is not _UNSET_PARAVIEW_OPTION:
            if series is not None and type(series) is not bool:
                raise TypeError("ParaView.series must be an exact bool or None")
            warnings.warn(
                "ParaView(series=...) is deprecated; use collection=... for the standard "
                "PVD collection",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy_collection = (
                selected_mode is not ParallelMode.PER_RANK if series is None else series)
            if collection is not _UNSET_PARAVIEW_OPTION \
                    and collection is not legacy_collection:
                raise ValueError("ParaView.collection and deprecated series disagree")
            collection = legacy_collection
        if collection is _UNSET_PARAVIEW_OPTION:
            collection = True
        from .paraview_state import MaterializedPVSM, PortableState

        if state is _DEFAULT_PARAVIEW_STATE:
            resolved_state: Any = PortableState() if collection else None
        else:
            resolved_state = state
        if type(resolved_state) not in {type(None), PortableState, MaterializedPVSM}:
            raise TypeError(
                "ParaView.state must be PortableState(), MaterializedPVSM(), or None")
        if resolved_state is not None and not collection:
            raise ValueError("ParaView.state requires collection=True")
        resolved_preset = ParaViewPreset() if preset is None else preset
        if type(resolved_preset) is not ParaViewPreset:
            raise TypeError("ParaView.preset must be an exact ParaViewPreset")
        resolved_placement = (
            MpiRelayToRoot()
            if placement is None and self.mode is ParallelMode.PER_RANK
            else SharedDirectory() if placement is None else placement
        )
        if type(resolved_placement) not in {SharedDirectory, MpiRelayToRoot}:
            raise TypeError(
                "ParaView.placement must be SharedDirectory() or MpiRelayToRoot()")
        if type(resolved_placement) is MpiRelayToRoot \
                and self.mode is not ParallelMode.PER_RANK:
            raise ValueError("MpiRelayToRoot is meaningful only for ParaView PER_RANK output")
        object.__setattr__(self, "compression", compression)
        object.__setattr__(self, "collection", collection)
        object.__setattr__(self, "preset", resolved_preset)
        object.__setattr__(self, "placement", resolved_placement)
        object.__setattr__(self, "state", resolved_state)

    def writer(self) -> Any:
        from ._writers.paraview import ParaViewWriter
        return ParaViewWriter(
            self.mode,
            compression=self.compression,
            collection=self.collection,
            preset=self.preset,
            placement=self.placement,
            state=self.state,
        )

    def options(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "compression": self.compression,
            "collection": self.collection,
            "preset": self.preset.to_data(),
            "placement": self.placement.to_data(),
            "state": None if self.state is None else self.state.to_data(),
        }

    def reopen(self, path: Any) -> Any:
        from ._writers.paraview import read_paraview
        return read_paraview(path)

    def reopen_series(self, path: Any) -> Any:
        from ._writers.paraview import read_paraview_series
        return read_paraview_series(path)

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.paraview-vtu.v1",
            "format_name": self.format_name,
            "extension": self.extension,
            "parallel_mode": self.mode.value,
            "target_policy": (
                "immutable_sample" if self.collection else "literal"),
            "selection_contract": {
                "schema_version": 1,
                "layout_cardinality": "single",
            },
            "options": self.options(),
        }


__all__ = [
    "FormatInterface", "ExternalWriter", "HDF5", "NPZ", "ParaView", "ParaViewPreset",
    "MpiRelayToRoot", "SharedDirectory",
]
