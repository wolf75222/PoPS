"""Optional real Catalyst 2 Python backend for post-commit PoPS observer frames.

The dependency is imported lazily.  Production uses the installed ``catalyst`` and ``conduit``
modules; focused tests inject API-compatible modules without pretending that Catalyst is present.
The native runtime currently supplies rank-2 cell-centered fields.  Live visualization is a
single-rank contract and never passes an MPI communicator to Catalyst.
"""
from __future__ import annotations

import importlib
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pops.output._consumer_contracts import ParallelMode
from pops.output.data import FieldPayload, LevelGeometry, _field_family_identity
from pops.output.observers import ObserverFrame, ObserverReceipt, ObserverRun
from pops.output._writers.paraview import _field_display_names, _field_families
from pops.mesh._layout_plan_contracts import (
    CARTESIAN_2D_COORDINATES,
    POLAR_ANNULUS_2D_COORDINATES,
)


def _module_version(module: Any) -> str:
    value = getattr(module, "__version__", "unknown")
    return value if isinstance(value, str) and value else "unknown"


def _call(api: Any, operation: str, node: Any) -> None:
    callback = getattr(api, operation, None)
    if not callable(callback):
        raise RuntimeError("Catalyst Python module does not expose %s()" % operation)
    result = callback(node)
    if result not in (None, 0):
        raise RuntimeError("Catalyst %s() returned failure code %r" % (operation, result))


def _piece_for_box(field: FieldPayload, box_index: int) -> Any:
    rows = [piece for piece in field.pieces if piece.global_box_index == box_index]
    if len(rows) != 1:
        raise ValueError(
            "Catalyst complete snapshot requires exactly one field piece for geometry box %d"
            % box_index)
    return rows[0]


def _domain_name(geometry: LevelGeometry, box_index: int) -> str:
    return "layout_%s_level_%04d_box_%06d" % (
        geometry.layout_identity.hexdigest[:16], geometry.level, box_index)


class CatalystPythonProvider:
    """Structural provider that owns the optional real Catalyst/Conduit Python modules."""

    def __init__(
        self,
        *,
        channel: str = "mesh",
        catalyst_module: Any = None,
        conduit_module: Any = None,
    ) -> None:
        if not isinstance(channel, str) or not channel or channel.strip() != channel:
            raise TypeError("Catalyst channel must be non-empty canonical text")
        self._channel = channel
        self._catalyst_module = catalyst_module
        self._conduit_module = conduit_module

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.catalyst-python.v1",
            "observer_kind": "catalyst",
            "channel": self._channel,
            "api": "Catalyst2/Conduit-Blueprint",
            "conduit_import_order": ["catalyst_conduit", "conduit"],
        }

    def _modules(self) -> tuple[Any, Any]:
        catalyst = self._catalyst_module
        conduit = self._conduit_module
        try:
            if catalyst is None:
                catalyst = importlib.import_module("catalyst")
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "Catalyst live visualization requires the optional catalyst Python module "
                "built against the selected ParaView installation") from error
        if conduit is None:
            errors = []
            for module_name in ("catalyst_conduit", "conduit"):
                try:
                    conduit = importlib.import_module(module_name)
                    break
                except (ImportError, ModuleNotFoundError) as error:
                    errors.append(error)
            if conduit is None:
                raise RuntimeError(
                    "Catalyst live visualization requires catalyst_conduit (ParaView builds) "
                    "or an external conduit Python module") from errors[-1]
        if not callable(getattr(conduit, "Node", None)):
            raise RuntimeError("Conduit Python module does not expose Node")
        missing = [
            name for name in ("initialize", "execute", "finalize", "about")
            if not callable(getattr(catalyst, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Catalyst Python module does not expose callable lifecycle methods: %s"
                % ", ".join(missing))
        return catalyst, conduit

    def open_session(
        self, configuration: Mapping[str, Any], execution_context: Any,
    ) -> _CatalystPythonSession:
        if not isinstance(configuration, Mapping) \
                or configuration.get("observer_kind") != "catalyst":
            raise TypeError("Catalyst provider received an invalid observer configuration")
        pipeline = configuration.get("pipeline")
        if not isinstance(pipeline, str) or not pipeline:
            raise TypeError("Catalyst configuration requires a pipeline path")
        path = Path(pipeline).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Catalyst pipeline does not exist: %s" % path)
        expected_digest = configuration.get("pipeline_sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise TypeError("Catalyst configuration requires a SHA-256 pipeline identity")
        current_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_digest != expected_digest:
            raise RuntimeError("Catalyst pipeline changed after its declaration was authenticated")
        implementation = configuration.get("implementation")
        if not isinstance(implementation, str) or not implementation \
                or implementation.strip() != implementation:
            raise TypeError("Catalyst configuration requires a canonical implementation name")
        search_paths = configuration.get("search_paths")
        args = configuration.get("args")
        if not isinstance(search_paths, (tuple, list)) \
                or any(not isinstance(value, str) or not value for value in search_paths):
            raise TypeError("Catalyst configuration search_paths must be a list of strings")
        if not isinstance(args, (tuple, list)) \
                or any(not isinstance(value, str) or not value for value in args):
            raise TypeError("Catalyst configuration args must be a list of strings")
        inherited_async = os.environ.get("CATALYST_ASYNC_ENABLED")
        if inherited_async is not None \
                and inherited_async.strip().lower() not in {"", "0", "false", "off", "no"}:
            raise RuntimeError(
                "PoPS owns the post-commit worker and requires Catalyst internal async to be "
                "disabled; unset CATALYST_ASYNC_ENABLED or set it to 0")
        prefer_environment = os.environ.get("CATALYST_IMPLEMENTATION_PREFER_ENV")
        if prefer_environment:
            raise RuntimeError(
                "PoPS authenticates catalyst_load/implementation and rejects "
                "CATALYST_IMPLEMENTATION_PREFER_ENV; unset it instead of overriding the "
                "declaration")
        communicator = getattr(execution_context, "communicator", None)
        communicator_id = getattr(communicator, "identity", None)
        worker_communicator = configuration.get("_pops_worker_communicator")
        if communicator_id != "serial" or worker_communicator is not None:
            raise ValueError(
                "Catalyst live visualization currently supports serial execution only")
        catalyst, conduit = self._modules()
        return _CatalystPythonSession(
            catalyst, conduit, path, self._channel,
            pipeline_sha256=expected_digest,
            implementation=implementation,
            search_paths=tuple(search_paths),
            args=tuple(args),
        )


class _CatalystPythonSession:
    def __init__(
        self,
        catalyst: Any,
        conduit: Any,
        pipeline: Path,
        channel: str,
        *,
        pipeline_sha256: str,
        implementation: str,
        search_paths: tuple[str, ...],
        args: tuple[str, ...],
    ) -> None:
        self._catalyst = catalyst
        self._conduit = conduit
        self._pipeline = pipeline
        self._pipeline_sha256 = pipeline_sha256
        self._channel = channel
        self._implementation = implementation
        self._search_paths = search_paths
        self._args = args
        self._conduit_module = getattr(conduit, "__name__", type(conduit).__name__)
        self._conduit_version = _module_version(conduit)
        self._initialized = False
        self._initialize_entered = False
        self._finalize_attempted = False
        self._finalized = False
        self._execution_failed = False
        self._accepted_run_identities: frozenset[Any] = frozenset()
        self._implementation_evidence: dict[str, str] | None = None

    @property
    def authority(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.catalyst-python.v1",
            "delivery": "post_commit",
            "threading": "dedicated_serial",
            "worker_mpi": False,
        }

    def _node(self) -> Any:
        return self._conduit.Node()

    def _agree_local_phase(self, phase: str, error: BaseException | None) -> None:
        del phase
        if error is not None:
            raise error

    def initialize(self, run: ObserverRun) -> None:
        node = None
        local_error = None
        try:
            if self._initialized or self._finalized:
                raise RuntimeError("Catalyst observer session cannot be initialized twice")
            if hashlib.sha256(self._pipeline.read_bytes()).hexdigest() \
                    != self._pipeline_sha256:
                raise RuntimeError(
                    "Catalyst pipeline changed between session authentication and initialize")
            node = self._node()
            node["catalyst_load/implementation"] = self._implementation
            if self._search_paths:
                node["catalyst_load/search_paths"] = list(self._search_paths)
            node["catalyst/scripts/pops/filename"] = self._pipeline.as_posix()
            if self._args:
                node["catalyst/scripts/pops/args"] = list(self._args)
            # PoPS already owns a bounded worker.  A second Catalyst worker would acknowledge
            # enqueue rather than completed processing, so the initialize parameter overrides
            # Catalyst's environment default.
            node["catalyst/async/enabled"] = 0
            node["catalyst/pops/run_identity"] = run.run_identity.token
            for index, identity in enumerate(run.recovery_run_identities):
                node[
                    "catalyst/pops/recovery_run_identities/%06d" % index
                ] = identity.token
        except BaseException as error:
            local_error = error
        self._agree_local_phase("initialize", local_error)
        if node is None:  # collective agreement cannot clear a local construction failure
            raise RuntimeError("Catalyst initialize lost its local node authority")
        # Catalyst may allocate process-global state and then raise.  Mark entry before the call so
        # the queue's partial-initialize abort can still invoke finalize exactly once.
        self._initialize_entered = True
        initialize_error = None
        try:
            _call(self._catalyst, "initialize", node)
        except BaseException as error:
            initialize_error = error
        self._agree_local_phase("initialize backend", initialize_error)
        implementation_evidence = None
        about_error = None
        try:
            about = self._node()
            _call(self._catalyst, "about", about)
            reported = about["catalyst/implementation"]
            version = about["catalyst/version"]
            if reported != self._implementation:
                raise RuntimeError(
                    "Catalyst loaded implementation %r instead of requested %r"
                    % (reported, self._implementation))
            if not isinstance(version, str) or not version:
                raise RuntimeError("Catalyst about() returned no implementation version")
            implementation_evidence = {
                "implementation": reported,
                "catalyst_api_version": version,
                "catalyst_module": _module_version(self._catalyst),
                "conduit_module": self._conduit_module,
                "conduit_version": self._conduit_version,
            }
        except BaseException as error:
            about_error = error
        self._agree_local_phase("implementation authentication", about_error)
        if implementation_evidence is None:
            raise RuntimeError("Catalyst implementation authentication lost its evidence")
        self._implementation_evidence = implementation_evidence
        self._accepted_run_identities = frozenset(run.accepted_run_identities)
        self._initialized = True

    @staticmethod
    def _geometry_fields(
        frame: ObserverFrame, geometry: LevelGeometry,
    ) -> tuple[FieldPayload, ...]:
        selected = frame.snapshot.select(frame.request)
        fields = tuple(
            field for field in selected
            if (field.key.layout_identity.token, field.key.level) == geometry.key)
        if not fields:
            raise ValueError("Catalyst selected geometry has no field payload")
        if any(field.centering != "cell" for field in fields):
            raise NotImplementedError(
                "Catalyst Python provider currently proves cell-centered fields only")
        return fields

    def _add_domain(
        self,
        root: Any,
        frame: ObserverFrame,
        geometry: LevelGeometry,
        box_index: int,
        domain_id: int,
        display_names: Mapping[str, str],
    ) -> None:
        import numpy as np

        if len(geometry.cell_shape) != 2:
            raise NotImplementedError("Catalyst Python provider currently proves rank-2 meshes")
        jlo, ilo, jhi, ihi = geometry.boxes[box_index]
        name = _domain_name(geometry, box_index)
        base = "catalyst/channels/%s/data/%s" % (self._channel, name)
        if geometry.coordinate_system == CARTESIAN_2D_COORDINATES:
            root[base + "/coordsets/coords/type"] = "uniform"
            root[base + "/coordsets/coords/dims/i"] = ihi - ilo + 1
            root[base + "/coordsets/coords/dims/j"] = jhi - jlo + 1
            root[base + "/coordsets/coords/origin/x"] = (
                geometry.origin[0] + ilo * geometry.spacing[0])
            root[base + "/coordsets/coords/origin/y"] = (
                geometry.origin[1] + jlo * geometry.spacing[1])
            root[base + "/coordsets/coords/spacing/dx"] = geometry.spacing[0]
            root[base + "/coordsets/coords/spacing/dy"] = geometry.spacing[1]
            root[base + "/topologies/mesh/type"] = "uniform"
            root[base + "/topologies/mesh/coordset"] = "coords"
        elif geometry.coordinate_system == POLAR_ANNULUS_2D_COORDINATES:
            radial = geometry.origin[0] + np.arange(
                ilo, ihi + 1, dtype=np.float64) * geometry.spacing[0]
            theta = geometry.origin[1] + np.arange(
                jlo, jhi + 1, dtype=np.float64) * geometry.spacing[1]
            theta_grid, radial_grid = np.meshgrid(theta, radial, indexing="ij")
            root[base + "/coordsets/coords/type"] = "explicit"
            root[base + "/coordsets/coords/values/x"] = np.ascontiguousarray(
                radial_grid * np.cos(theta_grid)).reshape(-1)
            root[base + "/coordsets/coords/values/y"] = np.ascontiguousarray(
                radial_grid * np.sin(theta_grid)).reshape(-1)
            ni = ihi - ilo
            nj = jhi - jlo
            lower_left = np.arange(nj * ni, dtype=np.int64).reshape(nj, ni)
            lower_left += np.arange(nj, dtype=np.int64)[:, None]
            connectivity = np.stack((
                lower_left,
                lower_left + 1,
                lower_left + ni + 2,
                lower_left + ni + 1,
            ), axis=-1)
            root[base + "/topologies/mesh/type"] = "unstructured"
            root[base + "/topologies/mesh/coordset"] = "coords"
            root[base + "/topologies/mesh/elements/shape"] = "quad"
            root[base + "/topologies/mesh/elements/connectivity"] = \
                np.ascontiguousarray(connectivity).reshape(-1)
        else:
            raise NotImplementedError(
                "Catalyst has no proved coordinate mapping for %s"
                % geometry.coordinate_system)
        root[base + "/state/domain_id"] = domain_id
        root[base + "/state/level"] = geometry.level
        root[base + "/state/cycle"] = frame.macro_step
        root[base + "/state/time"] = frame.physical_time

        def cell_field(
            field_name: str,
            values: Any,
            component_names: tuple[str, ...] = (),
        ) -> None:
            prefix = base + "/fields/" + field_name
            root[prefix + "/association"] = "element"
            root[prefix + "/topology"] = "mesh"
            if len(component_names) > 1:
                for index, component in enumerate(component_names):
                    root[prefix + "/values/" + component] = np.ascontiguousarray(
                        values[index]).reshape(-1)
            else:
                root[prefix + "/values"] = np.ascontiguousarray(values).reshape(-1)

        coverage = geometry.coverage[jlo:jhi, ilo:ihi].astype(np.uint8, copy=False)
        cell_field("pops_coverage", coverage)
        # VTK_REFINED_CELL=8; this hides covered coarse cells in ParaView without deleting their
        # scientific values from the live Blueprint domain.
        cell_field("vtkGhostType", coverage * np.uint8(8))
        root[
            base
            + "/state/metadata/vtk_fields/vtkGhostType/attribute_type"
        ] = "Ghosts"
        cell_field("pops_cell_volume", geometry.cell_volumes[jlo:jhi, ilo:ihi])

        names: set[str] = set()
        for field in self._geometry_fields(frame, geometry):
            piece = _piece_for_box(field, box_index)
            family = _field_family_identity(field.key).token
            field_name = display_names.get(family)
            if field_name is None:
                raise RuntimeError("Catalyst field family has no shared ParaView display name")
            if field_name in names:
                raise ValueError("Catalyst field name collision: %s" % field_name)
            names.add(field_name)
            if len(field.component_names) > 1:
                cell_field(field_name, piece.values, field.component_names)
            elif field.component_names:
                cell_field(field_name, piece.values[0])
            else:
                cell_field(field_name, piece.values)

    def _prepare_execute_node(self, frame: ObserverFrame) -> Any:
        if not self._initialized or self._finalized:
            raise RuntimeError("Catalyst observer session is not active")
        if self._execution_failed:
            raise RuntimeError("Catalyst observer session is poisoned after an execute failure")
        if frame.snapshot.provenance.run_identity not in self._accepted_run_identities:
            raise ValueError(
                "Catalyst frame is outside the active/recovery run authority")
        if frame.request.parallel_mode is not ParallelMode.SERIAL \
                or frame.request.rank != 0 or frame.request.size != 1:
            raise ValueError("SERIAL Catalyst received a distributed frame")
        node = self._node()
        node["catalyst/state/timestep"] = frame.macro_step
        node["catalyst/state/time"] = frame.physical_time
        node["catalyst/channels/%s/type" % self._channel] = "multimesh"
        selected_fields = frame.snapshot.select(frame.request)
        families = _field_families(selected_fields)
        names = _field_display_names(families)
        display_names = {
            family: name
            for name, (family, _members) in zip(names, families, strict=True)
        }
        geometry_keys = sorted({
            (field.key.layout_identity.token, field.key.level)
            for field in selected_fields
        })
        geometries = [
            geometry for geometry in frame.snapshot.geometries
            if geometry.key in geometry_keys
        ]
        if not geometries:
            raise ValueError("Catalyst frame has no selected geometry")
        geometry_offsets = {}
        next_domain_id = 0
        for geometry in sorted(
                frame.snapshot.geometries,
                key=lambda value: (value.layout_identity.token, value.level)):
            geometry_offsets[geometry.key] = next_domain_id
            next_domain_id += len(geometry.boxes)
        domain_names = []
        for geometry in geometries:
            fields = self._geometry_fields(frame, geometry)
            local_boxes = {
                piece.global_box_index for field in fields for piece in field.pieces
            }
            if any(
                    {piece.global_box_index for piece in field.pieces} != local_boxes
                    for field in fields):
                raise ValueError(
                    "Catalyst fields disagree on the local geometry-box ownership set")
            for box_index in sorted(local_boxes):
                if box_index < 0 or box_index >= len(geometry.boxes):
                    raise ValueError("Catalyst field references an unknown global geometry box")
                domain_names.append(_domain_name(geometry, box_index))
                self._add_domain(
                    node, frame, geometry, box_index,
                    geometry_offsets[geometry.key] + box_index,
                    display_names)

        blueprint = getattr(self._conduit, "blueprint", None)
        mesh = getattr(blueprint, "mesh", None)
        verify = getattr(mesh, "verify", None)
        if callable(verify):
            for domain_name in domain_names:
                info = self._node()
                domain = node[
                    "catalyst/channels/%s/data/%s" % (self._channel, domain_name)]
                if verify(domain, info) is not True:
                    raise ValueError(
                        "Catalyst Conduit Blueprint verification failed for domain %s: %s"
                        % (domain_name, info))
        return node

    def execute(self, frame: ObserverFrame) -> ObserverReceipt:
        node = None
        local_error = None
        try:
            node = self._prepare_execute_node(frame)
        except BaseException as error:
            local_error = error
        self._agree_local_phase("execute", local_error)
        if node is None:  # collective agreement cannot clear a local construction failure
            raise RuntimeError("Catalyst execute lost its local Blueprint node")
        backend_error = None
        try:
            _call(self._catalyst, "execute", node)
        except BaseException as error:
            backend_error = error
        try:
            self._agree_local_phase("execute backend", backend_error)
        except BaseException:
            self._execution_failed = True
            raise
        evidence = self._implementation_evidence
        if evidence is None:
            raise RuntimeError("Catalyst execute lost its implementation evidence")
        return ObserverReceipt(
            frame.identity,
            "pops.output.catalyst-python.v1",
            {
                "channel": self._channel,
                "catalyst_version": _module_version(self._catalyst),
                "conduit_module": self._conduit_module,
                "conduit_version": self._conduit_version,
                "implementation": evidence["implementation"],
                "catalyst_api_version": evidence["catalyst_api_version"],
                "macro_step": frame.macro_step,
            },
        )

    def finalize(self) -> None:
        node = None
        local_error = None
        try:
            if self._finalized:
                return None
            if not self._initialized:
                raise RuntimeError("uninitialized Catalyst observer cannot be finalized")
            if self._finalize_attempted:
                raise RuntimeError("Catalyst observer finalization already failed")
            node = self._node()
        except BaseException as error:
            local_error = error
        self._agree_local_phase("finalize", local_error)
        if node is None:
            raise RuntimeError("Catalyst finalize lost its local node authority")
        self._finalize_attempted = True
        _call(self._catalyst, "finalize", node)
        self._finalized = True
        return None

    def abort(self) -> None:
        if self._initialize_entered and not self._finalized \
                and not self._finalize_attempted:
            node = None
            local_error = None
            try:
                node = self._node()
            except BaseException as error:
                local_error = error
            self._agree_local_phase("abort", local_error)
            if node is None:
                raise RuntimeError("Catalyst abort lost its local node authority")
            self._finalize_attempted = True
            _call(self._catalyst, "finalize", node)
            self._finalized = True
        return None


__all__ = ["CatalystPythonProvider"]
