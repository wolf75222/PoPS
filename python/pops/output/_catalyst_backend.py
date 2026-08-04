"""Optional real Catalyst 2 Python backend for post-commit PoPS observer frames.

The dependency is imported lazily.  Production uses the installed ``catalyst`` and ``conduit``
modules; focused tests inject API-compatible modules without pretending that Catalyst is present.
Live visualization consumes the immutable rank-1/2/3 ``LevelGeometry`` and field-piece contract;
it accepts a serial frame or collective rank-local frames on an authenticated duplicated MPI
observer lane.
"""

from __future__ import annotations

import importlib
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pops._geometry_contracts import (
    CARTESIAN_1D_COORDINATES,
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_3D_COORDINATES,
    POLAR_ANNULUS_2D_COORDINATES,
)
from pops.output._consumer_contracts import ParallelMode
from pops.output.data import FieldPayload, LevelGeometry, _field_family_identity
from pops.output.observers import (
    ObserverFrame,
    ObserverReceipt,
    ObserverRun,
    ObserverWorkerCollectiveLost,
)
from pops.output._writers.paraview import (
    _cell_corner_offsets,
    _field_display_names,
    _field_families,
)
from pops.output._writers.common import validate_field_pieces


_BLUEPRINT_INDEX_AXES = ("i", "j", "k")
_BLUEPRINT_COORDINATE_AXES = ("x", "y", "z")
_BLUEPRINT_SPACING_AXES = ("dx", "dy", "dz")
_BLUEPRINT_CELL_SHAPES = {1: "line", 2: "quad", 3: "hex"}
_CARTESIAN_COORDINATES = {
    1: CARTESIAN_1D_COORDINATES,
    2: CARTESIAN_2D_COORDINATES,
    3: CARTESIAN_3D_COORDINATES,
}


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
            % box_index
        )
    return rows[0]


def _block_name(geometry: LevelGeometry) -> str:
    """Name one logical PDC block identically on every MPI rank."""

    return "layout_%s_level_%04d" % (geometry.layout_identity.hexdigest[:16], geometry.level)


def _box_bounds(
    geometry: LevelGeometry,
    box_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dimension = geometry.spatial_rank
    box = geometry.boxes[box_index]
    return box[:dimension], box[dimension:]


def _spatial_slices(
    lower: tuple[int, ...],
    upper: tuple[int, ...],
) -> tuple[slice, ...]:
    return tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))


def _structured_connectivity(cell_shape: tuple[int, ...]) -> Any:
    """Return VTK-wound line/quad/hex connectivity for one dense logical box."""

    import numpy as np

    dimension = len(cell_shape)
    point_shape = tuple(extent + 1 for extent in cell_shape)
    point_ids = np.arange(np.prod(point_shape), dtype=np.int64).reshape(point_shape)
    cell_indices = np.indices(cell_shape, dtype=np.int64)
    connectivity = np.stack(
        tuple(
            point_ids[tuple(
                cell_indices[axis] + corner[axis]
                for axis in range(dimension)
            )]
            for corner in _cell_corner_offsets(dimension)
        ),
        axis=-1,
    )
    return np.ascontiguousarray(connectivity).reshape(-1)


def _add_blueprint_topology(
    root: Any,
    base: str,
    geometry: LevelGeometry,
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    coordset: str,
    topology: str,
    *,
    empty: bool,
) -> None:
    """Materialize one exact Blueprint topology at the external Catalyst boundary."""

    import numpy as np

    dimension = geometry.spatial_rank
    expected_cartesian = _CARTESIAN_COORDINATES[dimension]
    coordset_base = base + "/coordsets/%s" % coordset
    topology_base = base + "/topologies/%s" % topology
    coordinate_lower = tuple(reversed(lower))
    coordinate_upper = tuple(reversed(upper))
    if geometry.coordinate_system == expected_cartesian:
        root[coordset_base + "/type"] = "uniform"
        for axis, (lo, hi, origin, spacing) in enumerate(zip(
            coordinate_lower,
            coordinate_upper,
            geometry.origin,
            geometry.spacing,
            strict=True,
        )):
            point_count = hi - lo + 1
            if empty:
                point_count = 0 if axis == 0 else 1
            root[
                coordset_base + "/dims/" + _BLUEPRINT_INDEX_AXES[axis]
            ] = point_count
            root[
                coordset_base + "/origin/" + _BLUEPRINT_COORDINATE_AXES[axis]
            ] = origin + lo * spacing
            root[
                coordset_base + "/spacing/" + _BLUEPRINT_SPACING_AXES[axis]
            ] = spacing
        root[topology_base + "/type"] = "uniform"
        root[topology_base + "/coordset"] = coordset
        return
    if geometry.coordinate_system == POLAR_ANNULUS_2D_COORDINATES:
        if dimension != 2:
            raise ValueError("polar-annulus coordinates require spatial rank two")
        root[coordset_base + "/type"] = "explicit"
        if empty:
            x = np.empty(0, dtype=np.float64)
            y = np.empty(0, dtype=np.float64)
            connectivity = np.empty(0, dtype=np.int64)
        else:
            logical = np.indices(
                tuple(hi - lo + 1 for lo, hi in zip(lower, upper, strict=True)),
                dtype=np.int64,
            )
            for axis, lo in enumerate(lower):
                logical[axis] += lo
            radius = geometry.origin[0] + logical[-1] * geometry.spacing[0]
            angle = geometry.origin[1] + logical[-2] * geometry.spacing[1]
            x = np.ascontiguousarray(radius * np.cos(angle)).reshape(-1)
            y = np.ascontiguousarray(radius * np.sin(angle)).reshape(-1)
            connectivity = _structured_connectivity(tuple(
                hi - lo for lo, hi in zip(lower, upper, strict=True)
            ))
        root[coordset_base + "/values/x"] = x
        root[coordset_base + "/values/y"] = y
        root[topology_base + "/type"] = "unstructured"
        root[topology_base + "/coordset"] = coordset
        root[topology_base + "/elements/shape"] = _BLUEPRINT_CELL_SHAPES[dimension]
        root[topology_base + "/elements/connectivity"] = connectivity
        return
    raise NotImplementedError(
        "Catalyst has no proved coordinate mapping for %s" % geometry.coordinate_system
    )


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
                "built against the selected ParaView installation"
            ) from error
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
                    "or an external conduit Python module"
                ) from errors[-1]
        if not callable(getattr(conduit, "Node", None)):
            raise RuntimeError("Conduit Python module does not expose Node")
        missing = [
            name
            for name in ("initialize", "execute", "finalize", "about")
            if not callable(getattr(catalyst, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Catalyst Python module does not expose callable lifecycle methods: %s"
                % ", ".join(missing)
            )
        return catalyst, conduit

    def open_session(
        self,
        configuration: Mapping[str, Any],
        execution_context: Any,
    ) -> _CatalystPythonSession:
        if (
            not isinstance(configuration, Mapping)
            or configuration.get("observer_kind") != "catalyst"
        ):
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
        if (
            not isinstance(implementation, str)
            or not implementation
            or implementation.strip() != implementation
        ):
            raise TypeError("Catalyst configuration requires a canonical implementation name")
        search_paths = configuration.get("search_paths")
        args = configuration.get("args")
        if not isinstance(search_paths, (tuple, list)) or any(
            not isinstance(value, str) or not value for value in search_paths
        ):
            raise TypeError("Catalyst configuration search_paths must be a list of strings")
        if not isinstance(args, (tuple, list)) or any(
            not isinstance(value, str) or not value for value in args
        ):
            raise TypeError("Catalyst configuration args must be a list of strings")
        inherited_async = os.environ.get("CATALYST_ASYNC_ENABLED")
        if inherited_async is not None and inherited_async.strip().lower() not in {
            "",
            "0",
            "false",
            "off",
            "no",
        }:
            raise RuntimeError(
                "PoPS owns the post-commit worker and requires Catalyst internal async to be "
                "disabled; unset CATALYST_ASYNC_ENABLED or set it to 0"
            )
        prefer_environment = os.environ.get("CATALYST_IMPLEMENTATION_PREFER_ENV")
        if prefer_environment:
            raise RuntimeError(
                "PoPS authenticates catalyst_load/implementation and rejects "
                "CATALYST_IMPLEMENTATION_PREFER_ENV; unset it instead of overriding the "
                "declaration"
            )
        communicator = getattr(execution_context, "communicator", None)
        communicator_id = getattr(communicator, "identity", None)
        worker_communicator = configuration.get("_pops_worker_communicator")
        if communicator_id == "serial" and worker_communicator is None:
            pass
        elif communicator_id == "MPI_COMM_WORLD" and worker_communicator is not None:
            from pops._native_collectives import require_communicator, require_world

            world = require_world(getattr(communicator, "handle", None))
            lane = require_communicator(worker_communicator, allow_world=False)
            if int(world.rank) != int(lane.rank) or int(world.size) != int(lane.size):
                raise ValueError("Catalyst worker lane topology differs from MPI_COMM_WORLD")
        else:
            raise ValueError(
                "Catalyst requires either serial execution or an exact duplicated "
                "MPI_COMM_WORLD observer lane"
            )
        catalyst, conduit = self._modules()
        return _CatalystPythonSession(
            catalyst,
            conduit,
            path,
            self._channel,
            pipeline_sha256=expected_digest,
            implementation=implementation,
            search_paths=tuple(search_paths),
            args=tuple(args),
            worker_communicator=worker_communicator,
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
        worker_communicator: Any = None,
    ) -> None:
        self._catalyst = catalyst
        self._conduit = conduit
        self._pipeline = pipeline
        self._pipeline_sha256 = pipeline_sha256
        self._channel = channel
        self._implementation = implementation
        self._search_paths = search_paths
        self._args = args
        self._worker_communicator = worker_communicator
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
        worker_mpi = self._worker_communicator is not None
        return {
            "schema_version": 1,
            "provider_id": "pops.output.catalyst-python.v1",
            "delivery": "post_commit",
            "threading": "dedicated_collective" if worker_mpi else "dedicated_serial",
            "worker_mpi": worker_mpi,
        }

    def _node(self) -> Any:
        return self._conduit.Node()

    def _agree_local_phase(self, phase: str, error: BaseException | None) -> None:
        if self._worker_communicator is None:
            if error is not None:
                raise error
            return
        from pops._native_collectives import allgather_value, rank, size

        rendered = None if error is None else "%s: %s" % (type(error).__name__, error)
        try:
            owner = rank(self._worker_communicator)
            peers = size(self._worker_communicator)
            rows = allgather_value(
                self._worker_communicator,
                {
                    "rank": owner,
                    "error": rendered,
                },
            )
        except BaseException as collective_error:
            raise ObserverWorkerCollectiveLost(
                "Catalyst %s lost its worker collective: %s: %s"
                % (phase, type(collective_error).__name__, collective_error)
            ) from collective_error
        if (
            not isinstance(rows, (tuple, list))
            or len(rows) != peers
            or any(
                not isinstance(row, dict)
                or set(row) != {"rank", "error"}
                or row["rank"] != owner
                or (row["error"] is not None and not isinstance(row["error"], str))
                for owner, row in enumerate(rows)
            )
        ):
            raise ObserverWorkerCollectiveLost(
                "Catalyst %s returned malformed worker-lane evidence" % phase
            )
        failures = [
            "rank %d: %s" % (owner, row["error"])
            for owner, row in enumerate(rows)
            if row["error"] is not None
        ]
        if failures:
            collective = RuntimeError(
                "Catalyst %s failed collectively: %s" % (phase, "; ".join(failures))
            )
            if error is not None:
                raise collective from error
            raise collective

    def _agree_exact_value(self, phase: str, value: Mapping[str, Any]) -> None:
        """Reject rank-divergent Catalyst authority before entering its collectives."""

        if self._worker_communicator is None:
            return
        from pops._native_collectives import allgather_value, rank, size

        try:
            owner = rank(self._worker_communicator)
            peers = size(self._worker_communicator)
            rows = allgather_value(
                self._worker_communicator,
                {
                    "rank": owner,
                    "value": dict(value),
                },
            )
        except BaseException as collective_error:
            raise ObserverWorkerCollectiveLost(
                "Catalyst %s lost its worker collective: %s: %s"
                % (phase, type(collective_error).__name__, collective_error)
            ) from collective_error
        if (
            not isinstance(rows, (tuple, list))
            or len(rows) != peers
            or any(
                not isinstance(row, dict)
                or set(row) != {"rank", "value"}
                or row["rank"] != owner
                or not isinstance(row["value"], dict)
                for owner, row in enumerate(rows)
            )
        ):
            raise ObserverWorkerCollectiveLost(
                "Catalyst %s returned malformed worker-lane evidence" % phase
            )
        canonical = rows[0]["value"]
        divergent = [owner for owner, row in enumerate(rows) if row["value"] != canonical]
        if divergent:
            raise RuntimeError(
                "Catalyst %s differs across ranks: %s"
                % (phase, ", ".join(str(owner) for owner in divergent))
            )

    def initialize(self, run: ObserverRun) -> None:
        node = None
        local_error = None
        try:
            if self._initialized or self._finalized:
                raise RuntimeError("Catalyst observer session cannot be initialized twice")
            if hashlib.sha256(self._pipeline.read_bytes()).hexdigest() != self._pipeline_sha256:
                raise RuntimeError(
                    "Catalyst pipeline changed between session authentication and initialize"
                )
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
            if self._worker_communicator is not None:
                node["catalyst/mpi_comm"] = int(self._worker_communicator.fortran_handle)
            node["catalyst/pops/run_identity"] = run.run_identity.token
            for index, identity in enumerate(run.recovery_run_identities):
                node["catalyst/pops/recovery_run_identities/%06d" % index] = identity.token
        except BaseException as error:
            local_error = error
        self._agree_local_phase("initialize", local_error)
        if node is None:  # collective agreement cannot clear a local construction failure
            raise RuntimeError("Catalyst initialize lost its local node authority")
        self._agree_exact_value(
            "initialize authority",
            {
                "args": list(self._args),
                "channel": self._channel,
                "implementation": self._implementation,
                "pipeline_sha256": self._pipeline_sha256,
                "recovery_run_identities": [
                    identity.token for identity in run.recovery_run_identities
                ],
                "run_identity": run.run_identity.token,
                "search_paths": list(self._search_paths),
            },
        )
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
                    % (reported, self._implementation)
                )
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
        self._agree_exact_value("implementation evidence", implementation_evidence)
        self._implementation_evidence = implementation_evidence
        self._accepted_run_identities = frozenset(run.accepted_run_identities)
        self._initialized = True

    @staticmethod
    def _geometry_fields(
        frame: ObserverFrame,
        geometry: LevelGeometry,
    ) -> tuple[FieldPayload, ...]:
        selected = frame.snapshot.select(frame.request)
        fields = tuple(
            field
            for field in selected
            if (field.key.layout_identity.token, field.key.level) == geometry.key
        )
        if not fields:
            raise ValueError("Catalyst selected geometry has no field payload")
        if any(field.centering != "cell" for field in fields):
            raise NotImplementedError(
                "Catalyst Python provider currently proves cell-centered fields only"
            )
        return fields

    def _add_domain(
        self,
        root: Any,
        frame: ObserverFrame,
        geometry: LevelGeometry,
        layout_ordinal: int,
        box_index: int,
        partition_index: int,
        domain_id: int,
        domain_name: str,
        display_names: Mapping[str, str],
    ) -> None:
        import numpy as np

        lower, upper = _box_bounds(geometry, box_index)
        base = "catalyst/channels/%s/data/%s" % (self._channel, domain_name)
        coordset = "coords_%06d" % partition_index
        topology = "mesh_%06d" % partition_index
        _add_blueprint_topology(
            root,
            base,
            geometry,
            lower,
            upper,
            coordset,
            topology,
            empty=False,
        )
        root[base + "/state/level"] = geometry.level
        root[base + "/state/level_id"] = geometry.level
        root[base + "/state/domain_id"] = domain_id
        root[base + "/state/cycle"] = frame.macro_step
        root[base + "/state/time"] = frame.physical_time

        field_slot = 0

        def cell_field(
            field_name: str,
            values: Any,
            component_names: tuple[str, ...] = (),
        ) -> str:
            nonlocal field_slot
            internal_name = "array_%06d_partition_%06d" % (field_slot, partition_index)
            field_slot += 1
            prefix = base + "/fields/" + internal_name
            root[prefix + "/association"] = "element"
            root[prefix + "/topology"] = topology
            root[prefix + "/display_name"] = field_name
            if len(component_names) > 1:
                for index, component in enumerate(component_names):
                    root[prefix + "/values/" + component] = np.ascontiguousarray(
                        values[index]
                    ).reshape(-1)
            else:
                root[prefix + "/values"] = np.ascontiguousarray(values).reshape(-1)
            return internal_name

        spatial = _spatial_slices(lower, upper)
        coverage = geometry.coverage[spatial].astype(np.uint8, copy=False)
        cell_field(
            "pops_layout",
            np.full(coverage.shape, layout_ordinal, dtype=np.int32),
        )
        cell_field(
            "pops_level",
            np.full(coverage.shape, geometry.level, dtype=np.int32),
        )
        cell_field("pops_coverage", coverage)
        # VTK_REFINED_CELL=8; this hides covered coarse cells in ParaView without deleting their
        # scientific values from the live Blueprint domain.
        ghost_field = cell_field("vtkGhostType", coverage * np.uint8(8))
        root[base + "/state/metadata/vtk_fields/%s/attribute_type" % ghost_field] = "Ghosts"
        cell_field("pops_cell_volume", geometry.cell_volumes[spatial])

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

    def _add_empty_domain(
        self,
        root: Any,
        frame: ObserverFrame,
        geometry: LevelGeometry,
        box_index: int,
        domain_id: int,
        domain_name: str,
        display_names: Mapping[str, str],
    ) -> None:
        """Publish one schema-complete zero-cell block for a box owned by another MPI rank."""
        import numpy as np

        lower, upper = _box_bounds(geometry, box_index)
        base = "catalyst/channels/%s/data/%s" % (self._channel, domain_name)
        coordset = "coords_%06d" % box_index
        topology = "mesh_%06d" % box_index
        _add_blueprint_topology(
            root,
            base,
            geometry,
            lower,
            upper,
            coordset,
            topology,
            empty=True,
        )
        root[base + "/state/level"] = geometry.level
        root[base + "/state/level_id"] = geometry.level
        root[base + "/state/domain_id"] = domain_id
        root[base + "/state/cycle"] = frame.macro_step
        root[base + "/state/time"] = frame.physical_time

        field_slot = 0

        def empty_cell_field(
            field_name: str,
            dtype: Any,
            component_names: tuple[str, ...] = (),
        ) -> str:
            nonlocal field_slot
            internal_name = "array_%06d_partition_%06d" % (field_slot, box_index)
            field_slot += 1
            prefix = base + "/fields/" + internal_name
            root[prefix + "/association"] = "element"
            root[prefix + "/topology"] = topology
            root[prefix + "/display_name"] = field_name
            if len(component_names) > 1:
                for component in component_names:
                    root[prefix + "/values/" + component] = np.empty(0, dtype=dtype)
            else:
                root[prefix + "/values"] = np.empty(0, dtype=dtype)
            return internal_name

        empty_cell_field("pops_layout", np.int32)
        empty_cell_field("pops_level", np.int32)
        empty_cell_field("pops_coverage", np.uint8)
        ghost_field = empty_cell_field("vtkGhostType", np.uint8)
        root[base + "/state/metadata/vtk_fields/%s/attribute_type" % ghost_field] = "Ghosts"
        empty_cell_field("pops_cell_volume", np.float64)

        names: set[str] = set()
        for field in self._geometry_fields(frame, geometry):
            family = _field_family_identity(field.key).token
            field_name = display_names.get(family)
            if field_name is None:
                raise RuntimeError("Catalyst field family has no shared ParaView display name")
            if field_name in names:
                raise ValueError("Catalyst field name collision: %s" % field_name)
            names.add(field_name)
            empty_cell_field(
                field_name,
                np.dtype(field.dtype),
                field.component_names,
            )

    def _prepare_execute_node(self, frame: ObserverFrame) -> Any:
        if not self._initialized or self._finalized:
            raise RuntimeError("Catalyst observer session is not active")
        if self._execution_failed:
            raise RuntimeError("Catalyst observer session is poisoned after an execute failure")
        if frame.snapshot.provenance.run_identity not in self._accepted_run_identities:
            raise ValueError("Catalyst frame is outside the active/recovery run authority")
        if self._worker_communicator is None:
            if (
                frame.request.parallel_mode is not ParallelMode.SERIAL
                or frame.request.rank != 0
                or frame.request.size != 1
            ):
                raise ValueError("SERIAL Catalyst received a distributed frame")
        else:
            from pops._native_collectives import rank, size

            if (
                frame.request.parallel_mode is not ParallelMode.COLLECTIVE
                or frame.request.rank != rank(self._worker_communicator)
                or frame.request.size != size(self._worker_communicator)
            ):
                raise ValueError("COLLECTIVE Catalyst requires its exact worker MPI lane topology")
        node = self._node()
        node["catalyst/state/timestep"] = frame.macro_step
        node["catalyst/state/time"] = frame.physical_time
        node["catalyst/channels/%s/type" % self._channel] = "multimesh"
        # An MPI rank may legitimately own no selected box.  Real Conduit ``fetch`` materializes
        # an empty object without assigning an unsupported Python dict value.
        fetch = getattr(node, "fetch", None)
        if callable(fetch):
            fetch("catalyst/channels/%s/data" % self._channel)
        selected_fields = frame.snapshot.select(frame.request)
        for field in selected_fields:
            validate_field_pieces(
                field,
                frame.snapshot.geometry(field.key),
                complete=frame.request.parallel_mode is ParallelMode.SERIAL,
                rank=frame.request.rank,
                size=frame.request.size,
            )
        families = _field_families(selected_fields)
        names = _field_display_names(families)
        display_names = {
            family: name for name, (family, _members) in zip(names, families, strict=True)
        }
        geometry_keys = sorted(
            {(field.key.layout_identity.token, field.key.level) for field in selected_fields}
        )
        geometries = [
            geometry for geometry in frame.snapshot.geometries if geometry.key in geometry_keys
        ]
        if not geometries:
            raise ValueError("Catalyst frame has no selected geometry")
        dimensions = {geometry.spatial_rank for geometry in geometries}
        if len(dimensions) != 1:
            raise ValueError("one Catalyst channel requires one common spatial rank")
        populated_blocks = []
        domain_id = 0
        for layout_ordinal, geometry in enumerate(geometries):
            block_name = _block_name(geometry)
            fields = self._geometry_fields(frame, geometry)
            local_boxes = {piece.global_box_index for field in fields for piece in field.pieces}
            if any(
                {piece.global_box_index for piece in field.pieces} != local_boxes
                for field in fields
            ):
                raise ValueError("Catalyst fields disagree on the local geometry-box ownership set")
            for box_index in range(len(geometry.boxes)):
                # ParaView's multimesh protocol defines every data child as one Blueprint mesh.
                # A global AMR box is that indivisible block; its qualified name stays unique and
                # stable across ranks; non-owning ranks publish its schema-complete zero-cell peer.
                domain_name = "%s_box_%06d" % (block_name, box_index)
                populated_blocks.append(domain_name)
                if box_index in local_boxes:
                    self._add_domain(
                        node,
                        frame,
                        geometry,
                        layout_ordinal,
                        box_index,
                        box_index,
                        domain_id,
                        domain_name,
                        display_names,
                    )
                else:
                    self._add_empty_domain(
                        node,
                        frame,
                        geometry,
                        box_index,
                        domain_id,
                        domain_name,
                        display_names,
                    )
                domain_id += 1

        blueprint = getattr(self._conduit, "blueprint", None)
        mesh = getattr(blueprint, "mesh", None)
        verify = getattr(mesh, "verify", None)
        if callable(verify):
            for block_name in populated_blocks:
                info = self._node()
                domain = node["catalyst/channels/%s/data/%s" % (self._channel, block_name)]
                if verify(domain, info) is not True:
                    raise ValueError(
                        "Catalyst Conduit Blueprint verification failed for block %s: %s"
                        % (block_name, info)
                    )
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
        backend_error = None
        try:
            _call(self._catalyst, "finalize", node)
        except BaseException as error:
            backend_error = error
        self._agree_local_phase("finalize backend", backend_error)
        self._finalized = True
        return None

    def abort(self) -> None:
        if self._finalized:
            return None
        if self._finalize_attempted:
            raise RuntimeError("Catalyst observer abort cannot retry failed finalization")
        if self._initialize_entered:
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
            backend_error = None
            try:
                _call(self._catalyst, "finalize", node)
            except BaseException as error:
                backend_error = error
            self._agree_local_phase("abort backend", backend_error)
            self._finalized = True
        return None


__all__ = ["CatalystPythonProvider"]
