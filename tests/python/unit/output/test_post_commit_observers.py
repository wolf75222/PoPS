"""Isolated contract tests for bounded post-commit observers and optional Catalyst 2."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pops._geometry_contracts import POLAR_ANNULUS_2D_COORDINATES
from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output._catalyst_backend import CatalystPythonProvider
from pops.output._consumer_contracts import ParallelMode
from pops.output.data import (
    ArrayPiece,
    FieldKey,
    FieldPayload,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    _NATIVE_GEOMETRY_ARRAYS,
)
from pops.output.observers import (
    Catalyst,
    LiveVisualization,
    ObserverFrame,
    ObserverReceipt,
    ObserverRun,
    detach_observer_frame,
)
from pops.time import AcceptedStep, Clock, Every, Schedule
from pops.runtime._observer_runtime import (
    ObserverDeliveryReport,
    PostCommitObserverQueue,
    PostCommitObserverWorker,
)
import pops.runtime._observer_runtime as observer_runtime
import pops.runtime._runtime_consumers as runtime_consumers


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def test_observer_report_has_an_exact_byte_free_collective_projection():
    from pops._native_collectives import decode_value, encode_value

    frame = _frame()
    receipt = ObserverReceipt(frame.identity, "test.observer", {
        "opaque": b"\x00\xff",
        "nested": {"values": [b"abc", 7, True, None]},
    })
    report = ObserverDeliveryReport(
        "test-consumer",
        frame.snapshot.provenance.run_identity,
        0,
        frame.identity,
        "delivered",
        1,
        receipt=receipt,
    )

    transported = decode_value(encode_value(report.to_collective_data()))
    reopened = ObserverDeliveryReport.from_collective_data(transported)

    assert reopened == report
    assert reopened.to_data() == report.to_data()


def _frame(
    *, mode: ParallelMode = ParallelMode.SERIAL, centering: str = "cell",
    native_geometry_arrays=None,
    coordinate_system: str = "pops://coordinates/cartesian-2d@1",
    origin=(0.0, 0.0),
    spacing=(0.5, 0.25),
    field_name="temperature",
    component_names=("temperature",),
):
    layout = _identity("layout-plan", "uniform")
    component = _identity("component-manifest", "heat")
    owner = OwnerPath.case("case").child(OwnerKind.BLOCK, "heat")
    reference = Handle(field_name, kind="state", owner=owner)
    key = FieldKey(reference, component, layout, 0, "accepted")
    coverage = np.asarray([[False, True], [False, False]])
    volumes = np.full((2, 2), 0.125)
    geometry_kwargs = {}
    if native_geometry_arrays is not None:
        valid, coverage, volumes = native_geometry_arrays
        geometry_kwargs = {
            "_native_valid_cells": valid,
            "_native_arrays": _NATIVE_GEOMETRY_ARRAYS,
        }
    geometry = LevelGeometry(
        layout,
        "uniform",
        0,
        origin,
        spacing,
        (2, 2),
        ((0, 0, 2, 2),),
        coverage,
        volumes,
        coordinate_system=coordinate_system,
        **geometry_kwargs,
    )
    spatial_shape = (2, 2) if centering == "cell" else (3, 3)
    piece = ArrayPiece(
        (0, 0),
        spatial_shape,
        np.arange(1, 1 + spatial_shape[0] * spatial_shape[1], dtype=np.float64).reshape(
            (1,) + spatial_shape),
        0,
        0,
        False,
    )
    field = FieldPayload(
        key, centering, "K", component_names, spatial_shape, (piece,))
    snapshot = OutputSnapshot(
        OutputClock.at("macro", 0.25, 4, stage="accepted"),
        OutputProvenance(
            _identity("resolved-plan", "plan"),
            _identity("bind", "bind"),
            _identity("run", "run"),
            "accepted-step-transaction",
        ),
        (geometry,),
        (field,),
        {"test": "post-commit-observer"},
    )
    request = OutputRequest(
        "live-temperature", (key,), mode, rank=0, size=(1 if mode is ParallelMode.SERIAL else 2))
    return ObserverFrame(snapshot, request)


def _without_local_pieces(base: ObserverFrame) -> ObserverFrame:
    field = base.snapshot.fields[0]
    empty_field = FieldPayload(
        field.key,
        field.centering,
        field.units,
        field.component_names,
        field.global_shape,
        (),
        dtype=field.array_dtype,
    )
    snapshot = OutputSnapshot(
        base.snapshot.clock,
        base.snapshot.provenance,
        base.snapshot.geometries,
        (empty_field,),
        dict(base.snapshot.metadata),
    )
    return ObserverFrame(snapshot, base.request)


class _Node:
    def __init__(self, values=None, prefix=""):
        self.values = {} if values is None else values
        self.prefix = prefix

    def _path(self, key):
        return key if not self.prefix else self.prefix + "/" + key

    def __setitem__(self, key, value):
        self.values[self._path(key)] = value

    def __getitem__(self, key):
        path = self._path(key)
        if path in self.values:
            return self.values[path]
        if any(candidate.startswith(path + "/") for candidate in self.values):
            return _Node(self.values, path)
        raise KeyError(path)

    def __str__(self):
        return repr(self.values)


class _FetchTrackingNode(_Node):
    def __init__(self, values=None, prefix="", fetched=None):
        super().__init__(values, prefix)
        self.fetched = [] if fetched is None else fetched

    def fetch(self, key):
        path = self._path(key)
        self.fetched.append(path)
        return _FetchTrackingNode(self.values, path, self.fetched)


class _CatalystModule:
    __version__ = "test-catalyst"

    def __init__(self):
        self.operations = []

    def initialize(self, node):
        self.operations.append(("initialize", node))

    def execute(self, node):
        self.operations.append(("execute", node))

    def finalize(self, node):
        self.operations.append(("finalize", node))

    def about(self, node):
        node["catalyst/implementation"] = "paraview"
        node["catalyst/version"] = "2.0-test"


class _StubReportingCatalyst(_CatalystModule):
    def about(self, node):
        node["catalyst/implementation"] = "stub"
        node["catalyst/version"] = "2.0-test"


class _PartiallyFailingInitializeCatalyst(_CatalystModule):
    def initialize(self, node):
        self.operations.append(("initialize", node))
        raise RuntimeError("Catalyst allocated state then failed")


class _BlueprintMesh:
    def __init__(self):
        self.verified_domains = []

    def verify(self, domain, _info):
        self.verified_domains.append(domain.prefix)
        paths = domain.values
        return all(any(
            candidate.startswith(domain.prefix + suffix)
            for candidate in paths
        ) for suffix in ("/coordsets/", "/topologies/", "/fields/"))


class _ConduitModule:
    __version__ = "test-conduit"
    Node = _Node

    def __init__(self):
        self.__name__ = "test_catalyst_conduit"
        self.blueprint = SimpleNamespace(mesh=_BlueprintMesh())


class _FetchTrackingConduitModule(_ConduitModule):
    Node = _FetchTrackingNode


def _serial_context():
    return SimpleNamespace(communicator=SimpleNamespace(identity="serial"))


def _collective_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int = 2,
    lane_size: int = 2,
    peer_error: str | None = None,
    divergent_initialize_authority: bool = False,
):
    import pops._native_collectives as native_collectives

    world = SimpleNamespace(
        identity="MPI_COMM_WORLD", active=True, rank=0, size=world_size)
    lane = SimpleNamespace(
        identity="MPI_COMM_WORLD/observer/catalyst-test",
        active=True,
        rank=0,
        size=lane_size,
        fortran_handle=73,
    )
    agreements = []

    def require_world(communicator):
        assert communicator is world
        return communicator

    def require_duplicate(communicator, *, allow_world=True):
        assert communicator is lane
        assert allow_world is False
        return communicator

    def allgather_value(communicator, value):
        assert communicator is lane
        agreements.append(dict(value))
        rows = [dict(value, rank=owner) for owner in range(lane.size)]
        if peer_error is not None and len(rows) > 1 and "error" in rows[1]:
            rows[1]["error"] = peer_error
        if divergent_initialize_authority and len(rows) > 1 \
                and isinstance(rows[1].get("value"), dict) \
                and "pipeline_sha256" in rows[1]["value"]:
            rows[1]["value"] = dict(
                rows[1]["value"], pipeline_sha256="0" * 64)
        return tuple(rows)

    monkeypatch.setattr(native_collectives, "require_world", require_world)
    monkeypatch.setattr(native_collectives, "require_communicator", require_duplicate)
    monkeypatch.setattr(native_collectives, "rank", lambda communicator: communicator.rank)
    monkeypatch.setattr(native_collectives, "size", lambda communicator: communicator.size)
    monkeypatch.setattr(native_collectives, "allgather_value", allgather_value)
    return (
        SimpleNamespace(communicator=SimpleNamespace(
            identity="MPI_COMM_WORLD", handle=world)),
        lane,
        agreements,
    )


@pytest.mark.parametrize(
    "mode",
    (ParallelMode.ROOT, ParallelMode.PER_RANK),
)
def test_live_visualization_rejects_noncollective_mpi_modes(mode: ParallelMode):
    class _Provider:
        def consumer_data(self):
            return {
                "schema_version": 1,
                "provider_id": "test.serial-live",
                "observer_kind": "test",
            }

        def open_session(self, _execution_context):
            raise AssertionError("rejected live declarations must not open a session")

    frame = _frame()
    with pytest.raises(ValueError, match="supports only SERIAL or COLLECTIVE mode"):
        LiveVisualization(
            observer=_Provider(),
            schedule=Schedule(Every(AcceptedStep(Clock("serial-live")), 1)),
            fields=(frame.snapshot.fields[0].key.reference,),
            mode=mode,
        )


def test_live_visualization_accepts_collective_catalyst_and_rejects_retry(
    tmp_path: Path,
):
    pipeline = tmp_path / "collective_declaration.py"
    pipeline.write_text("# collective Catalyst declaration\n")
    frame = _frame()
    observer = Catalyst(pipeline=str(pipeline))
    schedule = Schedule(Every(AcceptedStep(Clock("collective-live")), 1))

    declaration = LiveVisualization(
        observer=observer,
        schedule=schedule,
        fields=(frame.snapshot.fields[0].key.reference,),
        mode=ParallelMode.COLLECTIVE,
    )

    assert declaration.mode is ParallelMode.COLLECTIVE
    assert declaration.options()["mode"] == "collective"
    authoring = declaration.consumer_authoring()
    assert len(authoring) == 1
    assert authoring[0].parallel_mode is ParallelMode.COLLECTIVE
    assert authoring[0].operation.consumer_data()["parallel_mode"] == "collective"

    with pytest.raises(ValueError, match="requires max_attempts=1"):
        LiveVisualization(
            observer=observer,
            schedule=schedule,
            fields=(frame.snapshot.fields[0].key.reference,),
            mode=ParallelMode.COLLECTIVE,
            max_attempts=2,
        )


def test_optional_real_catalyst_provider_executes_blueprint_lifecycle(tmp_path: Path):
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("# injected Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    conduit_module = _ConduitModule()
    provider = CatalystPythonProvider(
        catalyst_module=catalyst_module,
        conduit_module=conduit_module,
    )
    declaration = Catalyst(
        pipeline=str(pipeline),
        search_paths=(str(tmp_path),),
        args=("--extract=volume",),
        provider=provider,
    )
    session = declaration.open_session(_serial_context())
    frame = _frame()
    run = ObserverRun(frame.snapshot.provenance.run_identity, {"case": "heat"})

    with PostCommitObserverQueue(
            session, run, consumer_id="live-temperature") as dispatcher:
        assert dispatcher.submit(frame) == 0
        reports = dispatcher.flush()

    assert len(reports) == 1
    assert reports[0].status == "delivered"
    assert [operation for operation, _ in catalyst_module.operations] == [
        "initialize", "execute", "finalize"]
    assert catalyst_module.operations[0][1].values["catalyst/async/enabled"] == 0
    assert catalyst_module.operations[0][1].values[
        "catalyst_load/implementation"] == "paraview"
    assert catalyst_module.operations[0][1].values[
        "catalyst_load/search_paths"] == [str(tmp_path.resolve())]
    assert catalyst_module.operations[0][1].values[
        "catalyst/scripts/pops/args"] == ["--extract=volume"]
    execute_node = catalyst_module.operations[1][1]
    paths = execute_node.values
    temperature_prefixes = [
        path.removesuffix("/display_name") for path, value in paths.items()
        if path.endswith("/display_name") and value == "temperature"
    ]
    assert len(temperature_prefixes) == 1
    assert np.array_equal(
        paths[temperature_prefixes[0] + "/values"],
        np.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    assert any(
        path.endswith("/display_name") and value == "vtkGhostType"
        for path, value in paths.items())
    assert paths["catalyst/channels/mesh/type"] == "multimesh"
    ghost_metadata = [
        value for path, value in paths.items()
        if "/state/metadata/vtk_fields/" in path
        and path.endswith("/attribute_type")
    ]
    assert ghost_metadata == ["Ghosts"]
    assert len(conduit_module.blueprint.mesh.verified_domains) == 1
    assert "/data/layout_" in conduit_module.blueprint.mesh.verified_domains[0]
    assert paths["catalyst/state/timestep"] == 4
    assert paths["catalyst/state/time"] == 0.25
    assert reports[0].receipt.detail["conduit_module"] == "test_catalyst_conduit"


def test_collective_catalyst_publishes_empty_mesh_when_rank_owns_no_geometry_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "empty_rank_pipeline.py"
    pipeline.write_text("# collective rank with no local mesh partition\n")
    catalyst_module = _CatalystModule()
    provider = CatalystPythonProvider(
        catalyst_module=catalyst_module,
        conduit_module=_FetchTrackingConduitModule(),
    )
    execution_context, worker_lane, _agreements = _collective_context(monkeypatch)
    session = Catalyst(
        pipeline=str(pipeline), provider=provider,
    ).open_runtime_session(
        {"worker_communicator": worker_lane}, execution_context,
    )

    frame = _without_local_pieces(_frame(mode=ParallelMode.COLLECTIVE))

    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    session.execute(frame)
    session.finalize()

    execute_node = catalyst_module.operations[1][1]
    data_path = "catalyst/channels/mesh/data"
    assert execute_node.fetched == [data_path]
    child_paths = {
        path: value for path, value in execute_node.values.items()
        if path.startswith(data_path + "/")
    }
    assert any(path.endswith("/topologies/mesh_000000/type") and value == "uniform"
               for path, value in child_paths.items())
    empty_arrays = [
        value for path, value in child_paths.items()
        if "/fields/" in path and path.endswith("/values")
    ]
    assert len(empty_arrays) == 4
    assert all(array.size == 0 for array in empty_arrays)


def test_collective_catalyst_rejects_unproved_polar_empty_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "polar_empty_rank_pipeline.py"
    pipeline.write_text("# polar collective zero-cell peer is not proved\n")
    provider = CatalystPythonProvider(
        catalyst_module=_CatalystModule(),
        conduit_module=_FetchTrackingConduitModule(),
    )
    execution_context, worker_lane, _agreements = _collective_context(monkeypatch)
    session = Catalyst(
        pipeline=str(pipeline), provider=provider,
    ).open_runtime_session(
        {"worker_communicator": worker_lane}, execution_context,
    )
    frame = _without_local_pieces(_frame(
        mode=ParallelMode.COLLECTIVE,
        coordinate_system=POLAR_ANNULUS_2D_COORDINATES,
    ))

    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    with pytest.raises(
        RuntimeError,
        match="zero-cell peers currently prove Cartesian 2D only",
    ):
        session.execute(frame)
    session.finalize()


def test_catalyst_rejects_nested_async_environment(tmp_path: Path, monkeypatch):
    pipeline = tmp_path / "nested_async.py"
    pipeline.write_text("# nested Catalyst async must remain disabled\n")
    monkeypatch.setenv("CATALYST_ASYNC_ENABLED", "1")
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=_CatalystModule(),
            conduit_module=_ConduitModule(),
        ),
    )

    with pytest.raises(RuntimeError, match="requires Catalyst internal async to be disabled"):
        declaration.open_session(_serial_context())


def test_catalyst_rejects_environment_loader_precedence(tmp_path: Path, monkeypatch):
    pipeline = tmp_path / "environment_loader.py"
    pipeline.write_text("# declaration authority must win\n")
    monkeypatch.setenv("CATALYST_IMPLEMENTATION_PREFER_ENV", "1")
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=_CatalystModule(),
            conduit_module=_ConduitModule(),
        ),
    )

    with pytest.raises(RuntimeError, match="rejects CATALYST_IMPLEMENTATION_PREFER_ENV"):
        declaration.open_session(_serial_context())


def test_catalyst_partial_initialize_is_finalized_exactly_once(tmp_path: Path):
    pipeline = tmp_path / "partial_initialize.py"
    pipeline.write_text("# partial initialize cleanup\n")
    catalyst_module = _PartiallyFailingInitializeCatalyst()
    session = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    ).open_session(_serial_context())

    with pytest.raises(RuntimeError, match="initialization failed"):
        PostCommitObserverQueue(
            session,
            ObserverRun(_identity("run", "partial-catalyst-initialize")),
            consumer_id="partial-catalyst-initialize",
        )

    assert [operation for operation, _node in catalyst_module.operations] == [
        "initialize", "finalize"]


def test_catalyst_rejects_stub_implementation_acknowledgement(tmp_path: Path):
    pipeline = tmp_path / "stub_pipeline.py"
    pipeline.write_text("# a successful stub must not count as visualization\n")
    catalyst_module = _StubReportingCatalyst()
    session = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    ).open_session(_serial_context())

    with pytest.raises(RuntimeError, match="initialization failed"):
        PostCommitObserverQueue(
            session,
            ObserverRun(_identity("run", "stub-rejected")),
            consumer_id="stub-rejected",
        )

    assert [operation for operation, _node in catalyst_module.operations] == [
        "initialize", "finalize"]


def test_catalyst_maps_polar_annulus_to_explicit_cartesian_quads(tmp_path: Path):
    pipeline = tmp_path / "polar_pipeline.py"
    pipeline.write_text("# injected polar Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    session = declaration.open_session(_serial_context())
    frame = _frame(
        coordinate_system="pops://coordinates/polar-annulus-2d@1",
        origin=(1.0, 0.0),
        spacing=(0.5, np.pi / 2.0),
    )
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    session.execute(frame)
    session.finalize()

    paths = catalyst_module.operations[1][1].values
    topology_types = [
        value for path, value in paths.items()
        if "/topologies/" in path and path.endswith("/type")
    ]
    assert topology_types == ["unstructured"]
    x = next(value for path, value in paths.items()
             if "/coordsets/" in path and path.endswith("/values/x"))
    y = next(value for path, value in paths.items()
             if "/coordsets/" in path and path.endswith("/values/y"))
    connectivity = next(value for path, value in paths.items()
                        if path.endswith("/elements/connectivity"))
    assert np.allclose(x[:3], [1.0, 1.5, 2.0])
    assert np.allclose(y[:3], [0.0, 0.0, 0.0])
    assert np.array_equal(connectivity[:4], [0, 1, 4, 3])


def test_catalyst_uses_same_block_disambiguated_names_as_paraview_files(tmp_path: Path):
    base_frame = _frame()
    geometry = base_frame.snapshot.geometries[0]
    component = _identity("component-manifest", "two-blocks")
    fields = []
    keys = []
    for index, block_name in enumerate(("fluid", "radiation")):
        owner = OwnerPath.case("case").child(OwnerKind.BLOCK, block_name)
        key = FieldKey(
            Handle("rho", kind="state", owner=owner),
            component,
            geometry.layout_identity,
            0,
            "accepted",
        )
        keys.append(key)
        fields.append(FieldPayload(
            key,
            "cell",
            "kg/m3",
            ("rho",),
            geometry.cell_shape,
            (ArrayPiece(
                (0, 0),
                (2, 2),
                np.full((1, 2, 2), index + 1.0),
                0,
                0,
                False,
            ),),
        ))
    snapshot = OutputSnapshot(
        base_frame.snapshot.clock,
        base_frame.snapshot.provenance,
        (geometry,),
        tuple(fields),
        {"test": "block-name-disambiguation"},
    )
    frame = ObserverFrame(
        snapshot,
        OutputRequest("two-block-live", tuple(keys), ParallelMode.SERIAL, 0, 1),
    )
    pipeline = tmp_path / "two_blocks.py"
    pipeline.write_text("# injected two-block Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    session = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    ).open_session(_serial_context())
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    session.execute(frame)
    session.finalize()

    paths = catalyst_module.operations[1][1].values
    display_names = {
        value for path, value in paths.items()
        if "/fields/" in path and path.endswith("/display_name")
    }
    assert {"fluid.rho", "radiation.rho"}.issubset(display_names)


class _RetrySession:
    authority = {
        "schema_version": 1,
        "provider_id": "test.observer",
        "delivery": "post_commit",
        "threading": "dedicated_serial",
        "worker_mpi": False,
    }

    def __init__(self, *, always_fail=False):
        self.calls = 0
        self.always_fail = always_fail

    def initialize(self, _run):
        return None

    def execute(self, frame):
        self.calls += 1
        if self.always_fail or self.calls == 1:
            raise RuntimeError("viewer unavailable")
        return ObserverReceipt(frame.identity, "test.observer")

    def finalize(self):
        return None

    def abort(self):
        return None


def test_live_observer_session_provider_must_match_authenticated_manifest():
    class _DeclaredProvider:
        def consumer_data(self):
            return {
                "schema_version": 1,
                "provider_id": "test.declared-observer",
                "observer_kind": "test",
            }

        def open_session(self, _execution_context):
            return _RetrySession()

    descriptor = LiveVisualization(
        observer=_DeclaredProvider(),
        schedule=Schedule(Every(AcceptedStep(Clock("provider-authority")), 1)),
        fields=(Handle(
            "temperature",
            kind="state",
            owner=OwnerPath.model("provider-authority"),
        ),),
    )
    operation = descriptor.consumer_authoring()[0].operation

    with pytest.raises(ValueError, match="provider_id differs from its authenticated manifest"):
        operation.open_session(_serial_context())


def test_catalyst_session_provider_must_match_authenticated_backend(tmp_path: Path):
    class _DeclaredCatalystBackend:
        def consumer_data(self):
            return {
                "schema_version": 1,
                "provider_id": "test.declared-catalyst-backend",
                "observer_kind": "catalyst",
            }

        def open_session(self, _configuration, _execution_context):
            return _RetrySession()

    pipeline = tmp_path / "provider_identity.py"
    pipeline.write_text("# provider identity test\n")
    declaration = Catalyst(
        pipeline=str(pipeline), provider=_DeclaredCatalystBackend())

    with pytest.raises(ValueError, match="provider_id differs from its authenticated provider"):
        declaration.open_session(_serial_context())


def test_bounded_dispatcher_retries_then_reports_without_compensation():
    session = _RetrySession()
    dispatcher = PostCommitObserverQueue(
        session, ObserverRun(_identity("run", "retry")),
        consumer_id="retry-observer", capacity=1, max_attempts=2)
    dispatcher.submit(_frame())
    reports = dispatcher.close()

    assert dispatcher.capacity == 1
    assert reports[0].status == "delivered"
    assert reports[0].attempts == 2
    assert session.calls == 2


def test_bounded_dispatcher_reports_exhausted_frame_as_skipped():
    session = _RetrySession(always_fail=True)
    dispatcher = PostCommitObserverQueue(
        session, ObserverRun(_identity("run", "skip")),
        consumer_id="skip-observer", capacity=1, max_attempts=2)
    dispatcher.submit(_frame())
    reports = dispatcher.close()

    assert reports[0].status == "skipped"
    assert reports[0].receipt is None
    assert "viewer unavailable" in reports[0].reason
    assert session.calls == 2


def test_serial_catalyst_rejects_unproved_centering_and_distributed_frame(tmp_path: Path):
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("# injected Catalyst pipeline\n")
    provider = CatalystPythonProvider(
        catalyst_module=_CatalystModule(), conduit_module=_ConduitModule())
    session = Catalyst(
        pipeline=str(pipeline), provider=provider).open_session(_serial_context())
    frame = _frame(centering="node")
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    with pytest.raises(NotImplementedError, match="cell-centered"):
        session.execute(frame)
    with pytest.raises(ValueError, match="SERIAL Catalyst received a distributed frame"):
        session.execute(_frame(mode=ParallelMode.PER_RANK))
    session.finalize()


def test_catalyst_rejects_an_mpi_execution_context_before_loading_modules(
    tmp_path: Path,
):
    pipeline = tmp_path / "mpi_pipeline.py"
    pipeline.write_text("# must never load without a duplicated lane\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    mpi_context = SimpleNamespace(
        communicator=SimpleNamespace(identity="MPI_COMM_WORLD"))

    with pytest.raises(ValueError, match="exact duplicated MPI_COMM_WORLD observer lane"):
        declaration.open_session(mpi_context)

    assert catalyst_module.operations == []


def test_catalyst_collective_session_authenticates_lane_and_passes_mpi_comm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "collective_pipeline.py"
    pipeline.write_text("# collective Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    mpi_context, lane, agreements = _collective_context(monkeypatch)

    session = declaration.open_runtime_session(
        {"worker_communicator": lane}, mpi_context)
    assert session.authority == {
        "schema_version": 1,
        "provider_id": "pops.output.catalyst-python.v1",
        "delivery": "post_commit",
        "threading": "dedicated_collective",
        "worker_mpi": True,
    }

    frame = _frame(mode=ParallelMode.COLLECTIVE)
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    receipt = session.execute(frame)
    session.finalize()

    assert receipt.frame_identity == frame.identity
    assert [operation for operation, _node in catalyst_module.operations] == [
        "initialize", "execute", "finalize"]
    assert catalyst_module.operations[0][1].values["catalyst/mpi_comm"] == 73
    assert agreements
    assert all(
        row == {"rank": 0, "error": None}
        or set(row) == {"rank", "value"}
        for row in agreements)


def test_catalyst_rejects_a_worker_lane_with_different_world_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "mismatched_collective_pipeline.py"
    pipeline.write_text("# mismatched collective Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    mpi_context, lane, _agreements = _collective_context(
        monkeypatch, world_size=3, lane_size=2)

    with pytest.raises(ValueError, match="lane topology differs from MPI_COMM_WORLD"):
        declaration.open_runtime_session(
            {"worker_communicator": lane}, mpi_context)

    assert catalyst_module.operations == []


def test_catalyst_collective_agreement_propagates_a_peer_initialize_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "peer_failure_pipeline.py"
    pipeline.write_text("# peer failure Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    mpi_context, lane, agreements = _collective_context(
        monkeypatch, peer_error="ValueError: rank-one pipeline failure")
    session = declaration.open_runtime_session(
        {"worker_communicator": lane}, mpi_context)

    with pytest.raises(
            RuntimeError,
            match="Catalyst initialize failed collectively:.*rank 1:.*pipeline failure"):
        session.initialize(ObserverRun(_identity("run", "peer-initialize-failure")))

    assert agreements == [{"rank": 0, "error": None}]
    assert catalyst_module.operations == []


def test_catalyst_collective_rejects_rank_divergent_initialize_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "divergent_collective_pipeline.py"
    pipeline.write_text("# divergent collective Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    mpi_context, lane, _agreements = _collective_context(
        monkeypatch, divergent_initialize_authority=True)
    session = declaration.open_runtime_session(
        {"worker_communicator": lane}, mpi_context)

    with pytest.raises(
            RuntimeError, match="Catalyst initialize authority differs across ranks: 1"):
        session.initialize(ObserverRun(_identity("run", "divergent-authority")))

    assert catalyst_module.operations == []


def test_catalyst_collective_rejects_a_frame_from_another_lane_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pipeline = tmp_path / "frame_topology_pipeline.py"
    pipeline.write_text("# frame topology Catalyst pipeline\n")
    catalyst_module = _CatalystModule()
    declaration = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    )
    mpi_context, lane, _agreements = _collective_context(monkeypatch)
    session = declaration.open_runtime_session(
        {"worker_communicator": lane}, mpi_context)
    frame = _frame(mode=ParallelMode.COLLECTIVE)
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    mismatched = ObserverFrame(
        frame.snapshot,
        OutputRequest(
            frame.request.consumer_id,
            frame.request.selection,
            ParallelMode.COLLECTIVE,
            rank=1,
            size=2,
        ),
    )

    with pytest.raises(
            RuntimeError,
            match="Catalyst execute failed collectively:.*exact worker MPI lane topology"):
        session.execute(mismatched)

    session.finalize()
    assert [operation for operation, _node in catalyst_module.operations] == [
        "initialize", "finalize"]


def test_catalyst_conduit_import_prefers_paraview_name_then_external_fallback(monkeypatch):
    from pops.output import _catalyst_backend as backend

    catalyst_module = _CatalystModule()
    conduit_module = _ConduitModule()
    calls = []

    def imported(name):
        calls.append(name)
        if name == "catalyst":
            return catalyst_module
        if name == "catalyst_conduit":
            raise ModuleNotFoundError(name)
        if name == "conduit":
            return conduit_module
        raise AssertionError(name)

    monkeypatch.setattr(backend.importlib, "import_module", imported)
    selected_catalyst, selected_conduit = CatalystPythonProvider()._modules()

    assert selected_catalyst is catalyst_module
    assert selected_conduit is conduit_module
    assert calls == ["catalyst", "catalyst_conduit", "conduit"]


def test_catalyst_prevalidates_complete_lifecycle_module() -> None:
    incomplete = SimpleNamespace(initialize=lambda _node: None, execute=lambda _node: None)
    provider = CatalystPythonProvider(
        catalyst_module=incomplete,
        conduit_module=_ConduitModule(),
    )

    with pytest.raises(RuntimeError, match="callable lifecycle methods: finalize, about"):
        provider._modules()


def test_builtin_catalyst_runtime_lifecycle_is_process_global(monkeypatch) -> None:
    monkeypatch.setattr(runtime_consumers, "_BUILTIN_CATALYST_PROCESS_STARTED", False)

    runtime_consumers._reserve_builtin_catalyst_process_lifecycle()
    with pytest.raises(RuntimeError, match="new process"):
        runtime_consumers._reserve_builtin_catalyst_process_lifecycle()


def test_real_catalyst_conduit_blueprint_when_available(tmp_path: Path):
    conduit = pytest.importorskip("catalyst_conduit")
    blueprint = getattr(conduit, "blueprint", None)
    if not callable(getattr(getattr(blueprint, "mesh", None), "verify", None)):
        pytest.skip("installed catalyst_conduit has no Blueprint mesh verifier")
    pipeline = tmp_path / "real_conduit_pipeline.py"
    pipeline.write_text("# real Conduit verification with fake Catalyst lifecycle\n")
    catalyst_module = _CatalystModule()
    session = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=conduit,
        ),
    ).open_session(_serial_context())
    frame = _frame()
    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    session.execute(frame)
    session.finalize()

    assert [name for name, _node in catalyst_module.operations] == [
        "initialize", "execute", "finalize"]


def test_scalar_tutorial_pipeline_executes_with_real_catalyst_when_available():
    pytest.importorskip("catalyst")
    pytest.importorskip("catalyst_conduit")
    pipeline = (
        Path(__file__).resolve().parents[4]
        / "docs/tuto/scalar_advection/catalyst_pipeline.py"
    )
    session = Catalyst(pipeline=str(pipeline)).open_session(_serial_context())
    frame = _frame(field_name="U", component_names=("rho",))

    session.initialize(ObserverRun(frame.snapshot.provenance.run_identity))
    receipt = session.execute(frame)
    session.finalize()

    assert receipt.frame_identity == frame.identity
    assert receipt.provider_id == "pops.output.catalyst-python.v1"


def test_catalyst_revalidates_pipeline_immediately_before_initialize(tmp_path: Path):
    pipeline = tmp_path / "mutable_pipeline.py"
    pipeline.write_text("# authenticated pipeline\n")
    catalyst_module = _CatalystModule()
    session = Catalyst(
        pipeline=str(pipeline),
        provider=CatalystPythonProvider(
            catalyst_module=catalyst_module,
            conduit_module=_ConduitModule(),
        ),
    ).open_session(_serial_context())
    pipeline.write_text("# changed after session open\n")

    with pytest.raises(RuntimeError, match="changed between session authentication"):
        session.initialize(ObserverRun(_identity("run", "pipeline-mutation")))

    assert catalyst_module.operations == []


def test_background_dispatcher_rejects_worker_mpi_without_a_duplicate_lane():
    session = _RetrySession()
    session.authority = dict(
        session.authority,
        threading="dedicated_collective",
        worker_mpi=True,
    )
    with pytest.raises(ValueError, match="explicit duplicated worker lane"):
        PostCommitObserverQueue(
            session, ObserverRun(_identity("run", "mpi")), consumer_id="mpi-observer")


def test_background_dispatcher_rejects_a_worker_lane_for_a_serial_session(
    monkeypatch: pytest.MonkeyPatch,
):
    import pops._native_collectives as native_collectives

    lane = SimpleNamespace(
        identity="MPI_COMM_WORLD/observer/serial-session",
        active=True,
        rank=0,
        size=2,
    )

    def require_duplicate(communicator, *, allow_world=True):
        assert communicator is lane
        assert allow_world is False
        return communicator

    monkeypatch.setattr(native_collectives, "require_communicator", require_duplicate)

    with pytest.raises(ValueError, match="serial observer session must not receive"):
        PostCommitObserverQueue(
            _RetrySession(),
            ObserverRun(_identity("run", "serial-session-with-lane")),
            consumer_id="serial-session-with-lane",
            worker_communicator=lane,
        )


def test_background_dispatcher_accepts_a_collective_session_with_duplicate_lane(
    monkeypatch: pytest.MonkeyPatch,
):
    import pops._native_collectives as native_collectives

    class _CollectiveSession(_RetrySession):
        authority = dict(
            _RetrySession.authority,
            threading="dedicated_collective",
            worker_mpi=True,
        )

        def execute(self, frame):
            self.calls += 1
            return ObserverReceipt(frame.identity, "test.observer")

    lane = SimpleNamespace(
        identity="MPI_COMM_WORLD/observer/queue-test",
        active=True,
        rank=0,
        size=2,
    )
    barriers = []

    def require_duplicate(communicator, *, allow_world=True):
        assert communicator is lane
        assert allow_world is False
        return communicator

    def allgather_value(communicator, value):
        assert communicator is lane
        return tuple(dict(value, rank=owner) for owner in range(communicator.size))

    monkeypatch.setattr(native_collectives, "require_communicator", require_duplicate)
    monkeypatch.setattr(native_collectives, "rank", lambda communicator: communicator.rank)
    monkeypatch.setattr(native_collectives, "size", lambda communicator: communicator.size)
    monkeypatch.setattr(native_collectives, "allgather_value", allgather_value)
    monkeypatch.setattr(
        native_collectives, "barrier", lambda communicator: barriers.append(communicator))

    session = _CollectiveSession()
    worker = PostCommitObserverWorker(thread_name="test-collective-observer-worker")
    queue = PostCommitObserverQueue(
        session,
        ObserverRun(_identity("run", "collective-queue")),
        consumer_id="collective-observer",
        worker_communicator=lane,
        shared_worker=worker,
    )
    queue.submit(_frame(mode=ParallelMode.COLLECTIVE))
    reports = queue.close()
    worker.close()

    assert len(reports) == 1
    assert reports[0].status == "delivered"
    assert session.calls == 1
    # Startup uses one explicit gate; shutdown reaches the stronger lifecycle allgather above.
    assert barriers == [lane]


def test_shared_post_commit_worker_preserves_cross_consumer_fifo_on_one_thread():
    events = []

    class _OrderedSession(_RetrySession):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def initialize(self, _run):
            events.append(("initialize", self.name, threading.get_ident()))

        def execute(self, frame):
            events.append(("execute", self.name, threading.get_ident()))
            return ObserverReceipt(frame.identity, "test.observer")

        def finalize(self):
            events.append(("finalize", self.name, threading.get_ident()))

    worker = PostCommitObserverWorker(thread_name="test-shared-post-commit-fifo")
    run = ObserverRun(_identity("run", "shared-fifo"))
    first = PostCommitObserverQueue(
        _OrderedSession("first"), run, consumer_id="first", shared_worker=worker)
    second = PostCommitObserverQueue(
        _OrderedSession("second"), run, consumer_id="second", shared_worker=worker)

    second.submit(_frame())
    first.submit(_frame())
    first.close()
    second.close()
    worker.close()

    assert [(phase, name) for phase, name, _thread in events] == [
        ("initialize", "first"),
        ("initialize", "second"),
        ("execute", "second"),
        ("execute", "first"),
        ("finalize", "first"),
        ("finalize", "second"),
    ]
    assert len({thread for _phase, _name, thread in events}) == 1


def test_observer_initialization_failure_aborts_partial_session_once():
    class _PartialInitializeSession(_RetrySession):
        def __init__(self):
            super().__init__()
            self.abort_calls = 0

        def initialize(self, _run):
            raise RuntimeError("partially initialized")

        def abort(self):
            self.abort_calls += 1

    session = _PartialInitializeSession()

    with pytest.raises(RuntimeError, match="initialization failed"):
        PostCommitObserverQueue(
            session,
            ObserverRun(_identity("run", "partial-initialize")),
            consumer_id="partial-initialize",
        )

    assert session.abort_calls == 1


def test_observer_receipt_must_match_authenticated_session_provider():
    class _WrongProviderReceiptSession(_RetrySession):
        def execute(self, frame):
            self.calls += 1
            return ObserverReceipt(frame.identity, "another.observer")

    dispatcher = PostCommitObserverQueue(
        _WrongProviderReceiptSession(),
        ObserverRun(_identity("run", "wrong-receipt-provider")),
        consumer_id="wrong-receipt-provider",
    )
    dispatcher.submit(_frame())
    (report,) = dispatcher.close()

    assert report.status == "skipped"
    assert "provider_id differs" in report.reason


def test_runtime_owned_submission_detaches_once_and_keeps_no_native_sharing(monkeypatch):
    valid = np.ones((2, 2), dtype=np.bool_)
    coverage = np.asarray([[False, True], [False, False]])
    volumes = np.full((2, 2), 0.125)
    source = _frame(native_geometry_arrays=(valid, coverage, volumes))
    real_detach = observer_runtime.detach_observer_frame
    calls = []

    def counted(frame):
        calls.append(frame.identity)
        return real_detach(frame)

    monkeypatch.setattr(observer_runtime, "detach_observer_frame", counted)
    owned = observer_runtime._detach_owned_observer_frame(source)

    class _CaptureSession(_RetrySession):
        def __init__(self):
            super().__init__()
            self.frames = []

        def execute(self, frame):
            self.frames.append(frame)
            return ObserverReceipt(frame.identity, "test.observer")

    session = _CaptureSession()
    dispatcher = PostCommitObserverQueue(
        session,
        ObserverRun(_identity("run", "single-detach")),
        consumer_id="single-detach",
    )
    dispatcher._submit_detached(owned)
    dispatcher.close()

    assert calls == [source.identity]
    detached_geometry = session.frames[0].snapshot.geometries[0]
    assert not np.shares_memory(detached_geometry.coverage, coverage)
    assert not np.shares_memory(detached_geometry.cell_volumes, volumes)


@pytest.mark.parametrize("rank", (0, 1))
def test_root_provider_preflight_reaches_one_consensus_before_local_failure(
    monkeypatch, rank,
):
    calls = []

    class _Operation:
        def consumer_data(self):
            raise AssertionError("manifest data is already resolved")

        def preflight(self, _context):
            if rank == 1:
                raise RuntimeError("rank-one preflight failed")

        def preopen_session(self, _context):
            if rank == 0:
                raise RuntimeError("rank-zero preopen failed")
            raise AssertionError("non-root must not preopen")

    operation = _Operation()
    manifest = SimpleNamespace(
        kind=runtime_consumers.ConsumerKind.MONITOR,
        qualified_id="observer/root-preflight",
        parallel_mode=ParallelMode.ROOT,
        operation=operation,
        operation_data={
            "parallel_mode": "root",
            "queue_capacity": 1,
            "max_attempts": 1,
            "on_failure": {"action": "raise_on_flush"},
        },
        diagnostic_quantities=(),
    )
    communicator = object()
    owner = SimpleNamespace(
        _consumer_graph=SimpleNamespace(nodes=(manifest,)),
        _execution_context=object(),
        _retain_output_recoveries=None,
        _component_manifests=(),
        _layout_plan=SimpleNamespace(layouts=()),
    )
    monkeypatch.setattr(
        runtime_consumers,
        "_execution_topology",
        lambda _owner: (rank, 2, communicator),
    )

    def gathered(actual_communicator, value):
        assert actual_communicator is communicator
        calls.append(value)
        errors = (
            "RuntimeError: rank-zero preopen failed",
            "RuntimeError: rank-one preflight failed",
        )
        return tuple({"rank": owner_rank, "error": error}
                     for owner_rank, error in enumerate(errors))

    monkeypatch.setattr(runtime_consumers, "allgather_value", gathered)

    with pytest.raises(RuntimeError, match="provider session preflight failed"):
        runtime_consumers.RuntimeConsumerPublisher(owner)

    assert len(calls) == 1
    assert calls[0]["rank"] == rank
    assert calls[0]["error"] is not None


@pytest.mark.parametrize("rank", (0, 1))
def test_root_frame_detach_failure_is_collective_before_prepare_returns(
    monkeypatch, rank,
):
    frame = _frame(mode=ParallelMode.ROOT)
    communicator = object()
    publisher = runtime_consumers.RuntimeConsumerPublisher.__new__(
        runtime_consumers.RuntimeConsumerPublisher)
    publisher._owner = SimpleNamespace(
        _output_snapshot=lambda _manifest: (frame.snapshot, frame.request))
    publisher._rank = rank
    publisher._size = 2
    publisher._communicator = communicator
    calls = []

    def detached(_frame_value):
        if rank == 0:
            raise RuntimeError("rank-zero detach failed")
        raise AssertionError("non-root must not detach")

    monkeypatch.setattr(
        runtime_consumers, "_detach_owned_observer_frame", detached)

    def gathered(actual_communicator, value):
        assert actual_communicator is communicator
        calls.append(value)
        return (
            {"rank": 0, "error": "RuntimeError: rank-zero detach failed"},
            {"rank": 1, "error": None},
        )

    monkeypatch.setattr(runtime_consumers, "allgather_value", gathered)
    manifest = SimpleNamespace(parallel_mode=ParallelMode.ROOT)

    with pytest.raises(RuntimeError, match="frame detachment failed"):
        publisher._prepare_live_visualization(object(), manifest)

    assert len(calls) == 1
    assert calls[0] == {
        "rank": rank,
        "error": "RuntimeError: rank-zero detach failed" if rank == 0 else None,
    }


def test_detached_frame_does_not_borrow_runtime_geometry_buffers():
    valid = np.ones((2, 2), dtype=np.bool_)
    coverage = np.asarray([[False, True], [False, False]])
    volumes = np.full((2, 2), 0.125)
    frame = _frame(native_geometry_arrays=(valid, coverage, volumes))
    detached = detach_observer_frame(frame)

    coverage.setflags(write=True)
    volumes.setflags(write=True)
    coverage[:, :] = False
    volumes[:, :] = 99.0

    detached_geometry = detached.snapshot.geometries[0]
    assert bool(detached_geometry.coverage[0, 1]) is True
    assert np.all(detached_geometry.cell_volumes == 0.125)
    assert not np.shares_memory(detached_geometry.coverage, coverage)
    assert not np.shares_memory(detached_geometry.cell_volumes, volumes)
