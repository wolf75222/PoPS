"""Dimension-generic VTK topology without weakening the native two-dimensional solver."""
from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import (
    ArrayPiece,
    FieldKey,
    FieldPayload,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    ParaViewWriter,
    PortableState,
    read_paraview,
    read_paraview_parallel,
)
from pops.output._consumer_contracts import ParallelMode
from pops.output._writers.paraview import _series_identity, _stage_pvtu, _vtu_schema


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _snapshot(
    cell_shape: tuple[int, ...],
    *,
    centering: str = "cell",
    values: np.ndarray | None = None,
) -> tuple[OutputSnapshot, OutputRequest]:
    dimension = len(cell_shape)
    layout = _identity("layout-plan", "cartesian-%dd" % dimension)
    manifest = _identity("component-manifest", "scalar")
    handle = Handle(
        "phi",
        kind="state",
        owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
    )
    key = FieldKey(handle, manifest, layout, 0, "accepted")
    global_shape = (
        tuple(extent + 1 for extent in cell_shape)
        if centering == "node" else cell_shape
    )
    if centering == "face_x":
        global_shape = cell_shape[:-1] + (cell_shape[-1] + 1,)
    if values is None:
        values = np.arange(np.prod(global_shape), dtype=np.float64).reshape(global_shape)
    lower = (0,) * dimension
    piece = ArrayPiece(lower, global_shape, values, 0, 0, False)
    field = FieldPayload(key, centering, "1", (), global_shape, (piece,))
    geometry = LevelGeometry(
        layout,
        "uniform",
        0,
        tuple(float(axis + 1) for axis in range(dimension)),
        tuple(0.25 * (axis + 1) for axis in range(dimension)),
        cell_shape,
        (lower + cell_shape,),
        np.zeros(cell_shape, dtype=np.bool_),
        np.ones(cell_shape, dtype=np.float64),
        coordinate_system="pops://coordinates/cartesian-%dd@1" % dimension,
        cell_measure="pops://cell-measures/cartesian-%dd@1" % dimension,
        axis_names=tuple("xyz"[:dimension]),
    )
    snapshot = OutputSnapshot(
        OutputClock.at("macro", 0.5, 2, stage="accepted"),
        OutputProvenance(
            _identity("resolved-plan", "plan"),
            _identity("bind", "bind"),
            _identity("run", "run"),
            "dimension-generic-test",
        ),
        (geometry,),
        (field,),
    )
    return snapshot, OutputRequest("vtk", (key,), ParallelMode.SERIAL)


def _stage(tmp_path, snapshot: OutputSnapshot, request: OutputRequest, name: str):
    session = ParaViewWriter(collection=False).prepare_session(
        snapshot, request, tmp_path / name)
    session.stage()
    return session


def test_vtu_round_trip_uses_shared_points_and_vtk_lines_in_one_dimension(tmp_path):
    snapshot, request = _snapshot((3,), values=np.asarray([10.0, 20.0, 30.0]))
    session = _stage(tmp_path, snapshot, request, "line.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    assert reopened.arrays["Points"].shape == (4, 3)
    assert np.array_equal(reopened.arrays["Points"][:, 0], [1.0, 1.25, 1.5, 1.75])
    assert np.all(reopened.arrays["Points"][:, 1:] == 0.0)
    assert np.array_equal(reopened.arrays["connectivity"], [0, 1, 1, 2, 2, 3])
    assert np.array_equal(reopened.arrays["offsets"], [2, 4, 6])
    assert np.array_equal(reopened.arrays["types"], [3, 3, 3])
    assert np.array_equal(reopened.arrays["phi"], [10.0, 20.0, 30.0])
    geometry = next(iter(reopened.manifest["datasets"]["geometries"].values()))
    assert geometry["spatial_rank"] == 1
    session.abort_prepare()


def test_vtu_round_trip_uses_shared_points_and_hexahedra_in_three_dimensions(tmp_path):
    snapshot, request = _snapshot((2, 1, 2))
    session = _stage(tmp_path, snapshot, request, "hex.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    assert reopened.arrays["Points"].shape == (18, 3)
    assert np.array_equal(reopened.arrays["connectivity"][:16], [
        0, 1, 4, 3, 6, 7, 10, 9,
        1, 2, 5, 4, 7, 8, 11, 10,
    ])
    assert np.array_equal(reopened.arrays["offsets"], [8, 16, 24, 32])
    assert np.array_equal(reopened.arrays["types"], [12, 12, 12, 12])
    assert np.array_equal(reopened.arrays["Points"][0], [1.0, 2.0, 3.0])
    assert np.array_equal(reopened.arrays["Points"][-1], [1.5, 2.5, 4.5])
    assert np.array_equal(reopened.arrays["phi"], np.arange(4, dtype=np.float64))
    session.abort_prepare()


def test_nodal_field_is_emitted_as_point_data_without_conversion(tmp_path):
    values = np.arange(9, dtype=np.float64).reshape((3, 3))
    snapshot, request = _snapshot((2, 2), centering="node", values=values)
    session = _stage(tmp_path, snapshot, request, "nodes.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    field_record = next(iter(reopened.manifest["datasets"]["fields"].values()))
    assert field_record["association"] == "point"
    assert field_record["point_range"] == [0, 9]
    assert np.array_equal(reopened.arrays["phi"], np.arange(9, dtype=np.float64))
    schema = _vtu_schema(session.temporary)
    assert [row["name"] for row in schema["point_arrays"]] == ["phi"]
    assert "phi" not in {row["name"] for row in schema["cell_arrays"]}
    point_data = ET.parse(session.temporary).getroot().find(
        "./UnstructuredGrid/Piece/PointData")
    assert point_data is not None and point_data.attrib["Scalars"] == "phi"
    tree = ET.parse(session.temporary)
    point_data = tree.getroot().find("./UnstructuredGrid/Piece/PointData")
    cell_data = tree.getroot().find("./UnstructuredGrid/Piece/CellData")
    assert point_data is not None and cell_data is not None
    moved = point_data.find("./DataArray[@Name='phi']")
    assert moved is not None
    point_data.remove(moved)
    cell_data.append(moved)
    tampered = tmp_path / "nodes-wrong-association.vtu"
    tree.write(tampered, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="differs from its exact data association"):
        read_paraview(tampered)
    session.abort_prepare()


def test_nodal_field_requires_state_none_until_state_association_is_generic(tmp_path):
    snapshot, request = _snapshot((2, 2), centering="node")
    session = ParaViewWriter(state=PortableState()).prepare_session(
        snapshot, request, tmp_path / "nodes-with-state.vtu")
    with pytest.raises(NotImplementedError, match="PointData output currently requires state=None"):
        session.stage()
    session.abort_prepare()


def test_pvtu_authenticates_nodal_schema_as_parallel_point_data(tmp_path):
    snapshot, serial_request = _snapshot((2, 2), centering="node")
    rank0_request = replace(
        serial_request, parallel_mode=ParallelMode.PER_RANK, rank=0, size=2)
    rank1_request = replace(rank0_request, rank=1)
    rank1_field = replace(
        snapshot.fields[0],
        pieces=(replace(snapshot.fields[0].pieces[0], owner_rank=1, replicated=True),),
    )
    rank1_snapshot = replace(snapshot, fields=(rank1_field,))
    writer = ParaViewWriter(
        mode=ParallelMode.PER_RANK, collection=False, state=None)
    targets = (tmp_path / "nodes-rank-0.vtu", tmp_path / "nodes-rank-1.vtu")
    leaves = (
        writer._stage_file(snapshot, rank0_request, targets[0]),
        writer._stage_file(rank1_snapshot, rank1_request, targets[1]),
    )
    rows = tuple({
        "rank": rank,
        "target": str(targets[rank]),
        "output_identity": leaves[rank].output_identity.token,
        "schema": _vtu_schema(leaves[rank].temporary),
    } for rank in range(2))
    pvtu = _stage_pvtu(
        tmp_path,
        snapshot,
        rank0_request,
        rows,
        _series_identity(snapshot, rank0_request, compression=6),
    )
    try:
        for leaf in leaves:
            leaf.publish()
        pvtu.publish()
        reopened = read_paraview_parallel(pvtu.target)
        assert [row["name"] for row in reopened.manifest["schema"]["point_arrays"]] \
            == ["phi"]
        point_data = ET.parse(pvtu.target).getroot().find(
            "./PUnstructuredGrid/PPointData")
        assert point_data is not None and point_data.attrib["Scalars"] == "phi"
        declaration = point_data.find("./PDataArray")
        assert declaration is not None and declaration.attrib["Name"] == "phi"
    finally:
        pvtu.rollback()
        for leaf in leaves:
            leaf.rollback()


def test_face_centered_field_requires_an_explicit_face_topology(tmp_path):
    snapshot, request = _snapshot((2, 2), centering="face_x")
    session = ParaViewWriter(collection=False).prepare_session(
        snapshot, request, tmp_path / "faces.vtu")
    with pytest.raises(NotImplementedError, match="multi-topology face mesh"):
        session.stage()
    session.abort_prepare()
