"""Dimension-generic geometry and VTK topology contracts."""
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
    ParaViewPreset,
    ParaViewWriter,
    PortableState,
    ReopenedParaViewMultiBlock,
    read_paraview,
    read_paraview_parallel,
)
from pops.output._consumer_contracts import ParallelMode
from pops.output.data import (
    EMBEDDED_BOUNDARY_ARRAY_NAMES,
    EmbeddedBoundaryPayload,
    _NATIVE_GEOMETRY_ARRAYS,
    _centering_shape,
)
from pops.output._writers.paraview import (
    _PVSM_SAVE_SCRIPT,
    _resolved_preset_data,
    _series_identity,
    _stage_pvtu,
    _vtu_schema,
)


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _snapshot(
    cell_shape: tuple[int, ...],
    *,
    centering: str = "cell",
    values: np.ndarray | None = None,
    embedded: bool = False,
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
    global_shape = _centering_shape(cell_shape, centering)
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
    )
    sidecars = ()
    if embedded:
        sidecar_values = {
            "pops_active": np.ones(cell_shape, dtype=np.float64),
            "pops_phi": np.linspace(
                -1.0, 1.0, num=int(np.prod(cell_shape)), dtype=np.float64
            ).reshape(cell_shape),
            "pops_kappa": np.full(cell_shape, 0.25, dtype=np.float64),
        }
        sidecar_values["pops_active"].reshape(-1)[-1] = 0.0
        sidecars = (
            EmbeddedBoundaryPayload(
                layout,
                0,
                cell_shape,
                {
                    name: (
                        ArrayPiece(
                            lower,
                            cell_shape,
                            value.reshape((1,) + cell_shape),
                            0,
                            0,
                            False,
                        ),
                    )
                    for name, value in sidecar_values.items()
                },
            ),
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
        embedded_boundaries=sidecars,
    )
    return snapshot, OutputRequest("vtk", (key,), ParallelMode.SERIAL)


def _stage(tmp_path, snapshot: OutputSnapshot, request: OutputRequest, name: str):
    session = ParaViewWriter(collection=False).prepare_session(
        snapshot, request, tmp_path / name)
    session.stage()
    return session


@pytest.mark.parametrize(
    ("cell_shape", "coordinate_system", "cell_measure", "axis_names"),
    (
        ((3,), "pops://coordinates/cartesian-1d@1",
         "pops://cell-measures/cartesian-length@1", ("x",)),
        ((2, 3), "pops://coordinates/cartesian-2d@1",
         "pops://cell-measures/cartesian-area@1", ("x", "y")),
        ((2, 1, 3), "pops://coordinates/cartesian-3d@1",
         "pops://cell-measures/cartesian-volume@1", ("x", "y", "z")),
    ),
)
def test_level_geometry_infers_cartesian_contract_from_cell_shape(
    cell_shape,
    coordinate_system,
    cell_measure,
    axis_names,
):
    snapshot, _request = _snapshot(cell_shape)
    geometry = snapshot.geometries[0]

    assert geometry.spatial_rank == len(cell_shape)
    assert geometry.coordinate_system == coordinate_system
    assert geometry.cell_measure == cell_measure
    assert geometry.axis_names == axis_names


@pytest.mark.parametrize("cell_shape", ((4,), (2, 1, 3)))
def test_native_level_geometry_borrows_rank_generic_arrays(cell_shape):
    dimension = len(cell_shape)
    valid = np.ones(cell_shape, dtype=np.bool_)
    coverage = np.zeros(cell_shape, dtype=np.bool_)
    volumes = np.ones(cell_shape, dtype=np.float64)
    geometry = LevelGeometry(
        _identity("layout-plan", "native-%dd" % dimension),
        "uniform",
        0,
        (0.0,) * dimension,
        (1.0,) * dimension,
        cell_shape,
        ((0,) * dimension + cell_shape,),
        coverage,
        volumes,
        _native_valid_cells=valid,
        _native_arrays=_NATIVE_GEOMETRY_ARRAYS,
    )

    assert geometry.spatial_rank == dimension
    assert np.shares_memory(geometry.valid_cells, valid)
    assert np.shares_memory(geometry.coverage, coverage)
    assert np.shares_memory(geometry.cell_volumes, volumes)


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


@pytest.mark.parametrize(
    ("cell_shape", "vtk_type"),
    (((3,), 3), ((2, 2), 9), ((2, 1, 2), 12)),
    ids=("line", "quad", "hex"),
)
def test_vtu_emits_exact_embedded_boundary_sidecars_without_replacing_cartesian_cells(
    tmp_path,
    cell_shape,
    vtk_type,
):
    snapshot, request = _snapshot(cell_shape, embedded=True)
    session = _stage(tmp_path, snapshot, request, "embedded.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    assert set(EMBEDDED_BOUNDARY_ARRAY_NAMES).issubset(reopened.arrays)
    assert np.array_equal(
        reopened.arrays["pops_active"],
        snapshot.embedded_boundaries[0].pieces("pops_active")[0].values.reshape(-1),
    )
    assert np.all(reopened.arrays["pops_kappa"] == 0.25)
    assert np.all(reopened.arrays["pops_cell_volume"] == 1.0)
    assert np.all(reopened.arrays["types"] == vtk_type)
    assert reopened.arrays["phi"].shape == (int(np.prod(cell_shape)),)
    sidecar_manifest = reopened.manifest["datasets"]["embedded_boundaries"]
    assert len(sidecar_manifest) == 1
    assert set(next(iter(sidecar_manifest.values()))["arrays"]) == set(
        EMBEDDED_BOUNDARY_ARRAY_NAMES
    )
    schema = _vtu_schema(session.temporary)
    materialized = _resolved_preset_data(schema, ParaViewPreset(color_by="phi"))
    assert materialized["threshold_active"] is True
    assert 'source.Scalars = ["CELLS", "pops_active"]' in _PVSM_SAVE_SCRIPT
    assert "source.LowerThreshold = 0.5" in _PVSM_SAVE_SCRIPT
    invalid_schema = dict(
        schema,
        cell_arrays=[
            dict(row, component_names=["invalid"])
            if row["name"] == "pops_active" else row
            for row in schema["cell_arrays"]
        ],
    )
    with pytest.raises(ValueError, match="invalid ParaView scalar schema"):
        _resolved_preset_data(invalid_schema, ParaViewPreset(color_by="phi"))
    session.abort_prepare()


@pytest.mark.parametrize("cell_shape", ((3,), (2, 2), (2, 1, 2)))
def test_nodal_field_is_emitted_as_point_data_without_conversion(tmp_path, cell_shape):
    values = np.arange(np.prod(_centering_shape(cell_shape, "node")), dtype=np.float64).reshape(
        _centering_shape(cell_shape, "node")
    )
    snapshot, request = _snapshot(cell_shape, centering="node", values=values)
    session = _stage(tmp_path, snapshot, request, "nodes.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    n_points = int(np.prod(values.shape))
    field_record = next(iter(reopened.manifest["datasets"]["fields"].values()))
    assert field_record["association"] == "point"
    assert field_record["point_range"] == [0, n_points]
    assert np.array_equal(reopened.arrays["phi"], np.arange(n_points, dtype=np.float64))
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


def test_nodal_field_portable_state_colors_by_point_data(tmp_path):
    snapshot, request = _snapshot((2, 2), centering="node")
    session = ParaViewWriter(state=PortableState()).prepare_session(
        snapshot, request, tmp_path / "nodes-with-state.vtu")
    session.stage()
    receipt = session.publish()
    session.finalize()

    assert receipt is not None
    recipe = tmp_path / "nodes-with-state.view.py"
    # collection names the PVD from the consumer/series, not the VTU stem
    scripts = list(tmp_path.glob("*.view.py"))
    assert scripts, "portable state script was not published"
    source = scripts[0].read_text(encoding="utf-8")
    assert 'association = "POINTS"' in source or '("POINTS", config["color_by"])' in source
    assert "if config[\"color_by\"] in point_names:" in source


def test_mixed_cell_and_face_fields_publish_sibling_multiblock_topologies(tmp_path):
    cell_snapshot, request = _snapshot((2, 2), centering="cell")
    face_snapshot, _face_request = _snapshot((2, 2), centering="face_x")
    face_key = replace(
        face_snapshot.fields[0].key,
        reference=Handle(
            "face_phi",
            kind="state",
            owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
        ),
    )
    face_field = replace(face_snapshot.fields[0], key=face_key)
    mixed = replace(cell_snapshot, fields=(cell_snapshot.fields[0], face_field))
    mixed_request = OutputRequest(
        "vtk",
        (cell_snapshot.fields[0].key, face_key),
        ParallelMode.SERIAL,
    )
    session = ParaViewWriter(collection=False).prepare_session(
        mixed, mixed_request, tmp_path / "mixed.vtu")
    session.stage()
    receipt = session.publish()
    session.finalize()

    assert receipt is not None
    reopened = read_paraview(receipt.path).require_selection(mixed_request)
    assert isinstance(reopened, ReopenedParaViewMultiBlock)
    assert [name for name, _block in reopened.blocks] == ["cell", "face_x"]
    cell_block = reopened.block("cell")
    face_block = reopened.block("face_x")
    n_cells = int(np.prod((2, 2)))
    n_faces = int(np.prod(_centering_shape((2, 2), "face_x")))
    assert cell_block.arrays["phi"].shape == (n_cells,)
    assert face_block.arrays["face_phi"].shape == (n_faces,)
    assert n_faces != n_cells
    assert "face_phi" not in cell_block.arrays
    assert "phi" not in face_block.arrays
    assert np.all(cell_block.arrays["types"] == 9)
    assert np.all(face_block.arrays["types"] == 3)


def test_distinct_face_centerings_publish_one_block_per_axis(tmp_path):
    face_x, request_x = _snapshot((2, 2), centering="face_x")
    face_y, _request_y = _snapshot((2, 2), centering="face_y")
    face_y_key = replace(
        face_y.fields[0].key,
        reference=Handle(
            "face_y_phi",
            kind="state",
            owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
        ),
    )
    mixed = replace(
        face_x,
        fields=(face_x.fields[0], replace(face_y.fields[0], key=face_y_key)),
    )
    mixed_request = OutputRequest(
        "vtk",
        (face_x.fields[0].key, face_y_key),
        ParallelMode.SERIAL,
    )
    session = ParaViewWriter(collection=False).prepare_session(
        mixed, mixed_request, tmp_path / "faces-mixed.vtu")
    session.stage()
    receipt = session.publish()
    session.finalize()

    reopened = read_paraview(receipt.path).require_selection(mixed_request)
    assert isinstance(reopened, ReopenedParaViewMultiBlock)
    assert [name for name, _block in reopened.blocks] == ["face_x", "face_y"]
    assert reopened.block("face_x").arrays["phi"].shape == (6,)
    assert reopened.block("face_y").arrays["face_y_phi"].shape == (6,)
    assert "face_y_phi" not in reopened.block("face_x").arrays
    assert "phi" not in reopened.block("face_y").arrays


@pytest.mark.parametrize("cell_shape", ((3,), (2, 2), (2, 1, 2)))
def test_face_fields_keep_eb_sidecars_on_the_cell_block(tmp_path, cell_shape):
    cell_snapshot, _cell_request = _snapshot(cell_shape, centering="cell", embedded=True)
    face_snapshot, _face_request = _snapshot(cell_shape, centering="face_x")
    face_key = replace(
        face_snapshot.fields[0].key,
        reference=Handle(
            "face_phi",
            kind="state",
            owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
        ),
    )
    mixed = replace(
        cell_snapshot,
        fields=(cell_snapshot.fields[0], replace(face_snapshot.fields[0], key=face_key)),
    )
    mixed_request = OutputRequest(
        "vtk",
        (cell_snapshot.fields[0].key, face_key),
        ParallelMode.SERIAL,
    )
    session = ParaViewWriter(collection=False).prepare_session(
        mixed, mixed_request, tmp_path / "faces-eb.vtu")
    session.stage()
    receipt = session.publish()
    session.finalize()

    reopened = read_paraview(receipt.path).require_selection(mixed_request)
    cell_block = reopened.block("cell")
    face_block = reopened.block("face_x")
    assert set(EMBEDDED_BOUNDARY_ARRAY_NAMES).issubset(cell_block.arrays)
    assert set(EMBEDDED_BOUNDARY_ARRAY_NAMES).isdisjoint(face_block.arrays)
    assert "face_phi" not in cell_block.arrays
    assert "phi" not in face_block.arrays


def test_face_only_embedded_boundary_still_emits_a_cell_block(tmp_path):
    face_snapshot, request = _snapshot((2, 2), centering="face_x", embedded=True)
    session = ParaViewWriter(collection=False).prepare_session(
        face_snapshot, request, tmp_path / "face-only-eb.vtu")
    session.stage()
    receipt = session.publish()
    session.finalize()

    reopened = read_paraview(receipt.path)
    assert isinstance(reopened, ReopenedParaViewMultiBlock)
    assert [name for name, _block in reopened.blocks] == ["cell", "face_x"]
    assert set(EMBEDDED_BOUNDARY_ARRAY_NAMES).issubset(reopened.block("cell").arrays)
    assert "phi" in reopened.block("face_x").arrays
    assert "phi" not in reopened.block("cell").arrays


def test_rank_three_face_axes_are_sibling_blocks(tmp_path):
    face_x, _request_x = _snapshot((2, 1, 2), centering="face_x")
    fields = [face_x.fields[0]]
    keys = [face_x.fields[0].key]
    for centering, name in (("face_y", "face_y_phi"), ("face_z", "face_z_phi")):
        extra, _request = _snapshot((2, 1, 2), centering=centering)
        key = replace(
            extra.fields[0].key,
            reference=Handle(
                name,
                kind="state",
                owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
            ),
        )
        fields.append(replace(extra.fields[0], key=key))
        keys.append(key)
    mixed = replace(face_x, fields=tuple(fields))
    mixed_request = OutputRequest("vtk", tuple(keys), ParallelMode.SERIAL)
    session = ParaViewWriter(collection=False).prepare_session(
        mixed, mixed_request, tmp_path / "faces-xyz.vtu")
    session.stage()
    receipt = session.publish()
    session.finalize()

    reopened = read_paraview(receipt.path).require_selection(mixed_request)
    assert [name for name, _block in reopened.blocks] == ["face_x", "face_y", "face_z"]


def test_per_rank_mixed_cell_and_face_publish_rank_local_multiblock(tmp_path):
    cell_snapshot, serial_request = _snapshot((2, 2), centering="cell")
    face_snapshot, _face_request = _snapshot((2, 2), centering="face_x")
    face_key = replace(
        face_snapshot.fields[0].key,
        reference=Handle(
            "face_phi",
            kind="state",
            owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
        ),
    )
    mixed = replace(cell_snapshot, fields=(cell_snapshot.fields[0], replace(
        face_snapshot.fields[0], key=face_key)))
    rank0_request = replace(
        OutputRequest("vtk", (cell_snapshot.fields[0].key, face_key), ParallelMode.SERIAL),
        parallel_mode=ParallelMode.PER_RANK,
        rank=0,
        size=2,
    )
    companions = []
    writer = ParaViewWriter(mode=ParallelMode.PER_RANK, collection=False, state=None)
    prepared = writer._stage_file(
        mixed, rank0_request, tmp_path / "mixed-rank-0.vtu", companions=companions)
    try:
        assert prepared.target.suffix == ".vtm"
        assert [item.target.name for item in companions] == [
            "mixed-rank-0__cell.vtu",
            "mixed-rank-0__face_x.vtu",
        ]
        for leaf in companions:
            leaf.publish()
        prepared.publish()
        reopened = read_paraview(prepared.target).require_selection(rank0_request)
        assert isinstance(reopened, ReopenedParaViewMultiBlock)
        assert [name for name, _block in reopened.blocks] == ["cell", "face_x"]
        n_cells = int(np.prod((2, 2)))
        n_faces = int(np.prod(_centering_shape((2, 2), "face_x")))
        assert reopened.block("cell").arrays["phi"].shape == (n_cells,)
        assert reopened.block("face_x").arrays["face_phi"].shape == (n_faces,)
        assert "face_phi" not in reopened.block("cell").arrays
        assert "phi" not in reopened.block("face_x").arrays
    finally:
        prepared.rollback()
        for leaf in companions:
            leaf.rollback()


def test_face_z_is_refused_on_spatial_rank_two():
    with pytest.raises(ValueError, match="not defined for spatial rank 2"):
        _snapshot((2, 2), centering="face_z")


@pytest.mark.parametrize("cell_shape", ((3,), (2, 2), (2, 1, 2)))
def test_pvtu_authenticates_nodal_schema_as_parallel_point_data(tmp_path, cell_shape):
    snapshot, serial_request = _snapshot(cell_shape, centering="node")
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


@pytest.mark.parametrize(
    ("cell_shape", "vtk_type", "point_count"),
    (((3,), 1, 4), ((2, 2), 3, 9), ((2, 1, 2), 9, 18)),
    ids=("vertex-1d", "line-2d", "quad-3d"),
)
def test_face_x_field_emits_a_vtk_face_mesh_without_recentering(
    tmp_path, cell_shape, vtk_type, point_count,
):
    snapshot, request = _snapshot(cell_shape, centering="face_x")
    session = _stage(tmp_path, snapshot, request, "faces.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    n_faces = int(np.prod(_centering_shape(cell_shape, "face_x")))
    n_cells = int(np.prod(cell_shape))
    field_record = next(iter(reopened.manifest["datasets"]["fields"].values()))
    geometry = next(iter(reopened.manifest["datasets"]["geometries"].values()))
    assert field_record["association"] == "cell"
    assert field_record["cell_range"] == [0, n_faces]
    assert geometry["topology"] == "face_x"
    assert geometry["spatial_rank"] == len(cell_shape)
    assert reopened.arrays["phi"].shape == (n_faces,)
    assert n_faces != n_cells
    assert np.array_equal(reopened.arrays["phi"], np.arange(n_faces, dtype=np.float64))
    assert np.all(reopened.arrays["types"] == vtk_type)
    assert reopened.arrays["Points"].shape == (point_count, 3)
    assert "pops_cell_volume" not in reopened.arrays
    schema = _vtu_schema(session.temporary)
    assert "phi" in {row["name"] for row in schema["cell_arrays"]}
    assert "phi" not in {row["name"] for row in schema["point_arrays"]}
    session.abort_prepare()


def test_face_x_two_dimensional_connectivity_is_lines_on_x_interfaces(tmp_path):
    snapshot, request = _snapshot((2, 2), centering="face_x")
    session = _stage(tmp_path, snapshot, request, "face-x-2d.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    assert np.array_equal(
        reopened.arrays["connectivity"],
        [0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8],
    )
    assert np.array_equal(reopened.arrays["offsets"], [2, 4, 6, 8, 10, 12])
    assert np.array_equal(reopened.arrays["types"], [3] * 6)
    session.abort_prepare()


def test_face_y_two_dimensional_field_uses_lines_on_y_interfaces(tmp_path):
    snapshot, request = _snapshot((2, 3), centering="face_y")
    session = _stage(tmp_path, snapshot, request, "face-y.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    assert reopened.arrays["phi"].shape == (9,)
    assert np.all(reopened.arrays["types"] == 3)
    geometry = next(iter(reopened.manifest["datasets"]["geometries"].values()))
    assert geometry["topology"] == "face_y"
    session.abort_prepare()


def test_face_z_three_dimensional_field_uses_quads_on_z_interfaces(tmp_path):
    snapshot, request = _snapshot((2, 1, 2), centering="face_z")
    session = _stage(tmp_path, snapshot, request, "face-z.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)

    assert reopened.arrays["phi"].shape == (6,)
    assert np.all(reopened.arrays["types"] == 9)
    geometry = next(iter(reopened.manifest["datasets"]["geometries"].values()))
    assert geometry["topology"] == "face_z"
    session.abort_prepare()


def test_polar_annulus_refuses_spatial_ranks_other_than_two(tmp_path):
    from pops._geometry_contracts import POLAR_ANNULUS_2D_COORDINATES, POLAR_ANNULUS_CELL_AREA

    snapshot, request = _snapshot((3,))
    polar = replace(
        snapshot,
        geometries=(
            replace(
                snapshot.geometries[0],
                coordinate_system=POLAR_ANNULUS_2D_COORDINATES,
                cell_measure=POLAR_ANNULUS_CELL_AREA,
                axis_names=("r",),
            ),
        ),
    )
    session = ParaViewWriter(collection=False).prepare_session(
        polar, request, tmp_path / "polar-1d.vtu")
    with pytest.raises(ValueError, match="spatial rank two"):
        session.stage()
    session.abort_prepare()


def test_polar_annulus_face_x_emits_lines_on_the_annulus(tmp_path):
    from pops._geometry_contracts import POLAR_ANNULUS_2D_COORDINATES, POLAR_ANNULUS_CELL_AREA

    snapshot, request = _snapshot((2, 2), centering="face_x")
    polar = replace(
        snapshot,
        geometries=(
            replace(
                snapshot.geometries[0],
                coordinate_system=POLAR_ANNULUS_2D_COORDINATES,
                cell_measure=POLAR_ANNULUS_CELL_AREA,
                axis_names=("r", "theta"),
            ),
        ),
    )
    session = _stage(tmp_path, polar, request, "polar-face.vtu")
    reopened = read_paraview(session.temporary).require_selection(request)
    n_faces = int(np.prod(_centering_shape((2, 2), "face_x")))
    points = reopened.arrays["Points"]
    geometry = snapshot.geometries[0]
    radius = np.hypot(points[:, 0], points[:, 1])
    assert n_faces == 6
    assert np.all(reopened.arrays["types"] == 3)
    assert reopened.arrays["phi"].shape == (n_faces,)
    assert np.min(radius) >= geometry.origin[0] - 1.0e-12
    session.abort_prepare()
