"""Mandatory format-native reopen proofs executed only by the explicit M4 gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import (
    ArrayPiece,
    FieldKey,
    FieldPayload,
    HDF5Writer,
    LevelGeometry,
    NPZWriter,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    ParaViewWriter,
    ParallelMode,
    read_hdf5,
)


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _snapshot_and_request():
    layout = _identity("layout-plan", "m4-native-reopen")
    component = _identity("component-manifest", "m4-native-reopen")
    owner = OwnerPath.case("m4-native-reopen").child(OwnerKind.BLOCK, "fluid")
    state = Handle("rho", kind="state", owner=owner)
    key = FieldKey(state, component, layout, 0, "accepted")
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    geometry = LevelGeometry(
        layout,
        "uniform",
        0,
        (0.0, 0.0),
        (0.5, 0.5),
        (2, 2),
        ((0, 0, 2, 2),),
        np.zeros((2, 2), dtype=np.bool_),
        np.full((2, 2), 0.25, dtype=np.float64),
    )
    field = FieldPayload(
        key,
        "cell",
        "kg.m-3",
        (),
        (2, 2),
        (ArrayPiece((0, 0), (2, 2), values, 0, 0, False),),
    )
    snapshot = OutputSnapshot(
        OutputClock.at("macro", 0.25, 4, stage="accepted"),
        OutputProvenance(
            _identity("resolved-plan", "m4-native-reopen"),
            _identity("bind", "m4-native-reopen"),
            _identity("run", "m4-native-reopen"),
            "accepted-step-transaction",
        ),
        (geometry,),
        (field,),
        {"case": "m4-native-reopen"},
    )
    request = OutputRequest("rho-output", (key,), ParallelMode.SERIAL)
    return snapshot, request, values


def _publish(writer, target):
    snapshot, request, expected = _snapshot_and_request()
    session = writer.prepare_session(snapshot, request, target)
    session.stage()
    receipt = session.publish()
    session.finalize()
    assert receipt.path == target
    return receipt.path, request, expected


def _field_dataset(manifest: dict, request: OutputRequest) -> str:
    key = request.selection[0].identity.token
    dataset = manifest["datasets"]["fields"][key]
    if isinstance(dataset, str):
        return dataset
    pieces = dataset["pieces"]
    assert len(pieces) == 1
    return pieces[0]["name"]


def test_npz_reopens_with_numpy_without_a_pops_reader(tmp_path):
    path, request, expected = _publish(NPZWriter(), tmp_path / "native.npz")

    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["pops_output_manifest"]))
        dataset = _field_dataset(manifest, request)
        np.testing.assert_array_equal(archive[dataset], expected)
        assert set(archive.files) == set(manifest["arrays"]) | {
            "pops_output_manifest"
        }
        assert manifest["snapshot"]["clock"]["time"] == float.hex(0.25)


def test_hdf5_reopens_with_h5py_without_a_pops_reader(tmp_path):
    import h5py

    path, request, expected = _publish(HDF5Writer(), tmp_path / "native.h5")

    with h5py.File(path, "r") as output:
        manifest = json.loads(str(output.attrs["pops_output_manifest"]))
        dataset = _field_dataset(manifest, request)
        np.testing.assert_array_equal(output[dataset][...], expected)
        assert set(output.attrs) == {"pops_output_manifest"}
        assert manifest["snapshot"]["clock"]["time"] == float.hex(0.25)


def test_hdf5_authenticated_reader_rejects_native_dataset_tampering(tmp_path):
    import h5py

    path, request, _expected = _publish(HDF5Writer(), tmp_path / "tampered.h5")
    with h5py.File(path, "r+") as output:
        manifest = json.loads(str(output.attrs["pops_output_manifest"]))
        dataset = _field_dataset(manifest, request)
        output[dataset][0, 0] = np.float64(99.0)

    with pytest.raises(ValueError, match="parallel piece failed verification"):
        read_hdf5(path)


def test_paraview_reopens_with_vtk_without_a_pops_reader(tmp_path):
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    path, _request, expected = _publish(
        ParaViewWriter(collection=False), tmp_path / "native.vtu"
    )

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == 4
    assert grid.GetNumberOfPoints() == 9
    rho = grid.GetCellData().GetArray("rho")
    assert rho is not None
    assert [rho.GetTuple1(index) for index in range(4)] == expected.ravel().tolist()
    assert grid.GetCellData().GetArray("field_0000") is None
    assert [
        grid.GetCellData().GetArray("pops_level").GetTuple1(index)
        for index in range(4)
    ] == [0.0, 0.0, 0.0, 0.0]
    assert grid.GetFieldData().GetArray("TimeValue").GetTuple1(0) == 0.25
