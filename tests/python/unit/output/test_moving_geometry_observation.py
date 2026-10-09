"""Physical nodes survive detachment and scientific/archive projections."""
from dataclasses import replace
import numpy as np
import pytest
from tests.python.unit.output.test_post_commit_observers import _frame
from pops.output.observers import detach_observer_frame
from pops.output._observer_archive import encode_observer_frame, decode_observer_frame
from pops.output._writers.common import piece_payload
from pops.output._writers.paraview import _physical_point_coordinates


def moving_frame():
    frame=_frame(cell_shape=(3,))
    nodes=np.array([[0.],[.2],[.65],[1.]])
    geometry=replace(frame.snapshot.geometries[0],
        coordinate_system="pops://coordinates/moving-cartesian-1d@1",
        cell_measure="pops://cell-measures/endpoint-length@1",
        cell_volumes=np.diff(nodes[:,0]),node_coordinates=nodes)
    return replace(frame,snapshot=replace(frame.snapshot,geometries=(geometry,))),nodes


def test_actual_nodes_are_owned_detached_archived_and_written():
    frame,nodes=moving_frame(); nodes[1,0]=.1
    detached=detach_observer_frame(frame)
    geometry=detached.snapshot.geometries[0]
    np.testing.assert_array_equal(geometry.node_coordinates[:,0],[0,.2,.65,1])
    assert not geometry.node_coordinates.flags.writeable
    restored=decode_observer_frame(encode_observer_frame(detached))
    np.testing.assert_array_equal(restored.snapshot.geometries[0].node_coordinates,geometry.node_coordinates)
    arrays,datasets,_=piece_payload(detached.snapshot,detached.request)
    row=next(iter(datasets["geometries"].values()))
    np.testing.assert_array_equal(arrays[row["node_coordinates"]],geometry.node_coordinates)
    points=_physical_point_coordinates(geometry,(np.arange(4),))
    np.testing.assert_array_equal(points[:,0],geometry.node_coordinates[:,0])
    np.testing.assert_array_equal(points[:,1:],0)


def test_moving_geometry_requires_exact_positive_endpoint_measures():
    frame,_=moving_frame(); geometry=frame.snapshot.geometries[0]
    with pytest.raises(ValueError,match="requires explicit"):
        replace(geometry,node_coordinates=None)
    with pytest.raises(ValueError,match="endpoint differences"):
        replace(geometry,cell_volumes=np.full(3,1/3))
    with pytest.raises(ValueError,match="finite shape"):
        replace(geometry,node_coordinates=np.zeros((3,1)))


@pytest.mark.parametrize("coordinate,measure", [
    ("pops://coordinates/moving-cartesian-1d@1", None),
    (None, "pops://cell-measures/endpoint-length@1"),
    ("pops://coordinates/moving-cartesian-1d@1",
     "pops://cell-measures/endpoint-length@1"),
])
def test_versioned_1d_geometry_refuses_rank_two(coordinate, measure):
    geometry = _frame(cell_shape=(2, 2)).snapshot.geometries[0]
    with pytest.raises(ValueError, match="spatial rank one"):
        replace(geometry,
                coordinate_system=coordinate or geometry.coordinate_system,
                cell_measure=measure or geometry.cell_measure,
                node_coordinates=np.zeros((3, 3, 2)))


@pytest.mark.parametrize("attribute,value", [
    ("coordinate_system", "pops://coordinates/cartesian-1d@1"),
    ("cell_measure", "pops://cell-measures/cartesian@1"),
])
def test_versioned_1d_geometry_requires_coherent_pair(attribute, value):
    frame, _ = moving_frame()
    with pytest.raises(ValueError, match="coordinate/measure pair"):
        replace(frame.snapshot.geometries[0], **{attribute: value})


def test_future_geometry_uris_are_not_restricted_to_current_1d_contract():
    geometry = _frame(cell_shape=(2, 2)).snapshot.geometries[0]
    future = replace(geometry,
                     coordinate_system="pops://coordinates/future-curvilinear-2d@1",
                     cell_measure="pops://cell-measures/future-area@1",
                     node_coordinates=np.zeros((3, 3, 2)))
    assert future.spatial_rank == 2
