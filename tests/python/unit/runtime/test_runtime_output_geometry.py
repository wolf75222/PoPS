from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pops._geometry_contracts import cartesian_geometry_contract
from pops.domain import CartesianDomain, Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, NormalizedGeometry, PolarMesh, normalize_layout_plan
from pops.mesh._layout_plan_contracts import LayoutLevel
from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import FieldKey
from pops.output._consumer_contracts import ParallelMode
from pops.runtime._runtime_consumers import (
    _active_output_levels,
    _native_cartesian_geometry,
    RuntimeOutputSnapshot,
)


@pytest.mark.parametrize("shape", ((5,), (4, 3), (4, 3, 2)))
def test_native_cartesian_geometry_contract_accepts_every_compiled_rank(shape):
    coordinate_system, cell_measure = cartesian_geometry_contract(len(shape))
    geometry = NormalizedGeometry(
        coordinate_system,
        cell_measure,
        ("x", "y", "z")[:len(shape)],
        (0.0,) * len(shape),
        (1.0,) * len(shape),
        shape,
    )

    assert _native_cartesian_geometry(geometry)
    assert not _native_cartesian_geometry(replace(
        geometry, cell_measure="pops://cell-measures/not-cartesian@1"
    ))


class _Engine:
    def __init__(
        self,
        nx: int | None = None,
        ny: int | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        if shape is None:
            if nx is None or ny is None:
                raise TypeError("the test engine requires nx/ny or one ranked shape")
            shape = (nx, ny)
        self._shape = tuple(shape)
        self._nx = self._shape[0]
        self._ny = self._shape[1] if len(self._shape) > 1 else 1
        self._L = 10_000.0  # legacy private state must not influence normalized output geometry
        self._s = self
        self.geometry_calls = 0
        self.geometry_shapes = []
        self.topology_epoch = 0
        self.boxes = ()
        self.active_levels = 1
        self.level_refinements = {}

    def nx(self) -> int:
        return self._nx

    def ny(self) -> int:
        return self._ny

    def checkpoint_topology_epoch(self) -> int:
        return self.topology_epoch

    def n_levels(self) -> int:
        return self.active_levels

    def _output_geometry_snapshot(self, *args):
        self.geometry_calls += 1
        if len(args) == 4:
            origin, spacing, shape, cell_measure = args
            level, ratio = 0, (0,) * len(shape)
            materialized = ((tuple(0 for _ in shape), tuple(item - 1 for item in shape)),)
        else:
            level, origin, spacing, shape, ratio, cell_measure = args
            materialized = (
                ((tuple(0 for _ in shape), tuple(item - 1 for item in shape)),)
                if level == 0
                else tuple(
                    (lower, upper)
                    for box_level, lower, upper in self.boxes
                    if box_level == level
                )
            )
        shape = tuple(shape)
        self.geometry_shapes.append(shape)
        refinement = self.level_refinements.get(
            level, (2 ** level,) * len(self._shape)
        )
        assert shape == tuple(
            item * refinement[axis] for axis, item in enumerate(self._shape)
        )
        cell_shape = tuple(reversed(shape))
        boxes = tuple(
            tuple(reversed(lower)) + tuple(item + 1 for item in reversed(upper))
            for lower, upper in materialized
        )
        valid = np.zeros(cell_shape, dtype=np.bool_)
        for box in boxes:
            valid[tuple(slice(low, high) for low, high in zip(
                box[:len(shape)], box[len(shape):], strict=True
            ))] = True
        coverage = np.zeros(cell_shape, dtype=np.bool_)
        if any(ratio):
            for box_level, lower, upper in self.boxes:
                if box_level == level + 1:
                    parent_lower = tuple(
                        item // ratio[axis] for axis, item in enumerate(lower)
                    )
                    parent_upper = tuple(
                        item // ratio[axis] + 1 for axis, item in enumerate(upper)
                    )
                    coverage[tuple(slice(low, high) for low, high in zip(
                        reversed(parent_lower), reversed(parent_upper), strict=True
                    ))] = True
        if cell_measure in {
            "pops://cell-measures/cartesian-length@1",
            "pops://cell-measures/cartesian-area@1",
            "pops://cell-measures/cartesian-volume@1",
        }:
            volumes = np.full(cell_shape, math.prod(spacing), dtype=np.float64)
        else:
            radial = origin[0] + np.arange(shape[0], dtype=np.float64) * spacing[0]
            areas = 0.5 * ((radial + spacing[0]) ** 2 - radial ** 2) * spacing[1]
            volumes = np.broadcast_to(areas, cell_shape).copy()
        for value in (valid, coverage, volumes):
            value.setflags(write=False)
        return {
            "dimension": len(shape),
            "topology_epoch": self.topology_epoch,
            "cell_shape": cell_shape,
            "boxes": boxes,
            "valid_cells": valid,
            "coverage": coverage,
            "cell_volumes": volumes,
        }


def _geometry(descriptor, engine):
    plan = normalize_layout_plan(descriptor, owner=OwnerPath.case("output-geometry"))
    layout = plan.layouts[0]
    owner = SimpleNamespace(
        _layout_plan=plan,
        _executor_for_layout=lambda layout_id: engine,
    )
    return RuntimeOutputSnapshot(owner)._geometry(layout, 0)


def test_runtime_output_uses_exact_rectangular_cartesian_geometry():
    frame = Rectangle("shifted", (1.0, -2.0), (5.0, 4.0)).frame(Cartesian2D())
    result = _geometry(
        Uniform(CartesianGrid(frame=frame, cells=(4, 3))),
        _Engine(nx=4, ny=3),
    )

    assert result.origin == (1.0, -2.0)
    assert result.coordinate_system == "pops://coordinates/cartesian-2d@1"
    assert result.cell_measure == "pops://cell-measures/cartesian-area@1"
    assert result.axis_names == ("x", "y")
    assert result.spacing == (1.0, 2.0)
    assert result.cell_shape == (3, 4)
    np.testing.assert_array_equal(result.cell_volumes, np.full((3, 4), 2.0))


@pytest.mark.parametrize(
    (
        "lower", "upper", "cells", "expected_shape", "expected_spacing", "expected_volume",
        "coordinate_system", "cell_measure", "axis_names",
    ),
    (
        (
            (-2.0,), (3.0,), (5,),
            (5,), (1.0,), 1.0,
            "pops://coordinates/cartesian-1d@1",
            "pops://cell-measures/cartesian-length@1",
            ("x",),
        ),
        (
            (1.0, -2.0), (5.0, 4.0), (4, 3),
            (3, 4), (1.0, 2.0), 2.0,
            "pops://coordinates/cartesian-2d@1",
            "pops://cell-measures/cartesian-area@1",
            ("x", "y"),
        ),
        (
            (0.0, -1.0, 2.0), (1.0, 1.0, 5.0), (4, 6, 8),
            (8, 6, 4), (0.25, 1.0 / 3.0, 0.375), 0.03125,
            "pops://coordinates/cartesian-3d@1",
            "pops://cell-measures/cartesian-volume@1",
            ("x", "y", "z"),
        ),
    ),
)
def test_runtime_output_uses_ranked_cartesian_domain_geometry(
    lower, upper, cells, expected_shape, expected_spacing, expected_volume,
    coordinate_system, cell_measure, axis_names,
):
    frame = CartesianDomain("ranked", lower, upper).frame()
    result = _geometry(
        Uniform(CartesianGrid(frame=frame, cells=cells)),
        _Engine(shape=cells),
    )

    assert result.origin == lower
    assert result.coordinate_system == coordinate_system
    assert result.cell_measure == cell_measure
    assert result.axis_names == axis_names
    assert result.spacing == expected_spacing
    assert result.cell_shape == expected_shape
    np.testing.assert_array_equal(
        result.cell_volumes, np.full(expected_shape, expected_volume)
    )


def test_runtime_output_geometry_is_deduplicated_and_invalidated_by_topology_epoch():
    frame = Rectangle("cache", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 4))),
        owner=OwnerPath.case("output-cache"),
    )
    layout = replace(
        plan.layouts[0], adaptive=True, transition_ratios=((2, 2),),
        levels=(LayoutLevel(0, (1, 1)), LayoutLevel(1, (2, 2))),
        capabilities={
            **dict(plan.layouts[0].capabilities),
            "supports_amr": True,
            "transition_ratios": [[2, 2]],
            "max_levels": 2,
        },
    )
    engine = _Engine(4, 4)
    engine.boxes = ((1, (2, 2), (5, 5)),)
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: engine,
    )
    builder = RuntimeOutputSnapshot(owner)

    first = builder._geometry(layout, 0)
    second = builder._geometry(layout, 0)
    assert second is first
    assert second.coverage is first.coverage
    assert engine.geometry_calls == 1

    engine.topology_epoch = 1
    third = builder._geometry(layout, 0)
    assert third is not first
    assert engine.geometry_calls == 2
    assert tuple(builder._geometry_cache) == ((third.layout_identity.token, 0, 1),)

    # Restart can reuse an old epoch number for a different accepted hierarchy.
    engine.boxes = ((1, (0, 0), (3, 3)),)
    builder.invalidate_geometry_cache()
    fourth = builder._geometry(layout, 0)
    assert fourth is not third
    assert engine.geometry_calls == 3


def test_runtime_output_intersects_authored_levels_with_live_amr_depth():
    frame = Rectangle("dynamic-depth", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 4))),
        owner=OwnerPath.case("dynamic-depth-output"),
    )
    layout = replace(
        plan.layouts[0],
        adaptive=True,
        transition_ratios=((2, 2), (2, 2)),
        levels=(
            LayoutLevel(0, (1, 1)),
            LayoutLevel(1, (2, 2)),
            LayoutLevel(2, (4, 4)),
        ),
        capabilities={
            **dict(plan.layouts[0].capabilities),
            "supports_amr": True,
            "transition_ratios": [[2, 2], [2, 2]],
            "max_levels": 3,
        },
    )
    engine = _Engine(4, 4)
    engine.active_levels = 2
    owner = SimpleNamespace(
        _executor_for_layout=lambda layout_id: engine,
    )

    assert _active_output_levels(owner, layout, (0, 1, 2)) == (0, 1)
    with pytest.raises(RuntimeError, match="no currently active AMR level"):
        _active_output_levels(owner, layout, (2,))

    engine.active_levels = 4
    with pytest.raises(RuntimeError, match="outside the configured"):
        _active_output_levels(owner, layout, (0, 1, 2))


def test_runtime_output_accepts_adaptive_rectangular_geometry():
    frame = Rectangle("adaptive-rectangle", (0.0, -1.0), (2.0, 2.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 3))),
        owner=OwnerPath.case("adaptive-rectangle-output"),
    )
    layout = replace(
        plan.layouts[0], adaptive=True, transition_ratios=((2, 2),),
        levels=(LayoutLevel(0, (1, 1)), LayoutLevel(1, (2, 2))),
        capabilities={
            **dict(plan.layouts[0].capabilities),
            "supports_amr": True,
            "transition_ratios": [[2, 2]],
            "max_levels": 2,
        },
    )
    engine = _Engine(4, 3)
    engine.boxes = ((1, (2, 1), (5, 4)),)
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: engine,
    )

    geometry = RuntimeOutputSnapshot(owner)._geometry(layout, 1)

    assert geometry.cell_shape == (6, 8)
    assert geometry.spacing == (0.25, 0.5)
    assert geometry.origin == (0.0, -1.0)


def test_runtime_output_composite_integral_forwards_exact_levels_to_native_reducer():
    calls = []

    class _Provider:
        def composite_reduce(self, *args):
            calls.append(args)
            return 3.25

    key = FieldKey(
        Handle(
            "rho", kind="state",
            owner=OwnerPath.case("native-integral").child(OwnerKind.BLOCK, "fluid"),
        ),
        make_identity("component-manifest", {"name": "scalar"}),
        make_identity("layout-plan", {"name": "amr"}),
        0,
        "accepted",
    )
    evidence = RuntimeOutputSnapshot._native_composite_integral({
        "native_engine": _Provider(),
        "reduction_method": "composite_reduce",
        "reduction_args": ("fluid", "sum", 0, [0, 2]),
        "reduction_levels": (0, 2),
    }, key)
    assert evidence is not None
    assert evidence.levels == (0, 2)
    assert evidence.value == 3.25
    assert calls == [("fluid", "sum", 0, [0, 2])]


def test_runtime_output_uses_exact_polar_annulus_cell_areas():
    result = _geometry(
        Uniform(PolarMesh(r_min=1.0, r_max=3.0, nr=4, ntheta=8)),
        _Engine(nx=4, ny=8),
    )

    assert result.origin == (1.0, 0.0)
    assert result.coordinate_system == "pops://coordinates/polar-annulus-2d@1"
    assert result.cell_measure == "pops://cell-measures/polar-annulus-area@1"
    assert result.axis_names == ("r", "theta")
    assert result.spacing == (0.5, math.tau / 8.0)
    assert result.cell_shape == (8, 4)
    assert not np.all(result.cell_volumes == result.cell_volumes[0, 0])
    np.testing.assert_allclose(
        np.sum(result.cell_volumes), math.pi * (3.0 ** 2 - 1.0 ** 2)
    )


def test_runtime_output_refuses_unknown_extension_cell_measure():
    frame = Rectangle("extension", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 4))),
        owner=OwnerPath.case("extension-output-geometry"),
    )
    layout = replace(
        plan.layouts[0],
        geometry=NormalizedGeometry(
            "pops://coordinates/extension-2d@1",
            "pops://cell-measures/extension-area@1",
            ("a", "b"), (0.0, 0.0), (1.0, 1.0), (4, 4),
        ),
        native_spatial_layout=None,
    )
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: _Engine(nx=4, ny=4),
    )

    with pytest.raises(NotImplementedError, match="does not implement normalized cell measure"):
        RuntimeOutputSnapshot(owner)._geometry(layout, 0)


def _generic_layout(geometry: NormalizedGeometry, *, owner: str):
    frame = Rectangle(owner, (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 4))),
        owner=OwnerPath.case(owner),
    )
    return replace(
        plan.layouts[0],
        geometry=geometry,
        levels=(LayoutLevel(0, (1,) * geometry.dimension),),
        native_spatial_layout=None,
    )


@pytest.mark.parametrize(
    ("geometry", "expected_shape", "expected_spacing", "expected_volume"),
    (
        (
            NormalizedGeometry(
                "pops://coordinates/cartesian-1d@1",
                "pops://cell-measures/cartesian-length@1",
                ("x",), (-2.0,), (3.0,), (5,),
            ),
            (5,),
            (1.0,),
            1.0,
        ),
        (
            NormalizedGeometry(
                "pops://coordinates/cartesian-3d@1",
                "pops://cell-measures/cartesian-volume@1",
                ("x", "y", "z"),
                (0.0, -1.0, 2.0),
                (1.0, 1.0, 5.0),
                (4, 6, 8),
            ),
            (8, 6, 4),
            (0.25, 1.0 / 3.0, 0.375),
            0.03125,
        ),
    ),
)
def test_runtime_output_infers_rank_from_normalized_geometry(
    geometry, expected_shape, expected_spacing, expected_volume
):
    assert NormalizedGeometry.from_data(geometry.to_data()) == geometry
    layout = _generic_layout(geometry, owner="ranked-output")
    engine = _Engine(shape=geometry.cells)
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: engine,
    )

    result = RuntimeOutputSnapshot(owner)._geometry(layout, 0)

    assert result.spatial_rank == geometry.dimension
    assert result.cell_shape == expected_shape
    assert result.spacing == expected_spacing
    assert result.axis_names == geometry.axis_names
    assert result.boxes == ((0,) * geometry.dimension + expected_shape,)
    np.testing.assert_array_equal(
        result.cell_volumes,
        np.full(expected_shape, expected_volume, dtype=np.float64),
    )
    assert engine.geometry_shapes == [geometry.cells]


def test_runtime_output_preserves_rank_three_amr_patch_bounds():
    geometry = NormalizedGeometry(
        "pops://coordinates/cartesian-3d@1",
        "pops://cell-measures/cartesian-volume@1",
        ("x", "y", "z"),
        (0.0, -1.0, 2.0),
        (4.0, 2.0, 4.0),
        (4, 3, 2),
    )
    base_layout = _generic_layout(geometry, owner="rank-three-amr")
    layout = replace(
        base_layout,
        adaptive=True,
        transition_ratios=((2, 1, 3),),
        levels=(LayoutLevel(0, (1, 1, 1)), LayoutLevel(1, (2, 1, 3))),
        capabilities={
            **dict(base_layout.capabilities),
            "supports_amr": True,
            "transition_ratios": [[2, 1, 3]],
            "max_levels": 2,
        },
    )
    engine = _Engine(shape=geometry.cells)
    engine.level_refinements[1] = (2, 1, 3)
    engine.boxes = ((1, (2, 0, 0), (5, 2, 5)),)
    engine.active_levels = 2
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: engine,
    )

    result = RuntimeOutputSnapshot(owner)._geometry(layout, 1)

    assert result.cell_shape == (6, 3, 8)
    assert result.boxes == ((0, 0, 2, 6, 3, 6),)
    assert int(np.count_nonzero(result.valid_cells)) == 4 * 3 * 6


def test_runtime_output_rejects_native_rank_that_differs_from_compiled_geometry():
    geometry = NormalizedGeometry(
        "pops://coordinates/cartesian-3d@1",
        "pops://cell-measures/cartesian-volume@1",
        ("x", "y", "z"),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2, 3, 4),
    )

    class _WrongRankEngine(_Engine):
        def _output_geometry_snapshot(self, *args):
            result = super()._output_geometry_snapshot(*args)
            result["dimension"] = 2
            return result

    layout = _generic_layout(geometry, owner="rank-mismatch")
    engine = _WrongRankEngine(shape=geometry.cells)
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: engine,
    )

    with pytest.raises(ValueError, match="rank differs"):
        RuntimeOutputSnapshot(owner)._geometry(layout, 0)


@pytest.mark.parametrize(
    ("lower", "upper"),
    (
        ((2,), (7,)),
        ((1, 2, 3), (3, 5, 7)),
    ),
)
def test_runtime_output_piece_lowering_uses_the_geometry_rank(lower, upper):
    dimension = len(lower)
    spatial_shape = tuple(high - low for low, high in zip(lower, upper, strict=True))
    values = np.arange(2 * math.prod(spatial_shape), dtype=np.float64).reshape(
        (2, *spatial_shape)
    )

    class _PieceProvider:
        @staticmethod
        def pieces():
            return (
                {
                    "lower": lower,
                    "upper": upper,
                    "values": values,
                    "global_box_index": 0,
                    "owner_rank": 0,
                    "replicated": False,
                },
            )

    pieces = RuntimeOutputSnapshot._local_pieces(
        _PieceProvider(),
        "pieces",
        (),
        mode=ParallelMode.SERIAL,
        rank=0,
        dimension=dimension,
    )

    assert pieces[0].lower == lower
    assert pieces[0].upper == upper
    assert pieces[0].values.shape == (2, *spatial_shape)
    RuntimeOutputSnapshot._validate_piece_bounds(
        pieces,
        (lower + upper,),
        dimension=dimension,
        complete=True,
        rank=0,
    )


def test_runtime_output_piece_lowering_rejects_a_bound_from_another_rank():
    class _PieceProvider:
        @staticmethod
        def pieces():
            return (
                {
                    "lower": (0, 0),
                    "upper": (2, 3),
                    "values": np.zeros((1, 2, 3), dtype=np.float64),
                    "global_box_index": 0,
                    "owner_rank": 0,
                    "replicated": False,
                },
            )

    with pytest.raises(TypeError, match="3 exact integer bounds"):
        RuntimeOutputSnapshot._local_pieces(
            _PieceProvider(),
            "pieces",
            (),
            mode=ParallelMode.SERIAL,
            rank=0,
            dimension=3,
        )
