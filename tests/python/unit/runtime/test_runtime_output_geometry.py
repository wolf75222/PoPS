from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, NormalizedGeometry, PolarMesh, normalize_layout_plan
from pops.mesh._layout_plan_contracts import LayoutLevel
from pops.model import OwnerPath
from pops.runtime._runtime_consumers import RuntimeOutputSnapshot


class _Engine:
    def __init__(self, nx: int, ny: int) -> None:
        self._nx = nx
        self._ny = ny
        self._L = 10_000.0  # legacy private state must not influence normalized output geometry
        self._s = self
        self.geometry_calls = 0
        self.topology_epoch = 0
        self.boxes = ()

    def nx(self) -> int:
        return self._nx

    def ny(self) -> int:
        return self._ny

    def checkpoint_topology_epoch(self) -> int:
        return self.topology_epoch

    def _output_geometry_snapshot(self, *args):
        self.geometry_calls += 1
        if len(args) == 4:
            origin, spacing, shape, cell_measure = args
            level, ratio, boxes = 0, 0, ((0, 0, shape[0], shape[1]),)
        else:
            level, origin, spacing, shape, ratio, cell_measure = args
            boxes = ((0, 0, shape[0], shape[1]),) if level == 0 else tuple(
                (jlo, ilo, jhi + 1, ihi + 1)
                for box_level, ilo, jlo, ihi, jhi in self.boxes
                if box_level == level
            )
        valid = np.zeros(shape, dtype=np.bool_)
        for jlo, ilo, jhi, ihi in boxes:
            valid[jlo:jhi, ilo:ihi] = True
        coverage = np.zeros(shape, dtype=np.bool_)
        if ratio:
            for box_level, ilo, jlo, ihi, jhi in self.boxes:
                if box_level == level + 1:
                    coverage[jlo // ratio:(jhi + 1 + ratio - 1) // ratio,
                             ilo // ratio:(ihi + 1 + ratio - 1) // ratio] = True
        if cell_measure.endswith("cartesian-area@1"):
            volumes = np.full(shape, spacing[0] * spacing[1], dtype=np.float64)
        else:
            radial = origin[0] + np.arange(shape[1], dtype=np.float64) * spacing[0]
            areas = 0.5 * ((radial + spacing[0]) ** 2 - radial ** 2) * spacing[1]
            volumes = np.broadcast_to(areas, shape).copy()
        for value in (valid, coverage, volumes):
            value.setflags(write=False)
        return {
            "topology_epoch": self.topology_epoch,
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


def test_runtime_output_geometry_is_deduplicated_and_invalidated_by_topology_epoch():
    frame = Rectangle("cache", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 4))),
        owner=OwnerPath.case("output-cache"),
    )
    layout = replace(
        plan.layouts[0], adaptive=True, transition_ratios=(2,),
        levels=(LayoutLevel(0, 1), LayoutLevel(1, 2)),
    )
    engine = _Engine(4, 4)
    engine.boxes = ((1, 2, 2, 5, 5),)
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
    )
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: _Engine(nx=4, ny=4),
    )

    with pytest.raises(NotImplementedError, match="does not implement normalized cell measure"):
        RuntimeOutputSnapshot(owner)._geometry(layout, 0)


def test_normalized_geometry_is_rank_generic_but_current_output_provider_refuses_3d():
    geometry = NormalizedGeometry(
        "pops://coordinates/cartesian-3d@1",
        "pops://cell-measures/cartesian-volume@1",
        ("x", "y", "z"), (0.0, -1.0, 2.0), (1.0, 1.0, 5.0), (4, 6, 8),
    )
    assert geometry.dimension == 3
    assert NormalizedGeometry.from_data(geometry.to_data()) == geometry

    frame = Rectangle("rank-gate", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    plan = normalize_layout_plan(
        Uniform(CartesianGrid(frame=frame, cells=(4, 6))),
        owner=OwnerPath.case("rank-gate"),
    )
    layout = replace(plan.layouts[0], geometry=geometry)
    owner = SimpleNamespace(
        _layout_plan=SimpleNamespace(layouts=(layout,)),
        _executor_for_layout=lambda layout_id: _Engine(nx=4, ny=6),
    )

    with pytest.raises(NotImplementedError, match="supports rank-2 geometry"):
        RuntimeOutputSnapshot(owner)._geometry(layout, 0)
