"""The Python domain is the sole automatic 1D/2D/3D dimension authority."""

import pytest

from pops._geometry_contracts import (
    CARTESIAN_1D_COORDINATES,
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_3D_COORDINATES,
    CARTESIAN_CELL_AREA,
    CARTESIAN_CELL_LENGTH,
    CARTESIAN_CELL_VOLUME,
)
from pops.domain import CartesianDomain, CartesianDomainFrame
from pops.frames import Cartesian
from pops.mesh import CartesianGrid, PeriodicAxes


@pytest.mark.parametrize(
    ("lower", "upper", "cells", "coordinates", "measure"),
    (
        ((-1.0,), (2.0,), (12,), CARTESIAN_1D_COORDINATES, CARTESIAN_CELL_LENGTH),
        (
            (-1.0, 0.0), (2.0, 4.0), (12, 8),
            CARTESIAN_2D_COORDINATES, CARTESIAN_CELL_AREA,
        ),
        (
            (-1.0, 0.0, 3.0), (2.0, 4.0, 5.0), (12, 8, 4),
            CARTESIAN_3D_COORDINATES, CARTESIAN_CELL_VOLUME,
        ),
    ),
)
def test_domain_vector_rank_selects_one_exact_grid_dimension(
    lower, upper, cells, coordinates, measure,
):
    domain = CartesianDomain("ranked", lower, upper)
    frame = domain.frame()
    grid = CartesianGrid(
        frame=frame,
        cells=cells,
        periodic=PeriodicAxes(frame.axes),
    )

    assert domain.dimension == len(cells)
    assert frame.coordinates == Cartesian(len(cells))
    assert grid.capabilities().to_dict()["dim"] == len(cells)
    assert grid.native_spatial_data()["periodicity"] == [True] * len(cells)
    geometry = grid.normalized_geometry()
    assert geometry.dimension == len(cells)
    assert geometry.coordinate_system == coordinates
    assert geometry.cell_measure == measure
    assert geometry.cells == cells
    assert CartesianGrid.from_dict(grid.to_dict()) == grid


def test_cartesian_domain_rejects_an_independent_or_ambiguous_rank():
    with pytest.raises(ValueError, match="common rank"):
        CartesianDomain("bad", (0.0,), (1.0, 2.0))
    with pytest.raises(ValueError, match="one, two, or three"):
        CartesianDomain("bad", (), ())
    with pytest.raises(ValueError, match="one, two, or three"):
        CartesianDomain("bad", (0.0,) * 4, (1.0,) * 4)


def test_cartesian_domain_frame_authenticates_the_inferred_rank():
    domain = CartesianDomain("line", (0.0,), (1.0,))
    with pytest.raises(ValueError, match="ranks differ"):
        CartesianDomainFrame(domain, Cartesian(2))
