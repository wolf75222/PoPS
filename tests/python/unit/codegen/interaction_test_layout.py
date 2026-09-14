"""Explicit two-band Cartesian descriptor for the predeclared MPI interaction matrix."""
import os

from pops.mesh.grid import CartesianGrid
from tests.python.support.layout_plan import cartesian_grid


class _TwoBands(CartesianGrid):
    __slots__ = ()

    def native_spatial_data(self):
        data = super().native_spatial_data()
        n, ny = self.cells
        if n % 2:
            raise ValueError("two-band qualification requires even x cells")
        data["decomposition"] = {
            "schema_version": 1, "kind": "axis_bands", "axis": 0,
            "boxes": [{"lower": [0, 0], "upper_exclusive": [n // 2, ny]},
                      {"lower": [n // 2, 0], "upper_exclusive": [n, ny]}],
        }
        return data

    def to_dict(self):
        return {**super().to_dict(), "test_decomposition": self.native_spatial_data()["decomposition"]}

    canonical_identity = to_dict


def interaction_grid(*, n, periodic=True):
    grid = cartesian_grid(n=n, periodic=periodic)
    bands = os.environ.get("POPS_TEST_INTERACTION_BANDS", "1")
    if bands == "1":
        return grid
    if bands != "2":
        raise ValueError("interaction qualification bands must be explicitly 1 or 2")
    return _TwoBands(frame=grid.frame, cells=grid.cells, periodic=grid.periodic)


def require_interaction_partition(simulation, *, n, context):
    """Check the declared serial or MPI partition using the bound launch context."""
    from pops._native_collectives import allgather_value, require_world

    bands = os.environ.get("POPS_TEST_INTERACTION_BANDS", "1")
    if bands not in ("1", "2"):
        raise ValueError("interaction qualification bands must be explicitly 1 or 2")
    if context.communicator.identity == "serial":
        assert bands == "1", "two-rank interaction qualification requires MPI"
        assert context.communicator.handle is None
        boxes = simulation.local_boxes("left")
        cells = sum((int(hi[0])-int(lo[0])) * (int(hi[1])-int(lo[1])) for lo, hi in boxes)
        assert cells == n*n
        return None
    assert context.communicator.identity == "MPI_COMM_WORLD"
    world = require_world(context.communicator.handle)
    if bands == "1":
        return world
    assert world.size == 2
    boxes = simulation.local_boxes("left")
    cells = sum((int(hi[0])-int(lo[0])) * (int(hi[1])-int(lo[1])) for lo, hi in boxes)
    owns_bad = any(all(int(lo[d]) <= n-1 < int(hi[d]) for d in (0, 1)) for lo, hi in boxes)
    assert allgather_value(world, cells) == (n*n//2, n*n//2)
    assert allgather_value(world, owns_bad) == (False, True)
    return world
