"""Versioned geometry URIs shared by mesh producers and scientific consumers."""
from __future__ import annotations


CARTESIAN_1D_COORDINATES = "pops://coordinates/cartesian-1d@1"
CARTESIAN_2D_COORDINATES = "pops://coordinates/cartesian-2d@1"
CARTESIAN_3D_COORDINATES = "pops://coordinates/cartesian-3d@1"
POLAR_ANNULUS_2D_COORDINATES = "pops://coordinates/polar-annulus-2d@1"
CARTESIAN_CELL_LENGTH = "pops://cell-measures/cartesian-length@1"
CARTESIAN_CELL_AREA = "pops://cell-measures/cartesian-area@1"
CARTESIAN_CELL_VOLUME = "pops://cell-measures/cartesian-volume@1"
POLAR_ANNULUS_CELL_AREA = "pops://cell-measures/polar-annulus-area@1"

CARTESIAN_COORDINATES_BY_DIMENSION = (
    CARTESIAN_1D_COORDINATES,
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_3D_COORDINATES,
)
CARTESIAN_CELL_MEASURES_BY_DIMENSION = (
    CARTESIAN_CELL_LENGTH,
    CARTESIAN_CELL_AREA,
    CARTESIAN_CELL_VOLUME,
)


def cartesian_geometry_contract(dimension: int) -> tuple[str, str]:
    """Return the exact coordinate/measure URIs for one compiled Cartesian rank."""
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("Cartesian geometry dimension must be an exact integer")
    if dimension not in (1, 2, 3):
        raise ValueError("Cartesian geometry dimension must be 1, 2, or 3")
    index = dimension - 1
    return (
        CARTESIAN_COORDINATES_BY_DIMENSION[index],
        CARTESIAN_CELL_MEASURES_BY_DIMENSION[index],
    )


__all__ = [
    "CARTESIAN_1D_COORDINATES",
    "CARTESIAN_2D_COORDINATES",
    "CARTESIAN_3D_COORDINATES",
    "CARTESIAN_CELL_LENGTH",
    "CARTESIAN_CELL_AREA",
    "CARTESIAN_CELL_VOLUME",
    "CARTESIAN_COORDINATES_BY_DIMENSION",
    "CARTESIAN_CELL_MEASURES_BY_DIMENSION",
    "cartesian_geometry_contract",
    "POLAR_ANNULUS_2D_COORDINATES",
    "POLAR_ANNULUS_CELL_AREA",
]
