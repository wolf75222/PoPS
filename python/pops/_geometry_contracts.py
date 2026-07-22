"""Versioned geometry URIs shared by mesh producers and scientific consumers."""
from __future__ import annotations


CARTESIAN_1D_COORDINATES = "pops://coordinates/cartesian-1d@1"
CARTESIAN_2D_COORDINATES = "pops://coordinates/cartesian-2d@1"
CARTESIAN_3D_COORDINATES = "pops://coordinates/cartesian-3d@1"
POLAR_ANNULUS_2D_COORDINATES = "pops://coordinates/polar-annulus-2d@1"
CARTESIAN_CELL_AREA = "pops://cell-measures/cartesian-area@1"
POLAR_ANNULUS_CELL_AREA = "pops://cell-measures/polar-annulus-area@1"


__all__ = [
    "CARTESIAN_1D_COORDINATES",
    "CARTESIAN_2D_COORDINATES",
    "CARTESIAN_3D_COORDINATES",
    "CARTESIAN_CELL_AREA",
    "POLAR_ANNULUS_2D_COORDINATES",
    "POLAR_ANNULUS_CELL_AREA",
]
