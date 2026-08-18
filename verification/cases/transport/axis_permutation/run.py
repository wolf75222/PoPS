"""TR-06 in-memory axis permutation / reflection. No compile, bind, or pops.run.

Sample the 2-d TR-01 product at mapped coordinates. After the inverse map,
the exact fields are identical (L∞ = 0).
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))


def mapped_permutation_fields(
    n_cells: int = _exact.N_CELLS, t: float = _exact.T
):
    """Return original, swapped-coordinate, and transpose-mapped product fields."""
    x, y, volumes, _axis = _exact.uniform_grid_2d(n_cells)
    swapped_x, swapped_y = _exact.permute_xy(x, y)
    original = _exact.exact_product(
        x, y, t, kx=_exact.KX, ky=_exact.KY, ax=_exact.AX, ay=_exact.AY
    )
    at_swapped = _exact.exact_product(
        swapped_x,
        swapped_y,
        t,
        kx=_exact.KX,
        ky=_exact.KY,
        ax=_exact.AX,
        ay=_exact.AY,
    )
    mapped = at_swapped.T
    return original, at_swapped, mapped, volumes


def mapped_reflection_fields(
    n_cells: int = _exact.N_CELLS, t: float = _exact.T
):
    """Return original, x-reflected, and axis-0-flipped product fields."""
    x, y, volumes, _axis = _exact.uniform_grid_2d(n_cells)
    original = _exact.exact_product(
        x, y, t, kx=_exact.KX, ky=_exact.KY, ax=_exact.AX, ay=_exact.AY
    )
    at_reflected = _exact.exact_product(
        _exact.reflect_x(x),
        y,
        t,
        kx=_exact.KX,
        ky=_exact.KY,
        ax=_exact.AX,
        ay=_exact.AY,
    )
    mapped = at_reflected[::-1, :]
    return original, at_reflected, mapped, volumes


def permutation_linf(n_cells: int = _exact.N_CELLS, t: float = _exact.T) -> float:
    """Return L∞ of the transpose-mapped permutation identity."""
    original, _at_swapped, mapped, volumes = mapped_permutation_fields(
        n_cells, t
    )
    return float(reference_errors(mapped, original, volumes).linf)


def reflection_linf(n_cells: int = _exact.N_CELLS, t: float = _exact.T) -> float:
    """Return L∞ of the x-flipped reflection identity."""
    original, _at_reflected, mapped, volumes = mapped_reflection_fields(
        n_cells, t
    )
    return float(reference_errors(mapped, original, volumes).linf)
