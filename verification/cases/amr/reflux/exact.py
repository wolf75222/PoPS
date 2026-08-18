"""AM-09 reflux conservation oracle: closed with reflux, open without.

Ratio-2 coarse-fine face: two fine subfaces restrict to their average.
The leftover storage change equals that flux mismatch. Task 16
``conservation_residual`` is closed only when the reflux term is present.

Does not import pops or read a PoPS output. Does not require a live runtime.
"""
from __future__ import annotations

RATIO = 2
COARSE_FACE_FLUX = 1.0
FINE_FACE_FLUXES = (1.2, 1.6)
OUTWARD_BOUNDARY_FLUX = 0.0
SOURCES = 0.0
PROJECTION = 0.0


def restricted_fine_flux(fine_face_fluxes=FINE_FACE_FLUXES) -> float:
    """Ratio-2 restriction: average of the two fine subfaces on one coarse face."""
    faces = tuple(float(value) for value in fine_face_fluxes)
    if len(faces) != int(RATIO):
        raise ValueError(f"ratio-{RATIO} restriction needs {RATIO} fine faces, got {len(faces)}")
    return 0.5 * (faces[0] + faces[1])


def reflux_correction(
    *,
    coarse_face_flux=COARSE_FACE_FLUX,
    fine_face_fluxes=FINE_FACE_FLUXES,
) -> float:
    """Fine-restricted flux minus the coarse face flux (the CF mismatch)."""
    return restricted_fine_flux(fine_face_fluxes) - float(coarse_face_flux)


def closed_balance_terms(
    *,
    coarse_face_flux=COARSE_FACE_FLUX,
    fine_face_fluxes=FINE_FACE_FLUXES,
) -> dict:
    """Already-reduced closed statement: reflux cancels the storage leftover."""
    reflux = reflux_correction(
        coarse_face_flux=coarse_face_flux,
        fine_face_fluxes=fine_face_fluxes,
    )
    return {
        "storage_change": float(reflux),
        "outward_boundary_flux": float(OUTWARD_BOUNDARY_FLUX),
        "sources": float(SOURCES),
        "reflux": float(reflux),
        "projection": float(PROJECTION),
    }


def open_balance_terms(
    *,
    coarse_face_flux=COARSE_FACE_FLUX,
    fine_face_fluxes=FINE_FACE_FLUXES,
) -> dict:
    """Negative control: same leftover, reflux omitted (open statement)."""
    terms = closed_balance_terms(
        coarse_face_flux=coarse_face_flux,
        fine_face_fluxes=fine_face_fluxes,
    )
    terms["reflux"] = 0.0
    return terms
