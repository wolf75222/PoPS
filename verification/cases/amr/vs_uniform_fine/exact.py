"""AM-07 manufactured spatial error: E ∝ h².

Does not import pops or read a PoPS output. Does not require a live runtime.
"""
from __future__ import annotations

ERROR_CONSTANT = 1.0
FINE_RATIO = 2
H = 1.0 / 16.0
H_SERIES = (1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0)


def fine_spacing(h) -> float:
    """Local spacing on the refined patch: h / FINE_RATIO."""
    return float(h) / float(FINE_RATIO)


def manufactured_error(h) -> float:
    """Manufactured second-order spatial error E = C h²."""
    spacing = float(h)
    return float(ERROR_CONSTANT) * spacing * spacing
