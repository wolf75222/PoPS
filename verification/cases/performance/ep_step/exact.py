"""PF-06 Euler–Poisson step stand-in: segmented toy pipeline timings.

Six pipeline stages plus a total: halo, hyperbolic, charge, poisson,
gradient, source, total. Each pipeline stage has a fake timing > 0.
The total is the exact sum. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

PIPELINE_SEGMENTS = ("halo", "hyperbolic", "charge", "poisson", "gradient", "source")
SEGMENT_NAMES = PIPELINE_SEGMENTS + ("total",)

# Deterministic fake wall times in seconds. Not measured.
_FAKE_SECONDS = {
    "halo": 1.0e-4,
    "hyperbolic": 4.0e-4,
    "charge": 1.5e-4,
    "poisson": 6.0e-4,
    "gradient": 1.2e-4,
    "source": 1.8e-4,
}


def fake_segment_time(name: str) -> float:
    """Return the fake timing of one pipeline stage."""
    if name not in _FAKE_SECONDS:
        raise ValueError(f"unknown pipeline segment {name!r}")
    return float(_FAKE_SECONDS[name])


def pipeline_total(timings) -> float:
    """Return the sum of the six pipeline-stage timings."""
    return sum(float(timings[name]) for name in PIPELINE_SEGMENTS)


def segment_timings() -> dict:
    """Return the seven-segment timing map, with total equal to the sum."""
    timings = {name: fake_segment_time(name) for name in PIPELINE_SEGMENTS}
    timings["total"] = pipeline_total(timings)
    return timings
