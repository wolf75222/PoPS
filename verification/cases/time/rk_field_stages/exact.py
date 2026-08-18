"""TM-07 field-solve contract: one elliptic update per required RK stage.

Does not import pops or read a PoPS output. Does not require a live runtime.
"""
from __future__ import annotations

SSPRK2_STAGES = 2
SSPRK3_STAGES = 3
FROZEN_FIELD_SOLVES_PER_STEP = 1

INTEGRATOR_STAGES = {
    "SSPRK2": SSPRK2_STAGES,
    "SSPRK3": SSPRK3_STAGES,
}


def stage_count(integrator: str) -> int:
    """Documented stage count of an explicit SSP integrator."""
    try:
        return INTEGRATOR_STAGES[integrator]
    except KeyError as exc:
        raise ValueError(f"unknown integrator {integrator!r}") from exc


def required_field_solves(stages: int) -> int:
    """Coupled field updates required per step equal the stage count."""
    return int(stages)
