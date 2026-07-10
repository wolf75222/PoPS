"""Structural validation of ProgramValue owner and authoring-region boundaries."""
from __future__ import annotations

from typing import Any

from pops.time.values import ProgramValue
from pops.time.program_value_validation import (
    require_affine_region, require_owned, require_region, require_top_level,
    validate_input_regions,
)


_BLOCK_KEYS = ("cond_block", "body_block", "apply_block", "residual_block")


def _block_region(program: Any, block: Any, where: str, hint: Any = None) -> int:
    if not block:
        if isinstance(hint, int) and hint > 0:
            return hint
        entry = program._recording_regions.get(id(block))
        if entry is None or entry[0] is not block:
            raise ValueError("%s: empty sub-block has no authoring region" % where)
        return entry[1]
    regions = {value.region for value in block}
    if len(regions) != 1 or 0 in regions:
        raise ValueError("%s: recorded nodes must share one non-top-level region" % where)
    region = next(iter(regions))
    if hint is not None and hint != region:
        raise ValueError("%s: recorded region metadata does not match its nodes" % where)
    for value in block:
        require_owned(program, value, where)
        validate_input_regions(program, value.inputs, region, where)
        for key in _BLOCK_KEYS:
            nested = value.attrs.get(key)
            if nested is not None:
                _block_region(
                    program, nested, "%s/%s" % (where, key),
                    value.attrs.get(key.replace("_block", "_region")))
    return region


def validate_program_regions(program: Any) -> None:
    """Fail if a value is foreign, fabricated, or escapes/crosses an undeclared region."""
    for value in program._values:
        require_top_level(program, value, "Program.validate top-level")
        validate_input_regions(program, value.inputs, 0, "Program.validate top-level")
        attrs = value.attrs
        schedule = attrs.get("schedule")
        cond = getattr(schedule, "params", {}).get("cond") if schedule is not None else None
        if isinstance(cond, ProgramValue):
            require_top_level(program, cond, "schedule when(cond)")
        regions = {
            key: _block_region(
                program, attrs[key], "%s %s" % (value.op, key),
                attrs.get(key.replace("_block", "_region")))
            for key in _BLOCK_KEYS if key in attrs and attrs[key] is not None
        }
        if value.op == "while":
            if "cond" in attrs:
                require_region(program, attrs["cond"], regions["cond_block"], "while cond")
            if "body" in attrs:
                require_region(
                    program, attrs["body"], regions["body_block"], "while body",
                    allow=value.inputs)
        elif value.op in ("range", "if"):
            require_region(
                program, attrs["body"], regions["body_block"], "%s body" % value.op,
                allow=value.inputs)
        elif value.op == "matrix_free_operator":
            result = attrs.get("apply_result")
            if result is not None:
                require_affine_region(program, result, regions["apply_block"], "set_apply")
        elif value.op == "solve_local_nonlinear":
            require_region(
                program, attrs["residual"], regions["residual_block"],
                "solve_local_nonlinear residual")
    for state in program._commits.values():
        require_top_level(program, state, "Program.validate commit")
    if program._dt_bound is not None:
        block, result = program._dt_bound
        require_region(program, result, _block_region(program, block, "dt_bound"), "dt_bound")


__all__ = ["validate_program_regions"]
