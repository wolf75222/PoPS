"""Generic ownership and region admission shared by temporal authoring leaves."""
from __future__ import annotations

from typing import Any

from pops.time.value_support import require_owned_value


TOP_LEVEL_REGION = 0


def require_top_level(program: Any, value: Any, where: str) -> Any:
    value = require_owned_value(program, value, where)
    if value.region != TOP_LEVEL_REGION:
        if "problem_identity" in value.attrs:
            from pops.time.solve_request import SolveRequestError

            raise SolveRequestError(
                "result_scope", "solved value cannot escape its authoring region", where=where)
        raise ValueError(
            "%s: sub-block value %r cannot escape its authoring region" % (where, value.name))
    return value


__all__ = ["TOP_LEVEL_REGION", "require_top_level"]
