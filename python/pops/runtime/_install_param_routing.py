"""Pure runtime-parameter routing cores for the unified install seam (host-testable, no engine call).

Extracted from ``pops.runtime._system_unified_install`` so the seam module fits the Spec-4 line budget.
``route_program_params`` maps a compiled time Program's declared runtime parameters to the per-PROGRAM-
block ``set_program_params`` vectors (ADC-510, Spec 5 C5); it is the mirror of ``_route_block_params``
(the AOT-block ``set_block_params`` route, P7-b) and is re-exposed as
``System._route_program_params`` for host tests.
"""
from __future__ import annotations

from typing import Any


def route_program_params(routes: Any, defaults: Any, params: Any) -> Any:
    """Map a flat ``{param_name: value}`` to the per-PROGRAM-block ``set_program_params`` vectors.

    @p routes maps a program block index to its param names in within-block index order; @p defaults
    maps a name to its declaration value; @p params the supplied values. Returns ``(per_block,
    unknown)``: per_block maps a PROGRAM block index to its COMPLETE value vector (the supplied value,
    else the declaration default); unknown lists the supplied names declared by no Program kernel
    (rejected upstream, no silent drop)."""
    consumed = set()
    per_block = {}
    for blk, names in routes.items():
        per_block[blk] = [float(params[n]) if n in params else float(defaults.get(n, 0.0))
                          for n in names]
        consumed |= {n for n in names if n in params}
    return per_block, sorted(set(params) - consumed)
