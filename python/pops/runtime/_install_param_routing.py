"""Pure runtime-parameter routing cores for the unified install seam (host-testable, no engine call).

Extracted from ``pops.runtime._system_unified_install`` so the seam module fits the Spec-4 line budget.
``route_program_params`` maps a compiled time Program's declared runtime parameters to the per-PROGRAM-
block ``set_program_params`` vectors (ADC-510, Spec 5 C5); ``route_block_params`` is its sibling for the
per-INSTANCE ``set_block_params`` route (P7-b on Uniform, ADC-514 on AMR). Both are re-exposed on
``System`` (``_route_program_params`` / ``_route_block_params``) for host tests.
"""
from __future__ import annotations

from typing import Any


def route_block_params(resolved_models: Any, params: Any) -> Any:
    """Map a flat ``{param_name: value}`` to ``{instance: sorted runtime-param value vector}``.

    The pure core of the per-instance ``set_block_params`` route, SHARED by the Uniform System path
    (P7-b) and the AMR path (ADC-514). @p resolved_models maps each instance name to its RESOLVED model
    (exposing ``runtime_param_names`` / ``runtime_param_values``); a model with no runtime param
    contributes nothing. For each declaring instance, build its COMPLETE value vector (the supplied
    value, else the declaration default) in ``runtime_param_names`` order. Returns ``(per_block,
    unknown)``: per_block only lists instances with params; unknown lists supplied names declared by no
    instance (rejected upstream, no silent drop). No engine call -> host-testable."""
    consumed = set()
    per_block = {}
    for name, model in resolved_models.items():
        rt_names = list(getattr(model, "runtime_param_names", []) or [])
        if not rt_names:
            continue
        values_fn = getattr(model, "runtime_param_values", None)
        raw_defaults: Any = values_fn() if callable(values_fn) else [None] * len(rt_names)
        defaults: Any = list(raw_defaults)
        values = []
        for k, pname in enumerate(rt_names):
            if pname in params:
                values.append(float(params[pname]))
                consumed.add(pname)
            else:
                values.append(float(defaults[k]) if defaults[k] is not None else 0.0)
        per_block[name] = values
    return per_block, sorted(set(params) - consumed)


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
