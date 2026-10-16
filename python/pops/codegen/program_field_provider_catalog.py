"""Install scientific Field producers independently of solve invocation identities."""

from __future__ import annotations

import json
from typing import Any

from pops._structural_data import canonical_json
from pops.fields._observation_contract import validate_field_gradient, validate_field_observation
from pops.time.canonical_data import _json_ready


def field_provider_contract(value: Any) -> str:
    """Authenticate the selected unknown before producing the exact codec key."""
    if value.op == "field_gradient":
        validate_field_gradient(value)
    else:
        validate_field_observation(value)
    return canonical_json(_json_ready(value.attrs["field_authority"]))


def emit_field_provider_catalog(program: Any, *, target: str) -> list[str]:
    """Run after solver installation, before initialization or checkpoint decoding."""
    from .program_lowerability import all_ops

    rows: set[tuple[str, int, int, int]] = set()
    for value in all_ops(program):
        if value.op not in ("field_component", "field_gradient"):
            continue
        if value.op == "field_gradient":
            components, solve = validate_field_gradient(value)
            _width, component, _source_solve = validate_field_observation(value.inputs[0])
        else:
            _width, component, solve = validate_field_observation(value)
            components = 1
        contract = field_provider_contract(value)
        # An AMR solve id is an exact native installation route, never the
        # persistent scientific producer key. Preserve every distinct route.
        rows.add((contract, solve.id if target == "amr_system" else -1, component, components))
    if rows and target not in ("system", "amr_system"):
        raise ValueError("independent Field catalog requires a native Program context")
    if target == "amr_system":
        return ["ctx.configure_hierarchy_continuation_field_provider(%s, %d, %d, %d, 1);"
                % (json.dumps(contract), solve, component, components)
                for contract, solve, component, components in sorted(rows)]
    return ["ctx.configure_continuation_field_provider(%s, %d, 1);" % (json.dumps(contract), components)
            for contract, components in sorted({(row[0], row[3]) for row in rows})]
