"""Exact consumed field-observation authority shared by authoring and emission."""
from __future__ import annotations

from typing import Any

from pops.identity import Identity, canonical_bytes


def validate_field_observation(value: Any) -> tuple[int, int, Any]:
    """Return packed width, selected component and solve after authenticating the observation."""
    if value.op != "field_component" or value.vtype != "scalar_field" \
            or type(value.attrs.get("ncomp")) is not int or value.attrs["ncomp"] != 1 \
            or len(value.inputs) != 1:
        raise ValueError("field observation requires one consumed packed source")
    identity = Identity.from_token(value.attrs.get("field_problem_identity"))
    if identity.domain != "field-problem":
        raise ValueError("field observation lost its physical field-problem identity")
    consumed = value.inputs[0]
    if consumed.op != "solve_outcome_component":
        raise ValueError("field observation requires an explicitly consumed solve result")
    selected, width = value.attrs.get("component"), consumed.attrs.get("ncomp")
    if type(selected) is not int or type(width) is not int or not 0 <= selected < width:
        raise ValueError("field observation component is outside its solved unknown tuple")
    from pops.model import Handle
    from pops.time._program.serialization import _json_ready
    from pops.time._graph.base import strict_data

    unknown = Handle.from_canonical_identity(_json_ready(value.attrs.get("field_unknown")))
    if unknown.kind != "field" or unknown.block_ref is not None:
        raise ValueError("field observation lost its independent solved-field identity")
    try:
        solve = consumed.inputs[0].inputs[0]
        physical = solve.attrs["solve_request"]["physical_problem"]
        expected_unknown = physical["unknown_components"][selected]
    except (IndexError, KeyError, AttributeError, TypeError) as exc:
        raise ValueError("field observation lost its consumed physical solve authority") from exc
    if canonical_bytes(strict_data(unknown.canonical_identity(), where="field unknown")) != \
            canonical_bytes(strict_data(expected_unknown, where="solved field unknown")):
        raise ValueError("field observation does not select its declared solved unknown")
    from pops.codegen.program_field_plan import _nodes, _reachable

    native_sources = tuple(node for node in _reachable(solve, _nodes(value.prog))
                           if node.op in ("field_problem_load", "field_problem_apply"))
    if not native_sources or any(node.attrs.get("field_problem_identity") != identity.token
                                 for node in native_sources):
        raise ValueError("field observation belongs to a different native field problem")
    return width, selected, solve
