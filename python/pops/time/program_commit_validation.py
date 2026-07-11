"""Pure validation for atomic endpoint-based Program commit groups."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.time.handles import StateEndpointHandle
from pops.time.references import block_name
from pops.time.values import ProgramValue
from pops.time.program_value_validation import require_compatible_spaces, require_top_level


def validate_commit_many(program: Any, mapping: Any) -> list[tuple[Any, Any]]:
    """Return validated ``(qualified_state, value)`` pairs without mutating ``program``."""
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(
            "commit_many: a non-empty {StateEndpointHandle: ProgramValue} mapping is required")

    validated: list[tuple[Any, Any]] = []
    for endpoint, state in mapping.items():
        if not isinstance(endpoint, StateEndpointHandle):
            raise TypeError(
                "commit_many: every target must be U.next (a StateEndpointHandle); "
                "block-name strings are not public commit targets")
        endpoint = program._require_endpoint(endpoint, "commit_many")
        block = endpoint.block
        if not (isinstance(state, ProgramValue) and state.vtype in ("state", "scalar_field")):
            raise TypeError(
                "commit_many: endpoint for block %r needs a State or scalar_field ProgramValue"
                % (block,))
        if state.prog is not program:
            raise ValueError("commit_many: the State for %r belongs to a different Program" % block)
        require_top_level(program, state, "commit_many")
        require_compatible_spaces(
            endpoint.space, state.space, "commit_many block %r" % block, typed_pair=True)
        if state.block != block:
            raise ValueError(
                "commit_many: endpoint for block %r cannot receive a value owned by block %r"
                % (block_name(block), block_name(state.block)))
        if endpoint.state in program._commits:
            raise ValueError("state %s committed more than once" % endpoint.state.qualified_id)
        validated.append((endpoint.state, state))
    return validated
