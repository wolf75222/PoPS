"""Shared owner, region and state-space provenance checks for Program authoring."""
from __future__ import annotations

from typing import Any

from pops.time.values import ProgramValue, _Affine


TOP_LEVEL_REGION = 0


def structural_state_space(space: Any) -> Any:
    """Return the complete StateSpace behind a State/Rate tag, when known."""
    kind = getattr(space, "kind", None)
    if kind == "state":
        return space
    if kind == "rate":
        return getattr(space, "base_space", None)
    return None


def state_space_name(space: Any) -> Any:
    """Return a State/Rate tag's base state name without claiming structural knowledge."""
    kind = getattr(space, "kind", None)
    if kind == "state":
        return space.name
    if kind == "rate":
        return getattr(space, "base_name", None)
    return None


def state_space_key(space: Any) -> Any:
    """Stable complete structure used internally (stronger than Rate's public equality)."""
    state = structural_state_space(space)
    return state.to_data() if state is not None else None


def rate_space_for(space: Any) -> Any:
    """Return Rate(StateSpace) for a typed state, or None for a legacy untyped state."""
    state = structural_state_space(space)
    if state is None:
        return None
    from pops.model.spaces import Rate
    return Rate(state)


def require_state_space(space: Any, where: str) -> None:
    if space is not None and getattr(space, "kind", None) != "state":
        raise TypeError("%s: space must be a StateSpace or None" % where)


def require_compatible_spaces(left: Any, right: Any, where: str, *, typed_pair: bool = False) -> None:
    """Reject incompatible State/Rate provenance, including same-name structural mismatches."""
    if typed_pair and (left is None) != (right is None):
        raise ValueError("%s: cannot mix typed and untyped state declarations" % where)
    if left is None or right is None:
        return
    left_name, right_name = state_space_name(left), state_space_name(right)
    if left_name != right_name:
        raise ValueError(
            "%s: incompatible state spaces %r and %r" % (where, left_name, right_name))
    left_key, right_key = state_space_key(left), state_space_key(right)
    if left_key is not None and right_key is not None and left_key != right_key:
        raise ValueError(
            "%s: state spaces named %r have incompatible structures (%r != %r)"
            % (where, left_name, left_key, right_key))


def merge_state_spaces(values: Any, where: str) -> Any:
    """Validate State/Rate tags and return the strongest known StateSpace provenance."""
    state_like = [value for value in values if value.vtype in ("state", "rhs")]
    if not state_like or all(value.space is None for value in state_like):
        return None
    if any(value.space is None for value in state_like):
        raise ValueError("%s: cannot mix typed and untyped State/Rate values" % where)
    tags = [value.space for value in state_like]
    first = tags[0]
    for tag in tags[1:]:
        require_compatible_spaces(first, tag, where)
        if structural_state_space(first) is None and structural_state_space(tag) is not None:
            first = tag
    return structural_state_space(first)


def require_owned(program: Any, value: Any, where: str, *, vtype: Any = None) -> ProgramValue:
    if not isinstance(value, ProgramValue):
        raise ValueError("%s: expected a ProgramValue, got %r" % (where, value))
    if value.prog is not program:
        raise ValueError("%s: value %r belongs to a different Program" % (where, value.name))
    if program._issued_values.get(id(value)) is not value:
        raise ValueError("%s: value %r was not authored by this Program" % (where, value.name))
    if vtype is not None and value.vtype != vtype:
        raise ValueError("%s: expected a %s value, got %s" % (where, vtype, value.vtype))
    return value


def require_top_level(program: Any, value: Any, where: str) -> ProgramValue:
    value = require_owned(program, value, where)
    if value.region != TOP_LEVEL_REGION:
        raise ValueError(
            "%s: sub-block value %r cannot escape its authoring region" % (where, value.name))
    return value


def require_region(program: Any, value: Any, region: int, where: str, *, vtype: Any = None,
                   allow: Any = ()) -> Any:
    value = require_owned(program, value, where, vtype=vtype)
    if any(value is candidate for candidate in allow):
        return value
    if value.region != region:
        raise ValueError(
            "%s: callback result %r was not authored in the expected sub-block"
            % (where, value.name))
    return value


def require_affine_region(program: Any, value: Any, region: int, where: str) -> Any:
    """Validate a ProgramValue/_Affine callback result against one exact sub-block region."""
    if isinstance(value, ProgramValue):
        return require_region(program, value, region, where)
    if isinstance(value, _Affine):
        if not value.terms:
            raise ValueError("%s: callback returned an empty affine result" % where)
        for term, _ in value.terms:
            require_region(program, term, region, where)
        return value
    raise ValueError("%s: callback must return a Program field value" % where)


def validate_input_regions(program: Any, inputs: Any, region: int, where: str) -> None:
    """Allow same-region inputs and top-level captures into a sub-block, never the reverse."""
    for value in inputs:
        if not isinstance(value, ProgramValue):
            continue
        require_owned(program, value, where)
        if value.region == region:
            continue
        if region != TOP_LEVEL_REGION and value.region == TOP_LEVEL_REGION:
            continue
        if value.region in program._region_imports.get(region, ()):
            continue
        raise ValueError(
            "%s: value %r from authoring region %s cannot be consumed in region %s"
            % (where, value.name, value.region, region))


def require_declared_state_space(program: Any, block: Any, space: Any) -> None:
    """Enforce one typed-or-untyped StateSpace contract for every declared block."""
    require_state_space(space, "state")
    missing = object()
    prior = program._state_spaces.get(block, missing)
    if prior is missing:
        program._state_spaces[block] = space
        return
    require_compatible_spaces(prior, space, "state block %r" % block, typed_pair=True)


def require_history_space(program: Any, name: str, space: Any) -> None:
    """Enforce one state-space provenance contract for a full-state history ring."""
    missing = object()
    prior = program._history_spaces.get(name, missing)
    if prior is missing:
        program._history_spaces[name] = space
        return
    require_compatible_spaces(prior, space, "history %r" % name, typed_pair=True)


__all__ = [
    "TOP_LEVEL_REGION", "merge_state_spaces", "rate_space_for", "require_affine_region",
    "require_compatible_spaces", "require_declared_state_space", "require_history_space",
    "require_owned", "require_region", "require_state_space", "require_top_level",
    "state_space_key", "state_space_name", "structural_state_space", "validate_input_regions",
]
