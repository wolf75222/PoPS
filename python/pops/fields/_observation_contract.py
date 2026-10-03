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
    declared = physical.get("field_problem", {}).get("observation_spaces")
    if declared is not None:
        from pops.model import FieldSpace
        if type(value.space) is not FieldSpace or selected >= len(declared) \
                or canonical_bytes(strict_data(value.space.to_data(), where="observed space")) != \
                canonical_bytes(strict_data(declared[selected], where="declared observed space")):
            raise ValueError("field observation changed its declared physical space")
    elif value.space is not None:
        raise ValueError("field observation has no declared physical space authority")

    from pops.codegen.program_field_plan import _nodes, _reachable

    if solve.op == "solve_spatial_field":
        from ._program_nonlinear_problem import validate_nonlinear_field_request
        validate_nonlinear_field_request(value.prog, solve)
        if solve.attrs["field_problem_identity"] != identity.token:
            raise ValueError("field observation belongs to a different original residual")
        return width, selected, solve

    native_sources = tuple(node for node in _reachable(solve, _nodes(value.prog))
                           if node.op in ("field_problem_load", "field_problem_apply"))
    if not native_sources or any(node.attrs.get("field_problem_identity") != identity.token
                                 for node in native_sources):
        raise ValueError("field observation belongs to a different native field problem")
    return width, selected, solve


def validate_field_gradient(value: Any) -> tuple[int, Any]:
    """Authenticate differentiation separately from the consumed potential identity."""
    if value.op != "field_gradient" or value.vtype != "scalar_field" or len(value.inputs) != 1:
        raise ValueError("field gradient requires one consumed scalar observation")
    source = value.inputs[0]
    _width, _selected, solve = validate_field_observation(source)
    dimension = value.attrs.get("spatial_dimension")
    if type(dimension) is not int or dimension not in (1, 2, 3) \
            or value.attrs.get("ncomp") != dimension:
        raise ValueError("field gradient must preserve its exact physical dimension")
    if value.point != source.point or value.region != source.region \
            or value.attrs.get("field_problem_identity") != source.attrs["field_problem_identity"]:
        raise ValueError("field gradient changes its consumed stage or physical problem")
    if value.attrs.get("differentiation") != "cell_centered_second_order" \
            or value.attrs.get("sampling") != "cell":
        raise ValueError("field gradient has no realized differentiation and sampling rule")
    if source.space is not None:
        from pops.model import FieldSpace, PhysicalDimension
        from pops.time._program.serialization import _json_ready
        metadata = _json_ready(solve.attrs["solve_request"]["physical_problem"])
        physical = metadata["field_problem"]
        coordinates = physical.get("coordinate_units", ())
        from pops.time._graph.base import _canonical_int
        axes = tuple(_canonical_int(axis) for axis in metadata.get("observation_axes", ()))
        if len(coordinates) != len(axes) or not axes or any(type(axis) is not int or not 0 <= axis < dimension for axis in axes) \
                or len(set(axes)) != len(axes) or type(value.space) is not FieldSpace:
            raise ValueError("field gradient lost its declared coordinate units")
        units = []
        for axis in range(dimension):
            if axis not in axes:
                units.append(None)
                continue
            coordinate_data = coordinates[axes.index(axis)]
            coordinate = PhysicalDimension.from_data({"kind": coordinate_data["kind"],
                "powers": [[name, _canonical_int(n), _canonical_int(d)]
                           for name, n, d in coordinate_data["powers"]]})
            powers = dict(source.space.units[0].powers)
            for base, exponent in coordinate.powers:
                powers[base] = powers.get(base, 0) - exponent
            units.append(PhysicalDimension(tuple(powers.items())))
        if value.space.units != tuple(units) or value.space.support != source.space.support \
                or value.space.sampling != "cell" or value.space.centering != "cell" \
                or value.space.representation != "field" or len(value.space.components) != dimension:
            raise ValueError("field gradient changed its physical units, support or sampling")
    elif value.space is not None:
        raise ValueError("field gradient has no declared physical space authority")
    return dimension, solve


def validate_field_state_cell_mean(value: Any) -> Any:
    """Recheck the exact solved observation and the cell-mean State witness."""
    from pops.model import StateSpace
    from pops.time.references import canonical_handle

    if value.op != "field_state_cell_mean" or value.vtype != "scalar_field" \
            or len(value.inputs) != 1 or value.attrs.get("projection_version") != 1 \
            or value.attrs.get("ncomp") != 1 or value.attrs.get("sampling") != "cell_average" \
            or value.attrs.get("measure") != "cell_volume":
        raise ValueError("field-to-State projection has no exact cell-mean contract")
    source = value.inputs[0]
    width, selected, solve = validate_field_observation(source)
    if width < 1 or selected >= width or value.prog is not source.prog \
            or value.point != source.point or value.region != source.region \
            or value.attrs.get("field_problem_identity") != source.attrs.get("field_problem_identity") \
            or value.attrs.get("field_unknown") != source.attrs.get("field_unknown"):
        raise ValueError("field-to-State projection changes its consumed observation")
    space = value.space
    if not isinstance(space, StateSpace) or len(space.components) != 1 \
            or space.representation != "conservative" or space.centering != "cell" \
            or space.sampling != "cell_average" or space.support is None \
            or value.state_ref is None or value.block is None \
            or value.attrs.get("target_state") != value.state_ref:
        raise ValueError("field-to-State projection has no qualified cell-mean target")
    load = solve.inputs[1]
    if load.op != "field_problem_load" or not load.inputs \
            or any(item.state_ref != value.state_ref or item.block != value.block
                   or item.space != space for item in load.inputs) \
            or value.attrs.get("source_state") != load.inputs[0].state_ref:
        raise ValueError("field-to-State projection changes its exact load support")
    applies = tuple(node for node in solve.inputs[0].attrs.get("apply_block", ())
                    if node.op == "field_problem_apply")
    if len(applies) != 1 or applies[0].inputs[2].attrs.get("field_dependencies"):
        raise ValueError("field-to-State projection has no constant-coefficient cell-mean stencil")
    # Both handles must survive detachment as the same canonical State identity;
    # no block name or component spelling resolves authority here.
    if canonical_handle(value.state_ref) != canonical_handle(load.inputs[0].state_ref):
        raise ValueError("field-to-State projection changed State identity")
    return source
