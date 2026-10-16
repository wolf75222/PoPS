"""Exact keeper and equation-input authority for composite temporal evaluation."""
from collections.abc import Mapping
from pops.time._history.evaluation import TemporalEvaluation
from pops.time._history.transfer import validate_transfer_data, transfer_dimensions
from .spatial_history_source import history_source_contract

CONTRACT = "pops.field-temporal-evaluation@1"


def bind_temporal_field(program, states, dependencies, temporal):
    if temporal is None:
        return None
    if not isinstance(temporal, Mapping) or len(temporal) != len(dependencies):
        raise ValueError("field temporal evaluation must bind every exact equation input")
    from pops.fields._program_problem import _identity
    rows = []
    for source, dependency in zip(states, dependencies):
        selected = [policy for key, policy in temporal.items() if _identity(key) == _identity(dependency)]
        if len(selected) != 1 or type(selected[0]) is not TemporalEvaluation:
            raise ValueError("field temporal evaluation has a missing or foreign input authority")
        policy = selected[0]
        history = program._resolve_history_handle(policy.history)
        seed = history_source_contract(history)
        if (source.state_ref != seed["state"] or source.clock != seed["clock"]
                or source.space != seed["space"]):
            raise ValueError("field temporal evaluation changes its exact State keeper")
        rows.append(dict(history=seed["history"], source=history.id, transfer=policy.transfer.to_data()))
    data = dict(contract=CONTRACT, inputs=tuple(rows))
    validate_temporal_field(program, states, data)
    return data


def validate_temporal_field(program, states, data):
    if (not isinstance(data, Mapping) or set(data) != {"contract", "inputs"}
            or data["contract"] != CONTRACT or not isinstance(data["inputs"], (tuple, list))
            or len(data["inputs"]) != len(states)):
        raise ValueError("invalid field temporal evaluation contract")
    values = {value.id: value for value in program._values}
    for state, row in zip(states, data["inputs"]):
        if not isinstance(row, Mapping) or set(row) != {"history", "source", "transfer"}:
            raise ValueError("invalid field temporal input contract")
        history = values.get(row["source"])
        if history is None:
            raise ValueError("field temporal evaluation lost its keeper source")
        seed = history_source_contract(history)
        if (seed["history"] != row["history"] or seed["state"] != state.state_ref
                or seed["space"] != state.space or seed["clock"] != state.clock):
            raise ValueError("field temporal evaluation has a foreign keeper authority")
        width = program._histories[row["history"]] + 2
        validate_transfer_data(row["transfer"])
        if transfer_dimensions(row["transfer"]) != (
                width, 1, tuple("available_%d" % i for i in range(width))):
            raise ValueError("field temporal evaluation changed its source bundle")
    return data


def is_temporal_field_solve(value):
    return (value.op == "solve_linear" and value.attrs.get("scope") == "hierarchy"
            and value.attrs.get("temporal_evaluation") is not None)
