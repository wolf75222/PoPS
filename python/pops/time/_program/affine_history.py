"""Closed retained-history source contracts for affine parent acceptance."""
from collections.abc import Mapping

PARAMETERS = ("macro_start", "source_duration", "accepted_duration", "scale")


def _has_state_endpoints(program, name):
    from pops.time.points import TimePoint
    # Merely sharing a StateSpace does not turn an observation or a stored RHS
    # into a State sample. Only keep_history certifies its exact State.n seed.
    matches = [(state, store) for state, store in program._time_history_stores.items()
               if store.attrs.get("history") == name]
    if len(matches) != 1:
        return False
    state, store = matches[0]
    if store.op != "store_history" or len(store.inputs) != 1:
        return False
    seed = program._canonical_value(store.inputs[0])
    return (seed.op == "state" and seed.vtype == "state"
            and seed.point == TimePoint(state.clock, 0)
            and seed.space == state.space == program._history_spaces.get(name)
            and seed.state_ref == state.state == program._history_state_refs.get(name)
            and seed.block == state.block == program._history_blocks.get(name))


def history_contract(program, histories):
    from pops.time._history.transfer import HistoryTransfer
    if histories is None:
        histories = {}
    if not isinstance(histories, Mapping):
        raise TypeError("accept_increment histories must map each retained ring to a HistoryTransfer")
    rows = {}
    for name, transfer in histories.items():
        if type(name) is not str or type(transfer) is not HistoryTransfer:
            raise TypeError("accept_increment histories require named closed HistoryTransfer equations")
        rows[name] = {"transfer": transfer.to_data(),
                      "state_endpoints": _has_state_endpoints(program, name)}
    validate_history_contract(program, rows)
    return rows


def validate_history_contract(program, rows):
    from pops.time._history.transfer import validate_transfer_data, transfer_dimensions
    if not isinstance(rows, Mapping) or set(rows) != set(program._histories):
        raise ValueError("affine acceptance requires one explicit transfer for every retained history")
    for name, row in rows.items():
        if set(row) != {"transfer", "state_endpoints"}:
            raise ValueError("affine history transfer has a foreign contract")
        endpoints = _has_state_endpoints(program, name)
        if type(row["state_endpoints"]) is not bool or row["state_endpoints"] != endpoints:
            raise ValueError("affine history endpoints require the exact declared State space")
        slots = program._histories[name]+1
        validate_transfer_data(row["transfer"])
        sources, targets, parameters = transfer_dimensions(row["transfer"])
        if targets != slots or sources != 2*slots+(2 if endpoints else 0) or parameters != PARAMETERS:
            raise ValueError("affine history source bundle, target ring or parameter binding differs")
