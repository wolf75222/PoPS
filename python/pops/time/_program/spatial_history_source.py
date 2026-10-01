"""Exact State.n keeper authority for nonlocal history snapshots (IR18)."""
from pops.time.points import TimePoint
from pops.time._history.policy import CopyCurrent
from pops.time._program.serialization import _json_ready
import json

CONTRACT = "pops.spatial-interaction-history-source@1"


def exact_image(value):
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def history_source_contract(source):
    if source.op != "history" or source.state_ref is None:
        raise ValueError("nonlocal history source requires a typed State keeper seed")
    program = source.prog
    name = source.attrs.get("history")
    lag = source.attrs.get("lag")
    if type(name) is not str or type(lag) is not int or lag < 1:
        raise ValueError("nonlocal history source requires an exact retained lag")
    matches = [(state, store) for state, store in program._time_history_stores.items()
               if store.attrs.get("history") == name]
    if len(matches) != 1:
        raise ValueError("nonlocal history source has no unique declared keeper seed closure")
    state, store = matches[0]
    config = program._time_history_configs[state]
    contract = program._history_contracts[state]
    if type(config[1]) is not CopyCurrent or lag > config[0]:
        raise ValueError("nonlocal history source lacks its declared CopyCurrent policy")
    if (source.block != state.block or source.state_ref != state.state or source.space != state.space
            or source.clock != state.clock or source.attrs.get("history_contract") is None
            or exact_image(source.attrs["history_contract"]) != exact_image(contract.to_data())):
        raise ValueError("nonlocal history source changes its keeper physical authority")
    if store.op != "store_history" or len(store.inputs) != 1:
        raise ValueError("nonlocal history keeper has no exact seed expression")
    from .spatial_interaction import _source_point
    _source_point(store.inputs[0])
    seed = program._canonical_value(store.inputs[0])
    if (seed.op != "state" or seed.point != TimePoint(state.clock, 0)
            or seed.state_ref != state.state or seed.block != state.block or seed.space != state.space):
        raise ValueError("nonlocal CopyCurrent seed must be exactly the issued State.n expression")
    return {"contract": CONTRACT, "history": name, "lag": lag,
            "state": state.state, "space": state.space, "clock": state.clock,
            "seed_id": seed.id, "seed_point": seed.point, "cold_start": "copy_current"}
