"""Hash projection must preserve every observable serialization protocol."""
import pytest

from examples.api_v02.finite_structural_depth import build_finite_structural_depth
from pops.problem import _snapshot_payload as payload
from pops.time._program import serialization
from pops.time._program.api import Program
from pops.time.points import Clock


def legacy(program):
    return {"type": type(program).__module__ + "." + type(program).__qualname__,
            "ir": program._serialize(include_provenance=False), "hash": program._ir_hash()}


@pytest.mark.parametrize("hook", ["cadence_contract", "transaction_plan", "clock", "json_ready", "handle_data"])
def test_second_projection_hook_failure_is_not_elided(monkeypatch, hook):
    program = build_finite_structural_depth(depth=2).case._time_registry.program
    owner, name = ((Clock, "to_data") if hook == "clock" else
                   (serialization, "_json_ready" if hook == "json_ready" else "handle_data")
                   if hook in {"json_ready", "handle_data"} else (Program, hook))
    original = getattr(owner, name)
    events = []

    def count(*args, **kwargs):
        events.append(hook)
        return original(*args, **kwargs)

    with monkeypatch.context() as trial:
        trial.setattr(owner, name, count)
        program._serialize(include_provenance=False)
    per_projection = len(events)
    assert per_projection > 0

    def fail_second(*args, **kwargs):
        events.append(hook)
        if len(events) > per_projection:
            raise ValueError("second projection through " + hook)
        return original(*args, **kwargs)

    failures = []
    with monkeypatch.context() as trial:
        trial.setattr(owner, name, fail_second)
        for operation in (legacy, lambda p: payload._time_snapshot_value(p, artifact=True)):
            events.clear()
            with pytest.raises(ValueError) as exc:
                operation(program)
            failures.append((str(exc.value), tuple(events)))
    assert failures[0] == failures[1]
    assert len(failures[0][1]) == per_projection + 1


def test_space_metadata_protocol_is_checked_again_by_the_hash():
    program = build_finite_structural_depth(depth=2).case._time_registry.program
    calls = []

    class Space:
        def to_data(self):
            calls.append("space")
            if len(calls) == 2:
                raise ValueError("second space projection refused")
            return {"version": 1}

    object.__setattr__(program._values[0], "space", Space())
    failures = []
    for operation in (legacy, lambda p: payload._time_snapshot_value(p, artifact=True)):
        calls.clear()
        with pytest.raises(ValueError) as exc:
            operation(program)
        failures.append((str(exc.value), tuple(calls)))
    assert failures[0] == failures[1] == ("second space projection refused", ("space", "space"))
