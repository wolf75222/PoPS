import json
import pathlib
import sys
import types

root = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "outputs/frontier-sol61-f6aad44-source/python"), str(root)]
import pops
# Control-flow probe only: no extension is loaded and no native execution is
# claimed. The controller needs the exception class, not a runtime backend.
class StepAttemptRejected(RuntimeError):
    pass
sys.modules["pops._bootstrap"] = types.SimpleNamespace(StepAttemptRejected=StepAttemptRejected)
from pops.runtime._step_strategy import prepare_program_run
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import ExternalTimeGrid
class _Engine:
    def __init__(self, strategy):
        self._step_strategy = strategy
        self._step_transaction_plan = None
        self._step_controller = None
        self._last_step_transaction_report = None

class _Native:
    def __init__(self):
        self.t = 0.
        self.cursor = 0
        self.calls = []
    def time(self): return self.t
    def macro_step(self): return self.cursor
    def step(self, dt):
        self.calls.append(dt)
        self.t += dt
        self.cursor += 1

assert pathlib.Path(pops.__file__).resolve().is_relative_to(
    root / "outputs/frontier-sol61-f6aad44-source")
policy = ExternalTimeGrid("grid", frontier="computed", endpoint_ulps=1)
engine = _Engine(policy)
temporal = TemporalRestartState(time_hex=(-.125).hex())
engine._temporal_restart_state = temporal
native = _Native()
native.t = -.125
prepared = prepare_program_run(engine, {"grid": (-.125, .125, .6)})
prepared.begin(temporal, time=-.125, macro_step=0)
prepared.run_step(native, t_end=.125)
saved = temporal.checkpoint_json(time=native.time(), macro_step=1)
payload = json.loads(saved)
print("honest_frontier", payload["controller_state"]["external_frontier"])
for corrupt in ("start", "index", "last_dt", "missing_receipt"):
    forged = json.loads(saved)
    row = forged["controller_state"]["external_frontier"]
    if corrupt == "index":
        row["index"] = 99
    elif corrupt == "last_dt":
        forged["controller_state"]["last_accepted_dt"] = (42.).hex()
    elif corrupt == "missing_receipt":
        forged["controller_state"].pop("external_frontier")
    else:
        # An algebraically self-consistent interval reaches the correct target,
        # but starts at an unrelated coordinate rather than the previous point.
        row["start"] = 0.0.hex()
        row["duration"] = float.fromhex(row["requested"]).hex()
        row["reached"] = row["requested"]
        forged["controller_state"]["last_accepted_dt"] = row["duration"]
    try:
        restored = TemporalRestartState.from_json(json.dumps(forged),
            time=float.fromhex(forged["clock"]["time"]), macro_step=1)
        print("restart_corruption", corrupt, "ACCEPTED")
    except Exception as error:
        print("restart_corruption", corrupt, "REJECTED", type(error).__name__, str(error))

temporal.controller_state["external_frontier"]["index"] = 99
try:
    prepared.run_step(native, t_end=.6)
    print("live_index_corruption ACCEPTED", native.time(), native.macro_step(),
          temporal.controller_state["external_frontier"]["index"])
except Exception as error:
    print("live_index_corruption REJECTED", type(error).__name__, str(error))

for start, target in ((-.1, .2), (-.3, .4)):
    engine = _Engine(policy)
    temporal = TemporalRestartState(time_hex=start.hex())
    engine._temporal_restart_state = temporal
    native = _Native()
    native.t = start
    prepared = prepare_program_run(engine, {"grid": (start, target, .6)})
    prepared.begin(temporal, time=start, macro_step=0)
    prepared.run_step(native, t_end=target)
    assert native.time() != target
    assert not prepared.pending(native, t_end=target)
    receipt = temporal.controller_state["external_frontier"]
    assert receipt["index"] == 1 and receipt["requested"] == target.hex()
    assert receipt["reached"] == native.time().hex()
    engine._temporal_restart_state = TemporalRestartState.from_json(
        temporal.checkpoint_json(time=native.time(), macro_step=1),
        time=native.time(), macro_step=1)
    prepared.begin(engine._temporal_restart_state, time=native.time(), macro_step=1)
    prepared.run_step(native, t_end=.6)
    assert native.time() == .6 and native.macro_step() == 2
    print("honest_ulp_and_restart PASS", start, target)
