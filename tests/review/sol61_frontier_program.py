# ruff: noqa: E402 -- import only after selecting the authenticated source package
"""Independent source/math/host counter-review; requires an explicitly frozen source tree.

Run this script, not pytest: the reviewed package is selected before importing pops.
No extension or PDE/JIT executes. The exception shim supports controller flow only.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
options = parser.parse_args()
source = options.source.resolve()
manifest = json.loads((source / "source_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
material = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()
assert hashlib.sha256(material).hexdigest() == manifest["source_tree_sha256"]
sys.path.insert(0, str(source / "python"))
import pops

assert Path(pops.__file__).resolve() == source / "python/pops/__init__.py"


class StepAttemptRejected(RuntimeError):
    status = "numerical_rejection"


sys.modules["pops._bootstrap"] = SimpleNamespace(StepAttemptRejected=StepAttemptRejected)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import ComputedDt, FixedDt
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.runtime._step_strategy import prepare_program_run
from pops.runtime._temporal_restart import TemporalRestartState

observations = {}


def case(*, spatial_branch=False, computed=True):
    frame = Rectangle("independent_frontier", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("independent_frontier_model", frame=frame)
    state = model.state("U", components=("x", "y"))
    x, y = state
    force = model.source("rotation", on=state, value=(-y, x))
    if spatial_branch:
        flux = model.flux(
            "F",
            frame=frame,
            state=state,
            components={axis: tuple(state) for axis in frame.axes},
            waves={axis: (1.0, 1.0) for axis in frame.axes},
        )
        rate = model.rate("balance", equation=ddt(state) == -div(flux))
        method = FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        )
    else:
        rate = model.rate("balance", equation=ddt(state) == force)
        method = StateStorage()
    owner = pops.Case("independent_frontier_case")
    block = owner.block("field", model)
    numerical = DiscretizationPlan()
    numerical.rates.add(rate, method)
    owner.numerics(numerical, block=block)
    program = pops.Program("independent_frontier_program")
    temporal = program.state(block[state])
    initial = temporal.n
    if spatial_branch:
        requested = program.requested_dt()
        rhs = program.branch(requested > 0, lambda P: rate(initial), lambda P: rate(initial))
        candidate = program.value("candidate", initial + program.dt * rhs, at=temporal.next.point)
        program.reached_duration(requested * 0.8)
    else:
        first = program.source(model.module.operator_handle("rotation"), initial)
        stage = program.stage("predictor", c=1)
        predictor = program.value("predictor", initial + program.dt * first, at=stage)
        last = program.source(model.module.operator_handle("rotation"), predictor)
        unrelaxed = program.value(
            "heun",
            initial + 0.5 * program.dt * first + 0.5 * program.dt * last,
            at=temporal.next.point,
        )
        increment = program.value("increment", unrelaxed - initial, at=temporal.next.point)
        gamma = -2.0 * program.dot(initial, increment) / program.dot(increment, increment)
        candidate = program.value("candidate", initial + gamma * increment, at=temporal.next.point)
        if computed:
            program.reached_duration(gamma * program.requested_dt())
    program.commit(temporal.next, candidate)
    program.step_strategy(ComputedDt(1.0) if computed else FixedDt(1.0))
    owner.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))
    return owner, layout, program


def emit(owner, layout, *, target="system"):
    resolved = pops.resolve(pops.validate(owner), layout=layout)
    return resolved.time, emit_cpp_program(
        resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target=target,
    )


def check_rotation_and_owned_scalar():
    from fractions import Fraction as F

    increment = (F(-1, 2), F(1))
    gamma = -2 * increment[0] / sum(value * value for value in increment)
    expected = (1 + gamma * increment[0], gamma * increment[1])
    assert gamma == F(4, 5) and expected == (F(3, 5), F(4, 5))
    assert sum(value * value for value in expected) == 1
    owner, layout, program = case()
    resolved, generated = emit(owner, layout)
    assert generated.index("ctx.reached_duration") < generated.index("ctx.commit_many")
    assert generated.count("ctx.dot(") == 2
    assert "0.8" not in generated
    source_values = [value for value in program._values if value.op == "source"]
    assert len(source_values) == 2
    assert source_values[0].point != source_values[1].point
    from pops.time._program.graph_conversion import program_to_graph

    original_identity = program._ir_hash()
    graph = program_to_graph(program)
    graph_data = graph.to_data()
    assert program._ir_hash() == original_identity
    graph_ops = [node["op"] for node in graph_data["nodes"] if "op" in node]
    assert graph_ops.count("requested_dt") == 1
    assert graph_ops.count("reached_duration") == 1
    scalar_values = [value for value in program._values if value.op in {"dot", "requested_dt"}]
    assert len({type(value) for value in scalar_values}) == 1
    assert all(value.vtype == "scalar" for value in scalar_values)
    receiving = pops.Program("receiving")
    foreign = pops.Program("foreign")
    try:
        receiving.reached_duration(foreign.requested_dt())
    except (ValueError, TypeError) as error:
        assert "different Program" in str(error)
    else:
        raise AssertionError("foreign Program Scalar accepted")
    try:
        emit(owner, layout, target="amr_system")
    except NotImplementedError as error:
        assert "AMR interval" in str(error)
    else:
        raise AssertionError("AMR frontier without exchange remapping accepted")
    observations["rotation"] = {
        "gamma": str(gamma),
        "candidate": list(map(str, expected)),
        "accepted_duration": str(gamma),
        "norm_squared": "1",
        "program_version": resolved._serialize()["version"],
    }
    observations["graph_conversion"] = (
        "Scalar type shared with dot; graph retains frontier ops and authoring identity"
    )


def check_scoped_spatial_refusal():
    owner, layout, program = case(spatial_branch=True)
    observations["scoped_spatial_top_ops"] = [value.op for value in program._values]
    try:
        _, generated = emit(owner, layout)
    except (ValueError, NotImplementedError) as error:
        assert "spatial" in str(error).lower() or "interval" in str(error).lower()
        observations["scoped_spatial"] = {"refused": True, "diagnostic": str(error)}
    else:
        observations["scoped_spatial"] = {
            "refused": False,
            "spatial_calls": generated.count("neg_div_flux"),
            "frontier_calls": generated.count("ctx.reached_duration"),
        }
        raise AssertionError(
            "ComputedDt v1 accepted FV interval operations hidden in branch regions"
        )


class Engine:
    def __init__(self, duration=0.8, **policy):
        self._step_strategy = ComputedDt(1.0, **policy)
        self._step_transaction_plan = None
        self._step_controller = None
        self._last_step_transaction_report = None
        self._temporal_restart_state = TemporalRestartState()
        self.duration = duration

    def program_diagnostics(self):
        return {"pops.frontier.duration": self.duration}


class Native:
    def __init__(self, engine):
        self.engine, self.t, self.cursor, self.calls = engine, 0.0, 0, []

    def time(self):
        return self.t

    def macro_step(self):
        return self.cursor

    def step(self, request):
        self.calls.append(request)
        self.t += self.engine.duration
        self.cursor += 1


def host(duration=0.8, **policy):
    engine = Engine(duration, **policy)
    native = Native(engine)
    prepared = prepare_program_run(engine)
    prepared.begin(engine._temporal_restart_state, time=0.0, macro_step=0)
    return engine, native, prepared


def check_controller_refusals_and_restart():
    engine, native, prepared = host()
    prepared.run_step(native, t_end=0.8)
    temporal = engine._temporal_restart_state
    saved = temporal.checkpoint_json(time=0.8, macro_step=1)
    restored = TemporalRestartState.from_json(saved, time=0.8, macro_step=1)
    assert restored.controller_state == temporal.controller_state
    for corruption in ("requested_duration", "last_dt", "missing", "limit", "rejections"):
        forged = json.loads(saved)
        controller = forged["controller_state"]
        row = controller["program_frontier"]
        if corruption == "last_dt":
            controller["last_accepted_dt"] = (42.0).hex()
        elif corruption == "missing":
            controller.pop("program_frontier")
        elif corruption == "rejections":
            row["rejections"] = 1
        else:
            row[corruption] = (0.5).hex()
        try:
            TemporalRestartState.from_json(json.dumps(forged), time=0.8, macro_step=1)
        except ValueError:
            pass
        else:
            raise AssertionError("computed restart corruption accepted: " + corruption)
    for duration in (0.0, -1.0, math.nan, math.inf, math.nextafter(0.8, math.inf)):
        engine, native, prepared = host(duration)
        try:
            prepared.run_step(native, t_end=0.8)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("invalid controller duration accepted: " + str(duration))
        assert engine._temporal_restart_state.time_hex == (0.0).hex()
        assert "program_frontier" not in engine._temporal_restart_state.controller_state
    for corruption in ("last_dt", "missing", "requested_duration", "rejections"):
        engine, native, prepared = host()
        prepared.run_step(native, t_end=0.8)
        controller = engine._temporal_restart_state.controller_state
        if corruption == "last_dt":
            controller["last_accepted_dt"] = (42.0).hex()
        elif corruption == "missing":
            controller.pop("program_frontier")
        elif corruption == "rejections":
            controller["program_frontier"]["rejections"] = 1
        else:
            controller["program_frontier"][corruption] = (0.5).hex()
        try:
            prepared.run_step(native, t_end=1.6)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("live computed receipt corruption accepted: " + corruption)
        assert native.calls == [1.0] and native.t == 0.8 and native.cursor == 1
    observations["controller_invalid"] = (
        "zero/negative/nan/inf/overshoot refused before temporal publication"
    )
    observations["live_receipt"] = (
        "last dt/missing/request/rejection corruption refused before next native.step"
    )


def check_policy_divergence_preflight():
    from pops.runtime import _step_strategy
    from pops import _native_collectives

    for field in range(5):
        engine, native, prepared = host()

        def gather(world, row, field=field):
            changed = list(row["contract"])
            changed[field] = "foreign"
            return row, {"contract": tuple(changed), "error": None}

        with (
            patch.object(_step_strategy, "_attempt_world", lambda *a, **k: SimpleNamespace(size=2)),
            patch.object(_native_collectives, "allgather_value", gather),
        ):
            try:
                prepared.run_step(native, t_end=0.8)
            except RuntimeError as error:
                assert "preparation differs" in str(error)
            else:
                raise AssertionError("divergent preflight contract accepted")
        assert native.calls == [] and native.t == 0.0 and native.cursor == 0
    observations["mpi_policy_seam"] = (
        "policy/entry time/step/endpoint/prior receipt divergence refused before native.step"
    )


def check_actual_duration_setter():
    header = (source / "include/pops/runtime/program/program_cadence_continuation.inc").read_text()
    start = header.index("void set_computed_duration(")
    end = header.index("\nvoid suspend_program_map", start)
    method = header[start:end]
    scaffold = (
        r"""#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
using Real=double;
enum class HistorySampleKind { Publication, Other };
struct Sample {
  HistorySampleKind kind=HistorySampleKind::Publication;
  std::uint64_t start_bits=0, interval_bits=std::bit_cast<std::uint64_t>(1.);
  void validate() const {
    if(!(std::bit_cast<double>(interval_bits)>0)) throw std::runtime_error("bad history interval");
  }
};
struct Frame {
  bool executing=true;
  double accepted_time=0, requested_dt=1;
  std::optional<double> computed_duration;
  struct Cadence { double window_end=1, effective_dt=1, numerical_dt=1; } cadence;
  struct Partition { double dt=1, end=1; } partition;
};
struct Probe {
  std::optional<Frame> cadence_continuation_{Frame{}};
  int stride_=1, substeps_=1;
  struct History {
    std::map<std::string,bool> store_pending{{"entry",true}};
    std::map<std::string,std::vector<Sample>> slot_sample{{"entry",{Sample{},Sample{}}}};
    std::map<std::string,std::vector<Real>> slot_dt{{"entry",{1.,1.}}};
  } hist_;
  Real last_dt_=1;
"""
        + method
        + r"""
};
int main() {
  Probe valid;
  valid.hist_.slot_sample.at("entry")[1].start_bits=std::bit_cast<std::uint64_t>(-.1);
  valid.set_computed_duration(.8);
  const auto& frame=*valid.cadence_continuation_;
  if(frame.cadence.window_end!=.8 || frame.partition.end!=.8 || frame.partition.dt!=1 ||
      frame.requested_dt!=1 || valid.last_dt_!=.8) return 2;
  if(valid.hist_.slot_dt.at("entry")[0]!=.8 || valid.hist_.slot_dt.at("entry")[1]!=1) return 3;
  if(valid.hist_.slot_sample.at("entry")[0].start_bits!=std::bit_cast<std::uint64_t>(0.)) return 4;
  try { valid.set_computed_duration(.7); return 5; } catch(const std::logic_error&) {}
  for(double duration:{0.,-1.,double(NAN),double(INFINITY)}) {
    Probe invalid;
    try { invalid.set_computed_duration(duration); return 6; } catch(const std::invalid_argument&) {}
    if(invalid.cadence_continuation_->computed_duration || invalid.last_dt_!=1 ||
        invalid.hist_.slot_dt.at("entry")[0]!=1) return 7;
  }
  Probe stalled; stalled.cadence_continuation_->accepted_time=1e300;
  try { stalled.set_computed_duration(1.); return 8; } catch(const std::invalid_argument&) {}
}
"""
    )
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "host C++20 compiler required"
    with tempfile.TemporaryDirectory() as directory:
        cpp, executable = Path(directory) / "setter.cpp", Path(directory) / "setter"
        cpp.write_text(scaffold)
        compiled = subprocess.run(
            [compiler, "-std=c++20", "-O2", str(cpp), "-o", str(executable)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert compiled.returncode == 0, compiled.stderr
        result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    observations["cpp_setter"] = (
        "actual method: request/stages preserved, outgoing history retimed, invalid/stalled/duplicate refused"
    )


checks = (
    check_rotation_and_owned_scalar,
    check_scoped_spatial_refusal,
    check_controller_refusals_and_restart,
    check_policy_divergence_preflight,
    check_actual_duration_setter,
)
results = []
for check in checks:
    try:
        check()
    except Exception as error:
        results.append(
            {
                "check": check.__name__,
                "passed": False,
                "exception": type(error).__name__,
                "diagnostic": str(error),
            }
        )
    else:
        results.append({"check": check.__name__, "passed": True})
receipt = {
    "base_commit": manifest["base_commit"],
    "source_tree_sha256": manifest["source_tree_sha256"],
    "source_package": pops.__file__,
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "host_compiler": shutil.which("clang++") or shutil.which("c++"),
    "header_sha256": {
        name: digest for name, digest in manifest["files"].items() if name.startswith("include/")
    },
    "scope": "source/math/host only; no native JIT, MPI or AMR execution",
    "results": results,
    "observations": observations,
}
options.receipt.parent.mkdir(parents=True, exist_ok=True)
options.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
sys.exit(int(any(not result["passed"] for result in results)))
