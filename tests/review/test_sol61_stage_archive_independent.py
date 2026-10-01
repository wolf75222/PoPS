"""Synthetic host protocol checks of Stage fixture@2; no native evidence."""
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/python/integration/runtime/test_public_evolved_original_stage.py"


def function():
    return next(n for n in ast.parse(FIXTURE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_public_"))


def actual_guard(namespace):
    nodes = list(ast.walk(function()))
    assignment = next(n for n in nodes if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "checkpoints" for t in n.targets))
    parent = next(n for n in nodes if isinstance(n, ast.If) and assignment in n.body)
    start = parent.body.index(assignment)
    stop = next(i for i in range(start + 1, len(parent.body))
                if isinstance(parent.body[i], ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "receipt" for t in parent.body[i].targets))
    module = ast.Module(body=parent.body[start:stop], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(FIXTURE), "exec"), namespace)


def images(tmp_path):
    cps, pins, phases = [], {}, {}
    for phase, data in (("accepted", b"synthetic-step1"),
                        ("continuous", b"synthetic-step2"), ("replay", b"synthetic-step2")):
        path = tmp_path / (phase + "-checkpoint.npz")
        path.write_bytes(data)
        cps.append((phase, path))
        pins[phase] = hashlib.sha256(data).hexdigest()
    for phase in ("accepted", "reloaded", "continuous", "replay"):
        path = tmp_path / (phase + ".npz")
        path.write_bytes(b"synthetic-observation")
        phases[phase] = {"path": str(path)}
    initial = tmp_path / "initial.npz"
    initial.write_bytes(b"synthetic-initial")
    return dict(checkpoint_files=cps, checkpoint_hashes=pins, phases=phases,
                initial_path=initial, bounded_bytes=lambda p: Path(p).read_bytes(),
                hashlib=hashlib, Path=Path)


def test_actual_guard_positive(tmp_path):
    ns = images(tmp_path)
    actual_guard(ns)
    assert len(ns["checkpoint_paths"]) == 3 and len(ns["observation_paths"]) == 5


@pytest.mark.parametrize("index", range(3))
def test_actual_guard_overwrite(tmp_path, index):
    ns = images(tmp_path)
    ns["checkpoint_files"][index][1].write_bytes(b"replaced-after-seal")
    with pytest.raises(AssertionError, match="^native checkpoint was overwritten$"):
        actual_guard(ns)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("alias", (False, True))
def test_actual_guard_collision_even_resealed(tmp_path, index, alias):
    ns = images(tmp_path)
    phase, _ = ns["checkpoint_files"][index]
    observation = Path(ns["phases"]["accepted"]["path"])
    path = observation
    if alias:
        path = tmp_path / (phase + "-alias.npz")
        path.symlink_to(observation)
    ns["checkpoint_files"][index] = (phase, path)
    ns["checkpoint_hashes"][phase] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(AssertionError, match="^checkpoints and observations must use distinct files$"):
        actual_guard(ns)


def test_actual_targets_and_hash_before_capture():
    nodes = list(ast.walk(function()))
    calls = sorted((n for n in nodes if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "checkpoint"),
                   key=lambda n: n.lineno)
    assert [ast.literal_eval(n.args[0].right) for n in calls] == [
        "accepted-checkpoint", "continuous-checkpoint", "replay-checkpoint"]
    for var, phase in (("checkpoint", "accepted"), ("continuous_checkpoint", "continuous"),
                       ("replay_checkpoint", "replay")):
        pins = [n.lineno for n in nodes if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "bounded_bytes"
                and len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == var]
        capture = next(n for n in nodes if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == phase for t in n.targets))
        assert min(pins) < capture.lineno
    assert '"pops.evolved-stage-native-fixture@2"' in FIXTURE.read_text()


def test_ir_cpp_hash_same_actual_component():
    loop = next(n for n in ast.walk(function()) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple) and "component" in ast.unparse(n.target))
    attrs = [n for n in ast.walk(loop) if isinstance(n, ast.Attribute)
             and n.attr in {"dump_cpp", "dump_ir", "program_hash"}]
    assert sorted(n.attr for n in attrs) == ["dump_cpp", "dump_ir", "program_hash"]
    assert all(isinstance(n.value, ast.Name) and n.value.id == "component" for n in attrs)
    source = ast.unparse(function())
    assert "row.program" in source and "artifact.layout_programs" in source
    assert "row.model" in source and "artifact.blocks" in source
    assert "pops.compile" in source and "compile_resolved_plan_once" in source


def test_actual_dump_ir_has_no_fallback(tmp_path):
    path = ROOT / "python/pops/codegen/_loader_dump.py"
    owner = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    owner.body = [n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name in {"dump_ir", "_require_program"}]
    module = ast.Module(body=[owner], type_ignores=[])
    ns = {"Any": Any}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    holder = ns[owner.name]()
    payload = {"protocol_only": True, "values": [3, -2]}

    class CarriedProgram:
        def _serialize(self):
            return payload

    holder.program = CarriedProgram()
    expected = json.dumps(payload, indent=2, sort_keys=True)
    output = tmp_path / "program.ir.json"
    assert holder.dump_ir(output) == output and output.read_text() == expected
    assert holder.dump_ir() == expected
    holder.program = None
    with pytest.raises(ValueError, match="^dump_ir: this CompiledProblem carries no Program"):
        holder.dump_ir(tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()
