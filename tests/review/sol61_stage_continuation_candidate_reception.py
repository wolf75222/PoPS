"""Opt-in host review of actual comparator against real read-only checkpoint files."""
import argparse
import ast
import copy
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def token(row):
    return "pops.%s.v%d:sha256:%s" % (row["domain"], row["schema_version"], row["hexdigest"])


def authorities(independent, trio):
    independent.review(*trio)  # independent derivation, no author helper as oracle
    rows = {}
    dt = float(trio[0]["t"])
    accepted = independent.authenticate(trio[0])
    strategy = json.loads(trio[0]["temporal_restart_state"].item())["strategy"]
    for phase, payload in zip(("accepted", "continuous", "replay"), trio, strict=True):
        manifest = independent.authenticate(payload)
        run_manifest = {"protocol": "pops.manifest", "kind": "run", "schema_version": 3,
            "payload": {"bind_identity": token(manifest["bind_identity"]),
                "continuation_identity": token(accepted["run_identity"]) if phase == "replay" else None,
                "start_time": 0. if phase == "accepted" else dt,
                "start_macro_step": 0 if phase == "accepted" else 1,
                "controls": {"t_end": float(payload["t"]), "max_steps": 1,
                    "step_transaction": strategy, "output_mode": "current-directory"},
                "run_identity": token(manifest["run_identity"])}}
        rows[phase] = {name: token(manifest[name + "_identity"]) for name in ("semantic", "artifact", "bind")}
        rows[phase].update(run=token(manifest["run_identity"]), run_manifest=run_manifest,
            restart=token(manifest["restart_identity"]), clock=manifest["clock"],
            last_restart=token(accepted["restart_identity"]) if phase == "replay" else None)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--independent-reader", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source / "python"))
    import pops
    assert Path(pops.__file__).resolve().is_relative_to(source / "python")
    spec = importlib.util.spec_from_file_location("independent_lineage", args.independent_reader)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    fixture = source / "tests/python/integration/runtime/test_public_evolved_original_stage.py"
    tree = ast.parse(fixture.read_text())
    from pops.runtime._runtime_instance import RuntimeInstance
    for name in ("_checkpoint_identities", "time", "macro_step"):
        assert callable(getattr(RuntimeInstance, name))
    for name in ("last_run_manifest", "last_run_identity", "last_restart_identity"):
        assert isinstance(getattr(RuntimeInstance, name), property)
    creator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_provenance")
    authentication = next(n for n in ast.walk(creator) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == "authenticate_checkpoint_payload")
    assert isinstance(authentication.args[0], ast.Name) and authentication.args[0].id == "runtime"
    capture = next(n for n in ast.walk(creator) if isinstance(n, ast.Return))
    assert authentication.lineno < capture.lineno
    comparator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compare_checkpoint_replay")
    ns = dict(np=np, BytesIO=BytesIO, json=json, bounded_bytes=lambda p: Path(p).read_bytes())
    exec(compile(ast.Module(body=[comparator], type_ignores=[]), str(fixture), "exec"), ns)
    compare = ns[comparator.name]
    refusals, positives = [], []
    with tempfile.TemporaryDirectory(prefix="sol61-stage-candidate-") as scratch:
        for case in range(8):
            root = args.campaign / "pytest-tmp" / ("test_public_evolved_original_s%d" % case) / "evolved-original-stage"
            paths = {phase: root / (phase + "-checkpoint.npz") for phase in ("accepted", "continuous", "replay")}
            trio = tuple(independent.load(p) for p in paths.values())
            authority = authorities(independent, trio)
            assert compare(paths, authority)["exact_payload_and_manifest"]
            positives.append(case)
            for attack in ("state", "history", "diagnostics", "clock", "lineage", "artifact"):
                payload = {k: v.copy() for k, v in trio[2].items()}
                manifest = independent.authenticate(payload)
                altered_authority = copy.deepcopy(authority)
                if attack in {"state", "history", "diagnostics"}:
                    key = {"state": "state_Q0", "history": "history_T0_0",
                           "diagnostics": "program_diagnostics_state"}[attack]
                    payload[key].flat[0] += 1
                elif attack == "clock":
                    payload["t"] = np.array(float(payload["t"]) * 2)
                    manifest["clock"]["time"] = float(payload["t"]).hex()
                elif attack == "lineage":
                    manifest["run_identity"] = independent.authenticate(trio[1])["run_identity"]
                else:
                    manifest["artifact_identity"]["hexdigest"] = "00" * 32
                    altered_authority["replay"]["artifact"] = token(manifest["artifact_identity"])
                independent.reseal(payload, manifest)
                altered_authority["replay"]["restart"] = token(manifest["restart_identity"])
                path = Path(scratch) / ("%d-%s.npz" % (case, attack))
                np.savez(path, **payload)
                try:
                    compare(dict(paths, replay=path), altered_authority)
                except (AssertionError, ValueError, TypeError):
                    refusals.append((case, attack))
                else:
                    raise AssertionError("candidate accepted mutation: " + attack)
    print(json.dumps({"candidate": "a0f92caa", "positive_pairs": positives,
                      "resealed_refusals": refusals, "native_execution": False,
                      "live_creator_tested": False, "source_runtime_api_checked": True}, sort_keys=True))


if __name__ == "__main__":
    main()
