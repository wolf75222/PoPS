"""Independent NumPy/stdlib Stage checkpoint review; never executes PoPS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

MANIFEST = "pops_checkpoint_manifest"
TOKEN = "pops_restart_identity"


def head(major, value):
    if value < 24:
        return bytes([major * 32 + value])
    for code, width in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if value < 1 << (8 * width):
            return bytes([major * 32 + code]) + value.to_bytes(width, "big")
    raise ValueError("integer too large")


def cbor(value):
    if value is None:
        return b"\xf6"
    if type(value) is bool:
        return b"\xf5" if value else b"\xf4"
    if type(value) is int:
        return head(0, value) if value >= 0 else head(1, -value - 1)
    if type(value) in (str, bytes):
        raw = value.encode() if type(value) is str else value
        return head(3 if type(value) is str else 2, len(raw)) + raw
    if type(value) is list:
        return head(4, len(value)) + b"".join(map(cbor, value))
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("non-text CBOR key")
        entries = sorted((cbor(k), cbor(v)) for k, v in value.items())
        entries.sort(key=lambda pair: (len(pair[0]), pair[0]))
        return head(5, len(entries)) + b"".join(k + v for k, v in entries)
    raise ValueError("unsupported CBOR value")


def digest(domain, payload):
    return hashlib.sha256(cbor(dict(protocol="pops.identity", domain=domain,
                                   schema_version=1, payload=payload))).hexdigest()


def identity(row, domain):
    assert set(row) == {"domain", "algorithm", "schema_version", "hexdigest"}
    assert row["domain"] == domain and row["algorithm"] == "sha256"
    assert type(row["schema_version"]) is int and row["schema_version"] > 0
    raw = bytes.fromhex(row["hexdigest"])
    assert len(raw) == 32 and raw.hex() == row["hexdigest"]
    return dict(domain=domain, algorithm="sha256", schema_version=row["schema_version"], digest=raw)


def strict_json(text):
    def pairs(rows):
        result = {}
        for k, v in rows:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive.files) == len(set(archive.files))
        return {k: archive[k].copy() for k in archive.files}


def authenticate(payload):
    manifest = strict_json(payload[MANIFEST].item())
    assert set(manifest) == {"schema_version", "runtime_kind", "semantic_identity",
                             "artifact_identity", "bind_identity", "run_identity",
                             "clock", "arrays", "restart_identity"}
    assert manifest["schema_version"] == 1 and type(manifest["schema_version"]) is int
    assert manifest["runtime_kind"] == "uniform"
    assert set(payload) == set(manifest["arrays"]) | {MANIFEST, TOKEN}
    for name, evidence in manifest["arrays"].items():
        value = np.asarray(payload[name], order="C")
        assert not value.dtype.hasobject
        header = dict(protocol="pops.array-evidence.v1", dtype=value.dtype.str, shape=list(value.shape))
        actual = dict(dtype=value.dtype.str, shape=list(value.shape),
                      content_sha256=hashlib.sha256(cbor(header) + value.tobytes()).hexdigest())
        assert evidence == actual, "array evidence mismatch: " + name
    for domain in ("semantic", "artifact", "bind", "run", "restart"):
        identity(manifest[domain + "_identity"], domain)
    base = {k: v for k, v in manifest.items() if k != "restart_identity"}
    expected = digest("restart", base)
    assert manifest["restart_identity"]["hexdigest"] == expected, "restart digest mismatch"
    assert payload[TOKEN].item() == "pops.restart.v1:sha256:" + expected
    assert manifest["clock"] == {"time": float(payload["t"]).hex(),
                                  "macro_step": int(payload["macro_step"])}
    return manifest


def run_digest(bind, strategy, start, end, step, lineage):
    payload = dict(schema_version=3, bind_identity=identity(bind, "bind"),
                   continuation_identity=None if lineage is None else identity(lineage, "run"),
                   start_time=start.hex(), start_macro_step=step,
                   controls=dict(t_end=end.hex(), step_transaction=strategy,
                                 max_steps=1, output_mode="current-directory"))
    return digest("run", payload)


def review(accepted, continuous, replay):
    am, cm, rm = map(authenticate, (accepted, continuous, replay))
    dt = float(accepted["t"])
    assert dt > 0 and int(accepted["macro_step"]) == 1
    assert float(continuous["t"]) == 2 * dt and int(continuous["macro_step"]) == 2
    strategy = strict_json(accepted["temporal_restart_state"].item())["strategy"]
    assert strategy == {"controls": {}, "strategy": {"kind": "fixed_dt",
                        "dt": {"kind": "binary64", "value": dt.hex()}}}
    expected = ((am, 0., dt, 0, None), (cm, dt, 2 * dt, 1, None),
                (rm, dt, 2 * dt, 1, am["run_identity"]))
    for manifest, start, end, step, lineage in expected:
        assert manifest["run_identity"]["hexdigest"] == run_digest(
            manifest["bind_identity"], strategy, start, end, step, lineage), "run lineage mismatch"
    assert set(continuous) == set(replay)
    for name in continuous.keys() - {MANIFEST, TOKEN}:
        a, b = continuous[name], replay[name]
        assert (a.dtype.str, a.shape, a.tobytes()) == (b.dtype.str, b.shape, b.tobytes()), name
    for key in cm.keys() - {"run_identity", "restart_identity"}:
        assert cm[key] == rm[key], key
    for key in ("semantic_identity", "artifact_identity", "bind_identity"):
        assert am[key] == cm[key], key
    return {"payload_leaves": len(continuous), "exact_payload_leaves": len(continuous) - 2,
            "run_continuous": cm["run_identity"], "run_replay": rm["run_identity"],
            "scope": "native_checkpoint_bytes_and_lineage_only_no_scientific_acceptance"}


def reseal(payload, manifest):
    for name in manifest["arrays"]:
        value = np.asarray(payload[name], order="C")
        header = dict(protocol="pops.array-evidence.v1", dtype=value.dtype.str, shape=list(value.shape))
        manifest["arrays"][name] = dict(dtype=value.dtype.str, shape=list(value.shape),
            content_sha256=hashlib.sha256(cbor(header) + value.tobytes()).hexdigest())
    base = {k: v for k, v in manifest.items() if k != "restart_identity"}
    manifest["restart_identity"]["hexdigest"] = digest("restart", base)
    payload[MANIFEST] = np.array(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    payload[TOKEN] = np.array("pops.restart.v1:sha256:" + manifest["restart_identity"]["hexdigest"])


def countermodels(accepted, continuous, replay):
    """Mutate disk COPIES of real failed-run CPs, never the donor or native code."""
    refused = []
    attacks = ("state", "history", "diagnostics", "clock", "lineage", "artifact")
    with tempfile.TemporaryDirectory(prefix="sol61-stage-CP-countermodels-") as scratch:
        for attack in attacks:
            mutated = {k: v.copy() for k, v in replay.items()}
            manifest = strict_json(mutated[MANIFEST].item())
            if attack in {"state", "history", "diagnostics"}:
                key = {"state": "state_Q0", "history": "history_T0_0",
                       "diagnostics": "program_diagnostics_state"}[attack]
                mutated[key].flat[0] += 1
            elif attack == "clock":
                mutated["t"] = np.array(float(mutated["t"]) * 2)
                manifest["clock"]["time"] = float(mutated["t"]).hex()
            elif attack == "lineage":
                manifest["run_identity"]["hexdigest"] = strict_json(continuous[MANIFEST].item())["run_identity"]["hexdigest"]
            else:
                manifest["artifact_identity"]["hexdigest"] = "00" * 32
            # Fully reseal content evidence and the outer restart digest. This
            # reaches lineage / byte-exact comparison, beyond transport integrity.
            reseal(mutated, manifest)
            path = Path(scratch) / (attack + ".npz")
            np.savez(path, **mutated)
            try:
                review(accepted, continuous, load(path))
            except (AssertionError, ValueError) as error:
                refused.append({"attack": attack, "refusal": str(error)})
            else:
                raise AssertionError("countermodel accepted: " + attack)
    return refused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--countermodels", action="store_true")
    args = parser.parse_args()
    records = {}
    for index in range(8):
        directory = args.campaign / "pytest-tmp" / ("test_public_evolved_original_s%d" % index) / "evolved-original-stage"
        trio = tuple(load(directory / (phase + "-checkpoint.npz"))
                     for phase in ("accepted", "continuous", "replay"))
        records[str(index)] = review(*trio)
        records[str(index)]["observed_files"] = {
            phase: {"path": str(directory / (phase + "-checkpoint.npz")),
                    "sha256": hashlib.sha256((directory / (phase + "-checkpoint.npz")).read_bytes()).hexdigest()}
            for phase in ("accepted", "continuous", "replay")}
        if args.countermodels:
            records[str(index)]["countermodels"] = countermodels(*trio)
    print(json.dumps(records, indent=2, sort_keys=True))
