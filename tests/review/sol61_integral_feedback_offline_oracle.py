"""Offline T5 receipt reception. NumPy/stdlib only; never construct saved runtime states.

An externally pinned complete campaign is mandatory. No pins, native outcome or
scientific PASS is inferred from these source contracts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import zipfile

import numpy as np

SOURCE_CONTRACT = "8eabb8e2875158e3f4f8a0ed0255b0a63db4b6c4"
SCHEMA = "sol61.integral-feedback.offline-pins@1"
DT, GAMMA, Q0, TOL = .01, .3, .7, 3e-13
PHASES = {"restart": ("initial", "accepted", "continuous", "restored", "replayed"),
          "retry": ("before", "rejected", "retried")}
# Reader budgets for this N8/N16 fixture, not limits on the PoPS runtime.
MAX_FILE_BYTES = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(token):
        raise ValueError("nonfinite JSON token " + token)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def cbor(value):
    """Independent narrow deterministic CBOR, sufficient for checkpoint envelopes."""
    def head(major, number):
        require(0 <= number <= (1 << 64) - 1, "CBOR length/integer overflow")
        if number < 24:
            return bytes([major * 32 + number])
        for width, marker in ((1, 24), (2, 25), (4, 26), (8, 27)):
            if number < 1 << (8 * width):
                return bytes([major * 32 + marker]) + number.to_bytes(width, "big")
        raise ValueError("CBOR overflow")

    if value is None:
        return b"\xf6"
    if type(value) is bool:
        return b"\xf5" if value else b"\xf4"
    if type(value) is int:
        require(-(1 << 63) <= value < (1 << 63), "CBOR integer outside int64")
        return head(0, value) if value >= 0 else head(1, -1 - value)
    if type(value) is str:
        raw = value.encode("utf-8")
        return head(3, len(raw)) + raw
    if type(value) is list:
        return head(4, len(value)) + b"".join(cbor(item) for item in value)
    if type(value) is dict:
        require(all(type(key) is str for key in value), "CBOR keys must be strings")
        rows = sorted(((cbor(key), cbor(item)) for key, item in value.items()),
                      key=lambda row: (len(row[0]), row[0]))
        return head(5, len(rows)) + b"".join(key + item for key, item in rows)
    raise ValueError("unsupported CBOR type; binary64 must use canonical hex strings")


def identity_token(value, domain):
    require(type(value) is dict and set(value) == {"domain", "schema_version", "algorithm", "hexdigest"},
            "invalid exact identity fields")
    require(value["domain"] == domain and value["algorithm"] == "sha256"
            and type(value["schema_version"]) is int and value["schema_version"] >= 1
            and re.fullmatch("[0-9a-f]{64}", value["hexdigest"]) is not None,
            "invalid identity domain/version/digest")
    return f"pops.{domain}.v{value['schema_version']}:sha256:{value['hexdigest']}"


def pinned_file(base, row):
    require(type(row) is dict and set(row) == {"path", "sha256"}, "invalid external file pin")
    require(type(row["path"]) is str and type(row["sha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None, "invalid file pin type")
    path = Path(row["path"])
    if not path.is_absolute():
        path = base / path
    require(path.is_file() and path.stat().st_size <= MAX_FILE_BYTES, "missing/oversized pinned file")
    raw = path.read_bytes()
    require(digest(raw) == row["sha256"], "external SHA256 pin mismatch: " + str(path))
    return path, raw


def archive(raw):
    # Bound decompression before NumPy allocates; reject duplicate/foreign ZIP members.
    with zipfile.ZipFile(io.BytesIO(raw)) as packed:
        infos = packed.infolist()
        names = [entry.filename for entry in infos]
        require(len(set(names)) == len(names) and all(name.endswith(".npy") and "/" not in name
                for name in names), "invalid NPZ member inventory")
        require(sum(entry.file_size for entry in infos) <= MAX_FILE_BYTES, "NPZ decompression budget exceeded")
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def scalar(value, kind):
    require(value.shape == (), "receipt scalar must be rank zero")
    result = value.item()
    if kind == "real":
        require(value.dtype == np.dtype("float64") and type(result) is float
                and math.isfinite(result), "nonfinite/non-binary64 scalar")
    else:
        require(type(result) is int and result >= 0, "invalid integer clock")
    return result


def exchange_image(raw):
    """Independent strict POPSEX02 decoder, bounded by its actual byte image."""
    require(raw[:8] == b"POPSEX02", "T5 needs exact POPSEX02 ledger")
    position = 8

    def word():
        nonlocal position
        require(position + 8 <= len(raw), "truncated ledger word")
        result = int.from_bytes(raw[position:position + 8], "little")
        position += 8
        return result

    def text():
        nonlocal position
        length = word()
        require(length <= len(raw) - position, "ledger string exceeds image")
        result = raw[position:position + length].decode("utf-8")
        position += length
        return result

    def number():
        result = struct.unpack("<d", word().to_bytes(8, "little"))[0]
        require(math.isfinite(result), "nonfinite ledger value")
        return result

    def count(minimum):
        result = word()
        require(result <= (len(raw) - position) // minimum, "ledger count exceeds image")
        return result

    records, keys = [], set()
    for _ in range(count(112)):
        operation, occurrence, context, quadrature = (text() for _ in range(4))
        require(all((operation, occurrence, context, quadrature)), "empty ledger identity")
        orientation = word()
        orientation = orientation if orientation < (1 << 63) else orientation - (1 << 64)
        measure, flux, weight = number(), number(), number()
        multiplicity = word()
        axis, side, component = (word() - 1 for _ in range(3))
        exterior = word()
        evaluation = text()
        require(orientation in (-1, 1) and measure > 0 and multiplicity > 0
                and exterior in (0, 1), "invalid ledger incidence")
        require(math.isfinite(orientation * measure * flux * weight * multiplicity), "nonfinite ledger amount")
        require(not exterior or (axis >= 0 and side in (0, 1) and component >= 0 and evaluation),
                "incomplete exterior trace metadata")
        key = (operation, occurrence, context, quadrature)
        require(key not in keys, "duplicate local exchange occurrence")
        keys.add(key)
        records.append(dict(operation=operation, occurrence=occurrence, context=context,
                            quadrature=quadrature, orientation=orientation, measure=measure,
                            flux=flux, weight=weight, multiplicity=multiplicity, axis=axis,
                            side=side, component=component, exterior=bool(exterior), evaluation=evaluation))
    quantities = {}
    for _ in range(count(24)):
        name, initial, value = text(), number(), number()
        require(name and name not in quantities, "duplicate/empty integral declaration")
        quantities[name] = [initial, value]
    consumed = [list(text() for _ in range(4)) for _ in range(count(32))]
    require(len({tuple(key) for key in consumed}) == len(consumed), "duplicate consumed exchange key")
    require(position == len(raw), "trailing ledger bytes")
    exterior_keys = {(r["operation"], r["occurrence"], r["context"], r["quadrature"])
                     for r in records if r["exterior"]}
    require(all(tuple(key) in exterior_keys for key in consumed), "consumed key lacks exterior record")
    require(quantities or not consumed, "consumption without integral declaration")
    return dict(records=records, quantities=quantities, consumed=consumed)


def frame(record, previous_time, previous_step):
    value = record["context"]
    prefix = "pops.exchange.frame.v1/"
    require(value.startswith(prefix), "missing exact runtime exchange frame")
    remainder = value[len(prefix):]
    length, remainder = remainder.split(":", 1)
    require(length.isdecimal(), "malformed clock length")
    count = int(length)
    require(len(remainder.encode("utf-8")) >= count, "truncated clock")
    # T5's declared clock is ASCII; avoid accidentally treating byte lengths as Unicode positions.
    raw = remainder.encode("utf-8")
    clock, remainder = raw[:count].decode("utf-8"), raw[count:].decode("utf-8")
    require(clock and remainder.startswith("/"), "malformed exact clock frame")
    fields = remainder[1:].split("/", 8)
    require(len(fields) == 9, "incomplete runtime exchange frame")
    tick, level, substep, stage, numerator, denominator, dt_bits, time_bits = map(int, fields[:8])
    evaluation_length, evaluation = fields[8].split(":", 1)
    require(evaluation_length.isdecimal() and int(evaluation_length) == len(evaluation.encode("utf-8"))
            and evaluation == record["evaluation"], "source evaluation authority mismatch")
    expected_dt = int.from_bytes(struct.pack("<d", DT), "little")
    expected_time = int.from_bytes(struct.pack("<d", previous_time), "little")
    require((tick, level, substep, stage, numerator, denominator) == (previous_step, 0, 0, 0, 0, 1)
            and dt_bits == expected_dt and time_bits == expected_time, "wrong runtime interval/duration authority")
    return clock


@dataclass
class Snapshot:
    state: np.ndarray
    q: float
    time: float
    step: int
    images: tuple[bytes, ...]
    ledgers: list[dict]
    hashes: dict
    temporal: dict


def load_snapshot(base, pins, case, phase):
    row = case["phases"][phase]
    require(set(row) == {"receipt", "state", "checkpoint"}, "incomplete phase file pins")
    checked = {name: pinned_file(base, value) for name, value in row.items()}
    receipt = strict_json(checked["receipt"][1])
    require(receipt["phase"] == phase and receipt["artifact"] == case["artifact"]
            and receipt["quantity_identity"] == case["quantity_identity"]
            and receipt["platform"] == case["platform"] and receipt["dimension"] == 2,
            "phase/artifact/quantity/platform identity mismatch")
    require(receipt["checkpoint_sha256"] == digest(checked["checkpoint"][1]), "receipt checkpoint SHA mismatch")
    require(Path(receipt["checkpoint"]).name == checked["checkpoint"][0].name, "checkpoint basename mismatch")
    data = archive(checked["state"][1])
    require(set(data) == {"state", "q", "time", "step", "level_0"}, "unexpected T5 saved state keys")
    u = data["state"]
    require(u.dtype == np.dtype("float64") and u.shape == (1, 1, case["cells"])
            and np.isfinite(u).all(), "T5 state shape/type/finitude mismatch")
    require(data["level_0"].dtype == u.dtype and data["level_0"].shape == u.shape
            and data["level_0"].tobytes() == u.tobytes(), "level0 differs from saved global state")
    q, time, step = scalar(data["q"], "real"), scalar(data["time"], "real"), scalar(data["step"], "int")
    require(type(receipt["quantity"]) is float and type(receipt["time"]) is float
            and type(receipt["macro_step"]) is int
            and q == receipt["quantity"] and time == receipt["time"] and step == receipt["macro_step"],
            "saved state clock/integral differs from receipt")
    checkpoint = archive(checked["checkpoint"][1])
    require(checkpoint["state_fluid"].dtype == u.dtype and checkpoint["state_fluid"].shape == u.shape
            and checkpoint["state_fluid"].tobytes() == u.tobytes(), "checkpoint physical state differs")
    require(str(checkpoint["abi_key"].item()) == pins["abi_key"], "checkpoint SDK/native ABI mismatch")
    require(scalar(checkpoint["t"], "real") == time and scalar(checkpoint["macro_step"], "int") == step,
            "checkpoint clock differs")
    temporal = strict_json(str(checkpoint["temporal_restart_state"].item()))
    require(cbor(temporal["clock"]) == cbor({"time": time.hex(), "macro_step": step}),
            "temporal accepted clock differs")
    schedule = temporal["program_schedule"]
    require(type(schedule) is dict and schedule["kind"] == "pops.temporal-program-schedule"
            and len(schedule["clocks"]) == 1
            and schedule["clocks"][0]["id"] == schedule["primary_clock"]
            and type(schedule["clocks"][0]["ticks_per_macro"]) is int
            and schedule["clocks"][0]["ticks_per_macro"] == 1,
            "T5 does not carry its ordinary root clock authority")
    manifest = strict_json(str(checkpoint["pops_checkpoint_manifest"].item()))
    require(manifest["runtime_kind"] == "uniform"
            and cbor(manifest["clock"]) == cbor({"time": time.hex(), "macro_step": step}),
            "checkpoint envelope clock/runtime mismatch")
    require(identity_token(manifest["artifact_identity"], "artifact") == case["artifact"], "checkpoint artifact differs")
    identity_token(manifest["semantic_identity"], "semantic")
    identity_token(manifest["bind_identity"], "bind")
    version = manifest["schema_version"]
    bound_initial = phase in {"initial", "before"}
    require(type(version) is int and version == (2 if bound_initial else 1), "wrong initial/run envelope version")
    if bound_initial:
        require(time.hex() == "0x0.0p+0" and manifest["run_identity"] is None
                and cbor(manifest["origin"]) == cbor({"schema_version": 1, "kind": "bound_initial"}), "invalid initial provenance")
    else:
        identity_token(manifest["run_identity"], "run")
        require("origin" not in manifest, "run envelope invents initial origin")
    require(set(checkpoint) == set(manifest["arrays"]) | {"pops_checkpoint_manifest", "pops_restart_identity"},
            "checkpoint manifest member mismatch")
    for name, expected in manifest["arrays"].items():
        array = np.ascontiguousarray(checkpoint[name]) if checkpoint[name].shape else checkpoint[name]
        header = cbor(dict(protocol="pops.array-evidence.v1", dtype=array.dtype.str, shape=list(array.shape)))
        actual = dict(dtype=array.dtype.str, shape=list(array.shape), content_sha256=digest(header + array.tobytes()))
        require(expected == actual, "checkpoint typed array digest mismatch: " + name)
    base_manifest = {key: value for key, value in manifest.items() if key != "restart_identity"}
    envelope = dict(protocol="pops.identity", domain="restart", schema_version=1, payload=base_manifest)
    restart = identity_token(manifest["restart_identity"], "restart")
    require(manifest["restart_identity"]["hexdigest"] == digest(cbor(envelope))
            and str(checkpoint["pops_restart_identity"].item()) == restart, "checkpoint envelope digest mismatch")
    raw, offsets = checkpoint["program_exchange_state"], checkpoint["program_exchange_offsets"]
    require(raw.dtype == np.dtype("uint8") and raw.ndim == 1 and offsets.ndim == 1
            and offsets.dtype.kind in "iu" and len(offsets) == pins["size"] + 1, "rank ledger offsets mismatch")
    bounds = [int(value) for value in offsets]
    require(bounds[0] == 0 and bounds[-1] == raw.size
            and all(a < b for a, b in zip(bounds[:-1], bounds[1:], strict=True)), "ledger offsets not exact partition")
    wire = raw.tobytes()
    images = tuple(wire[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True))
    ledgers = [exchange_image(image) for image in images]
    require([digest(image) for image in images] == receipt["ledger_sha256"] and ledgers == receipt["ledgers"],
            "native ledger bytes disagree with JSON receipt")
    require(all(ledger["quantities"] == {case["quantity_identity"]: [Q0, q]} for ledger in ledgers),
            "rank-local q declarations/values disagree")
    return Snapshot(u.reshape(-1), q, time, step, images, ledgers,
                    {name: digest(value[1]) for name, value in checked.items()}, temporal)


def same_accepted(a, b, label):
    require(a.state.tobytes() == b.state.tobytes() and struct.pack("<d", a.q) == struct.pack("<d", b.q)
            and a.time.hex() == b.time.hex() and a.step == b.step and a.images == b.images,
            label + " did not preserve exact state/q/clock/rank-ledger bytes")
    for field in ("program_schedule", "clock_cursors", "schedule_cursors", "synchronization_cursors",
                  "history_cursors", "cache_cursors"):
        require(cbor(a.temporal[field]) == cbor(b.temporal[field]), label + " changed accepted " + field)


def initial(snapshot, cells):
    require(snapshot.time.hex() == "0x0.0p+0" and snapshot.step == 0 and snapshot.q == Q0, "initial clock/q mismatch")
    require(max(abs(float(value) - (1. + .2 * (i + .5) / cells))
                for i, value in enumerate(snapshot.state)) < TOL, "saved initial projection differs from declared affine IC")
    require(all(not ledger["records"] and not ledger["consumed"] for ledger in snapshot.ledgers), "initial has accepted records")


def step(before, after):
    require(after.time == before.time + DT and after.step == before.step + 1, "wrong accepted clock increment")
    n = before.state.size
    # Evaluate the declared discrete equations by independent face incidences.
    reacted = [float(value) * (1. - GAMMA * before.q * DT) for value in before.state]
    expected = [math.fsum((reacted[i], -DT * n * reacted[i], DT * n * reacted[max(0, i - 1)]))
                for i in range(n)]
    error = max(abs(float(actual) - target) for actual, target in zip(after.state, expected, strict=True))
    delivered = DT * reacted[-1]
    require(error < TOL and abs(after.q - before.q - delivered) < TOL, "reaction/candidate q capture or FV field update mismatch")
    source_inventory = -GAMMA * before.q * DT * math.fsum(map(float, before.state)) / n
    boundary_inventory = DT * (reacted[0] - reacted[-1])
    inventory_change = math.fsum(map(float, after.state)) / n - math.fsum(map(float, before.state)) / n
    balance_error = abs(math.fsum((inventory_change, -source_inventory, -boundary_inventory)))
    require(balance_error < TOL, "source plus external flux balance failed")
    all_records = [record for ledger in after.ledgers for record in ledger["records"]]
    records = [record for record in all_records
               if record["exterior"] and (record["axis"], record["side"], record["component"]) == (0, 1, 0)]
    consumed = [key for ledger in after.ledgers for key in ledger["consumed"]]
    require(len(records) == len(consumed) == 1, "right trace must be unique across actual rank images")
    record = records[0]
    require(record["orientation"] == -1 and record["multiplicity"] == 1 and record["measure"] == 1.
            and record["weight"] == DT and abs(record["flux"] - reacted[-1]) < TOL
            and abs(record["measure"] * record["flux"] * record["weight"] - delivered) < TOL,
            "ledger physical flux/measure/duration/orientation mismatch")
    require(consumed[0] == [record[key] for key in ("operation", "occurrence", "context", "quadrature")],
            "selected trace consumed a different occurrence")
    require(record["quadrature"] == f"cell:{n-1}:0/axis:0/side:1", "wrong physical right endpoint cell")
    clock = frame(record, before.time, before.step)
    require(clock == before.temporal["program_schedule"]["primary_clock"]
            == after.temporal["program_schedule"]["primary_clock"], "trace uses a foreign declared clock")
    expected_quadratures = {f"cell:{i}:0/axis:{axis}/side:{side}"
                            for i in range(n) for axis in range(2) for side in range(2)}
    require(len(all_records) == 4 * n
            and {r["quadrature"] for r in all_records} == expected_quadratures,
            "native ledger omits or duplicates an owned cell incidence")
    static = tuple(record[key] for key in ("operation", "occurrence", "context", "evaluation"))
    for incidence in all_records:
        require(tuple(incidence[key] for key in ("operation", "occurrence", "context", "evaluation")) == static,
                "incidence belongs to a foreign FV/source/interval occurrence")
        match = re.fullmatch(r"cell:([0-9]+):0/axis:([01])/side:([01])", incidence["quadrature"])
        require(match is not None, "foreign spatial quadrature")
        i, axis, side = map(int, match.groups())
        flux = reacted[i if side else max(0, i - 1)] if axis == 0 else 0.
        exterior = axis == 0 and ((side == 0 and i == 0) or (side == 1 and i == n - 1))
        require((incidence["axis"], incidence["side"], incidence["component"]) == (axis, side, 0)
                and incidence["orientation"] == (1 if side == 0 else -1)
                and incidence["measure"] == (1. if axis == 0 else 1. / n)
                and incidence["weight"] == DT and incidence["multiplicity"] == 1
                and incidence["exterior"] == exterior and abs(incidence["flux"] - flux) < TOL,
                "native face flux/measure/support differs from the saved-state FV oracle")
    ledger_inventory = math.fsum(r["orientation"] * r["measure"] * r["flux"] * r["weight"]
                                * r["multiplicity"] for r in all_records)
    ledger_error = abs(ledger_inventory - boundary_inventory)
    require(ledger_error < TOL, "native ledger incidences fail telescoping external balance")
    return dict(field_linf_error=error, inventory_residual=balance_error,
                native_ledger_residual=ledger_error, native_incidence_count=len(all_records),
                q_before=before.q, q_after=after.q, right_delivery=delivered,
                source_inventory=source_inventory, boundary_inventory=boundary_inventory,
                record=record, clock=clock)


def receive(path):
    base, raw = path.resolve().parent, path.read_bytes()
    pins = strict_json(raw)
    require(pins["schema"] == SCHEMA and type(pins["size"]) is int and pins["size"] > 0
            and pins["dimension"] == 2, "unsupported T5 pin schema/rank/dimension")
    require(re.fullmatch("[0-9a-f]{40}", pins["source_commit"]) is not None
            and re.fullmatch("[0-9a-f]{64}", pins["native_sha256"]) is not None
            and type(pins["abi_key"]) is str and "dim=2" in pins["abi_key"], "missing explicit native/source pins")
    _, identity_raw = pinned_file(base, pins["identity_file"])
    identity = strict_json(identity_raw)
    require(all(identity[key] == pins[key] for key in ("source_commit", "abi_key", "native_sha256")),
            "external runtime identity differs from pins")
    labels = [(case["kind"], case["cells"]) for case in pins["cases"]]
    require(sorted(labels) == [("restart", 8), ("restart", 16), ("retry", 8)], "incomplete/duplicate T5 campaign")
    results = []
    accepted_n8 = None
    for case in pins["cases"]:
        require(set(case["phases"]) == set(PHASES[case["kind"]]), "wrong case phase inventory")
        require(type(case["quantity_identity"]) is str and case["quantity_identity"].startswith("pops.integral.v2/"),
                "T5 q must have typed integral authority")
        snapshots = {phase: load_snapshot(base, pins, case, phase) for phase in PHASES[case["kind"]]}
        first = snapshots["initial" if case["kind"] == "restart" else "before"]
        initial(first, case["cells"])
        if case["kind"] == "restart":
            one = step(first, snapshots["accepted"])
            two = step(snapshots["accepted"], snapshots["continuous"])
            same_accepted(snapshots["accepted"], snapshots["restored"], "restart restore")
            same_accepted(snapshots["continuous"], snapshots["replayed"], "restart replay")
            replay = step(snapshots["restored"], snapshots["replayed"])
            require((one["record"]["operation"], one["record"]["occurrence"], one["record"]["evaluation"], one["clock"])
                    == (two["record"]["operation"], two["record"]["occurrence"], two["record"]["evaluation"], two["clock"]),
                    "changed static occurrence/source/clock between accepted steps")
            result = dict(kind=case["kind"], cells=case["cells"], steps=[one, two, replay])
            if case["cells"] == 8:
                accepted_n8 = snapshots["accepted"]
        else:
            same_accepted(first, snapshots["rejected"], "rejected attempt")
            result = dict(kind=case["kind"], cells=case["cells"], steps=[step(first, snapshots["retried"])])
        result["file_sha256"] = {phase: saved.hashes for phase, saved in snapshots.items()}
        results.append(result)
    retry_case = next(case for case in pins["cases"] if case["kind"] == "retry")
    retried = load_snapshot(base, pins, retry_case, "retried")
    require(accepted_n8 is not None and accepted_n8.state.tobytes() == retried.state.tobytes()
            and accepted_n8.q == retried.q, "safe retry differs from fresh N8 accepted step")
    return dict(schema="sol61.integral-feedback.offline-reception@1", scientific_status="PASS",
                source_contract=SOURCE_CONTRACT, source_commit=pins["source_commit"],
                abi_key=pins["abi_key"], native_sha256=pins["native_sha256"], dimension=2,
                ranks=pins["size"], pins_sha256=digest(raw), constants=dict(dt=DT, gamma=GAMMA, q0=Q0, tolerance=TOL),
                cases=results, limitation="Offline saved-state/ledger reception; no new PoPS/native/MPI execution.")


def contract():
    return dict(schema=SCHEMA, scientific_status="pending_receipts", source_contract=SOURCE_CONTRACT,
                expected_cases=[dict(kind=kind, cells=n, phases=PHASES[kind])
                                for kind, n in (("restart", 8), ("restart", 16), ("retry", 8))],
                required_external_pins=["source_commit", "abi_key", "native_sha256", "identity_file", "dimension", "size"],
                case_pins=["artifact", "platform", "quantity_identity", "kind", "cells", "phases"],
                phase_pins={name: {"path": "actual file path", "sha256": "externally pinned SHA256"}
                            for name in ("receipt", "state", "checkpoint")},
                equations=["U*=U(1-gamma*q*h)", "U'_i=U*_i-h*N*(U*_i-U*_{max(0,i-1)})", "q'=q+h*U*_{N-1}"],
                tolerance=TOL, consumes_no_pops_package=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--describe-contract", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(args.describe_contract != (args.pins is not None), "choose exact pins or describe-contract")
    result = contract() if args.describe_contract else receive(args.pins)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
