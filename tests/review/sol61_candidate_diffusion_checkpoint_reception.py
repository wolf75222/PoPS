"""Prospective Candidate fixture@2 strict checkpoint receiver; no PoPS/native import."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "candidate_d_frozen_states_v1",
    Path(__file__).with_name("sol61_candidate_diffusion_saved_reception.py"),
)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
need, exact, digest, read, pinned, strict_json, leaf, same = (
    getattr(base, n)
    for n in ("need", "exact", "digest", "read", "pinned", "strict_json", "leaf", "same")
)
wire, protocol, components, CASES, PHASES, CLOCKS, DT, MAX_BYTES = (
    getattr(base, n)
    for n in ("wire", "protocol", "components", "CASES", "PHASES", "CLOCKS", "DT", "MAX_BYTES")
)
science, history_identity, diagnostic_images, diagnostic_science = (
    getattr(base, n)
    for n in ("science", "history_identity", "diagnostic_images", "diagnostic_science")
)

SCHEMA = "sol61.candidate-d-owner-pins@2"
QUALIFICATION = "saved-candidate-d-checkpoint-original-residual@2"
FIXTURE_REVISION = "766f078fd63e484c1814fe09b7ed37cee291f19f"
SOURCE_FINGERPRINTS = dict(base.SOURCE_FINGERPRINTS)
SOURCE_FINGERPRINTS["fixture"] = (
    base.SOURCE_FINGERPRINTS["fixture"][0],
    "d9e2ed69d84b0a63f3e01f270a17d8bb07879bc4f79d5958e7fd340ecffb5844",
)
EMITTER_FINGERPRINTS = {
    base.SOURCE_FINGERPRINTS["emitter"][1],
    "61add876ccaed3838d4229e528c831bf471eee0cebda40637453c822bf743576",
}


def origins(owner, roots):
    exact(
        owner,
        (
            "schema",
            "source_commit",
            "native_build_source_commit",
            "abi_key",
            "python_package",
            "sdk",
            "native",
            "sources",
            "execution_association",
        ),
        "owner@2",
    )
    need(owner["schema"] == "sol61.candidate-d-execution-owner@2", "owner schema@2 required")
    need(
        all(
            type(owner[k]) is str and re.fullmatch("[0-9a-f]{40}", owner[k])
            for k in ("source_commit", "native_build_source_commit")
        ),
        "source/build source identity absent",
    )
    need(type(owner["abi_key"]) is str and bool(owner["abi_key"]), "ABI absent")
    for key in ("python_package", "sdk", "native"):
        pinned(owner[key], roots, MAX_BYTES if key == "native" else protocol.MAX_FILE_BYTES)
    exact(owner["sources"], SOURCE_FINGERPRINTS, "four source identities")
    for key, (_, sha) in SOURCE_FINGERPRINTS.items():
        actual = digest(pinned(owner["sources"][key], roots)[1])
        need(
            actual in (EMITTER_FINGERPRINTS if key == "emitter" else {sha}),
            "reviewed source fingerprint differs: " + key,
        )
    exact(owner["execution_association"], CASES, "ROOT execution association")


def receipt_contract(receipt, width, order, ranks, owner):
    need(
        receipt["fixture_schema"] == "pops.candidate-diffusion-native-fixture@2"
        and "program_irs" in receipt,
        "Candidate fixture@2 with actual carried IR required",
    )
    common = {k: v for k, v in receipt.items() if k != "program_irs"}
    common["fixture_schema"] = "pops.candidate-diffusion-native-fixture@1"
    # Shared unchanged physical fields only; neither @1 pins nor approval are accepted.
    base.receipt_contract(common, width, order, ranks, owner)
    need(
        type(receipt["program_irs"]) is list and len(receipt["program_irs"]) == 1,
        "one actual compiled Program IR required",
    )
    row = receipt["program_irs"][0]
    exact(row, ("component", "path", "sha256", "program_hash"), "actual IR row")
    need(
        row["component"] == receipt["sources"][0]["component"]
        and type(row["program_hash"]) is str
        and re.fullmatch("[0-9a-f]{64}", row["program_hash"]),
        "IR component/Program hash differs",
    )


def case_inventory(directory):
    directory = wire.canonical(directory)
    receipt = strict_json(read(directory / "receipt.json")[1])
    need(
        receipt.get("fixture_schema") == "pops.candidate-diffusion-native-fixture@2",
        "historical fixture is not checkpoint@2",
    )
    exact(receipt["phases"], PHASES, "phase inventory")
    initial = leaf(receipt["initial_npz"])
    need(initial["sha256"] == receipt["initial_sha256"], "initial hash differs")
    phases = {}
    for phase, row in receipt["phases"].items():
        exact(row, ("npz", "sha256", "checks"), "phase row")
        phases[phase] = leaf(row["npz"])
        need(phases[phase]["sha256"] == row["sha256"], "phase hash differs")
    exact(receipt["checkpoints"], ("accepted", "continuous", "replay"), "checkpoint phases")
    observations = {initial["path"]} | {v["path"] for v in phases.values()}
    cp_paths = [r["path"] for r in receipt["checkpoints"].values()]
    need(
        len(set(cp_paths)) == 3 and set(cp_paths).isdisjoint(observations),
        "checkpoint/observation paths must be distinct before reading checkpoint bodies",
    )
    checkpoints = {phase: leaf(row["path"]) for phase, row in receipt["checkpoints"].items()}
    need(checkpoints == receipt["checkpoints"], "checkpoint bytes changed after capture")
    need(
        type(receipt["sources"]) is list
        and len(receipt["sources"]) == 1
        and type(receipt["program_irs"]) is list
        and len(receipt["program_irs"]) == 1,
        "CPP/carried IR inventory differs",
    )
    cpp, ir = leaf(receipt["sources"][0]["path"]), leaf(receipt["program_irs"][0]["path"])
    need(
        cpp["sha256"] == receipt["sources"][0]["sha256"]
        and ir["sha256"] == receipt["program_irs"][0]["sha256"],
        "CPP/IR bytes differ",
    )
    files = {directory / "receipt.json"} | {
        Path(v["path"]) for v in [initial, cpp, ir, *phases.values(), *checkpoints.values()]
    }
    need(
        len(files) == 11
        and all(p.parent == directory for p in files)
        and set(directory.iterdir()) == files,
        "closed eleven-file Candidate@2 inventory differs",
    )
    return dict(
        directory=str(directory),
        artifact=receipt["artifact"],
        receipt=leaf(directory / "receipt.json"),
        initial=initial,
        phases=phases,
        checkpoints=checkpoints,
        cpp=cpp,
        ir=ir,
    )


def ir_contract(ir_raw, cpp_raw, width, claimed_hash):
    names, ir = base.cpp_contract(cpp_raw, width, ir_raw)

    def prune(node):
        if type(node) is dict:
            return {k: prune(v) for k, v in node.items() if k != "provenance"}
        if type(node) is list:
            return [prune(v) for v in node]
        return node

    actual = digest(json.dumps(prune(ir), sort_keys=True, separators=(",", ":")).encode())
    need(actual == claimed_hash, "carried IR scientific digest differs")
    need(
        re.findall(
            r'extern "C" const char\* pops_program_hash\(\) \{ return "([0-9a-f]{64})"; \}',
            cpp_raw.decode(),
        )
        == [actual],
        "actual CPP/carried IR Program hash differs",
    )
    for n in ir["nodes"]:
        if n["op"] == "state":
            p = n["point"]
            need(
                type(p["schema_version"]) is int
                and p["schema_version"] == 1
                and type(p["step"]) is int
                and p["step"] == 0,
                "readonly point schema/clock index differs",
            )
    return {name.rsplit(".", 1)[1]: name for name in names}, ir


def temporal_contract(temporal, phase, width, ir):
    integer_keys = {
        "schema_version",
        "macro_step",
        "steps",
        "tick",
        "ticks_per_macro",
        "depth",
        "ring_slots",
        "ncomp",
        "lag",
        "newest_lag",
        "oldest_lag",
        "newest_tick",
        "oldest_tick",
        "valid_lags",
        "accepted",
        "failed",
        "rejected",
    }

    def typed(node):
        if type(node) is dict:
            for key, value in node.items():
                if key in integer_keys and not (key == "macro_step" and type(value) is dict):
                    need(type(value) is int, "temporal integer authority has wrong type")
                typed(value)
        elif type(node) is list:
            for value in node:
                typed(value)

    typed(temporal)
    typed(ir["clock"])
    time, step = CLOCKS[phase]
    exact(
        temporal,
        (
            "schema_version",
            "clock",
            "status",
            "synchronized",
            "controller_state",
            "event_queue",
            "clock_cursors",
            "schedule_cursors",
            "synchronization_cursors",
            "history_cursors",
            "cache_cursors",
            "program_schedule",
            "strategy",
            "transaction_stats",
        ),
        "TemporalV2 closed schema",
    )
    need(
        type(temporal["schema_version"]) is int
        and temporal["schema_version"] == 2
        and temporal["clock"] == dict(time=time.hex(), macro_step=step)
        and type(temporal["clock"]["macro_step"]) is int
        and temporal["status"] == "accepted"
        and temporal["synchronized"] is True,
        "temporal accepted clock differs",
    )
    clock = "pops.clock.v1::sha256:" + digest(
        json.dumps(ir["clock"], sort_keys=True, separators=(",", ":")).encode()
    )
    need(
        temporal["clock_cursors"] == {clock: dict(phase="accepted", tick=step, time=time.hex())},
        "logical clock cursor differs",
    )
    need(
        temporal["schedule_cursors"] == {"macro_step": dict(phase="accepted", macro_step=step)}
        and temporal["synchronization_cursors"] == {}
        and temporal["cache_cursors"] == {}
        and temporal["event_queue"] == [],
        "schedule/synchronization/cache/event publication differs",
    )
    need(
        temporal["controller_state"]
        == dict(
            last_accepted_dt=DT.hex(),
            fixed_dt_grid=dict(
                schema_version=1, origin=(0.0).hex(), steps=step, macro_step=step, time=time.hex()
            ),
        ),
        "fixed dt issued grid differs",
    )
    strategy = dict(
        strategy=dict(kind="fixed_dt", dt=dict(kind="binary64", value=DT.hex())), controls={}
    )
    need(
        temporal["strategy"] == strategy
        and ir["step_transaction"]["strategy"] == strategy["strategy"],
        "exact temporal strategy differs",
    )
    need(
        temporal["transaction_stats"] == dict(accepted=step, failed=0, rejected=0)
        and all(type(v) is int for v in temporal["transaction_stats"].values()),
        "attempt publication counts differ",
    )
    expected = {
        f"q{i}": dict(
            clock=clock,
            cold_start_extended=False,
            initialized=True,
            newest_tick=step,
            oldest_tick=step - 1,
            valid_lags=1,
        )
        for i in range(width)
    }
    need(
        temporal["history_cursors"] == expected
        and all(
            type(v["newest_tick"]) is int
            and type(v["oldest_tick"]) is int
            and type(v["valid_lags"]) is int
            and type(v["initialized"]) is bool
            and type(v["cold_start_extended"]) is bool
            for v in temporal["history_cursors"].values()
        ),
        "history cursor validity differs",
    )
    schedule = temporal["program_schedule"]
    exact(
        schedule,
        (
            "kind",
            "schema_version",
            "primary_clock",
            "clocks",
            "histories",
            "schedules",
            "subcycles",
            "synchronizations",
        ),
        "temporal Program schedule",
    )
    need(
        schedule["kind"] == "pops.temporal-program-schedule"
        and type(schedule["schema_version"]) is int
        and schedule["schema_version"] == 1
        and schedule["primary_clock"] == clock
        and schedule["clocks"] == [dict(id=clock, descriptor=ir["clock"], ticks_per_macro=1)]
        and all(schedule[k] == [] for k in ("schedules", "subcycles", "synchronizations")),
        "declared clock schedule differs",
    )
    histories = []
    for i in range(width):
        name = f"q{i}"
        histories.append(
            dict(
                checkpoint_policy=dict(
                    kind="history-persistence",
                    payload=dict(policy="dense"),
                    protocol="pops.manifest",
                    schema_version=1,
                ),
                clock=clock,
                depth=1,
                interpolation=dict(dense_output=False, provider="exact", schema_version=1),
                name=name,
                ncomp=1,
                owner=None,
                ring_slots=2,
                space=dict(kind="scalar_field"),
                state=dict(kind="scalar_history", qualified_id="scalar-history:" + name),
                validity=dict(domain="accepted_clock_ticks", newest_lag=0, oldest_lag=1),
            )
        )
    need(
        schedule["histories"] == histories
        and ir["histories"]
        == [dict(lag=1, name=f"q{i}", ncomp=1, state=None) for i in range(width)],
        "declared exact history schedule differs",
    )


def checkpoint(raw, phase, saved, first, artifact, abi, ranks, names, ir, program_hash):
    arrays, manifest = wire.envelope(raw, "accepted", abi, artifact=artifact)
    need(
        manifest["runtime_kind"] == "uniform"
        and protocol.scalar(arrays["pops_checkpoint_version"], "int") == 8,
        "checkpoint version/runtime differs",
    )
    time, step = CLOCKS[phase]
    need(
        protocol.scalar(arrays["t"], "real").hex() == time.hex()
        and protocol.scalar(arrays["macro_step"], "int") == step,
        "checkpoint phase clock differs",
    )
    geometry = strict_json(str(arrays["pops_spatial_contract"].item()))
    exact(
        geometry,
        (
            "schema_version",
            "dimension",
            "shape",
            "lower",
            "upper",
            "periodicity",
            "refinement_ratios",
            "native_layout_identity",
            "identity",
        ),
        "spatial contract",
    )
    need(
        type(geometry["schema_version"]) is int
        and geometry["schema_version"] == 1
        and type(geometry["dimension"]) is int
        and geometry["dimension"] == 2
        and geometry["shape"] == [16, 16]
        and all(type(v) is int for v in geometry["shape"])
        and geometry["lower"] == [(0.0).hex()] * 2
        and geometry["upper"] == [(1.0).hex()] * 2
        and geometry["periodicity"] == [True, True]
        and all(type(v) is bool for v in geometry["periodicity"])
        and geometry["refinement_ratios"] == [],
        "checkpoint physical geometry differs",
    )
    need(
        re.fullmatch(
            r"pops\.native-spatial-layout\.v1:sha256:[0-9a-f]{64}",
            geometry["native_layout_identity"],
        ),
        "native layout identity absent",
    )
    need(
        geometry["identity"]
        == "pops.checkpoint-spatial-layout.v1:sha256:"
        + wire.identity_hash(
            "checkpoint-spatial-layout", {k: v for k, v in geometry.items() if k != "identity"}
        ),
        "spatial identity differs",
    )
    width = saved["solution"].shape[0]
    need(
        list(arrays["blocks"]) == ["forcing", "material", "response"],
        "checkpoint block order differs",
    )
    for block, labels in (
        ("response", [f"u{i}" for i in range(width)]),
        ("forcing", [f"f{i}" for i in range(width)]),
        ("material", ["alpha"]),
    ):
        need(
            list(arrays["names_" + block]) == labels
            and protocol.scalar(arrays["ncomp_" + block], "int") == len(labels),
            "component ordering differs",
        )
        value = arrays["state_" + block]
        need(
            value.dtype == np.float64 and value.size == len(labels) * 256,
            "checkpoint physical shape differs",
        )
        same(value.reshape(len(labels), 16, 16), saved[block], "checkpoint physical " + block)
    need(
        list(arrays["history_names"]) == [f"q{i}" for i in range(width)], "history registry differs"
    )
    for i in range(width):
        name = f"q{i}"
        need(
            protocol.scalar(arrays["history_depth_" + name], "int") == 2
            and protocol.scalar(arrays["history_ncomp_" + name], "int") == 1
            and arrays["history_init_" + name].item() is True
            and protocol.scalar(arrays["history_fill_count_" + name], "int") == min(step, 2)
            and arrays["history_stored_slots_" + name].dtype == np.int64
            and arrays["history_stored_slots_" + name].tolist() == [0, 1],
            "history accepted storage differs",
        )
        same(arrays["history_slot_dt_" + name], np.array([DT, DT]), "history durations")
        sample = arrays["history_sample_identity_" + name]
        need(sample.dtype == np.uint8 and sample.ndim == 1, "history identity storage differs")
        history_identity(sample.tobytes(), name, step)
        for slot, expected in enumerate((first["solution"][i], saved["solution"][i])):
            value = arrays[f"history_{name}_{slot}"]
            need(value.dtype == np.float64 and value.size == 256, "history geometry differs")
            same(value.reshape(16, 16), expected, "history solution")
    diagnostic_science(diagnostic_images(arrays, ranks), names)
    auxiliary = arrays["auxiliary_checkpoint"]
    need(
        auxiliary.dtype == np.uint8
        and auxiliary.ndim == 1
        and auxiliary.size >= 8
        and auxiliary[:8].tobytes() == b"POPSAUX2",
        "exact accepted auxiliary image absent",
    )
    exchanges, offsets = arrays["program_exchange_state"], arrays["program_exchange_offsets"]
    need(
        exchanges.dtype == np.uint8
        and exchanges.ndim == 1
        and offsets.dtype == np.int64
        and offsets.shape == (ranks + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(exchanges)
        and np.all(offsets[1:] > offsets[:-1]),
        "exchange rank offsets differ",
    )
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        image = exchanges[int(lo) : int(hi)].tobytes()
        need(
            image in (b"POPSEX01" + bytes(8), b"POPSEX02" + bytes(24)),
            "unexpected accepted physical exchange/consumption",
        )
    need(arrays["program_hash"].item() == program_hash, "checkpoint Program identity differs")
    temporal = strict_json(str(arrays["temporal_restart_state"].item()))
    temporal_contract(temporal, phase, width, ir)
    return arrays


def junit_others(raw):
    root = ET.fromstring(raw)
    prefix = "test_public_candidate_diffusion_nonconstant_saved_and_exact_replay["
    rows = [
        dict(classname=t.get("classname"), name=t.get("name"))
        for t in root.findall(".//testcase")
        if not t.get("name", "").startswith(prefix)
    ]
    need(
        all(type(v) is str and bool(v) for row in rows for v in row.values())
        and len({(r["classname"], r["name"]) for r in rows}) == len(rows),
        "foreign JUnit inventory duplicates/unnamed",
    )
    return rows


def junit(raw, rank, ranks, cases, others=None):
    root = ET.fromstring(raw)
    need(
        not any(root.findall(".//" + tag) for tag in ("failure", "error", "skipped")),
        "complete raw Native batch has failure/error/skip",
    )
    if others is not None:
        need(
            type(others) is list and junit_others(raw) == others,
            "ROOT-sealed other JUnit cases differ",
        )
    return base.junit(raw, rank, ranks, cases)


def assemble(root, directories, junits, owner, roots):
    root = wire.canonical(root)
    need(
        type(roots) is list
        and str(root) in roots
        and len(roots) == len(set(roots))
        and len(directories) == 2
        and len(junits) in (1, 2),
        "owner root/topology inventory differs",
    )
    origins(owner, roots)
    cases = {}
    for directory in directories:
        case = case_inventory(directory)
        need(Path(case["directory"]).is_relative_to(root), "case outside archive")
        receipt = strict_json(pinned(case["receipt"], roots)[1])
        key = next(
            (k for k, (w, o) in CASES.items() if (receipt["width"], receipt["order"]) == (w, o)),
            None,
        )
        need(key is not None and key not in cases, "foreign/duplicate case")
        receipt_contract(receipt, *CASES[key], len(junits), owner)
        base.linkage(owner["execution_association"][key], receipt, case["cpp"], roots)
        ir_contract(
            pinned(case["ir"], roots)[1],
            pinned(case["cpp"], roots)[1],
            CASES[key][0],
            receipt["program_irs"][0]["program_hash"],
        )
        cases[key] = case
    exact(cases, CASES, "two case inventory")
    junit_pins = [leaf(p) for p in junits]
    for rank, pin in enumerate(junit_pins):
        junit(pinned(pin, roots)[1], rank, len(junits), cases)
    return dict(
        schema=SCHEMA,
        qualification=QUALIFICATION,
        archive_root=str(root),
        file_roots=roots,
        mode="serial" if len(junits) == 1 else "mpi2",
        ranks=len(junits),
        owner=owner,
        junit=junit_pins,
        junit_others=[junit_others(pinned(pin, roots)[1]) for pin in junit_pins],
        cases=cases,
    )


def receive(pins_path, pins_sha, approval_path, approval_sha):
    need(
        all(type(v) is str and re.fullmatch("[0-9a-f]{64}", v) for v in (pins_sha, approval_sha))
        and pins_path is not None
        and approval_path is not None,
        "two external seals and ROOT approval required before reception",
    )
    raw, approved = read(pins_path)[1], read(approval_path)[1]
    need(digest(raw) == pins_sha and digest(approved) == approval_sha, "external seals differ")
    need(
        strict_json(approved)
        == dict(
            schema="sol61.candidate-d-root-approval@2",
            approved_by="ROOT",
            pins_sha256=pins_sha,
            qualification=QUALIFICATION,
        ),
        "ROOT approval@2 required",
    )
    pins = strict_json(raw)
    exact(
        pins,
        (
            "schema",
            "qualification",
            "archive_root",
            "file_roots",
            "mode",
            "ranks",
            "owner",
            "junit",
            "junit_others",
            "cases",
        ),
        "owner pins@2",
    )
    need(
        pins["schema"] == SCHEMA
        and pins["qualification"] == QUALIFICATION
        and type(pins["ranks"]) is int
        and pins["ranks"] in (1, 2)
        and pins["mode"] == ("serial" if pins["ranks"] == 1 else "mpi2"),
        "checkpoint@2 scope differs",
    )
    roots = pins["file_roots"]
    need(
        type(roots) is list and pins["archive_root"] in roots and len(roots) == len(set(roots)),
        "file roots differ",
    )
    origins(pins["owner"], roots)
    exact(pins["cases"], CASES, "actual cases")
    need(
        len(pins["junit"]) == pins["ranks"]
        and len({r["path"] for r in pins["junit"]}) == pins["ranks"],
        "all-rank JUnit pins differ",
    )
    need(
        type(pins["junit_others"]) is list and len(pins["junit_others"]) == pins["ranks"],
        "all-rank sealed JUnit extras absent",
    )
    batches = [
        junit(pinned(row, roots)[1], rank, pins["ranks"], pins["cases"], pins["junit_others"][rank])
        for rank, row in enumerate(pins["junit"])
    ]
    reports = {}
    for key, case in pins["cases"].items():
        need(
            Path(case["directory"]).is_relative_to(wire.canonical(pins["archive_root"])),
            "case escapes archive",
        )
        need(case_inventory(case["directory"]) == case, "sealed case inventory differs")
        receipt = strict_json(pinned(case["receipt"], roots)[1])
        width, order = CASES[key]
        receipt_contract(receipt, width, order, pins["ranks"], pins["owner"])
        binaries = base.linkage(
            pins["owner"]["execution_association"][key], receipt, case["cpp"], roots
        )
        names, ir = ir_contract(
            pinned(case["ir"], roots)[1],
            pinned(case["cpp"], roots)[1],
            width,
            receipt["program_irs"][0]["program_hash"],
        )
        initial = protocol.archive(pinned(case["initial"], roots)[1])
        states = {p: protocol.archive(pinned(row, roots)[1]) for p, row in case["phases"].items()}
        math_result = science(initial, states, width)
        images = {
            p: checkpoint(
                pinned(row, roots)[1],
                p,
                states[p],
                states["accepted"],
                receipt["artifact"],
                pins["owner"]["abi_key"],
                pins["ranks"],
                names,
                ir,
                receipt["program_irs"][0]["program_hash"],
            )
            for p, row in case["checkpoints"].items()
        }
        checkpoint(
            pinned(case["checkpoints"]["accepted"], roots)[1],
            "reloaded",
            states["reloaded"],
            states["accepted"],
            receipt["artifact"],
            pins["owner"]["abi_key"],
            pins["ranks"],
            names,
            ir,
            receipt["program_irs"][0]["program_hash"],
        )
        need(
            set(images["continuous"]) == set(images["replay"]),
            "exact replay checkpoint inventory differs",
        )
        for name in set(images["continuous"]) - {
            "pops_checkpoint_manifest",
            "pops_restart_identity",
        }:
            same(images["continuous"][name], images["replay"][name], "exact native replay " + name)
        reports[key] = dict(
            science=math_result,
            component_bytes=binaries,
            checkpoint_arrays=len(images["accepted"]),
            diagnostics_and_history="received",
        )
    return dict(
        schema="sol61.candidate-d-scientific-reception@2",
        qualification=QUALIFICATION,
        status="received",
        reviewed_fixture=FIXTURE_REVISION,
        source_commit=pins["owner"]["source_commit"],
        native_build_source_commit=pins["owner"]["native_build_source_commit"],
        abi_key=pins["owner"]["abi_key"],
        native_sha256=pins["owner"]["native"]["sha256"],
        sdk_sha256=pins["owner"]["sdk"]["sha256"],
        ranks=pins["ranks"],
        junit_batches=batches,
        cases=reports,
        checkpoint_reception=True,
        documentary_ir_reception=True,
        component_binary_binding_recomputed=True,
        cryptographic_aggregate_binding=False,
        execution_association=base.ASSOCIATION,
        gaps=[
            "reloaded checkpoint not independently saved; accepted checkpoint anchored to exact reloaded NPZ",
            "no aggregate payload/block CPP/private lease snapshot/CPP-to-DSO graph proof",
            "no AMR/GPU/convergence/arbitrary-D/SPD/partition qualification",
        ],
        evidence="externally ROOT-approved originals; no Native execution by reader",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble")
    for name in ("archive-root", "owner", "output"):
        a.add_argument("--" + name, required=True)
    for name in ("case", "junit", "file-root"):
        a.add_argument("--" + name, action="append", required=True)
    r = sub.add_parser("receive")
    for name in ("pins", "pins-sha256", "approval", "approval-sha256"):
        r.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "assemble":
        result = assemble(
            args.archive_root,
            args.case,
            args.junit,
            strict_json(read(args.owner)[1]),
            args.file_root,
        )
        target = Path(args.output)
        need(not target.exists(), "pending inventory output already exists")
        target.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print("pending ROOT external approval; no Native/scientific reception")
    else:
        print(
            json.dumps(
                receive(args.pins, args.pins_sha256, args.approval, args.approval_sha256),
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
