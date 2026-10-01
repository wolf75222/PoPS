"""Synthetic protocol/math tests only. No PoPS, native execution or native NPZ.

All temporary arrays and fake identities are explicitly synthetic test inputs;
no external ROOT approval or positive native archive is created.
"""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("captured_d_review", Path(__file__).with_name("sol61_captured_diffusion_saved_reception.py"))
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def independent_initial(width, n=16):
    # Independent cell-centre evaluation with scalar libm and direct per-cell stencil.
    q = np.empty((width, n, n))
    alpha = np.empty((n, n))
    for y in range(n):
        for x in range(n):
            px, py = 2*math.pi*(x+.5)/n, 2*math.pi*(y+.5)/n
            alpha[y, x] = .25*math.sin(px)+.15*math.cos(py)
            for c in range(width):
                q[c, y, x] = .15+.025*math.cos((c+1)*px)+.02*math.sin(py)
    d = np.array([[.012]]) if width == 1 else np.array([[.012, .002, 0], [-.001, .014, 0], [.001, 0, 0]])
    reaction = np.array([[1.1]]) if width == 1 else np.array([[1.1, .03, -.01], [-.02, 1.3, .02], [.01, -.03, 1.5]])
    f = np.zeros_like(q)
    for c in range(width):
        for y in range(n):
            for x in range(n):
                local = sum(reaction[c, j]*q[j, y, x] for j in range(width))+.2*q[c, y, x]**3
                for j in range(width):
                    for yy, xx in (((y-1) % n, x), ((y+1) % n, x), (y, (x-1) % n), (y, (x+1) % n)):
                        local += n*n*d[c, j]*(1+(alpha[y, x]+alpha[yy, xx])/2)*(q[j, y, x]-q[j, yy, xx])
                f[c, y, x] = local
    initial = dict(response=np.zeros_like(q), forcing=f, material=alpha[None], target=q)
    phases = {phase: dict(response=q*step*r.DT, forcing=f.copy(), material=alpha[None].copy(), solution=q.copy(),
                          time=np.array(time), step=np.array(step)) for phase, (time, step) in r.CLOCKS.items()}
    return initial, phases


@pytest.mark.parametrize("width,n", [(1, 8), (1, 16), (3, 8), (3, 16)])
def test_independent_faces_and_declared_samples(width, n):
    initial, states = independent_initial(width, n)
    reports = r.science(initial, states, width, n)
    assert max(row["original_residual_relative_l2"] for row in reports.values()) < 5e-14
    _, flux = r.original(initial["target"], initial["material"][0])
    np.testing.assert_allclose(flux.sum(axis=(1, 2)), 0, atol=2e-13)
    q = initial["target"]
    _, constant_flux = r.original(q, np.zeros((n, n)))
    centres = (np.arange(n)+.5)/n
    modes = np.stack([4*n*n*(.025*math.sin(math.pi*(c+1)/n)**2*np.cos(2*math.pi*(c+1)*centres)[None, :]
                             +.02*math.sin(math.pi/n)**2*np.sin(2*math.pi*centres)[:, None]) for c in range(width)])
    d, _ = r.matrices(width)
    np.testing.assert_allclose(constant_flux, np.einsum("ij,jyx->iyx", np.array(d, dtype=float), modes), atol=2e-13, rtol=0)


def test_general_singular_signed_matrix_is_not_restricted_to_spd():
    d, _ = r.matrices(3)
    assert d[1][0] < 0 and d[0][1] != d[1][0] and all(row[2] == 0 for row in d)
    initial, states = independent_initial(3)
    assert r.science(initial, states, 3)


@pytest.mark.parametrize("substitute", ("harmonic", "symmetrized_psd", "uniform"))
def test_resealed_false_scientific_operator_still_fails(substitute, tmp_path):
    initial, states = independent_initial(3)
    d, _ = r.matrices(3)
    if substitute == "harmonic":
        wrong, _ = r.original(initial["target"], initial["material"][0], harmonic=True)
    elif substitute == "uniform":
        wrong, _ = r.original(initial["target"], np.zeros((16, 16)))
    else:
        d = np.array(d, dtype=float)
        eigen, vectors = np.linalg.eigh((d+d.T)/2)
        fake = (vectors*np.maximum(eigen, 0))@vectors.T
        wrong, _ = r.original(initial["target"], initial["material"][0], diffusion=fake)
    initial["forcing"] = wrong
    for row in states.values():
        row["forcing"] = wrong.copy()
    # Entire synthetic payload and receipt hashes are freshly sealed, so physics
    # is tested beyond the integrity guard. No positive native receipt is forged.
    path = tmp_path/"synthetic-resealed.npz"
    np.savez(path, **initial)
    record = r.leaf(path)
    receipt = dict(kind="synthetic-protocol-only", state=record)
    receipt_path = tmp_path/"synthetic-resealed.json"
    receipt_path.write_text(json.dumps(receipt))
    assert r.pinned(r.leaf(receipt_path), [str(tmp_path)])[1]
    loaded = r.protocol.archive(r.pinned(record, [str(tmp_path)])[1])
    with pytest.raises(ValueError, match="initial forcing differs from original declared operator"):
        r.science(loaded, states, 3)


@pytest.mark.parametrize("attack,message", [("doubled_dt", "consumer duration"), ("missing_capture", "exact keys"),
    ("mutated_material", "readonly capture"), ("constant_q", "constant/incorrectly"),
    ("permuted_q", "independent declared target"), ("clock", "exact clock"), ("replay_signzero", "differs in bytes")])
def test_source_consumer_capture_clock_and_replay_attacks(attack, message):
    initial, states = independent_initial(3)
    row = states["accepted"]
    if attack == "doubled_dt":
        row["response"] *= 2
    elif attack == "missing_capture":
        del row["material"]
    elif attack == "mutated_material":
        row["material"][0, 0, 0] += .01
    elif attack == "constant_q":
        row["solution"][:] = .15
    elif attack == "permuted_q":
        row["solution"] = row["solution"][[2, 0, 1]]
    elif attack == "clock":
        row["time"] = np.array(.02)
    else:
        # Deliberately sub-tolerance perturbation, caught only by byte replay guard.
        states["replay"]["solution"][0, 0, 0] = np.nextafter(states["replay"]["solution"][0, 0, 0], math.inf)
    with pytest.raises(ValueError, match=message):
        r.science(initial, states, 3)


def synthetic_point(step, name="q0"):
    header = b"POPSHID1"+struct.pack("<Q", len(name))+name.encode()+struct.pack("<qQ", -1, 2)
    return header+b"".join(struct.pack("<QQQQ", 2,
        int.from_bytes(struct.pack("<d", start), "little"),
        int.from_bytes(struct.pack("<d", r.DT), "little"), 1) for start in (0., (step-1)*r.DT))


@pytest.mark.parametrize("step", (1, 2))
def test_history_point_actual_window(step):
    r.history_point(synthetic_point(step), "q0", step)


@pytest.mark.parametrize("attack", ("start", "duration", "ordinal", "ring"))
def test_stale_history_point_resealed(attack):
    raw = bytearray(synthetic_point(2))
    offset = len(raw)-32
    if attack == "ring":
        raw[16] = ord("x")
    else:
        word = {"start": 1, "duration": 2, "ordinal": 3}[attack]
        raw[offset+8*word:offset+8*(word+1)] = struct.pack("<Q", 0)
    with pytest.raises(ValueError, match="history identity|stale captured"):
        r.history_point(bytes(raw), "q0", 2)


def fake_token(domain):
    return dict(domain=domain, schema_version=1, algorithm="sha256", hexdigest="a"*64)


def synthetic_checkpoint(saved, first_accepted):
    width = saved["solution"].shape[0]
    time, step = float(saved["time"]), int(saved["step"])
    geometry = dict(schema_version=1, dimension=2, shape=[16, 16], lower=[0..hex()]*2, upper=[1..hex()]*2,
                    periodicity=[True, True], refinement_ratios=[], native_layout_identity="pops.native-spatial-layout.v1:sha256:"+"b"*64)
    geometry["identity"] = "pops.checkpoint-spatial-layout.v1:sha256:"+r.wire.identity_hash("checkpoint-spatial-layout", geometry)
    temporal = dict(clock=dict(time=time.hex(), macro_step=step), status="accepted", synchronized=True,
                    controller_state=dict(last_accepted_dt=r.DT.hex()), **{key: {} for key in (
                        "clock_cursors", "schedule_cursors", "synchronization_cursors", "history_cursors", "cache_cursors")})
    out = dict(t=np.array(time), macro_step=np.array(step), abi_key=np.array("synthetic-ABI-only"),
               pops_spatial_contract=np.array(json.dumps(geometry)), blocks=np.array(["response", "forcing", "material"]),
               temporal_restart_state=np.array(json.dumps(temporal)), auxiliary_checkpoint=np.array([1], dtype=np.uint8),
               program_exchange_state=np.array([], dtype=np.uint8), program_exchange_offsets=np.array([0], dtype=np.int64),
               history_names=np.array([f"q{i}" for i in range(width)]))
    for block, names in (("response", [f"u{i}" for i in range(width)]), ("forcing", [f"f{i}" for i in range(width)]), ("material", ["alpha"])):
        out["state_"+block] = saved[block].copy()
        out["ncomp_"+block], out["names_"+block] = np.array(len(names)), np.array(names)
    for i in range(width):
        name = f"q{i}"
        for prefix, value in (("history_depth_", 2), ("history_ncomp_", 1), ("history_init_", True), ("history_fill_count_", min(step, 2)),
                              ("history_stored_slots_", [0, 1]), ("history_slot_dt_", [r.DT, r.DT])):
            out[prefix+name] = np.array(value)
        out["history_"+name+"_0"] = first_accepted["solution"][i:i+1].copy()
        out["history_"+name+"_1"] = saved["solution"][i:i+1].copy()
        out["history_sample_identity_"+name] = np.frombuffer(synthetic_point(step, name), dtype=np.uint8).copy()
    return out


def reseal_checkpoint(arrays):
    arrays = {k: v for k, v in arrays.items() if k not in ("pops_checkpoint_manifest", "pops_restart_identity")}
    manifest = dict(schema_version=1, runtime_kind="uniform", **{domain+"_identity": fake_token(domain)
                    for domain in ("semantic", "artifact", "bind", "run")},
                    clock=dict(time=float(arrays["t"]).hex(), macro_step=int(arrays["macro_step"])),
                    arrays={k: r.wire.typed_array(v) for k, v in arrays.items()})
    hashed = r.digest(r.protocol.cbor(dict(protocol="pops.identity", domain="restart", schema_version=1, payload=manifest)))
    manifest["restart_identity"] = dict(domain="restart", schema_version=1, algorithm="sha256", hexdigest=hashed)
    arrays["pops_checkpoint_manifest"] = np.array(json.dumps(manifest))
    arrays["pops_restart_identity"] = np.array("pops.restart.v1:sha256:"+hashed)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


@pytest.mark.parametrize("width", (1, 3))
def test_checkpoint_protocol_synthetic_baseline(width):
    _, states = independent_initial(width)
    for phase in ("accepted", "continuous", "replay"):
        raw = reseal_checkpoint(synthetic_checkpoint(states[phase], states["accepted"]))
        r.checkpoint(raw, phase, states[phase], "pops.artifact.v1:sha256:"+"a"*64, "synthetic-ABI-only", first_accepted=states["accepted"])


@pytest.mark.parametrize("attack,message", [("axis", "coordinate/axis"), ("periodic", "coordinate/axis"),
    ("state", "checkpoint state"), ("historypoint", "stale captured"), ("cursor", "cursor point"), ("duration", "boundary/window")])
def test_fully_resealed_checkpoint_scientific_guards(attack, message):
    _, states = independent_initial(3)
    out = synthetic_checkpoint(states["continuous"], states["accepted"])
    if attack in ("axis", "periodic"):
        geometry = json.loads(str(out["pops_spatial_contract"]))
        if attack == "axis":
            geometry["upper"][1] = 2..hex()
        else:
            geometry["periodicity"][1] = False
        payload = {k: v for k, v in geometry.items() if k != "identity"}
        geometry["identity"] = "pops.checkpoint-spatial-layout.v1:sha256:"+r.wire.identity_hash("checkpoint-spatial-layout", payload)
        out["pops_spatial_contract"] = np.array(json.dumps(geometry))
    elif attack == "state":
        out["state_forcing"][0, 0, 0] += .01
    elif attack == "historypoint":
        out["history_sample_identity_q0"] = np.frombuffer(synthetic_point(1), dtype=np.uint8).copy()
    else:
        temporal = json.loads(str(out["temporal_restart_state"]))
        if attack == "cursor":
            temporal["clock_cursors"]["synthetic-clock"] = dict(phase="accepted", time=.01.hex())
        else:
            temporal["controller_state"]["last_accepted_dt"] = .02.hex()
        out["temporal_restart_state"] = np.array(json.dumps(temporal))
    raw = reseal_checkpoint(out)
    with pytest.raises(ValueError, match=message):
        r.checkpoint(raw, "continuous", states["continuous"], "pops.artifact.v1:sha256:"+"a"*64, "synthetic-ABI-only", first_accepted=states["accepted"])


def junit_image(rank=0, size=2):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    cases = {key: dict(directory="/synthetic-only/"+key, artifact="synthetic-artifact", receipt=dict(path="/synthetic-only/"+key+"/receipt.json")) for key in r.CASES}
    for key, case in cases.items():
        suffix = "1-order0" if key == "scalar1" else "3-order1"
        test = ET.SubElement(suite, "testcase", classname="tests.python.integration.runtime.test_public_captured_diffusion",
                             name="test_public_captured_diffusion_nonconstant_saved_and_exact_replay["+suffix+"]")
        properties = ET.SubElement(test, "properties")
        for name, value in dict(artifact_identity=case["artifact"], dimension=2, rank=rank, size=size,
                                evidence_path=case["directory"], captured_diffusion_receipt=case["receipt"]["path"]).items():
            ET.SubElement(properties, "property", name=name, value=str(value))
    return root, cases


@pytest.mark.parametrize("rank,size", [(0, 1), (0, 2), (1, 2)])
def test_exact_junit_each_rank(rank, size):
    root, cases = junit_image(rank, size)
    r.junit(ET.tostring(root), rank, size, cases)


@pytest.mark.parametrize("attack", ("failure", "skip", "foreignrealm", "rank", "duplicate"))
def test_junit_no_false_positive(attack):
    root, cases = junit_image()
    test = root.find(".//testcase")
    if attack in ("failure", "skip"):
        ET.SubElement(test, "failure" if attack == "failure" else "skipped")
    elif attack == "foreignrealm":
        test.set("classname", "synthetic.other_module")
    elif attack == "rank":
        test.find("./properties/property[@name='rank']").set("value", "1")
    else:
        root.find("testsuite").append(copy.deepcopy(test))
    with pytest.raises(ValueError, match="JUnit"):
        r.junit(ET.tostring(root), 0, 2, cases)


@pytest.mark.parametrize("attack", ("alias", "parent_alias", "bytes", "escape"))
def test_closed_root_alias_and_unresealed_bytes(tmp_path, attack):
    origin = tmp_path/"origin"
    origin.mkdir()
    file = origin/"synthetic-only.bin"
    file.write_bytes(b"synthetic payload")
    record = r.leaf(file)
    if attack == "bytes":
        file.write_bytes(b"corrupt synthetic payload")
    elif attack == "alias":
        alias = origin/"alias"
        alias.symlink_to(file)
        record["path"] = str(alias)
    elif attack == "parent_alias":
        alias = tmp_path/"alias-parent"
        alias.symlink_to(origin, target_is_directory=True)
        record["path"] = str(alias/file.name)
    else:
        record["path"] = str(file)
        (tmp_path/"foreign-root").mkdir()
    with pytest.raises(ValueError, match="aliases|digest|outside"):
        r.pinned(record, [str(origin if attack != "escape" else tmp_path/"foreign-root")])


def test_missing_external_seals_fail_before_any_native_claim(tmp_path):
    pins, approval = tmp_path/"synthetic-pins.json", tmp_path/"synthetic-approval.json"
    pins.write_text('{}')
    approval.write_text('{}')
    with pytest.raises(ValueError, match="external seal"):
        r.receive(pins, "0"*64, approval, "0"*64)


def test_strict_duplicate_and_nonfinite_json():
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e1000}'):
        with pytest.raises(ValueError):
            r.strict_json(raw)


def synthetic_case_directory(tmp_path):
    directory = tmp_path/"synthetic-protocol-case"
    directory.mkdir()
    initial = directory/"initial.npz"
    initial.write_bytes(b"synthetic unopened NPZ placeholder")
    phases = {}
    for phase in r.PHASES:
        path = directory/(phase+".npz")
        path.write_bytes(b"synthetic unopened NPZ placeholder")
        pin = r.leaf(path)
        phases[phase] = dict(npz=pin["path"], sha256=pin["sha256"], checks={})
    cps = {}
    for phase in ("accepted", "continuous", "replay"):
        path = directory/(phase+"-checkpoint.npz")
        path.write_bytes(b"synthetic unopened checkpoint placeholder")
        cps[phase] = r.leaf(path)
    cpp = directory/"synthetic-program.cpp"
    cpp.write_text("// synthetic unopened compiler source placeholder")
    source = r.leaf(cpp)
    receipt = dict(kind="synthetic-protocol-only", artifact="synthetic-artifact", initial_npz=str(initial),
                   initial_sha256=r.leaf(initial)["sha256"], phases=phases, checkpoints=cps,
                   sources=[dict(component="synthetic-program", **source)])
    path = directory/"receipt.json"
    path.write_text(json.dumps(receipt))
    return directory, path, receipt


def test_closed_inventory_synthetic_structure(tmp_path):
    directory, _, _ = synthetic_case_directory(tmp_path)
    assert len(list(directory.iterdir())) == 10
    assert r.case_inventory(directory)["directory"] == str(directory)


@pytest.mark.parametrize("attack", ("extra", "phase_reuse", "missing", "foreign"))
def test_closed_inventory_attacks(tmp_path, attack):
    directory, path, receipt = synthetic_case_directory(tmp_path)
    if attack == "extra":
        (directory/"foreign.bin").write_text("synthetic unrelated")
    elif attack == "phase_reuse":
        receipt["phases"]["replay"] = receipt["phases"]["accepted"]
    elif attack == "missing":
        del receipt["phases"]["reloaded"]
    else:
        outside = tmp_path/"foreign-source.cpp"
        outside.write_text("// synthetic foreign source")
        receipt["sources"] = [dict(component="synthetic-program", **r.leaf(outside))]
        (directory/"synthetic-program.cpp").unlink()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inventory|exact keys"):
        r.case_inventory(directory)


@pytest.fixture(scope="module")
def original_source_bytes():
    # Read the frozen original declaration as data only. No imported author oracle.
    return subprocess.check_output(["git", "show", "b39f4a9906f904cd2f857c2874878da8aa84084f:tests/python/support/captured_diffusion_mms.py"])


def test_real_frozen_source_declares_original_signed_recipe(original_source_bytes):
    r.declared_source(original_source_bytes)


@pytest.mark.parametrize("before,after", [(b"[-.001, .014, 0.]", b"[.001, .014, 0.]"),
    (b"lhs -= DivCoeffGrad", b"lhs += DivCoeffGrad"),
    (b"*(1+material[0])", b"*(1+2*material[0])"),
    (b"c=0)", b"c=1)"), (b"current.n+program.dt*rhs", b"current.n+2*program.dt*rhs")])
def test_source_original_body_point_and_consumer_mutations(original_source_bytes, before, after):
    assert before in original_source_bytes
    with pytest.raises(ValueError, match="canonicals|original body"):
        r.declared_source(original_source_bytes.replace(before, after))


def synthetic_owner(tmp_path, original_source_bytes):
    paths = {}
    for role in ("python_package", "sdk", "native", "fixture", "physical_helper"):
        path = tmp_path/("synthetic-"+role+".bin")
        path.write_bytes(original_source_bytes if role == "physical_helper" else b"synthetic origin bytes")
        paths[role] = r.leaf(path)
    return dict(schema="sol61.captured-d-execution-owner@2", source_commit="0"*40, native_build_source_commit=None,
                abi_key="synthetic-ABI-only", **{role: paths[role] for role in ("python_package", "sdk", "native")},
                sources={role: paths[role] for role in ("fixture", "physical_helper")}, cpp_dso_links=None)


def test_synthetic_owner_leaf_protocol_only(tmp_path, original_source_bytes):
    owner = synthetic_owner(tmp_path, original_source_bytes)
    r.origins(owner, [str(tmp_path)])


@pytest.mark.parametrize("attack", ("missing_sdk", "changed_native", "invented_link", "resealed_body"))
def test_missing_or_mutated_owner_never_inferred(tmp_path, original_source_bytes, attack):
    owner = synthetic_owner(tmp_path, original_source_bytes)
    if attack == "missing_sdk":
        del owner["sdk"]
    elif attack == "changed_native":
        Path(owner["native"]["path"]).write_bytes(b"drift synthetic native")
    elif attack == "invented_link":
        owner["cpp_dso_links"] = dict(synthetic="unreviewed")
    else:
        path = Path(owner["sources"]["physical_helper"]["path"])
        path.write_bytes(original_source_bytes.replace(b"lhs -= DivCoeffGrad", b"lhs += DivCoeffGrad"))
        owner["sources"]["physical_helper"] = r.leaf(path)
    with pytest.raises(ValueError, match="exact keys|digest|link format|original body"):
        r.origins(owner, [str(tmp_path)])


@pytest.mark.parametrize("attack", ("geometry_bool", "ncomp_bool", "history_bool"))
def test_resealed_typed_metadata_aliases(attack):
    _, states = independent_initial(1)
    out = synthetic_checkpoint(states["accepted"], states["accepted"])
    if attack == "geometry_bool":
        geom = json.loads(str(out["pops_spatial_contract"]))
        geom["periodicity"] = [1, 1]
        geom["identity"] = "pops.checkpoint-spatial-layout.v1:sha256:"+r.wire.identity_hash("checkpoint-spatial-layout", {k: v for k, v in geom.items() if k != "identity"})
        out["pops_spatial_contract"] = np.array(json.dumps(geom))
    elif attack == "ncomp_bool":
        out["ncomp_response"] = np.array(True)
    else:
        out["history_ncomp_q0"] = np.array(True)
    with pytest.raises(ValueError, match="geometry|integer clock"):
        r.checkpoint(reseal_checkpoint(out), "accepted", states["accepted"], "pops.artifact.v1:sha256:"+"a"*64, "synthetic-ABI-only", first_accepted=states["accepted"])


@pytest.mark.parametrize("phase", ("accepted", "continuous", "reloaded", "replay"))
def test_physical_history_rotation_distinct_stores(phase):
    # Protocol only: distinct arbitrary q values expose swaps hidden by a
    # stationary MMS. These are never passed to the scientific qualification.
    _, states = independent_initial(3)
    states["accepted"]["solution"] += .125
    states["reloaded"]["solution"] = states["accepted"]["solution"].copy()
    states["continuous"]["solution"] -= .25
    states["replay"]["solution"] = states["continuous"]["solution"].copy()
    arrays = synthetic_checkpoint(states[phase], states["accepted"])
    r.checkpoint(reseal_checkpoint(arrays), phase, states[phase],
        "pops.artifact.v1:sha256:"+"a"*64, "synthetic-ABI-only", first_accepted=states["accepted"])


@pytest.mark.parametrize("attack", ("max_lag_as_size", "fill_cold", "fill_high", "missing_slot", "slot_swap",
    "slot0_current", "slot1_prior", "window_swap", "slot0_start", "slot1_start", "ordinal_macrostep", "duration", "kind"))
def test_resealed_two_slot_history_guards(attack):
    _, states = independent_initial(1)
    states["accepted"]["solution"] += .125
    out = synthetic_checkpoint(states["continuous"], states["accepted"])
    if attack == "max_lag_as_size":
        out["history_depth_q0"] = np.array(1)
    elif attack in ("fill_cold", "fill_high"):
        out["history_fill_count_q0"] = np.array(1 if attack == "fill_cold" else 3)
    elif attack == "missing_slot":
        out["history_stored_slots_q0"] = np.array([1], dtype=np.int64)
        del out["history_q0_0"]
    elif attack == "slot_swap":
        out["history_q0_0"], out["history_q0_1"] = out["history_q0_1"], out["history_q0_0"]
    elif attack == "slot0_current":
        out["history_q0_0"] = out["history_q0_1"].copy()
    elif attack == "slot1_prior":
        out["history_q0_1"] = out["history_q0_0"].copy()
    elif attack == "duration":
        out["history_slot_dt_q0"][0] = .02
    else:
        raw = bytearray(out["history_sample_identity_q0"].tobytes())
        offset = len(raw)-64
        if attack == "window_swap":
            raw[offset:offset+32], raw[offset+32:] = raw[offset+32:], raw[offset:offset+32]
        else:
            slot, word, value = {"slot0_start": (0, 1, .01), "slot1_start": (1, 1, 0.),
                "ordinal_macrostep": (1, 3, 2), "kind": (0, 0, 1)}[attack]
            encoded = struct.pack("<d", value) if word == 1 else struct.pack("<Q", value)
            raw[offset+slot*32+word*8:offset+slot*32+(word+1)*8] = encoded
        out["history_sample_identity_q0"] = np.frombuffer(raw, dtype=np.uint8).copy()
    with pytest.raises(ValueError, match="history|publication point"):
        r.checkpoint(reseal_checkpoint(out), "continuous", states["continuous"],
            "pops.artifact.v1:sha256:"+"a"*64, "synthetic-ABI-only", first_accepted=states["accepted"])


def test_old_reception_scope_not_implicitly_upcast():
    assert r.QUALIFICATION == "saved-states-original-residual@2"
    _, states = independent_initial(1)
    old = synthetic_checkpoint(states["accepted"], states["accepted"])
    old["history_depth_q0"] = np.array(1)
    old["history_stored_slots_q0"] = np.array([0], dtype=np.int64)
    old["history_slot_dt_q0"] = np.array([r.DT])
    del old["history_q0_1"]
    with pytest.raises(ValueError, match="history initialization/storage"):
        r.checkpoint(reseal_checkpoint(old), "accepted", states["accepted"],
            "pops.artifact.v1:sha256:"+"a"*64, "synthetic-ABI-only", first_accepted=states["accepted"])


DIAGNOSTIC_NAMES = ["field_residual_4."+suffix for suffix in (
    "residual_norm", "reference_residual_norm", "rel_residual", "full_residual_evaluations", "finite_difference_jvps")]
DIAGNOSTIC_CPP = "\n".join('ctx.record_scalar('+json.dumps(name)+', value);' for name in DIAGNOSTIC_NAMES)


def synthetic_diagnostics(ranks=2):
    chunks = []
    for rank in range(ranks):
        values = {name.encode(): float(rank+1) for name in DIAGNOSTIC_NAMES}
        chunks.append(b"POPSDIA1"+struct.pack("<QQQQ", 64, rank, ranks, len(values))+
                      b"".join(struct.pack("<Q", len(name))+name+struct.pack("<d", value)
                               for name, value in sorted(values.items())))
    return dict(program_diagnostics_state=np.frombuffer(b"".join(chunks), dtype=np.uint8).copy(),
                program_diagnostics_offsets=np.array([0, *np.cumsum(list(map(len, chunks)))], dtype=np.int64))


@pytest.mark.parametrize("ranks", (1, 2))
def test_v3_actual_format_synthetic_rank_bits_and_payload8(ranks):
    arrays = dict(synthetic_diagnostics(ranks), pops_checkpoint_version=np.array(8), program_hash=np.array("b"*64))
    reports = r.checkpoint_v3(arrays, ranks, DIAGNOSTIC_CPP, "b"*64)
    assert len(reports) == ranks and (ranks == 1 or reports[0] != reports[1])
    assert r.QUALIFICATION == "saved-states-original-residual@2"
    assert r.scope(3) == "saved-states-original-residual@3"


@pytest.mark.parametrize("attack", ("absent", "dtype", "offset", "rankdup", "truncated", "count", "trailing",
                                  "nameorder", "nonfinite", "negative", "counter", "version", "programhash"))
def test_v3_resealed_diagnostics_or_payload_refusals(attack):
    arrays = dict(synthetic_diagnostics(), pops_checkpoint_version=np.array(8), program_hash=np.array("b"*64))
    raw = arrays["program_diagnostics_state"]
    if attack == "absent":
        del arrays["program_diagnostics_state"]
    elif attack == "dtype":
        arrays["program_diagnostics_offsets"] = arrays["program_diagnostics_offsets"].astype(np.uint64)
    elif attack == "offset":
        arrays["program_diagnostics_offsets"][1] += 1
    elif attack == "rankdup":
        start = arrays["program_diagnostics_offsets"][1]
        raw[start+16:start+24] = 0
    elif attack == "truncated":
        arrays["program_diagnostics_state"] = raw[:39].copy()
        arrays["program_diagnostics_offsets"] = np.array([0, 39], dtype=np.int64)
    elif attack == "count":
        raw[32:40] = 255
    elif attack == "trailing":
        arrays["program_diagnostics_state"] = np.concatenate((raw, np.array([0], dtype=np.uint8)))
        arrays["program_diagnostics_offsets"][-1] += 1
    elif attack == "nameorder":
        raw[48] = 255
    elif attack in ("nonfinite", "negative", "counter"):
        name_length = int.from_bytes(raw[40:48].tobytes(), "little")
        value = {"nonfinite": math.nan, "negative": -1., "counter": .5}[attack]
        raw[48+name_length:56+name_length] = np.frombuffer(struct.pack("<d", value), dtype=np.uint8)
    elif attack == "version":
        arrays["pops_checkpoint_version"] = np.array(7)
    else:
        arrays["program_hash"] = np.array("c"*64)
    with pytest.raises(ValueError):
        r.checkpoint_v3(arrays, 1 if attack == "truncated" else 2, DIAGNOSTIC_CPP, "b"*64)


def synthetic_ir_and_cpp():
    node = dict(id=0, name="synthetic-only", vtype="state", op="post_synchronization", attrs={"body_block": [
        dict(id=1, name="nested", vtype="scalar", op="scalar_op", attrs={"provenance": "semantic-attribute-kept"},
             inputs=[], point=None, provenance={"file": "synthetic/nested.py"})]},
        inputs=[], point=None, provenance={"file": "synthetic/caller.py"})
    ir = dict(version=10, nodes=[node])
    projected = copy.deepcopy(ir)
    del projected["nodes"][0]["provenance"], projected["nodes"][0]["attrs"]["body_block"][0]["provenance"]
    hashed = r.digest(json.dumps(projected, sort_keys=True, separators=(",", ":")).encode())
    cpp = 'extern "C" const char* pops_program_hash() { return "'+hashed+'"; }'
    return ir, cpp, hashed


def test_v3_program_projection_only_excludes_node_provenance():
    ir, cpp, hashed = synthetic_ir_and_cpp()
    assert r.program_identity(json.dumps(ir), cpp, hashed) == hashed
    ir["nodes"][0]["provenance"]["file"] = "different/synthetic-residence.py"
    assert r.program_identity(json.dumps(ir), cpp, hashed) == hashed
    ir["nodes"][0]["attrs"]["body_block"][0]["attrs"]["provenance"] = "changed-semantic-attribute"
    with pytest.raises(ValueError, match="hash link"):
        r.program_identity(json.dumps(ir), cpp, hashed)


@pytest.mark.parametrize("attack", ("body", "input", "point", "version", "cpp", "receipt", "doubleliteral"))
def test_v3_program_links_refuse_resealed_leaf_mutations(attack):
    ir, cpp, hashed = synthetic_ir_and_cpp()
    if attack == "body":
        ir["nodes"][0]["attrs"]["body_block"][0]["op"] = "different_body"
    elif attack == "input":
        ir["nodes"][0]["inputs"] = [99]
    elif attack == "point":
        ir["nodes"][0]["point"] = {"stage": "stale"}
    elif attack == "version":
        ir["version"] = 11
    elif attack == "cpp":
        cpp = cpp.replace(hashed, "c"*64)
    elif attack == "receipt":
        hashed = "c"*64
    else:
        cpp += "\n"+cpp
    with pytest.raises(ValueError):
        r.program_identity(json.dumps(ir), cpp, hashed)


@pytest.mark.parametrize("attack", (None, "unlisted", "failure", "skip", "duplicate"))
def test_v3_authentic_full_junit_scope_protocol(attack):
    root, cases = junit_image(0, 1)
    suite = root.find("testsuite")
    extra = ET.SubElement(suite, "testcase", name="synthetic-other-passed-test")
    names = [extra.get("name")]
    if attack == "failure":
        ET.SubElement(extra, "failure")
    elif attack == "skip":
        ET.SubElement(extra, "skipped")
    elif attack == "duplicate":
        ET.SubElement(suite, "testcase", name=extra.get("name"))
    elif attack == "unlisted":
        names = []
    if attack is None:
        r.junit(ET.tostring(root), 0, 1, cases, other_cases=names)
    else:
        with pytest.raises(ValueError):
            r.junit(ET.tostring(root), 0, 1, cases, other_cases=names)


def test_v3_closed_inventory_adds_real_ir_and_keeps_legacy_scope(tmp_path):
    directory, path, receipt = synthetic_case_directory(tmp_path)
    receipt["fixture_schema"] = "pops.captured-diffusion-native-fixture@3"
    ir_path = directory/"synthetic-program.ir.json"
    ir_path.write_text("synthetic unopened IR placeholder")
    receipt["program_irs"] = [dict(component="synthetic-program", program_hash="b"*64, **r.leaf(ir_path))]
    path.write_text(json.dumps(receipt))
    assert len(list(directory.iterdir())) == 11
    assert r.case_inventory(directory, fixture_version=3)["program_irs"] == [r.leaf(ir_path)]
    with pytest.raises(ValueError, match="inventory"):
        r.case_inventory(directory)
    receipt["checkpoints"]["accepted"] = receipt["phases"]["accepted"]
    path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, KeyError)):
        r.case_inventory(directory, fixture_version=3)


def synthetic_v3_temporal(width, step):
    # Protocol-only witness: an independent explicit lag/ring descriptor.
    clock = dict(schema_version=1, name="synthetic-macro", owner=None)
    identity = "pops.clock.v1::sha256:"+r.digest(json.dumps(clock, sort_keys=True, separators=(",", ":")).encode())
    policy = dict(kind="history-persistence", payload=dict(policy="dense"), protocol="pops.manifest", schema_version=1)
    names = [f"q{i}" for i in range(width)]
    ir = dict(clock=clock, histories=[dict(name=n, lag=1, ncomp=1, state=None) for n in names],
              history_persistence=[dict(name=n, depth=2, policy=policy) for n in names])
    time = .01 if step == 1 else .02
    schedule = dict(kind="pops.temporal-program-schedule", schema_version=1, primary_clock=identity,
                    clocks=[dict(descriptor=clock, id=identity, ticks_per_macro=1)], schedules=[],
                    synchronizations=[], subcycles=[], histories=[])
    cursors = {}
    for n in names:
        schedule["histories"].append(dict(name=n, depth=1, ring_slots=2, ncomp=1, clock=identity,
            owner=None, checkpoint_policy=policy, space=dict(kind="scalar_field"),
            state=dict(kind="scalar_history", qualified_id="scalar-history:"+n),
            validity=dict(domain="accepted_clock_ticks", newest_lag=0, oldest_lag=1),
            interpolation=dict(schema_version=1, dense_output=False, provider="exact")))
        cursors[n] = dict(clock=identity, newest_tick=step, oldest_tick=step-1, valid_lags=1,
                          cold_start_extended=False, initialized=True)
    temporal = dict(schema_version=2, clock=dict(time=time.hex(), macro_step=step), status="accepted", synchronized=True,
        strategy=dict(strategy=dict(kind="fixed_dt", dt=dict(kind="binary64", value=.01.hex())), controls={}),
        controller_state=dict(last_accepted_dt=.01.hex(), fixed_dt_grid=dict(schema_version=1,
            origin=0..hex(), macro_step=step, steps=step, time=time.hex())), event_queue=[], program_schedule=schedule,
        clock_cursors={identity:dict(time=time.hex(), tick=step, phase="accepted")},
        schedule_cursors=dict(macro_step=dict(macro_step=step, phase="accepted")), synchronization_cursors={},
        cache_cursors={}, history_cursors=cursors, transaction_stats=dict(accepted=step, rejected=0, failed=0))
    return ir, temporal


@pytest.mark.parametrize("width,phase", [(1,"accepted"),(3,"accepted"),(1,"continuous"),(3,"continuous")])
def test_v3_resealed_history_lags_are_linked_to_ir_without_false_phase(width, phase):
    _, states = independent_initial(width)
    ir, temporal = synthetic_v3_temporal(width, r.CLOCKS[phase][1])
    arrays = synthetic_checkpoint(states[phase], states["accepted"])
    arrays["blocks"] = np.array(["forcing","material","response"])
    arrays["temporal_restart_state"] = np.array(json.dumps(temporal))
    assert "phase" not in temporal["history_cursors"]["q0"]
    r.checkpoint(reseal_checkpoint(arrays), phase, states[phase], "pops.artifact.v1:sha256:"+"a"*64,
                 "synthetic-ABI-only", first_accepted=states["accepted"],
                 block_order=["forcing","material","response"], program_ir=ir)


@pytest.mark.parametrize("attack", ["wronglag", "wrongring", "staleoldest", "stalenewest", "foreignclock",
    "missinghistory", "fakephase", "booltick", "clockbody", "clocktime", "schedule", "controller", "queue", "stats"])
def test_v3_fully_resealed_clock_lag_controller_mutants_refuse(attack):
    _, states = independent_initial(3)
    ir, temporal = synthetic_v3_temporal(3,2)
    if attack == "wronglag":
        temporal["history_cursors"]["q0"]["valid_lags"] = 2
    elif attack == "wrongring":
        temporal["program_schedule"]["histories"][0]["ring_slots"] = 1
    elif attack == "staleoldest":
        temporal["history_cursors"]["q0"]["oldest_tick"] = 0
    elif attack == "stalenewest":
        temporal["history_cursors"]["q0"]["newest_tick"] = 1
    elif attack == "foreignclock":
        temporal["history_cursors"]["q0"]["clock"] = "synthetic-foreign-clock"
    elif attack == "missinghistory":
        del temporal["history_cursors"]["q2"]
    elif attack == "fakephase":
        temporal["history_cursors"]["q0"]["phase"] = "accepted"
    elif attack == "booltick":
        temporal["history_cursors"]["q0"]["oldest_tick"] = True
    elif attack == "clockbody":
        temporal["program_schedule"]["clocks"][0]["descriptor"] = dict(ir["clock"], name="foreign")
    elif attack == "clocktime":
        next(iter(temporal["clock_cursors"].values()))["time"] = .01.hex()
    elif attack == "schedule":
        temporal["schedule_cursors"]["macro_step"]["macro_step"] = 1
    elif attack == "controller":
        temporal["controller_state"]["fixed_dt_grid"]["origin"] = .01.hex()
    elif attack == "queue":
        temporal["event_queue"] = [{"synthetic_pending": True}]
    else:
        temporal["transaction_stats"]["accepted"] = 1
    arrays = synthetic_checkpoint(states["continuous"], states["accepted"])
    arrays["blocks"] = np.array(["forcing","material","response"])
    arrays["temporal_restart_state"] = np.array(json.dumps(temporal))
    with pytest.raises(ValueError):
        r.checkpoint(reseal_checkpoint(arrays), "continuous", states["continuous"], "pops.artifact.v1:sha256:"+"a"*64,
                     "synthetic-ABI-only", first_accepted=states["accepted"],
                     block_order=["forcing","material","response"], program_ir=ir)


@pytest.mark.parametrize("attack", [None,"reorder", "foreign", "duplicateexport"])
def test_v3_block_order_linked_to_actual_cpp_not_old_literal(attack):
    names = ["forcing","material","response"]
    ir = dict(block_order=[dict(local_id=name) for name in names])
    cpp = 'extern "C" const char* pops_program_block_name(int i) {\n switch(i) {\n'+"\n".join(
        'case %d: return "%s";'%(index,name) for index,name in enumerate(names))+'\n default:return "";\n }\n}'
    if attack == "reorder":
        ir["block_order"].reverse()
    elif attack == "foreign":
        ir["block_order"][0]["local_id"] = "foreign"
    elif attack == "duplicateexport":
        cpp += "\n"+cpp
    if attack is None:
        assert r.program_blocks(json.dumps(ir),cpp) == names
    else:
        with pytest.raises(ValueError):
            r.program_blocks(json.dumps(ir),cpp)
