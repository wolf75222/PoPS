"""Synthetic protocol tests and optional mutations of real, unapproved archives.

The CLI only creates negative-test copies; it cannot issue ROOT approval.
It imports no PoPS module or native library.
"""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("m19_sdk9f_batch_reader", Path(__file__).with_name("sol61_m19_saved_reception.py"))
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def synthetic_batch():
    cases = {r.case_id(*case): dict(directory="/SYNTHETIC/" + r.case_id(*case), native_sha256="a" * 64) for case in r.CASES}
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="13", failures="0", errors="0", skipped="0")
    for key, case in cases.items():
        test = ET.SubElement(suite, "testcase", classname="tests.python.integration.runtime.test_m19_product_support_runtime",
                             name=f"test_native_product_reduce_lift_restart[{key}]")
        properties = ET.SubElement(test, "properties")
        for name, value in dict(native_dimension=2, mpi_rank=0, mpi_size=2, native_sha256=case["native_sha256"],
                                saved_receipts=case["directory"]).items():
            ET.SubElement(properties, "property", name=name, value=str(value))
    others = [dict(classname="synthetic.other", name=f"protocol_only_{i}") for i in range(7)]
    for identity in others:
        ET.SubElement(suite, "testcase", **identity)
    return root, cases, others


def test_full_mixed_batch_is_explicit_and_old_six_case_reader_remains_strict():
    root, cases, others = synthetic_batch()
    r.junit(ET.tostring(root), 0, 2, cases, others)
    with pytest.raises(ValueError, match="JUnit"):
        r.junit(ET.tostring(root), 0, 2, cases)


@pytest.mark.parametrize("attack", ("failure", "error", "skip", "duplicate", "foreign", "missing", "counter",
                                    "duplicate_seal", "m19_in_extras", "malformed_seal", "rank"))
def test_no_extra_case_can_be_filtered_or_hide_a_red_batch(attack):
    root, cases, others = synthetic_batch()
    suite = root.find("testsuite")
    extras = list(suite)[6:]
    if attack in ("failure", "error", "skip"):
        ET.SubElement(extras[0], "skipped" if attack == "skip" else attack)
    elif attack == "duplicate":
        extras[1].set("name", extras[0].get("name"))
    elif attack == "foreign":
        extras[0].set("classname", "changed.realm")
    elif attack == "missing":
        suite.remove(extras[0])
        suite.set("tests", "12")
    elif attack == "counter":
        suite.set("tests", "6")
    elif attack == "duplicate_seal":
        others.append(copy.deepcopy(others[0]))
    elif attack == "m19_in_extras":
        others[0]["classname"] = "tests.python.integration.runtime.test_m19_product_support_runtime"
    elif attack == "malformed_seal":
        others[0]["name"] = True
    else:
        suite.find("testcase/properties/property[@name='mpi_rank']").set("value", "1")
    with pytest.raises(ValueError, match="JUnit"):
        r.junit(ET.tostring(root), 0, 2, cases, others)


def test_new_archive_requires_all_members_pinned_without_live_fallback(tmp_path):
    raw = b"SYNTHETIC protocol only, no native image"
    sha = r.digest(raw)
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("sha256/" + sha, raw)
        z.writestr("unsealed-extra", b"extra")
    pins = dict(schema="sol61.m19-owner-pins@3", supporting_evidence=[dict(path="/NO_DONOR/native", sha256=sha)],
                backing=dict(archive=dict(path=str(path), sha256=r.digest(path.read_bytes())),
                             members={"/NO_DONOR/native": dict(member="sha256/" + sha, sha256=sha)}))
    with pytest.raises(ValueError, match="unpinned members"):
        with r.archived_context(pins):
            pass
    pins["schema"] = "sol61.m19-owner-pins@2"
    with r.archived_context(pins):  # Historical @2 behavior remains explicit.
        assert r.read_bytes("/NO_DONOR/native") == raw


def packed(arrays):
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def reseal_checkpoint(arrays):
    """Negative harness reproduces envelope hashes, never authenticates an owner."""
    manifest = r.strict_json(str(arrays["pops_checkpoint_manifest"].item()))
    manifest["arrays"] = {name: r.typed_array(value) for name, value in arrays.items()
                          if name not in ("pops_checkpoint_manifest", "pops_restart_identity")}
    manifest["clock"] = dict(time=float(arrays["t"]).hex(), macro_step=int(arrays["macro_step"]))
    payload = {k: v for k, v in manifest.items() if k != "restart_identity"}
    manifest["restart_identity"]["hexdigest"] = r.digest(r.protocol.cbor(
        dict(protocol="pops.identity", domain="restart", schema_version=1, payload=payload)))
    arrays["pops_checkpoint_manifest"] = np.array(json.dumps(manifest))
    arrays["pops_restart_identity"] = np.array(r.protocol.identity_token(manifest["restart_identity"], "restart"))
    return packed(arrays)


def checkpoint_state(raw, block, value):
    arrays = r.protocol.archive(raw)
    changed = 0
    for index in range(3):
        key = "layout_checkpoint_" + str(index)
        child = r.protocol.archive(arrays[key].tobytes())
        if "state_" + block in child:
            child["state_" + block] = value.copy()
            arrays[key] = np.frombuffer(reseal_checkpoint(child), dtype=np.uint8).copy()
            changed += 1
    assert changed == 1
    return reseal_checkpoint(arrays)


def mutated_archive(original, directory, attack):
    """Every changed file, receipt, CP array and content pin is resealed on copies."""
    pins = copy.deepcopy(original)
    contents = {}
    with zipfile.ZipFile(pins["backing"]["archive"]["path"]) as z:
        for member in {row["member"] for row in pins["backing"]["members"].values()}:
            contents[member] = z.read(member)
    replacements = {}
    case = next(iter(pins["cases"].values()))

    def get(leaf):
        return replacements.get(leaf["path"], contents[pins["backing"]["members"][leaf["path"]]["member"]])

    def put(leaf, raw):
        replacements[leaf["path"]] = raw

    def receipt(phase, mutation):
        row = case["files"][phase]
        data = r.strict_json(get(row["receipt"]))
        mutation(data)
        put(row["receipt"], (json.dumps(data, indent=2) + "\n").encode())

    if attack in ("reduce", "weighted", "lift", "population", "restart_state"):
        block = dict(reduce="integral", weighted="weighted", lift="extended", population="population",
                     restart_state="integral")[attack]
        phases = ("restored",) if attack == "restart_state" else r.PHASES[1:]
        for phase in phases:
            row = case["files"][phase]
            arrays = r.protocol.archive(get(row["state"]))
            if attack == "restart_state":
                arrays[block].flat[0] = np.nextafter(arrays[block].flat[0], np.inf)
            else:
                arrays[block].flat[0] += .125
            state = packed(arrays)
            cp = checkpoint_state(get(row["checkpoint"]), block, arrays[block])
            put(row["state"], state)
            put(row["checkpoint"], cp)
            receipt(phase, lambda d, state=state, cp=cp: d.update(saved_state_sha256=r.digest(state), checkpoint_sha256=r.digest(cp)))
    elif attack == "clock":
        receipt("accepted", lambda d: d.update(time=.02, time_hex=.02.hex(), macro_step=2))
    elif attack == "ownership":
        receipt("accepted", lambda d: d["local_boxes_by_rank"][0]["population"].append(
            copy.deepcopy(d["local_boxes_by_rank"][0]["population"][0])))
    elif attack == "cadence":
        receipt("accepted", lambda d: d["mapping_counts"].update({next(iter(d["mapping_counts"])): 2}))
    elif attack == "provenance":
        data = r.strict_json(get(case["files"]["provenance"]))
        data["physical_equations"] = "changed to local reaction"
        put(case["files"]["provenance"], (json.dumps(data) + "\n").encode())
    elif attack == "extra_xml_failure":
        assert pins["other_junit_cases"]
        row = pins["junit"][0]
        xml = ET.fromstring(get(row))
        target = pins["other_junit_cases"][0]
        matches = [t for t in xml.iter("testcase") if (t.get("classname"), t.get("name")) == (target["classname"], target["name"])]
        assert len(matches) == 1
        ET.SubElement(matches[0], "failure", message="negative-test-only")
        put(row, ET.tostring(xml))
    else:
        raise AssertionError(attack)

    def update(row):
        if type(row) is dict:
            if set(row) == {"path", "sha256"} and row["path"] in replacements:
                row["sha256"] = r.digest(replacements[row["path"]])
            else:
                for child in row.values():
                    update(child)
        elif type(row) is list:
            for child in row:
                update(child)
    update({k: v for k, v in pins.items() if k != "backing"})
    for path, raw in replacements.items():
        sha = r.digest(raw)
        member = "sha256/" + sha
        pins["backing"]["members"][path] = dict(member=member, sha256=sha)
        contents[member] = raw
    directory.mkdir()
    archive = directory / "negative-test-only.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for member in sorted({row["member"] for row in pins["backing"]["members"].values()}):
            z.writestr(member, contents[member])
    pins["backing"]["archive"] = dict(path=str(archive), sha256=r.digest(archive.read_bytes()))
    (directory / "negative-test-only-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    return pins


def real_countermodels(pin_paths, output):
    output.mkdir(exist_ok=False)
    results = {}
    donor_hashes = {}
    for path in pin_paths:
        original = r.strict_json(path.read_bytes())
        archive = Path(original["backing"]["archive"]["path"])
        donor_hashes[str(path)] = r.digest(path.read_bytes())
        donor_hashes[str(archive)] = r.digest(archive.read_bytes())
        with r.archived_context(original):
            baseline = r._receive_pins(original)
        assert baseline["evidence_backing"] == "unapproved_archive@3"
        attacks = ["reduce", "weighted", "lift", "population", "restart_state", "clock", "ownership", "cadence", "provenance"]
        if original["other_junit_cases"]:
            attacks.append("extra_xml_failure")
        mode = original["mode"]
        results[mode] = {}
        for attack in attacks:
            pins = mutated_archive(original, output / (mode + "-" + attack), attack)
            try:
                with r.archived_context(pins):
                    r._receive_pins(pins)
            except ValueError as error:
                results[mode][attack] = dict(status="REFUSED", diagnostic=str(error), authority="negative-test-only; no ROOT approval")
            else:
                raise AssertionError("scientific countermodel was accepted: " + mode + ":" + attack)
        for path, sha in donor_hashes.items():
            assert r.digest(Path(path).read_bytes()) == sha
    assert not any(n == "pops" or n.startswith("pops.") for n in sys.modules)
    report = dict(schema="sol61.m19-sdk9f-negative-reception@1", scope="offline-copied-real-files-only",
                  cases=results, original_hashes=donor_hashes, origins_unchanged=True,
                  root_authenticity_claim=False, native_execution_performed=False)
    (output / "countermodels-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(real_countermodels(args.pins, args.output), indent=2))
