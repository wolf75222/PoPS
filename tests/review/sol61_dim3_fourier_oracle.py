#!/usr/bin/env python3
"""Second Dim3 audit: closed modal solution, original faces, caller-owned seals.

No PoPS, producer fixture, or existing checker is imported. A successful audit
of generated unit data is never a native reception. Native authenticity rests
on the external identity and manifest hashes supplied by the execution owner.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

CELLS = (6, 4, 3)
H = (1 / 6, 1 / 2, 1.)
DT = 1.e-5
VOLUME = 1 / 12
D = np.array(((.25, .05), (.05, .10)))
R = np.array(((0., -.3), (.3, 0.)))
B = D + R
WEIGHTS = (.5, 1.5)
CRITERIA = dict(state_atol=4.e-12, flux_atol=3.e-12, amount_atol=4.e-13,
                expected_records=3456, initial_bound_atol=2.e-14, minimum_energy_drop=1.e-9)


class Rejected(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Rejected(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise Rejected("nonfinite JSON: " + value)
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    def visit(value):
        if isinstance(value, float):
            need(math.isfinite(value), "nonfinite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result)
    return result


def close(actual, expected, atol, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    need(actual.shape == expected.shape and np.isfinite(actual).all(), "shape/finitude: " + label)
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
    need(error <= atol, label + " error=" + repr(error))
    return error


def modal_states(matrix=B, spacing=H, occurrence_sum=2., shift=(0., 0.), stale=False):
    """Closed six-mode polynomial; no graph, FFT, roll, or opaque solver."""
    x = 2 * np.pi * (np.arange(6) + .5) / 6
    y = 2 * np.pi * (np.arange(4) + .5) / 4
    z = 2 * np.pi * (np.arange(3) + .5) / 3
    phases = (x[None, None, :], y[None, :, None], z[:, None, None])
    ax, ay, az = 3 / np.pi, 2 * math.sqrt(2) / np.pi, 3 * math.sqrt(3) / (2 * np.pi)
    modes = (((1, 1, 1), np.cos, (.1 * ax * ay * az, 0.)),
             ((1, 1, 1), np.sin, (0., .07 * ax * ay * az)),
             ((0, 1, 0), np.cos, (.03 * ay, 0.)),
             ((0, 0, 1), np.sin, (.02 * az, 0.)),
             ((1, 0, 0), np.sin, (0., .02 * ax)),
             ((0, 0, 1), np.cos, (0., .015 * az)))
    fields = [np.broadcast_to(np.array((1. + shift[0], -.2 + shift[1]))[:, None, None, None],
                              (2, 3, 4, 6)).copy() for _ in range(3)]
    for frequencies, trig, coefficient in modes:
        pattern = trig(sum(f * p for f, p in zip(frequencies, phases, strict=True)))
        eigenvalue = -4 * sum(math.sin(math.pi * f / n)**2 / h**2
                             for f, n, h in zip(frequencies, CELLS, spacing, strict=True))
        a = DT * occurrence_sum * eigenvalue * matrix
        c = np.asarray(coefficient)
        coefficients = (c, (np.eye(2) + a) @ c,
                        (np.eye(2) + a + (0. if stale else .5) * (a @ a)) @ c)
        for field, value in zip(fields, coefficients, strict=True):
            field += value[:, None, None, None] * pattern
    return tuple(fields)


def rows_from_modal(order, case, initial, predictor, matrix=B, spacing=H, weights=WEIGHTS):
    """Harness generator only; not a producer of native evidence."""
    rows = []
    for stage, values in enumerate((initial, predictor)):
        for occurrence, k, j, i, axis, side, component in itertools.product(
                range(2), range(3), range(4), range(6), range(3), range(2), range(2)):
            left, right = [i, j, k], [i, j, k]
            left[axis] = (left[axis] - (side == 0)) % CELLS[axis]
            right[axis] = (right[axis] + (side == 1)) % CELLS[axis]
            physical = order[component]
            flux = sum(matrix[physical, q] * (values[q, right[2], right[1], right[0]] -
                       values[q, left[2], left[1], left[0]]) for q in range(2)) / spacing[axis]
            area = math.prod(spacing[t] for t in range(3) if t != axis)
            row = dict(operation_identity=case["operation_identity"],
                       occurrence_identity=case["operation_identity"] + "/occurrence:" + str(occurrence),
                       evaluation_context=case["stage_contexts"][str(stage)],
                       quadrature_identity=f"cell:{i}:{j}:{k}/axis:{axis}/side:{side}/component:{component}",
                       orientation=2 * side - 1, face_measure=area, numerical_flux=float(flux),
                       temporal_weight=DT / 2 * weights[occurrence], multiplicity=1)
            row["integrated_amount"] = row["orientation"] * area * row["numerical_flux"] * row["temporal_weight"]
            rows.append(row)
    return rows


def wire_rows(path):
    data, cursor = Path(path).read_bytes(), 8
    need(data[:8] == b"POPSEX01", "wire profile: expected actual fixture POPSEX01")
    def word():
        nonlocal cursor
        need(cursor + 8 <= len(data), "truncated wire")
        value = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8
        return value
    def text():
        nonlocal cursor
        n = word()
        need(n <= len(data) - cursor, "truncated wire string")
        value = data[cursor:cursor + n].decode("utf-8")
        cursor += n
        return value
    rows = []
    count = word()
    need(count <= 3456, "wire count")
    for _ in range(count):
        row = {name: text() for name in ("operation_identity", "occurrence_identity", "evaluation_context", "quadrature_identity")}
        orientation = word()
        row["orientation"] = orientation if orientation < 1 << 63 else orientation - (1 << 64)
        for name in ("face_measure", "numerical_flux", "temporal_weight"):
            row[name] = struct.unpack("<d", struct.pack("<Q", word()))[0]
            need(math.isfinite(row[name]), "nonfinite wire")
        row["multiplicity"] = word()
        row["integrated_amount"] = row["orientation"] * row["face_measure"] * row["numerical_flux"] * row["temporal_weight"] * row["multiplicity"]
        rows.append(row)
    need(cursor == len(data), "trailing wire")
    return rows


def audit_science(directory, order, case):
    def npz(name, keys):
        with np.load(directory / name, allow_pickle=False) as archive:
            need(len(archive.files) == len(set(archive.files)), "duplicate NPZ key")
            values = {key: archive[key].copy() for key in archive.files}
        need(all(v.dtype.kind in "fiu" and np.isfinite(v).all() for v in values.values()), "nonfinite NPZ")
        need(set(values) == set(keys.split()), "exact NPZ fields")
        return values
    observed = npz("initial.npz", "initial requested_initial bound_input order cells lengths")
    accepted = npz("accepted.npz", "initial final state_increment oracle_predictor oracle_final order dt cells lengths spacing cell_volume dissipative reversible occurrence_weights")
    increments = npz("increments.npz", "native_face_amounts state_amounts")
    for values in (observed, accepted):
        for name, target in dict(order=order, cells=CELLS, lengths=(1., 2., 3.)).items():
            close(values[name], target, 0., "physical metadata " + name)
    for name, target in dict(dt=DT, spacing=H, cell_volume=VOLUME, dissipative=D,
                             reversible=R, occurrence_weights=WEIGHTS).items():
        close(accepted[name], target, 0., "physical metadata " + name)
    initial, predictor, final = modal_states()
    for value in (observed["initial"], observed["requested_initial"], accepted["initial"]):
        close(value, initial, 2.e-14, "prescribed initial")
    close(observed["bound_input"], initial[list(order)], 2.e-14, "exact subject component input")
    state_error = close(accepted["final"], final, 4.e-12, "original SSPRK2 equation")
    close(accepted["state_increment"], accepted["final"] - accepted["initial"], 0., "state delta")
    close(accepted["oracle_predictor"], predictor, 2.e-14, "diagnostic predictor")
    close(accepted["oracle_final"], final, 2.e-14, "diagnostic final")
    batches = read_json(directory / "native-ledger.json")
    need(type(batches) is list, "rank batches")
    receipt = read_json(directory / "receipt.json")
    need(len(batches) == receipt["ranks"] == len(receipt["ledger_sha256_by_rank"]), "exact rank batches")
    need(receipt["records_by_rank"] == [len(batch) for batch in batches], "rank record counts")
    rows = []
    for rank, batch in enumerate(batches):
        wire = directory / f"ledger-rank-{rank:04d}.bin"
        need(sha(wire) == receipt["ledger_sha256_by_rank"][rank], "wire digest")
        decoded = wire_rows(wire)
        need(len(decoded) == len(batch), "wire/JSON count")
        for a, b in zip(decoded, batch, strict=True):
            need(set(a) == set(b), "wire/JSON fields")
            for key in a:
                if isinstance(a[key], float):
                    need(type(b[key]) is float and struct.pack("<d", a[key]) == struct.pack("<d", b[key]), "wire/JSON double")
                else:
                    need(type(a[key]) is type(b[key]) and a[key] == b[key], "wire/JSON identity")
        rows.extend(batch)
    expected = rows_from_modal(order, case, initial, predictor)
    def incidence(row):
        return row["evaluation_context"], row["occurrence_identity"], row["quadrature_identity"]
    reference = {incidence(row): row for row in expected}
    seen, amounts = set(), np.zeros_like(initial)
    flux_error = amount_error = 0.
    need(len(rows) == len(reference) == 3456, "incidence count")
    for row in rows:
        key = incidence(row)
        need(key in reference and key not in seen, "original stage/occurrence/incidence identity")
        seen.add(key)
        target = reference[key]
        for name in ("operation_identity", "orientation", "multiplicity", "face_measure", "temporal_weight"):
            need(type(row[name]) is type(target[name]) and row[name] == target[name], "original face metadata: " + name)
        flux_error = max(flux_error, close(row["numerical_flux"], target["numerical_flux"], 3.e-12, "original face flux"))
        amount_error = max(amount_error, close(row["integrated_amount"], target["integrated_amount"], 4.e-13, "original face amount"))
        i, j, k, _, _, component = map(int, re.fullmatch(r"cell:(\d+):(\d+):(\d+)/axis:([012])/side:([01])/component:([01])", row["quadrature_identity"]).groups())
        amounts[order[component], k, j, i] += row["integrated_amount"]
    close(increments["native_face_amounts"], amounts, 0., "saved face sums")
    state_amounts = (accepted["final"] - accepted["initial"]) * VOLUME
    close(increments["state_amounts"], state_amounts, 0., "saved state amounts")
    balance_error = close(amounts, state_amounts, 4.e-13, "original cell balance")
    close(amounts.sum(axis=(1, 2, 3)), np.zeros(2), 4.e-13, "global conservation")
    energies = [float(np.sum(q * q) * VOLUME) for q in (accepted["initial"], accepted["final"])]
    need(energies[0] - energies[1] > 1.e-9, "dissipative energy")
    return dict(spatial_cells=72, scalar_unknowns=144, records=3456, state_error=state_error,
                flux_error=flux_error, amount_error=amount_error, balance_error=balance_error,
                energy_drop=energies[0] - energies[1])


def archive_source_check(repository, commit, files):
    """Hash exact Git blobs, not the changing checkout or installed package."""
    for name in files:
        need(not Path(name).is_absolute() and ".." not in Path(name).parts and "\n" not in name,
             "source path grammar")
    requests = "".join(commit + ":" + name + "\n" for name in files).encode()
    result = subprocess.run(["rtk", "proxy", "git", "cat-file", "--batch"], cwd=repository,
                            input=requests, capture_output=True, check=True).stdout
    cursor = 0
    for name, expected in files.items():
        end = result.index(b"\n", cursor)
        fields = result[cursor:end].split()
        need(len(fields) == 3 and fields[1] == b"blob", "missing historical source blob")
        size = int(fields[2])
        cursor = end + 1
        need(hashlib.sha256(result[cursor:cursor + size]).hexdigest() == expected,
             "historical source mismatch: " + name)
        cursor += size + 1
    need(cursor == len(result), "historical source batch boundary")


def launch_provenance(identity, repository=None):
    for key in ("authenticated_source_manifest", "build_log", "native_launch_identity", "native_launch_result"):
        ref = identity[key]
        need(sha(ref["path"]) == ref["sha256"], "external launch file: " + key)
    source_ref = identity["authenticated_source_manifest"]
    sources = read_json(source_ref["path"])
    need(len(sources) == source_ref["files"], "source inventory count")
    launch = read_json(identity["native_launch_identity"]["path"])
    for key, recorded in (("native_source_commit", "source_commit"), ("native_sha256", "native_sha256"),
                          ("native_abi", "abi_key"), ("package_path", "package_file"), ("native_path", "native_file")):
        need(identity[key] == launch[recorded], "launch identity: " + key)
    need(launch["source_files_sha256"] == source_ref["sha256"] and
         launch["verified_source_files"] == len(sources) and
         launch["source_diff_sha256"] == hashlib.sha256(b"").hexdigest(), "launch source authentication")
    result_ref = identity["native_launch_result"]
    result = read_json(result_ref["path"])
    need(type(result["returncode"]) is int and result["returncode"] == 0 and result["status"] == "passed", "launch result")
    counts = dict(tests=2, failures=0, errors=0, skipped=0)
    if identity["ranks"] == 1:
        need(result["counts"] == counts and result["identity_sha256"] == identity["native_launch_identity"]["sha256"], "Serial complete result")
        need(sha(Path(result_ref["path"]).parent / "pytest.log") == result["log_sha256"], "Serial log")
    else:
        need(result["ranks"] == 2 and result["dimension"] == 3 and result["timeout"] is False and
             all(result[key] is True for key in ("same_installation", "test_sources_unchanged", "rank_test_parity")) and
             result["authentication_before"] == result["authentication_after"] == 0 and
             result["native_sha256"] == identity["native_sha256"], "MPI authentic complete result")
        need(len(result["rank_results"]) == 2, "MPI rank results")
        for rank, entry in enumerate(result["rank_results"]):
            need(type(entry["rank"]) is int and entry["rank"] == rank and entry["counts"] == counts and
                 entry["xml_sha256"] == identity["junit_by_rank"][rank]["sha256"], "MPI result/JUnit parity")
            need(sha(Path(result_ref["path"]).parent / f"rank{rank}.log") == entry["log_sha256"], "MPI rank log")
    if repository is not None:
        pinned = {**sources, "tests/python/integration/runtime/test_coupled_gradient_dim3_fourier_runtime.py": identity["fixture_sha256"]}
        archive_source_check(repository, identity["native_source_commit"], pinned)
    return len(sources)


def receive(root, identity_path, identity_sha, manifest_sha, repository=None):
    root, identity_path = Path(root), Path(identity_path)
    need(sha(identity_path) == identity_sha and sha(root / "manifest.json") == manifest_sha, "external owner seal")
    need(root.resolve() not in identity_path.resolve().parents, "identity must be external")
    identity, manifest = read_json(identity_path), read_json(root / "manifest.json")
    need(set(manifest) == {"contract", "external_identity_sha256", "files"}, "exact manifest fields")
    need(identity["contract"] == "pops.coupled-gradient-dim3.external-identity@1", "identity contract")
    need(manifest["contract"] == "pops.coupled-gradient-dim3.offline-manifest@1" and
         manifest["external_identity_sha256"] == identity_sha, "manifest contract/identity")
    need(identity["evidence_kind"] == "native-reception" and type(identity["native_exit_status"]) is int and
         identity["native_exit_status"] == 0, "actual execution required")
    need(type(identity["native_dimension"]) is int and identity["native_dimension"] == 3 and
         type(identity["ranks"]) is int and identity["ranks"] in (1, 2), "Dim3 Serial/MPI2 profile")
    need(re.fullmatch(r"[0-9a-f]{40}", identity["native_source_commit"]) is not None, "source commit")
    need("dim=3" in identity["native_abi"] and "mpi=1" in identity["native_abi"], "native ABI")
    need(set(identity["cases"]) == {"01", "10"}, "physical permutation inventory")
    source_count = launch_provenance(identity, repository)
    names = {"initial.npz", "accepted.npz", "increments.npz", "receipt.json", "native-ledger.json"}
    names |= {f"ledger-rank-{rank:04d}.bin" for rank in range(identity["ranks"])}
    files = {f"coupled-gradient-dim3-order-{order}/{name}" for order in ("01", "10") for name in names}
    paths = tuple(root.rglob("*"))
    need(not root.is_symlink() and not any(p.is_symlink() for p in paths), "symlink evidence")
    need({str(p.relative_to(root)) for p in paths if p.is_file() and p.name != "manifest.json"} == files and
         set(manifest["files"]) == files, "exact payload inventory")
    for name in files:
        need(sha(root / name) == manifest["files"][name], "payload seal: " + name)
    entries = identity["junit_by_rank"]
    need(len(entries) == identity["ranks"] and len({e["path"] for e in entries}) == len(entries), "all-rank JUnit inventory")
    for rank, entry in enumerate(entries):
        need(type(entry["rank"]) is int and entry["rank"] == rank and sha(entry["path"]) == entry["sha256"], "JUnit rank/digest")
        need(set(entry["testcases"]) == {"01", "10"}, "JUnit both permutations")
        xml = ET.parse(entry["path"])
        for order, name in entry["testcases"].items():
            cases = [c for c in xml.iter("testcase") if c.get("name") == name]
            need(len(cases) == 1 and not any(cases[0].find(tag) is not None for tag in ("failure", "error", "skipped")), "JUnit true passed case")
            props = {}
            for p in cases[0].iter("property"):
                need(p.get("name") not in props or props[p.get("name")] == p.get("value"), "conflicting JUnit property")
                props[p.get("name")] = p.get("value")
            need(props.get("mpi_rank") == str(rank) and props.get("mpi_size") == str(identity["ranks"]) and
                 props.get("native_dimension") == "3" and props.get("artifact_identity") == identity["cases"][order]["artifact_identity"], "JUnit runtime identity")
            need(Path(props.get("coupled_gradient_dim3_receipts", "")).name == "coupled-gradient-dim3-order-" + order, "JUnit receipt name")
    reports = {}
    for order in ("01", "10"):
        directory = root / ("coupled-gradient-dim3-order-" + order)
        receipt = read_json(directory / "receipt.json")
        need(type(receipt["ranks"]) is int and type(receipt["native_dimension"]) is int,
             "receipt rank/dimension types")
        for key in ("native_dimension", "ranks", "native_sha256", "native_abi", "fixture_sha256", "native_path", "package_path"):
            need(receipt[key] == identity[key], "receipt native identity: " + key)
        for key in ("artifact_identity", "platform_manifest"):
            need(receipt[key] == identity["cases"][order][key], "receipt artifact identity: " + key)
        need(type(receipt["native_macro_step"]) is int and receipt["native_macro_step"] == 1 and
             type(receipt["native_time"]) is float and receipt["native_time"] == DT, "actual accepted clock")
        need(receipt["component_order"] == list(map(int, order)) and receipt["criteria"] == CRITERIA, "physical ordering/criteria")
        need(receipt["storage"] == "canonical physical components,z,y,x", "physical storage axes")
        for name, target in dict(cells=list(CELLS), lengths=[1., 2., 3.], dt=DT, dissipative=D.tolist(),
                                 reversible=R.tolist(), occurrence_weights=list(WEIGHTS)).items():
            need(receipt[name] == target, "receipt physical metadata: " + name)
        reports[order] = audit_science(directory, tuple(map(int, order)), identity["cases"][order])
    with np.load(root / "coupled-gradient-dim3-order-01/accepted.npz", allow_pickle=False) as left:
        with np.load(root / "coupled-gradient-dim3-order-10/accepted.npz", allow_pickle=False) as right:
            close(left["final"], right["final"], 4.e-12, "physical component permutation equivalence")
    return dict(status="received", scope="saved Dim3 periodic Uniform SSPRK2; owner-sealed execution",
                native_source_commit=identity["native_source_commit"], native_sha256=identity["native_sha256"],
                ranks=identity["ranks"], identity_sha256=identity_sha, manifest_sha256=manifest_sha,
                verified_source_files=source_count, historical_git_checked=repository is not None, cases=reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--repository", type=Path, help="read-only Git repository for historical source blob verification")
    args = parser.parse_args()
    print(json.dumps(receive(args.root, args.identity, args.identity_sha256, args.manifest_sha256, args.repository), indent=2, allow_nan=False))
