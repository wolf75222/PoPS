#!/usr/bin/env python3
"""Independent bounded corruption probes; no PoPS import, native run, or oracle import.

Every experiment copies the genuine retained archive. Resealing only updates the
outer manifest's digest/size, leaving native/witness receipts untouched. A false
positive produces exit 1; output preserves observations for the checker author.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def reseal(root, paths):
    manifest = load_json(root / "manifest.json")
    for record in manifest["files"]:
        if record["path"] in paths:
            path = root / record["path"]
            record.update(bytes=path.stat().st_size, sha256=digest(path))
    write_json(root / "manifest.json", manifest)


def edit_npz(root, relative, edit):
    path = root / relative
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    edit(arrays)
    np.savez_compressed(path, **arrays)


def audit_receipt_links(root):
    """Check retained internal links independently, without current installation.

    Not every recorded native binary/source file is retained; unavailable inputs
    are counted, never substituted from a later installation.
    """
    manifest = load_json(root / "manifest.json")
    scope = manifest["native_scope"]
    checked = {"identity": 0, "source_snapshot": 0, "junit": 0,
               "saved_state_receipt": 0, "mpi_before_after": 0}
    issues = []
    for record in manifest["files"]:
        path = root / record["path"]
        if path.name == "identity.json":
            identity = load_json(path)
            dimension = 2 if ";dim=2" in identity["abi_key"] else 1
            if identity["native_sha256"] != scope[f"dimension{dimension}_native_sha256"]:
                issues.append(f"native identity: {record['path']}")
            if f"headers={scope['sdk_sha256']};" not in identity["abi_key"]:
                issues.append(f"SDK identity: {record['path']}")
            source_files = path.with_name("source-files.json")
            if source_files.is_file() and digest(source_files) != identity["source_files_sha256"]:
                issues.append(f"source files identity: {record['path']}")
            checked["identity"] += 1
        if record["path"].endswith("source-snapshot/manifest.json"):
            for relative, expected in load_json(path).items():
                if digest(path.parent / relative) != expected:
                    issues.append(f"snapshot: {record['path']}:{relative}")
                checked["source_snapshot"] += 1
        if path.name == "receipt.json":
            receipt = load_json(path)
            for case in receipt.get("records", []):
                relative = case.get("saved_state")
                if relative:
                    saved = path.parent / relative
                    if digest(saved) != case["saved_state_sha256"]:
                        issues.append(f"saved receipt: {record['path']}:{relative}")
                    checked["saved_state_receipt"] += 1
                if "native_sha256" in case and case["native_sha256"] != scope["dimension1_native_sha256"]:
                    issues.append(f"witness native identity: {record['path']}")
                if "artifact_abi_key" in case and not case["artifact_abi_key"].startswith(scope["sdk_sha256"] + "|"):
                    issues.append(f"witness artifact SDK: {record['path']}")
    for group in manifest["groups"]:
        base = root / "runs" / group["name"]
        for relative, summary in group["junit"].items():
            xml = ET.parse(base / relative).getroot()
            cases = list(xml.iter("testcase"))
            counts = {"total": len(cases),
                      "failed": sum(c.find("failure") is not None or c.find("error") is not None
                                    for c in cases),
                      "skipped": sum(c.find("skipped") is not None for c in cases)}
            if counts != summary:
                issues.append(f"JUnit summary: {group['name']}:{relative}")
            checked["junit"] += 1
        if (base / "before/identity.json").is_file():
            before, after = (load_json(base / part / "identity.json") for part in ("before", "after"))
            for name in ("native_sha256", "abi_key", "source_files_sha256"):
                if before[name] != after[name]:
                    issues.append(f"MPI changed installation: {name}")
            checked["mpi_before_after"] += 1
    return {"checked": checked, "issues": issues,
            "unavailable": "Archived binaries/full installed package not retained; SDK/source fingerprints are assertions linked to retained identity receipts."}


def audit_scientific_equations(root, cases):
    """Alternative mathematics on actual retained fields; never writes payloads."""
    metrics = {"m26_analytic_potential_error": 0., "m27_dense_mixed_error": 0.,
               "nd2_frozen_initial_error": 0., "nd2_spectral_ssprk2_error": 0.}
    for case in cases:
        with np.load(root / case["path"], allow_pickle=False) as saved:
            if case["kind"] == "m26_finite12_v1":
                angle = 2 * np.pi * (np.arange(12) + .5) / 12
                if "variation" in case["path"]:
                    first, second = .075 * np.cos(angle), -.055 * np.sin(angle)
                else:
                    first = .1 * np.cos(angle)
                    second = .06 * np.sin(angle)
                metrics["m26_analytic_potential_error"] = max(
                    metrics["m26_analytic_potential_error"],
                    float(np.max(np.abs(saved["potential_first"] - first))),
                    float(np.max(np.abs(saved["potential_second"] - second))))
            elif case["kind"] == "m27_mixed_linear_v1":
                n = saved["c"].shape[1]
                edges = np.arange(n + 1) / n
                current = .4 + .1 * n * np.diff(np.sin(2 * np.pi * edges)) / (2 * np.pi) \
                    - .05 * n * np.diff(np.cos(4 * np.pi * edges)) / (4 * np.pi)
                lap = n ** 2 * (np.roll(np.eye(n), -1, axis=1)
                               - 2 * np.eye(n) + np.roll(np.eye(n), 1, axis=1))
                block = np.block([[np.eye(n), -.01 * lap],
                                  [-np.eye(n) + .08 ** 2 * lap, np.eye(n)]])
                for step in range(10):
                    solved = np.linalg.solve(block, np.concatenate((current, np.zeros(n))))
                    current, mu = solved[:n], solved[n:]
                    metrics["m27_dense_mixed_error"] = max(
                        metrics["m27_dense_mixed_error"],
                        float(np.max(np.abs(saved["c"][step + 1] - current))),
                        float(np.max(np.abs(saved["mu"][step] - mu))))
            elif case["kind"] == "coupled_gradient_dim2_v1":
                x, y = (np.arange(8) + .5) / 8, (np.arange(6) + .5) / 6
                phase = 2 * np.pi * (x[None, :] + 2 * y[:, None])
                oblique = np.sinc(1 / 8) * np.sinc(2 / 6)
                initial = np.stack((1 + .1 * oblique * np.cos(phase)
                                    + .03 * np.sinc(2 / 6) * np.cos(4 * np.pi * y[:, None]) * np.ones((1, 8)),
                                    -.2 + .07 * oblique * np.sin(phase)
                                    + .02 * np.sinc(1 / 8) * np.sin(2 * np.pi * x)[None, :]))
                lam = -4 * (np.sin(np.pi * np.fft.fftfreq(8))[None, :] ** 2 / (1 / 8) ** 2
                            + np.sin(np.pi * np.fft.fftfreq(6))[:, None] ** 2 / (2 / 6) ** 2)
                transformed = np.fft.fft2(initial)
                matrix = np.array(((.25, -.25), (.35, .10)))
                first = 2e-5 * lam * np.einsum("ab,bji->aji", matrix, transformed)
                second = 2e-5 * lam * np.einsum("ab,bji->aji", matrix, first)
                expected = np.fft.ifft2(transformed + first + .5 * second).real
                metrics["nd2_frozen_initial_error"] = max(metrics["nd2_frozen_initial_error"],
                    float(np.max(np.abs(saved["initial"] - initial))))
                metrics["nd2_spectral_ssprk2_error"] = max(metrics["nd2_spectral_ssprk2_error"],
                    float(np.max(np.abs(saved["final"] - expected))))
    if not all(np.isfinite(value) and value < 3e-10 for value in metrics.values()):
        raise AssertionError(f"independent scientific equations: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--complete-witness-reseal", action="store_true",
                        help="also reseal saved-state witness digests for two equation-only corruptions")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("reviewed_saved_checker", args.checker)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    manifest = load_json(args.archive / "manifest.json")
    by_kind = {kind: next(case for case in manifest["scientific_states"] if case["kind"] == kind)
               for kind in ("m26_finite12_v1", "m27_mixed_linear_v1", "coupled_gradient_dim2_v1")}
    m26, m27, nd2 = (by_kind[k] for k in by_kind)
    observations = []

    def experiment(name, mutate, *, sealed, required_reason):
        with tempfile.TemporaryDirectory(prefix="pops-reception-injection-") as temporary:
            root = Path(temporary) / "archive"
            shutil.copytree(args.archive, root)
            changed = mutate(root)
            if sealed:
                reseal(root, changed)
            try:
                result = checker.check_archive(root, recompute=True)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                observations.append({"name": name, "resealed": sealed, "status": "rejected",
                                     "correct_reason": any(reason in message for reason in required_reason),
                                     "reason": message})
            else:
                observations.append({"name": name, "resealed": sealed, "status": "FALSE_POSITIVE",
                                     "correct_reason": False, "recomputed_states": result["recomputed_states"]})

    def byte_corruption(root):
        path = root / m27["path"]
        content = bytearray(path.read_bytes())
        content[len(content) // 2] ^= 1
        path.write_bytes(content)
        return [m27["path"]]

    def ledger_corruption(root):
        path = root / nd2["ledger"]
        rows = load_json(path)
        rows[0]["numerical_flux"] += .1
        write_json(path, rows)
        return [nd2["ledger"]]

    experiment("unsealed_npz_byte", byte_corruption, sealed=False,
               required_reason=("receipt integrity failed",))
    experiment("unsealed_ledger_flux", ledger_corruption, sealed=False,
               required_reason=("receipt integrity failed",))

    def m26_pairing(root):
        edit_npz(root, m26["path"], lambda a: a["measured_pairing"].__setitem__(0, a["measured_pairing"][0] + .001))
        return [m26["path"]]

    def m27_mu(root):
        edit_npz(root, m27["path"], lambda a: a["mu"].__setitem__((4, 3), a["mu"][4, 3] + .001))
        return [m27["path"]]

    def nd2_final(root):
        edit_npz(root, nd2["path"], lambda a: a["final"].__setitem__((1, 2, 3), a["final"][1, 2, 3] + .001))
        return [nd2["path"]]

    experiment("resealed_m26_pairing", m26_pairing, sealed=True,
               required_reason=("pairing_error", "saved-state receipt"))
    experiment("resealed_m27_mu", m27_mu, sealed=True,
               required_reason=("mu_error", "chemical_equation_residual", "saved-state receipt"))
    experiment("resealed_nd2_final", nd2_final, sealed=True,
               required_reason=("state_error",))

    def nd2_zero_initial(root):
        def zero(a):
            for key in ("initial", "final", "increments"):
                a[key].fill(0.)
        edit_npz(root, nd2["path"], zero)
        rows = load_json(root / nd2["ledger"])
        for row in rows:
            row["numerical_flux"] = row["integrated_amount"] = 0.
        write_json(root / nd2["ledger"], rows)
        return [nd2["path"], nd2["ledger"]]

    experiment("resealed_nd2_zero_initial_and_coherent_zero_output", nd2_zero_initial,
               sealed=True, required_reason=("initial_error", "frozen physical benchmark"))

    def nd2_nan_flux(root):
        rows = load_json(root / nd2["ledger"])
        rows[0]["numerical_flux"] = float("nan")
        write_json(root / nd2["ledger"], rows)
        return [nd2["ledger"]]

    def nd2_nan_measure(root):
        rows = load_json(root / nd2["ledger"])
        rows[0]["face_measure"] = float("nan")
        write_json(root / nd2["ledger"], rows)
        return [nd2["ledger"]]

    experiment("resealed_nd2_nan_flux", nd2_nan_flux, sealed=True,
               required_reason=("nonfinite", "numeric", "number", "finite"))
    experiment("resealed_nd2_nan_measure", nd2_nan_measure, sealed=True,
               required_reason=("nonfinite", "numeric", "number", "finite"))

    def omitted_science(root):
        current = load_json(root / "manifest.json")
        current["scientific_states"] = []
        write_json(root / "manifest.json", current)
        return []

    def foreign_native(root):
        relative = "runs/installed-integral-nd-native-dim2-fc0-20260930/identity.json"
        identity = load_json(root / relative)
        identity["native_sha256"] = "0" * 64
        write_json(root / relative, identity)
        return [relative]

    experiment("manifest_omits_all_scientific_states", omitted_science, sealed=True,
               required_reason=("scientific", "inventory", "states"))
    experiment("resealed_foreign_native_identity", foreign_native, sealed=True,
               required_reason=("identity", "native", "provenance"))

    def stale_saved_receipt(root):
        relative = str(Path(m27["path"]).with_name("receipt.json"))
        receipt = load_json(root / relative)
        receipt["records"][0]["saved_state_sha256"] = "0" * 64
        write_json(root / relative, receipt)
        return [relative]

    def forged_junit_summary(root):
        current = load_json(root / "manifest.json")
        current["groups"][0]["junit"]["pytest.xml"]["failed"] = 0
        write_json(root / "manifest.json", current)
        return []

    experiment("resealed_stale_saved_state_receipt", stale_saved_receipt, sealed=True,
               required_reason=("saved", "receipt", "digest"))
    experiment("manifest_forged_failed_junit_count", forged_junit_summary, sealed=True,
               required_reason=("JUnit", "junit", "counts", "summary"))
    if args.complete_witness_reseal:
        def complete_reseal(root, case, edit):
            changed = edit(root)
            relative = str(Path(case["path"]).with_name("receipt.json"))
            receipt = load_json(root / relative)
            matched = [row for row in receipt["records"]
                       if row["saved_state"] == Path(case["path"]).name]
            if len(matched) != 1:
                raise AssertionError("equation probe requires one genuine saved-state witness")
            matched[0]["saved_state_sha256"] = digest(root / case["path"])
            write_json(root / relative, receipt)
            return changed + [relative]

        experiment("fully_resealed_m26_pairing_equation", lambda root: complete_reseal(root, m26, m26_pairing),
                   sealed=True, required_reason=("pairing_error",))
        experiment("fully_resealed_m27_mu_equation", lambda root: complete_reseal(root, m27, m27_mu),
                   sealed=True, required_reason=("mu_error", "chemical_equation_residual"))
    baseline = checker.check_archive(args.archive, recompute=True)
    report = {"checker_sha256": digest(args.checker),
              "archive_manifest_sha256": digest(args.archive / "manifest.json"),
              "baseline": baseline, "independent_receipt_link_audit": audit_receipt_links(args.archive),
              "independent_scientific_audit": audit_scientific_equations(args.archive, manifest["scientific_states"]),
              "injections": observations,
              "complete_witness_reseal": args.complete_witness_reseal,
              "limits": ["No native/JIT/MPI rerun; verifies genuine retained arrays only.",
                         "Manifest digests are integrity links, not external signatures.",
                         "Two unsealed integrity, four finite scientific, two nonfinite ledger and four metadata probes; no exhaustive malicious-file/ZIP parser audit."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"checker_sha256": report["checker_sha256"],
                      "baseline_states": baseline["recomputed_states"],
                      "receipt_audit": report["independent_receipt_link_audit"],
                      "scientific_audit": report["independent_scientific_audit"],
                      "injections": observations}, indent=2))
    return int(any(not row["correct_reason"] for row in observations))


if __name__ == "__main__":
    raise SystemExit(main())
