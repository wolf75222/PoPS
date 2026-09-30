#!/usr/bin/env python3
"""Authenticate retained receipts and independently recompute native saved states.

This offline checker imports NumPy and the standard library only. It neither
imports PoPS/the witness oracles nor claims to rerun a native backend. The
equations and tolerances below belong to the named, frozen benchmark variants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def _json(path: Path):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError(f"duplicate receipt key: {name}")
            result[name] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite receipt JSON value: {value}")

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != names:
            raise ValueError(f"unexpected saved-state arrays: {path}")
        result = {name: saved[name].copy() for name in names}
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"nonfinite saved state: {path}")
    return result


def _bound(metrics: dict[str, float], limits: dict[str, float]) -> dict[str, float]:
    for name, limit in limits.items():
        if not math.isfinite(metrics[name]) or metrics[name] > limit:
            raise ValueError(f"saved-state criterion {name}: {metrics[name]} > {limit}")
    return metrics


def check_m27(path: Path) -> dict[str, float]:
    saved = _arrays(path, {"c", "mu", "times"})
    c, mu, times = saved["c"], saved["mu"], saved["times"]
    if c.ndim != 2 or c.shape[0] != 11 or c.shape[1] not in (16, 32, 64) \
            or mu.shape != (10, c.shape[1]) or times.shape != (11,):
        raise ValueError("M27 reception requires eleven retained c states and ten actual mu fields")
    n, dt, epsilon = c.shape[1], .01, .08
    h = 1. / n
    x = (np.arange(n) + .5) * h
    initial = .4 + .1 * np.sinc(h) * np.cos(2 * np.pi * x) \
        + .05 * np.sinc(2 * h) * np.sin(4 * np.pi * x)
    eigenvalues = 4 * np.sin(np.pi * np.fft.fftfreq(n)) ** 2 / h ** 2
    amplification = 1 / (1 + dt * eigenvalues * (1 + epsilon ** 2 * eigenvalues))
    initial_fourier = np.fft.fft(initial)
    expected = np.stack([np.fft.ifft(initial_fourier * amplification ** step).real
                         for step in range(11)])
    expected_mu = np.fft.ifft(np.fft.fft(expected[1:], axis=1)
                             * (1 + epsilon ** 2 * eigenvalues), axis=1).real
    lap_c = (np.roll(c[1:], -1, axis=1) - 2 * c[1:] + np.roll(c[1:], 1, axis=1)) / h ** 2
    lap_mu = (np.roll(mu, -1, axis=1) - 2 * mu + np.roll(mu, 1, axis=1)) / h ** 2
    energy = .5 * np.mean(c * c, axis=1) + .5 * epsilon ** 2 * np.mean(
        ((np.roll(c, -1, axis=1) - c) / h) ** 2, axis=1)
    return _bound({
        "initial_error": float(np.max(np.abs(c[0] - initial))),
        "c_error": float(np.max(np.abs(c - expected))),
        "mu_error": float(np.max(np.abs(mu - expected_mu))),
        "mass_equation_residual": float(np.max(np.abs(c[1:] - c[:-1] - dt * lap_mu))),
        "chemical_equation_residual": float(np.max(np.abs(mu - c[1:] + epsilon ** 2 * lap_c))),
        "mass_drift": float(np.max(np.abs(c.mean(axis=1) - c[0].mean()))),
        "energy_increase": float(max(0., np.max(np.diff(energy)))),
        "time_error": float(np.max(np.abs(times - np.arange(11) * dt))),
    }, {"initial_error": 2e-14, "c_error": 3e-10, "mu_error": 3e-10,
        "mass_equation_residual": 1e-10, "chemical_equation_residual": 1e-10,
        "mass_drift": 2e-12, "energy_increase": 3e-12, "time_error": 3e-14})


def check_m26(path: Path) -> dict[str, float]:
    saved = _arrays(path, {"density_first", "density_second", "potential_first",
                           "potential_second", "measured_pairing", "time"})
    a, b = saved["density_first"], saved["density_second"]
    pa, pb, paired = saved["potential_first"], saved["potential_second"], saved["measured_pairing"]
    if any(value.shape != (12,) for value in (a, b, pa, pb)) \
            or paired.shape != (7,) or saved["time"].shape != ():
        raise ValueError("M26 finite reception has twelve physical quadrature nodes and seven pairings")
    n = 12
    x = (np.arange(n) + .5) / n
    if "variation" in path.name:
        a0 = .9 + .15 * np.cos(2 * np.pi * x) - .03 * np.sin(6 * np.pi * x)
        b0 = 1.2 - .11 * np.sin(2 * np.pi * x) + .04 * np.cos(4 * np.pi * x)
    else:
        a0 = 1 + .2 * np.cos(2 * np.pi * x) + .1 * np.sin(4 * np.pi * x)
        b0 = .85 + .12 * np.sin(2 * np.pi * x) + .07 * np.cos(6 * np.pi * x)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("M26 witness density lost strict positivity")
    # Dense independent quadrature; the author oracle uses Fourier coefficients.
    operator = np.cos(2 * np.pi * (x[:, None] - x[None, :])) / n
    wa, wb = operator @ a0, operator.T @ b0
    expected = np.array([.5 * a0 @ wa / n, .5 * b0 @ wb / n,
                         (b0 - a0) @ wa / n, a0 @ wb / n,
                         wa @ b0 / n, a0.mean(), b0.mean()])
    return _bound({
        "density_error": float(max(np.max(np.abs(a - a0)), np.max(np.abs(b - b0)))),
        "potential_error": float(max(np.max(np.abs(pa - wa)), np.max(np.abs(pb - wb)))),
        "pairing_error": float(np.max(np.abs(paired - expected))),
        "adjoint_defect": float(abs(paired[3] - paired[4])),
        "energy_increment_error": float(abs(paired[2] + .5 * (b - a) @ (pb - pa) / n
                                              - (expected[1] - expected[0]))),
        "time_error": abs(float(saved["time"]) - 1.),
    }, {"density_error": 2e-14, "potential_error": 3e-13, "pairing_error": 3e-13,
        "adjoint_defect": 3e-13, "energy_increment_error": 3e-13, "time_error": 2e-14})


def check_nd2(path: Path, ledger_path: Path) -> dict[str, float]:
    saved = _arrays(path, {"initial", "final", "increments", "order", "dt", "cells", "lengths",
                           "dissipative", "reversible", "occurrence_weights"})
    initial, actual, retained = saved["initial"], saved["final"], saved["increments"]
    if initial.shape != (2, 6, 8) or actual.shape != initial.shape or retained.shape != initial.shape \
            or not np.array_equal(saved["cells"], (8, 6)) \
            or not np.array_equal(saved["lengths"], (1., 2.)) \
            or saved["dt"].shape != () or float(saved["dt"]) != 1e-5 \
            or saved["order"].shape != (2,) or sorted(saved["order"].tolist()) != [0, 1] \
            or not np.array_equal(saved["dissipative"], ((.25, .05), (.05, .10))) \
            or not np.array_equal(saved["reversible"], ((0., -.3), (.3, 0.))) \
            or not np.array_equal(saved["occurrence_weights"], (.5, 1.5)):
        raise ValueError("ND2 snapshot differs from the frozen physical benchmark")
    dt, hx, hy = 1e-5, 1/8, 2/6
    x, y = (np.arange(8) + .5) / 8, (np.arange(6) + .5) / 6
    phase = 2 * np.pi * (x[None, :] + 2 * y[:, None])
    oblique = np.sinc(1 / 8) * np.sinc(2 / 6)
    prescribed = np.stack((1 + .1 * oblique * np.cos(phase)
                           + .03 * np.sinc(2 / 6) * np.cos(4 * np.pi * y[:, None]),
                           -.2 + .07 * oblique * np.sin(phase)
                           + .02 * np.sinc(1 / 8) * np.sin(2 * np.pi * x)[None, :]))
    matrix = saved["dissipative"] + saved["reversible"]
    def rate(value):
        laplacian = (np.roll(value, -1, axis=2) - 2 * value + np.roll(value, 1, axis=2)) / hx ** 2 \
            + (np.roll(value, -1, axis=1) - 2 * value + np.roll(value, 1, axis=1)) / hy ** 2
        return 2 * np.einsum("ab,bji->aji", matrix, laplacian)
    predictor = initial + dt * rate(initial)
    expected = .5 * (initial + predictor + dt * rate(predictor))
    stage_values = (np.einsum("ab,bji->aji", matrix, initial),
                    np.einsum("ab,bji->aji", matrix, predictor))
    rows = _json(ledger_path)
    if type(rows) is not list or len(rows) != 1536:
        raise ValueError("ND2 ledger must retain all two-stage, two-occurrence face contributions")
    seen, accumulated = set(), np.zeros_like(initial)
    flux_error = amount_error = measure_error = weight_error = 0.
    for row in rows:
        if type(row) is not dict or set(row) != {
            "evaluation_context", "face_measure", "integrated_amount", "multiplicity",
            "numerical_flux", "occurrence_identity", "operation_identity", "orientation",
            "quadrature_identity", "temporal_weight",
        }:
            raise ValueError("ND2 ledger record has unexpected keys")
        for name in ("numerical_flux", "face_measure", "temporal_weight", "integrated_amount"):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]):
                raise ValueError(f"ND2 ledger has nonfinite or nonnumeric {name}")
        if type(row["orientation"]) is not int or type(row["multiplicity"]) is not int:
            raise ValueError("ND2 ledger orientation and multiplicity must be exact integers")
        cell = re.fullmatch(r"cell:(\d+):(\d+)/axis:(\d+)/side:(\d+)/component:(\d+)",
                            row["quadrature_identity"])
        stage = re.search(r"ssprk2_stage_([01])", row["evaluation_context"])
        occurrence = re.search(r"/occurrence:([01])$", row["occurrence_identity"])
        if not cell or not stage or not occurrence:
            raise ValueError("ND2 ledger lost exact cell/stage/physical occurrence identity")
        i, j, axis, side, component = map(int, cell.groups())
        k, ordinal = int(stage[1]), int(occurrence[1])
        key = (k, ordinal, i, j, axis, side, component)
        if key in seen or not (i < 8 and j < 6 and axis < 2 and side < 2 and component < 2):
            raise ValueError("duplicate or invalid ND2 ledger face")
        seen.add(key)
        if row["orientation"] != (-1 if side == 0 else 1) or row["multiplicity"] != 1 \
                or row["occurrence_identity"] != row["operation_identity"] + f"/occurrence:{ordinal}":
            raise ValueError("ND2 ledger orientation/multiplicity/occurrence is inconsistent")
        physical_component = int(saved["order"][component])
        value = stage_values[k][physical_component]
        if axis == 0:
            left, right = (i - 1, i) if side == 0 else (i, i + 1)
            flux = (value[j, right % 8] - value[j, left % 8]) / hx
            area = hy
        else:
            lower, upper = (j - 1, j) if side == 0 else (j, j + 1)
            flux = (value[upper % 6, i] - value[lower % 6, i]) / hy
            area = hx
        weight = .5 * dt * saved["occurrence_weights"][ordinal]
        amount = (-1 if side == 0 else 1) * flux * area * weight
        flux_error = max(flux_error, abs(row["numerical_flux"] - flux))
        measure_error = max(measure_error, abs(row["face_measure"] - area))
        weight_error = max(weight_error, abs(row["temporal_weight"] - weight))
        amount_error = max(amount_error, abs(row["integrated_amount"] - amount))
        accumulated[physical_component, j, i] += row["integrated_amount"]
    return _bound({
        "initial_error": float(np.max(np.abs(initial - prescribed))),
        "state_error": float(np.max(np.abs(actual - expected))),
        "face_flux_error": float(flux_error), "face_amount_error": float(amount_error),
        "face_measure_error": float(measure_error), "temporal_weight_error": float(weight_error),
        "retained_increment_error": float(np.max(np.abs(accumulated - retained))),
        "reynolds_amount_error": float(np.max(np.abs(accumulated - (actual - initial) * hx * hy))),
        "global_amount_defect": float(np.max(np.abs(accumulated.sum(axis=(1, 2))))),
    }, {"initial_error": 2e-14, "state_error": 4e-12, "face_flux_error": 3e-12, "face_amount_error": 3e-13,
        "face_measure_error": 3e-15, "temporal_weight_error": 3e-18,
        "retained_increment_error": 3e-13, "reynolds_amount_error": 4e-13,
        "global_amount_defect": 4e-13})


def _check_receipt_links(root: Path, manifest: dict, declared: set[str]) -> dict[str, int]:
    scope = manifest["native_scope"]
    sdk = scope["sdk_sha256"]
    if not isinstance(sdk, str) or not re.fullmatch(r"[0-9a-f]{64}", sdk) \
            or type(scope["native_abi_version"]) is not int or scope["native_abi_version"] != 4:
        raise ValueError("invalid frozen native scope")
    digests = {dim: scope[f"dimension{dim}_native_sha256"] for dim in (1, 2)}
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in digests.values()):
        raise ValueError("invalid frozen native module digest")
    identities = saved_states = xml_files = snapshot_files = 0
    for relative in sorted(declared):
        path = root / relative
        if path.name == "manifest.json" and path.parent.name == "source-snapshot":
            for name, digest in _json(path).items():
                saved = path.parent / name
                if not saved.resolve().is_relative_to(root) \
                        or str(saved.relative_to(root)) not in declared or _digest(saved) != digest:
                    raise ValueError(f"source snapshot digest differs: {relative}/{name}")
                snapshot_files += 1
        if path.name.endswith("identity.json"):
            data = _json(path)
            native_file = data.get("native_file", "")
            match = re.search(r"/site-packages/pops/_native/dim([12])/_pops[^/]*\.(?:so|pyd)$", native_file)
            if not match or data.get("native_sha256") != digests[int(match[1])] \
                    or not data.get("package_file", "").endswith("/site-packages/pops/__init__.py"):
                raise ValueError(f"native identity contradicts frozen scope: {relative}")
            if "abi_key" in data:
                fields = dict(piece.split("=", 1) for piece in data["abi_key"].split(";"))
                if fields.get("headers") != sdk or fields.get("dim") != match[1]:
                    raise ValueError(f"SDK/dimension receipt contradicts frozen scope: {relative}")
                source = path.parent / "source-files.json"
                if str(source.relative_to(root)) not in declared \
                        or data.get("source_files_sha256") != _digest(source):
                    raise ValueError(f"source identity link differs: {relative}")
            identities += 1
        if path.name == "receipt.json":
            receipt = _json(path)
            if "native_sha256" in receipt and receipt["native_sha256"] not in digests.values():
                raise ValueError(f"scientific receipt native digest differs: {relative}")
            for row in receipt.get("records", ()):
                if "saved_state" not in row:
                    continue
                name = row["saved_state"]
                if type(name) is not str or Path(name).name != name:
                    raise ValueError("saved-state receipt must name one local payload")
                saved = path.parent / name
                if str(saved.relative_to(root)) not in declared \
                        or row.get("saved_state_sha256") != _digest(saved):
                    raise ValueError(f"saved-state receipt digest differs: {relative}/{name}")
                saved_states += 1
    for group in manifest["groups"]:
        directory = root / "runs" / group["name"]
        counts = {}
        for name, claimed in group["junit"].items():
            path = directory / name
            if str(path.relative_to(root)) not in declared:
                raise ValueError("JUnit receipt is absent from declared inventory")
            cases = list(ET.parse(path).getroot().iter("testcase"))
            actual = {"total": len(cases),
                      "failed": sum(case.find("failure") is not None or case.find("error") is not None
                                    for case in cases),
                      "skipped": sum(case.find("skipped") is not None for case in cases)}
            if claimed != actual or any(type(value) is not int for value in claimed.values()):
                raise ValueError(f"JUnit receipt counts differ: {group['name']}/{name}")
            counts[name] = actual
            xml_files += 1
        if group["outer_result"] is None:
            continue
        path = root / group["outer_result"]
        if str(path.relative_to(root)) not in declared or path.parent != directory:
            raise ValueError("outer result is absent or belongs to a different run")
        result = _json(path)
        for key, name in (("identity_sha256", "identity.json"), ("log_sha256", "pytest.log")):
            if key in result and result[key] != _digest(directory / name):
                raise ValueError(f"outer result {key} link differs: {group['name']}")
        if "counts" in result:
            row = result["counts"]
            expected = {"total": row["tests"], "failed": row["failures"] + row["errors"],
                        "skipped": row["skipped"]}
            if counts.get("pytest.xml") != expected:
                raise ValueError("serial result differs from its JUnit receipt")
        if result.get("schema_version") == 3 and "rank_results" in result:
            if not result["rank_results"] or all(row.get("status") == "missing"
                                                for row in result["rank_results"]):
                if group["junit"] or result["status"] == "passed" \
                        or result["authentication_before"] == 0:
                    raise ValueError("MPI reception without rank receipts must retain its preflight refusal")
                continue
            if result["native_sha256"] != digests[result["dimension"]]:
                raise ValueError("MPI result native digest differs from frozen scope")
            before = _json(directory / "before" / "identity.json")
            after = _json(directory / "after" / "identity.json")
            fields = ("python", "package_file", "native_file", "native_sha256", "abi_key",
                      "source_files_sha256", "verified_source_files", "execution_environment")
            if any(before[name] != after[name] for name in fields) \
                    or result["same_installation"] is not True \
                    or result["authentication_before"] != 0 or result["authentication_after"] != 0:
                raise ValueError("MPI before/after installation authentication differs")
            for row in result["rank_results"]:
                prefix, rank_counts = "rank%d" % row["rank"], row["counts"]
                expected = {"total": rank_counts["tests"],
                            "failed": rank_counts["failures"] + rank_counts["errors"],
                            "skipped": rank_counts["skipped"]}
                if counts.get(prefix + ".xml") != expected \
                        or row["xml_sha256"] != _digest(directory / (prefix + ".xml")) \
                        or row["log_sha256"] != _digest(directory / (prefix + ".log")):
                    raise ValueError("MPI result differs from its rank receipts")
    return {"native_identities": identities, "saved_state_digest_links": saved_states,
            "junit_receipts": xml_files, "source_snapshot_digests": snapshot_files}


def check_archive(root: Path, *, recompute: bool) -> dict:
    root = root.resolve()
    manifest = _json(root / "manifest.json")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2:
        raise ValueError("unsupported reception archive")
    declared = set()
    for record in manifest["files"]:
        relative = record["path"]
        path = root / relative
        if relative in declared or not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("duplicate/escaping archive path")
        declared.add(relative)
        content = path.read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"receipt integrity failed: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != declared | {"manifest.json"}:
        raise ValueError("archive inventory differs from the authenticated manifest")
    states = manifest["scientific_states"]
    state_paths = [case["path"] for case in states]
    if not state_paths or len(state_paths) != len(set(state_paths)) \
            or set(state_paths) != {name for name in declared if name.endswith(".npz")}:
        raise ValueError("scientific saved-state inventory must cover every declared NPZ exactly once")
    links = _check_receipt_links(root, manifest, declared)
    results = []
    if recompute:
        for case in manifest["scientific_states"]:
            path = root / case["path"]
            if case["path"] not in declared:
                raise ValueError("unbound scientific saved state")
            if case["kind"] == "m27_mixed_linear_v1":
                metrics = check_m27(path)
            elif case["kind"] == "m26_finite12_v1":
                metrics = check_m26(path)
            elif case["kind"] == "coupled_gradient_dim2_v1" and case["ledger"] in declared:
                metrics = check_nd2(path, root / case["ledger"])
            else:
                raise ValueError("unsupported saved-state recomputation")
            results.append({"path": case["path"], "kind": case["kind"], "metrics": metrics})
    return {"status": "passed", "scope": "offline integrity and independent saved-state equations",
            "schema_version": 2, "receipt_links": links,
            "payloads": len(declared), "recomputed_states": len(results), "results": results,
            "native_scope": manifest["native_scope"], "limitations": manifest["limitations"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_archive(args.archive, recompute=args.recompute), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
