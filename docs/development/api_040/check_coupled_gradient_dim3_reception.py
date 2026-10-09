#!/usr/bin/env python3
"""Offline scientific/integrity checker; imports only NumPy and the stdlib.

A trusted external identity and caller-pinned manifest digest are required.
Sealing fingerprints saved evidence; it neither creates native results nor proves
execution. The fixed physical fixture is recalculated without PoPS/helpers.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

CELLS = (6, 4, 3)
LENGTHS = (1., 2., 3.)
H = tuple(length / count for length, count in zip(LENGTHS, CELLS, strict=True))
VOLUME = math.prod(H)
DT = 1.e-5
D = np.array([[.25, .05], [.05, .10]])
R = np.array([[0., -.3], [.3, 0.]])
B = D + R
WEIGHTS = (.5, 1.5)
CRITERIA = dict(state_atol=4.e-12, flux_atol=3.e-12, amount_atol=4.e-13,
                expected_records=3456, initial_bound_atol=2.e-14, minimum_energy_drop=1.e-9)
CONTRACT = "pops.coupled-gradient-dim3.offline-manifest@1"
IDENTITY_CONTRACT = "pops.coupled-gradient-dim3.external-identity@1"


class Rejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Rejected(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "json: duplicate key")
            result[key] = value
        return result
    def constant(value):
        raise Rejected("json: nonfinite constant " + value)
    data = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    def walk(value):
        if isinstance(value, float):
            require(math.isfinite(value), "json: nonfinite number")
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(data)
    return data


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "schema: duplicate NPZ fields")
        require(set(archive.files) == set(keys), "schema: NPZ field inventory " + path.name)
        values = {key: archive[key].copy() for key in archive.files}
    for name, value in values.items():
        require(value.dtype.kind in "fiu" and np.isfinite(value).all(), "schema: nonfinite/dtype " + name)
    return values


def close(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all(), "science: shape/finitude " + label)
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
    require(error <= tolerance, "science: " + label + " error=" + repr(error))
    return error


def prescribed_cell_means():
    # Integrate exp(i*k*x) over each cell, independently of midpoint/sinc helpers.
    means = []
    for count, length in zip(CELLS, LENGTHS, strict=True):
        edges = np.linspace(0., length, count + 1)
        wave = 2 * np.pi / length
        means.append((np.exp(1j * wave * edges[1:]) - np.exp(1j * wave * edges[:-1])) /
                     (1j * wave * (length / count)))
    x, y, z = means[0][None, None, :], means[1][None, :, None], means[2][:, None, None]
    oblique = z * y * x
    return np.stack((1 + .1 * oblique.real + .03 * y.real + .02 * z.imag,
                     -.2 + .07 * oblique.imag + .02 * x.imag + .015 * z.real))


def periodic_connectivity_laplacian():
    # Assemble the DOF graph explicitly; x varies fastest. No np.roll stencil.
    count = math.prod(CELLS)
    matrix = np.zeros((count, count))
    for k, j, i in itertools.product(range(CELLS[2]), range(CELLS[1]), range(CELLS[0])):
        cell = (i, j, k)
        row = (k * CELLS[1] + j) * CELLS[0] + i
        for axis in range(3):
            matrix[row, row] -= 2 / H[axis]**2
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] = (neighbor[axis] + sign) % CELLS[axis]
                column = (neighbor[2] * CELLS[1] + neighbor[1]) * CELLS[0] + neighbor[0]
                matrix[row, column] += 1 / H[axis]**2
    return matrix


def independent_solution():
    initial = prescribed_cell_means()
    laplacian = periodic_connectivity_laplacian()
    operator = sum(WEIGHTS) * np.kron(B, laplacian)
    require(np.array_equal(D, D.T) and np.array_equal(R, -R.T) and
            np.linalg.eigvalsh(D).min() > 0, "science: constitutive symmetric/skew contract")
    close(operator + operator.T, 2 * sum(WEIGHTS) * np.kron(D, laplacian), 1.e-13,
          "skew energy cancellation")
    vector = initial.ravel()
    predictor = vector + DT * (operator @ vector)
    final = .5 * (vector + predictor + DT * (operator @ predictor))
    # Independent discrete Fourier amplification, not a second copy of graph application.
    modes = np.fft.fftn(initial, axes=(1, 2, 3))
    spectral = np.empty_like(modes)
    for k, j, i in itertools.product(range(CELLS[2]), range(CELLS[1]), range(CELLS[0])):
        eigenvalue = -4 * sum(math.sin(math.pi * mode / n)**2 / h**2
                              for mode, n, h in zip((i, j, k), CELLS, H, strict=True))
        a = sum(WEIGHTS) * eigenvalue * B
        amplification = np.eye(2) + DT * a + .5 * DT**2 * (a @ a)
        spectral[:, k, j, i] = amplification @ modes[:, k, j, i]
    spectral = np.fft.ifftn(spectral, axes=(1, 2, 3))
    close(spectral.imag, np.zeros_like(spectral.imag), 2.e-14, "Fourier reality")
    close(final.reshape(initial.shape), spectral.real, 2.e-14, "graph/Fourier agreement")
    return initial, predictor.reshape(initial.shape), final.reshape(initial.shape)


def decode_wire(path):
    data, cursor = Path(path).read_bytes(), 8
    require(data[:8] in (b"POPSEX01", b"POPSEX02"), "wire: foreign profile")
    extended = data[:8] == b"POPSEX02"
    def word():
        nonlocal cursor
        require(cursor + 8 <= len(data), "wire: truncated word")
        value = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8
        return value
    def text():
        nonlocal cursor
        length = word()
        require(length <= len(data) - cursor, "wire: truncated text")
        value = data[cursor:cursor + length].decode("utf-8")
        cursor += length
        return value
    def number():
        value = struct.unpack("<d", struct.pack("<Q", word()))[0]
        require(math.isfinite(value), "wire: nonfinite double")
        return value
    records = []
    count = word()
    require(count <= 3456, "wire: excessive count")
    for _ in range(count):
        row = dict(zip(("operation_identity", "occurrence_identity", "evaluation_context", "quadrature_identity"),
                       (text() for _ in range(4)), strict=True))
        orientation = word()
        row["orientation"] = orientation - (1 << 64) if orientation >= 1 << 63 else orientation
        row.update(face_measure=number(), numerical_flux=number(), temporal_weight=number(), multiplicity=word())
        if extended:
            axis, side, component = (word() - 1 for _ in range(3))
            exterior, source = word(), text()
            require(axis in (-1, 0, 1, 2) and side in (-1, 0, 1) and component in (-1, 0, 1)
                    and exterior == 0, "wire: unexpected extended trace")
            if axis >= 0:
                require(row["quadrature_identity"].endswith(
                    f"/axis:{axis}/side:{side}/component:{component}"), "wire: trace/quadrature discrepancy")
            if source:
                require(row["evaluation_context"].endswith(f"/{len(source)}:{source}"),
                        "wire: source evaluation discrepancy")
        row["integrated_amount"] = row["orientation"] * row["face_measure"] * row["numerical_flux"] * row["temporal_weight"] * row["multiplicity"]
        require(math.isfinite(row["integrated_amount"]), "wire: amount overflow")
        records.append(row)
    if extended:
        require(word() == 0 and word() == 0, "wire: unexpected integrals/consumed traces")
    require(cursor == len(data), "wire: trailing bytes")
    return records


def metadata(data, order):
    for key, value in {"order": order, "cells": CELLS, "lengths": LENGTHS}.items():
        close(data[key], value, 0., "metadata " + key)
    for key, value in {"dt": DT, "spacing": H, "cell_volume": VOLUME,
                       "dissipative": D, "reversible": R, "occurrence_weights": WEIGHTS}.items():
        if key in data:
            close(data[key], value, 0., "metadata " + key)


def check_case(directory, identity, case_identity, order):
    receipt = strict_json(directory / "receipt.json")
    require(type(receipt.get("native_dimension")) is int and type(receipt.get("ranks")) is int,
            "schema: dimension/rank types")
    for key in ("native_dimension", "ranks", "native_sha256", "native_abi", "fixture_sha256", "native_path", "package_path"):
        require(receipt[key] == identity[key], "provenance: " + key)
    for key in ("artifact_identity", "platform_manifest"):
        require(receipt[key] == case_identity[key], "provenance: " + key)
    require(receipt["component_order"] == list(order) and receipt["storage"] == "canonical physical components,z,y,x",
            "schema: component order/storage")
    require(receipt["criteria"] == CRITERIA, "schema: criteria changed")
    require(type(receipt.get("native_macro_step")) is int and receipt["native_macro_step"] == 1 and
            type(receipt.get("native_time")) is float and receipt["native_time"] == DT,
            "provenance: missing/wrong native clock or macro step")
    for key, expected in {"cells": list(CELLS), "lengths": list(LENGTHS), "dt": DT,
                          "dissipative": D.tolist(), "reversible": R.tolist(),
                          "occurrence_weights": list(WEIGHTS)}.items():
        require(receipt[key] == expected, "schema: physical metadata " + key)
    initial_data = load_npz(directory / "initial.npz", ("initial", "requested_initial", "bound_input", "order", "cells", "lengths"))
    data = load_npz(directory / "accepted.npz", ("initial", "final", "state_increment", "oracle_predictor", "oracle_final", "order", "dt", "cells", "lengths", "spacing", "cell_volume", "dissipative", "reversible", "occurrence_weights"))
    increments = load_npz(directory / "increments.npz", ("native_face_amounts", "state_amounts"))
    metadata(initial_data, order)
    metadata(data, order)
    initial, predictor, final = independent_solution()
    for name, observed in (("bound native initial", initial_data["initial"]), ("requested initial", initial_data["requested_initial"]),
                           ("accepted initial", data["initial"])):
        close(observed, initial, 2.e-14, name)
    close(initial_data["bound_input"], initial[list(order)], 2.e-14, "permuted input")
    state_error = close(data["final"], final, CRITERIA["state_atol"], "SSPRK2 final state")
    close(data["state_increment"], data["final"] - data["initial"], 0., "state increment")
    close(data["oracle_predictor"], predictor, 2.e-14, "saved predictor diagnostic")
    close(data["oracle_final"], final, 2.e-14, "saved final diagnostic")
    batches = strict_json(directory / "native-ledger.json")
    require(type(batches) is list and len(batches) == identity["ranks"], "schema: rank batches")
    require(all(type(value) is int for value in receipt["records_by_rank"]), "schema: rank count types")
    require(receipt["records_by_rank"] == [len(batch) for batch in batches], "integrity: rank counts")
    require(len(receipt["ledger_sha256_by_rank"]) == identity["ranks"], "integrity: rank digest count")
    records = []
    for rank, batch in enumerate(batches):
        wire = directory / f"ledger-rank-{rank:04d}.bin"
        require(digest(wire) == receipt["ledger_sha256_by_rank"][rank], "integrity: receipt wire digest")
        decoded = decode_wire(wire)
        require(type(batch) is list and len(decoded) == len(batch), "integrity: native JSON/wire discrepancy")
        numeric = {"face_measure", "numerical_flux", "temporal_weight", "integrated_amount"}
        for native, projected in zip(decoded, batch, strict=True):
            require(type(projected) is dict and set(native) == set(projected), "integrity: native JSON/wire discrepancy")
            for key, value in native.items():
                if key in numeric:
                    require(type(projected[key]) is float and struct.pack("<d", value) == struct.pack("<d", projected[key]),
                            "integrity: native JSON/wire discrepancy")
                else:
                    require(type(projected[key]) is type(value) and projected[key] == value,
                            "integrity: native JSON/wire discrepancy")
        records.extend(batch)
    require(len(records) == 3456, "science: incidence count")
    amounts, seen = np.zeros_like(initial), set()
    flux_error, amount_error = 0., 0.
    stage_fields = (initial.reshape(2, -1), predictor.reshape(2, -1))
    stage_potentials = tuple(B @ value for value in stage_fields)
    for row in records:
        require(all(type(row[name]) is float for name in ("face_measure", "numerical_flux", "temporal_weight", "integrated_amount")),
                "schema: binary64 record numbers")
        require(row["operation_identity"] == case_identity["operation_identity"], "provenance: operation identity")
        context = row["evaluation_context"]
        contexts = case_identity["stage_contexts"]
        require(context in contexts.values(), "provenance: stage context")
        stage = next(int(key) for key, value in contexts.items() if value == context)
        tokens = re.fullmatch(r"cell:(\d+):(\d+):(\d+)/axis:([012])/side:([01])/component:([01])", row["quadrature_identity"])
        require(tokens is not None, "science: quadrature grammar")
        i, j, k, axis, side, component = map(int, tokens.groups())
        require(all(v < n for v, n in zip((i, j, k), CELLS, strict=True)), "science: quadrature domain")
        match = re.fullmatch(re.escape(row["operation_identity"]) + r"/occurrence:([01])", row["occurrence_identity"])
        require(match is not None, "science: occurrence grammar")
        occurrence = int(match.group(1))
        incidence = (i, j, k, axis, side, component, occurrence, stage)
        require(incidence not in seen, "science: duplicate incidence")
        seen.add(incidence)
        require(type(row["orientation"]) is int and row["orientation"] == 2 * side - 1 and
                type(row["multiplicity"]) is int and row["multiplicity"] == 1, "science: orientation/multiplicity")
        area = math.prod(H[tangent] for tangent in range(3) if tangent != axis)
        require(row["face_measure"] == area and row["temporal_weight"] == DT / 2 * WEIGHTS[occurrence],
                "science: face measure/quadrature duration")
        left, right = [i, j, k], [i, j, k]
        left[axis] = (left[axis] - (side == 0)) % CELLS[axis]
        right[axis] = (right[axis] + (side == 1)) % CELLS[axis]
        def flat(xyz):
            return (xyz[2] * CELLS[1] + xyz[1]) * CELLS[0] + xyz[0]
        physical = order[component]
        flux = (stage_potentials[stage][physical, flat(right)] - stage_potentials[stage][physical, flat(left)]) / H[axis]
        amount = (2 * side - 1) * area * flux * DT / 2 * WEIGHTS[occurrence]
        flux_error = max(flux_error, close(row["numerical_flux"], flux, CRITERIA["flux_atol"], "face flux"))
        amount_error = max(amount_error, close(row["integrated_amount"], amount, CRITERIA["amount_atol"], "face amount"))
        amounts[physical, k, j, i] += row["integrated_amount"]
    close(increments["native_face_amounts"], amounts, 0., "saved face accumulation")
    close(increments["state_amounts"], (data["final"] - data["initial"]) * VOLUME, 0., "saved state amounts")
    close(amounts, (data["final"] - data["initial"]) * VOLUME, CRITERIA["amount_atol"], "cell balance")
    close(amounts.sum(axis=(1, 2, 3)), np.zeros(2), CRITERIA["amount_atol"], "periodic global balance")
    initial_energy, final_energy = float(np.sum(data["initial"]**2) * VOLUME), float(np.sum(data["final"]**2) * VOLUME)
    require(final_energy < initial_energy - CRITERIA["minimum_energy_drop"], "science: dissipative energy")
    return dict(records=len(records), state_error=state_error, flux_error=flux_error,
                amount_error=amount_error, initial_energy=initial_energy, final_energy=final_energy)


def inventory(root, ranks):
    require(root.is_dir() and not root.is_symlink(), "integrity: evidence directory")
    names = ["initial.npz", "accepted.npz", "increments.npz", "native-ledger.json", "receipt.json"]
    names += [f"ledger-rank-{rank:04d}.bin" for rank in range(ranks)]
    expected = {f"coupled-gradient-dim3-order-{order}/{name}" for order in ("01", "10") for name in names}
    paths = tuple(root.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "integrity: symlink payload")
    actual = {str(path.relative_to(root)) for path in paths if path.is_file() and path != root / "manifest.json"}
    require(actual == expected, "integrity: payload file inventory")
    return expected


def external_identity(path):
    identity = strict_json(path)
    require(identity["contract"] == IDENTITY_CONTRACT and identity["native_dimension"] == 3 and
            type(identity["ranks"]) is int and identity["ranks"] in (1, 2), "provenance: identity profile")
    for key in ("native_sha256", "fixture_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", identity[key]) is not None, "provenance: hash syntax")
    require(identity["evidence_kind"] in ("native-reception", "synthetic-unit"), "provenance: evidence kind")
    require(type(identity["native_abi"]) is str and identity["native_abi"], "provenance: native ABI")
    require(set(identity["cases"]) == {"01", "10"}, "provenance: permutation inventory")
    if identity["evidence_kind"] == "native-reception":
        require(identity.get("native_exit_status") == 0 and type(identity.get("native_exit_status")) is int,
                "provenance: native launch status")
        require(re.fullmatch(r"[0-9a-f]{40}", identity.get("native_source_commit", "")) is not None,
                "provenance: native source commit")
        require(len(identity["junit_by_rank"]) == identity["ranks"], "provenance: JUnit rank inventory")
        require(len({entry["path"] for entry in identity["junit_by_rank"]}) == identity["ranks"],
                "provenance: duplicate JUnit path")
        for rank, entry in enumerate(identity["junit_by_rank"]):
            require(type(entry.get("rank")) is int and entry["rank"] == rank, "provenance: JUnit rank index")
            require(digest(entry["path"]) == entry["sha256"], "provenance: JUnit digest")
            tree = ET.parse(entry["path"])
            for order, name in entry["testcases"].items():
                require(order in ("01", "10"), "provenance: JUnit permutation")
                nodes = [node for node in tree.iter("testcase") if node.get("name") == name]
                require(len(nodes) == 1 and not any(nodes[0].find(tag) is not None
                    for tag in ("failure", "error", "skipped")), "provenance: JUnit case did not pass")
                props = {}
                for prop in nodes[0].iter("property"):
                    name, value = prop.get("name"), prop.get("value")
                    require(name not in props or props[name] == value, "provenance: JUnit conflicting properties")
                    props[name] = value
                require(props.get("mpi_rank") == str(rank) and props.get("mpi_size") == str(identity["ranks"])
                        and props.get("native_dimension") == "3", "provenance: JUnit rank/dimension links")
                require(props.get("artifact_identity") == identity["cases"][order]["artifact_identity"],
                        "provenance: JUnit artifact link")
                require(Path(props.get("coupled_gradient_dim3_receipts", "")).name ==
                        "coupled-gradient-dim3-order-" + order, "provenance: JUnit receipt link")
            require(set(entry["testcases"]) == {"01", "10"}, "provenance: JUnit case inventory")
    for case in identity["cases"].values():
        require(set(case["stage_contexts"]) == {"0", "1"} and len(set(case["stage_contexts"].values())) == 2,
                "provenance: stage identity inventory")
        for stage, context in case["stage_contexts"].items():
            require(re.findall(r"StagePoint\(name='ssprk2_stage_([01])',", context) == [stage],
                    "provenance: stage identity discriminator")
    return identity


def seal(root, identity_path):
    root = Path(root)
    identity = external_identity(identity_path)
    files = inventory(root, identity["ranks"])
    manifest = dict(contract=CONTRACT, external_identity_sha256=digest(identity_path),
                    files={name: digest(root / name) for name in sorted(files)})
    target = root / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return digest(target)


def verify(root, identity_path, manifest_sha256, identity_sha256):
    require(digest(identity_path) == identity_sha256, "provenance: caller-pinned identity digest")
    root = Path(root)
    require(digest(root / "manifest.json") == manifest_sha256, "integrity: pinned manifest digest")
    manifest, identity = strict_json(root / "manifest.json"), external_identity(identity_path)
    require(set(manifest) == {"contract", "external_identity_sha256", "files"} and manifest["contract"] == CONTRACT,
            "integrity: manifest contract")
    require(manifest["external_identity_sha256"] == digest(identity_path), "provenance: identity digest")
    files = inventory(root, identity["ranks"])
    require(set(manifest["files"]) == files, "integrity: manifest inventory")
    for name in files:
        require(digest(root / name) == manifest["files"][name], "integrity: payload digest " + name)
    reports = {order: check_case(root / ("coupled-gradient-dim3-order-" + order), identity,
                               identity["cases"][order], tuple(map(int, order))) for order in ("01", "10")}
    left = np.load(root / "coupled-gradient-dim3-order-01/accepted.npz", allow_pickle=False)
    right = np.load(root / "coupled-gradient-dim3-order-10/accepted.npz", allow_pickle=False)
    with left, right:
        close(left["final"], right["final"], CRITERIA["state_atol"], "permutation equivalence")
    return dict(status="PASS", evidence_kind=identity["evidence_kind"],
                scope=("synthetic checker unit control; no native results" if identity["evidence_kind"] == "synthetic-unit" else
                       "saved Dim3 Uniform scientific evidence; native execution authenticity requires external pins"),
                native_dimension=3, ranks=identity["ranks"], cases=reports)


def draft_identity(root):
    """Copy opaque metadata only; pending identity cannot authorize verification."""
    root = Path(root)
    identity = {"contract": IDENTITY_CONTRACT, "evidence_kind": "pending-owner-pin", "cases": {}}
    for order in ("01", "10"):
        directory = root / ("coupled-gradient-dim3-order-" + order)
        receipt = strict_json(directory / "receipt.json")
        for key in ("native_dimension", "ranks", "native_sha256", "native_abi", "fixture_sha256", "native_path", "package_path"):
            if key in identity:
                require(identity[key] == receipt[key], "provenance: draft identity divergence " + key)
            identity[key] = receipt[key]
        records = [row for batch in strict_json(directory / "native-ledger.json") for row in batch]
        operations = {row["operation_identity"] for row in records}
        require(len(operations) == 1, "provenance: draft operation inventory")
        stages = {}
        for context in {row["evaluation_context"] for row in records}:
            matches = re.findall(r"StagePoint\(name='ssprk2_stage_([01])',", context)
            require(len(matches) == 1 and matches[0] not in stages, "provenance: draft stage identity")
            stages[matches[0]] = context
        require(set(stages) == {"0", "1"}, "provenance: draft stage inventory")
        identity["cases"][order] = dict(artifact_identity=receipt["artifact_identity"],
            platform_manifest=receipt["platform_manifest"], operation_identity=operations.pop(), stage_contexts=stages)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--draft-identity", action="store_true")
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--identity-sha256")
    args = parser.parse_args()
    try:
        if args.draft_identity:
            print(json.dumps(draft_identity(args.root), indent=2, allow_nan=False))
        elif args.seal:
            require(args.identity is not None, "provenance: external identity required")
            print(json.dumps(dict(manifest_sha256=seal(args.root, args.identity), identity_sha256=digest(args.identity))))
        else:
            require(args.identity is not None, "provenance: external identity required")
            require(args.manifest_sha256 is not None and args.identity_sha256 is not None,
                    "integrity: caller-pinned manifest and identity digests required")
            print(json.dumps(verify(args.root, args.identity, args.manifest_sha256, args.identity_sha256), indent=2, allow_nan=False))
    except (Rejected, KeyError, TypeError, ValueError, AttributeError, OSError, struct.error, ET.ParseError) as error:
        parser.exit(1, "REJECTED: " + str(error) + "\n")


if __name__ == "__main__":
    main()
