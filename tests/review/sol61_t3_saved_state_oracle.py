"""Independent saved-state T3 reception: NumPy/stdlib only, no PoPS imports.

Inventory is deliberately unsealed. Verification requires externally supplied
owner pins and their external SHA, never a receipt's own self-declared digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = "sol61.t3.saved-state-pins@1"
CELLS = (5, 4)  # physical x,y; arrays have y,x storage
LENGTHS = (1., 1.5)
ORDERS = ((0, 1), (1, 0), (2, 0, 1))
DIFFUSION = np.array(((.04, .006, 0.), (-.003, .05, 0.), (0., 0., .03)))
STATE_ATOL = RESIDUAL_ATOL = 2e-8


class ReceptionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReceptionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def json_read(path):
    return json.loads(Path(path).read_text())


def inventory(runs, fixture):
    result = {"schema": SCHEMA, "authority": "unsealed-inventory",
              "oracle": file_record(__file__), "fixture": file_record(fixture), "runs": []}
    for name, directory in runs:
        directory = Path(directory).resolve()
        mpi = name == "mpi2"
        identity = directory / ("before/identity.json" if mpi else "identity.json")
        identity_data = json_read(identity)
        files = sorted(directory.glob("*.json")) + sorted(directory.glob("*.xml"))
        files += sorted(directory.glob("*.log"))
        if mpi:
            files += [directory / phase / filename for phase in ("before", "after")
                      for filename in ("identity.json", "source-files.json")]
        datasets = []
        for order in ORDERS:
            suffix = "nonlinear-field-order-" + "".join(map(str, order))
            matches = list(directory.glob("**/" + suffix + "/native.npz"))
            require(len(matches) == 1, "expected exactly one actual rank-zero NPZ for " + suffix)
            npz = matches[0]
            receipt = npz.with_name("receipt.json")
            received = json_read(receipt)
            require(sha(npz) == received["saved_state_sha256"], "NPZ differs from receipt")
            require(tuple(received["unknown_order"]) == order, "dataset unknown order mismatch")
            cache = npz.parent.parent / "pops-native-cache"
            compiled = sorted(cache.glob("*.so")) + sorted(cache.glob("*.so.pops-artifact.json"))
            require(len(compiled) == 6, "expected three actual compiled component DSOs/manifests")
            datasets.append({"order": list(order), "npz": file_record(npz),
                             "receipt": file_record(receipt),
                             "artifact_identity": received["artifact_identity"],
                             "compiled_components": [file_record(p) for p in compiled]})
        result["runs"].append({"name": name, "directory": str(directory),
            "mpi_size": 2 if mpi else 1, "source_commit": identity_data["source_commit"],
            "native_sha256": identity_data["native_sha256"], "native_abi": identity_data["abi_key"],
            "files": [file_record(p) for p in files], "datasets": datasets})
    return result


def periodic_second_derivative(count, length):
    # Explicit physical-axis stencil matrix; no author np.roll/einsum oracle.
    matrix = np.zeros((count, count), dtype=np.float64)
    spacing = length / count
    for i in range(count):
        matrix[i, i] -= 2. / spacing**2
        matrix[i, (i + 1) % count] += 1. / spacing**2
        matrix[i, (i - 1) % count] += 1. / spacing**2
    return matrix


def laplace(values):
    dx = periodic_second_derivative(CELLS[0], LENGTHS[0])
    dy = periodic_second_derivative(CELLS[1], LENGTHS[1])
    return np.stack([dy @ component + component @ dx.T for component in values])


def original_lhs(values, parameter, diffusion=DIFFUSION):
    q, v = values[:2]
    rows = [q * (parameter[0] + q * q + .03 * v) + .02 * v * v,
            v * (1.3 + .4 * v * v + .02 * q) + .01 * q * q]
    if len(values) == 3:
        z = values[2]
        rows.append(z * (1.4 + z * z))
    lap = laplace(values)
    for row in range(len(values)):
        for column in range(len(values)):
            rows[row] = rows[row] - float(diffusion[row, column]) * lap[column]
    return np.stack(rows)


def declared_cell_averages(width):
    # Integrate trigonometric data over physical cells by antiderivatives.
    # The author uses np.sinc at normalized centers; no author function imported.
    means = []
    for count, length in zip(CELLS, LENGTHS, strict=True):
        spacing, wave = length / count, 2. * math.pi / length
        lo = np.arange(count, dtype=np.float64) * spacing
        hi = lo + spacing
        cos_mean = (np.sin(wave * hi) - np.sin(wave * lo)) / (wave * spacing)
        sin_mean = (np.cos(wave * lo) - np.cos(wave * hi)) / (wave * spacing)
        means.append((cos_mean, sin_mean))
    cx, sx = means[0]
    cy, sy = means[1]
    mixed_cos = cy[:, None] * cx[None, :] - sy[:, None] * sx[None, :]
    target = np.stack([np.broadcast_to(.15 + .02 * cx[None, :], (4, 5)),
                       np.broadcast_to(.25 + .015 * sy[:, None], (4, 5)),
                       .18 + .01 * mixed_cos][:width])
    parameter = (1. + .04 * mixed_cos)[None, :, :]
    return target, parameter


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"solved", "forcing", "parameter"}, "unexpected saved state keys")
        data = {key: archive[key].copy() for key in archive.files}
    width = data["forcing"].shape[0]
    require(width in (2, 3), "unsupported observed field product")
    for key, shape in (("solved", (2, width, 4, 5)), ("forcing", (width, 4, 5)),
                       ("parameter", (1, 4, 5))):
        require(data[key].shape == shape, "shape/physical-axis mismatch: " + key)
        require(data[key].dtype == np.dtype("float64"), "non-float64 observation: " + key)
        require(np.isfinite(data[key]).all(), "nonfinite actual saved observations: " + key)
    return data


def scientific_report(data):
    values, forcing, parameter = data["solved"], data["forcing"], data["parameter"]
    target, expected_parameter = declared_cell_averages(len(forcing))
    capture_error = float(np.max(np.abs(parameter - expected_parameter)))
    forcing_error = float(np.max(np.abs(forcing - original_lhs(target, expected_parameter))))
    require(capture_error < 2e-13, "original captured parameter differs from declared cell averages")
    require(forcing_error < 2e-13, "original forcing differs from declared equations/cell averages")
    state_errors = [float(np.max(np.abs(row - target))) for row in values]
    residuals = [float(np.max(np.abs(original_lhs(row, parameter) - forcing))) for row in values]
    require(max(state_errors) < STATE_ATOL, "actual solution differs from exact manufactured target")
    require(max(residuals) < RESIDUAL_ATOL, "actual solution fails original nonlinear field residual")
    seed = .8 * values[0]  # actual first solution, used only to diagnose initialization
    seed_residual = float(np.max(np.abs(original_lhs(seed, parameter) - forcing)))
    require(seed_residual > 1e-3, "distinct second seed unexpectedly solves original equations")
    attacks = {
        "transpose_diffusion": float(np.max(np.abs(original_lhs(values[1], parameter, DIFFUSION.T) - forcing))),
        "wrong_capture_route": float(np.max(np.abs(original_lhs(values[1], values[0, :1]) - forcing))),
        "scaled_stale_forcing": float(np.max(np.abs(original_lhs(values[1], parameter) - .8 * forcing))),
        "seed_substituted_as_rhs": float(np.max(np.abs(original_lhs(values[1], parameter) - seed))),
        "opaque_local_only_equations": float(np.max(np.abs(original_lhs(values[1], parameter, np.zeros((3, 3))) - forcing))),
    }
    require(all(value > RESIDUAL_ATOL for value in attacks.values()), "counter-equation escaped scientific guard")
    return {"state_errors": state_errors, "original_residuals": residuals,
            "parameter_error": capture_error, "forcing_error": forcing_error,
            "second_seed_original_residual": seed_residual, "counter_equation_residuals": attacks,
            "two_solutions_difference": float(np.max(np.abs(values[0] - values[1])))}


def check_file(record):
    require(set(record) == {"path", "sha256", "bytes"}, "invalid external file pin shape")
    require(Path(record["path"]).stat().st_size == record["bytes"], "external file size mismatch")
    require(sha(record["path"]) == record["sha256"], "external file hash mismatch: " + record["path"])


def verify(pins_path, pins_sha256):
    require(sha(pins_path) == pins_sha256, "external owner pins SHA mismatch")
    pins = json_read(pins_path)
    require(pins["schema"] == SCHEMA and pins["authority"] == "root-reviewed",
            "owner must independently review and seal inventory")
    check_file(pins["oracle"])
    require(sha(__file__) == pins["oracle"]["sha256"], "executing oracle differs from owner pin")
    check_file(pins["fixture"])
    received = []
    for run in pins["runs"]:
        for record in run["files"]:
            check_file(record)
        directory = Path(run["directory"])
        result = json_read(directory / "result.json")
        require(result["status"] == "passed" and result["returncode"] == 0, "native run did not pass")
        mpi = run["mpi_size"] == 2
        identity = json_read(directory / ("before/identity.json" if mpi else "identity.json"))
        require(identity["source_commit"] == run["source_commit"], "native source identity mismatch")
        require(identity["native_sha256"] == run["native_sha256"] and identity["abi_key"] == run["native_abi"],
                "native binary/SDK ABI mismatch")
        if mpi:
            require(result["same_installation"] and result["test_sources_unchanged"] and
                    result["rank_test_parity"] and not result["timeout"], "MPI receipt authority incomplete")
            after = json_read(directory / "after/identity.json")
            require(after["native_sha256"] == identity["native_sha256"] and
                    after["abi_key"] == identity["abi_key"] and
                    after["source_files_sha256"] == identity["source_files_sha256"],
                    "production installation changed during MPI run")
        else:
            require(result["identity_sha256"] == sha(directory / "identity.json"), "run identity was resealed")
        xmls = [directory / ("rank%d.xml" % rank) for rank in range(2)] if mpi else [directory / "pytest.xml"]
        for xml in xmls:
            suites = list(ET.parse(xml).iter("testsuite"))
            require(len(suites) == 1 and suites[0].get("tests") == "8" and
                    all(suites[0].get(k) == "0" for k in ("errors", "failures", "skipped")),
                    "actual native JUnit run incomplete")
            cases = [case for case in suites[0] if case.tag == "testcase" and
                     case.get("classname", "").endswith("test_nonlinear_mixed_field_runtime")]
            require(len(cases) == 5, "expected three T3 successes and two transactional refusals")
        require([tuple(d["order"]) for d in run["datasets"]] == list(ORDERS), "dataset routes/order mismatch")
        for dataset in run["datasets"]:
            for record in [dataset["npz"], dataset["receipt"], *dataset["compiled_components"]]:
                check_file(record)
            receipt = json_read(dataset["receipt"]["path"])
            require(receipt["saved_state_sha256"] == dataset["npz"]["sha256"], "NPZ self-reseal mismatch")
            require(receipt["contract"] == "pops.spatial-field-residual@1" and receipt["native_dimension"] == 2,
                    "original equation contract mismatch")
            require(receipt["native_sha256"] == run["native_sha256"] and receipt["native_abi"] == run["native_abi"] and
                    receipt["fixture_sha256"] == pins["fixture"]["sha256"], "receipt provenance mismatch")
            require(receipt["native_path"] == identity["native_file"] and
                    receipt["package_path"] == identity["package_file"], "installed package/native path mismatch")
            platform = receipt["platform_manifest"]
            header_signature = dict(piece.split("=", 1) for piece in run["native_abi"].split(";"))["headers"]
            require(platform["backend"]["value"] == "production" and platform["target"]["value"] == "system" and
                    platform["device"]["value"] == "cpu" and platform["communicator"]["value"] == "MPI_COMM_WORLD" and
                    platform["abi"]["value"].split("|")[0] == header_signature and
                    all(platform["precision"][kind]["value"] == "float64"
                        for kind in ("storage", "compute", "accumulation", "reduction")),
                    "platform/SDK/precision contract mismatch")
            require(receipt["mpi_size"] == run["mpi_size"] and receipt["native_time"] == .01 and
                    type(receipt["native_macro_step"]) is int and receipt["native_macro_step"] == 1,
                    "actual accepted clock mismatch")
            require(receipt["cells"] == list(CELLS) and receipt["lengths"] == list(LENGTHS) and
                    receipt["unknown_order"] == dataset["order"] and receipt["artifact_identity"] == dataset["artifact_identity"],
                    "physical grid or artifact route mismatch")
            require(receipt["state_atol"] == STATE_ATOL and receipt["original_residual_atol"] == RESIDUAL_ATOL,
                    "scientific admission was weakened")
            numerical = scientific_report(load_arrays(dataset["npz"]["path"]))
            received.append({"run": run["name"], "order": dataset["order"], **numerical})
    require(len(received) == 6, "expected three actual datasets per serial/MPI2 run")
    return {"status": "received", "owner_pins_sha256": pins_sha256, "datasets": received,
            "scope": "actual saved nonlinear field solutions only; no new PoPS execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    inv = modes.add_parser("inventory")
    inv.add_argument("--run", action="append", required=True, help="serial=PATH or mpi2=PATH")
    inv.add_argument("--fixture", required=True)
    inv.add_argument("--output", required=True)
    check = modes.add_parser("verify")
    check.add_argument("--pins", required=True)
    check.add_argument("--pins-sha256", required=True)
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = inventory([tuple(item.split("=", 1)) for item in args.run], args.fixture) if args.mode == "inventory" else verify(args.pins, args.pins_sha256)
        Path(args.output).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        print(value.get("status", "unsealed inventory prepared"))
    except (ReceptionError, OSError, KeyError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
