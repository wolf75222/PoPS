"""Actual owner-sealed reception and harness-only, fully resealed counter-cases."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sol61_dim3_fourier_oracle import (B, H, VOLUME, Rejected,
                                     modal_states, read_json, receive, rows_from_modal, sha)

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = Path(os.environ["POPS_DIM3_OWNER_OUTPUTS"]) if "POPS_DIM3_OWNER_OUTPUTS" in os.environ else None
PINS = {
    "serial": ("5ff0b84abb22189d56d66dac26a00d13e0ba2fd10162652347ca2c636b86405d",
               "652357acd9fef343ab0a7a9723ee34a9d520f69d66cbecf7c2f956d116882cbf"),
    "mpi2": ("439694a9377920b26d63af81d36c7e1ce007b3236fd47de2b9df0538063a431f",
             "5c2d35ef17db2536269b214f97eec6dc238a0b3143733fa2917a705b7f65916c"),
}
# This is the independent verifier under attack, not our mathematical oracle.
SPEC = importlib.util.spec_from_file_location("dim3_checker_subject", ROOT / "docs/development/api_040/check_coupled_gradient_dim3_reception.py")
SUBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBJECT)


def actual(kind):
    if OUTPUTS is None:
        pytest.skip("explicit authentic saved-state inventory is required")
    directory = OUTPUTS / f"coupled-gradient-dim3-sdk20d-saved-{kind}-20260930"
    identity = OUTPUTS / f"coupled-gradient-dim3-sdk20d-{kind}-owner-identity-20260930.json"
    return directory, identity, *PINS[kind]


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, sort_keys=True) + "\n")


def wire(rows):
    output = bytearray(b"POPSEX01")
    def word(value):
        output.extend(struct.pack("<Q", value % (1 << 64)))
    word(len(rows))
    for row in rows:
        for name in ("operation_identity", "occurrence_identity", "evaluation_context", "quadrature_identity"):
            value = row[name].encode()
            word(len(value))
            output.extend(value)
        word(row["orientation"])
        for name in ("face_measure", "numerical_flux", "temporal_weight"):
            output.extend(struct.pack("<d", row[name]))
        word(row["multiplicity"])
    return bytes(output)


def accumulate(batches, order):
    result = np.zeros((2, 3, 4, 6))
    for batch in batches:
        for row in batch:
            tokens = row["quadrature_identity"].replace("cell:", "").split("/")
            i, j, k = map(int, tokens[0].split(":"))
            component = int(tokens[-1].split(":")[1])
            result[order[component], k, j, i] += row["integrated_amount"]
    return result


def rewrite_rows(directory, batches, order, final, initial):
    write_json(directory / "native-ledger.json", batches)
    receipt = read_json(directory / "receipt.json")
    digests = []
    for rank, batch in enumerate(batches):
        path = directory / f"ledger-rank-{rank:04d}.bin"
        path.write_bytes(wire(batch))
        digests.append(sha(path))
    receipt.update(records_by_rank=[len(b) for b in batches], ledger_sha256_by_rank=digests)
    # Metrics are producer diagnostics, not authority. Refresh them to the forged
    # equation's own balanced state, never retaining an old passing diagnostic.
    receipt["metrics"].update(record_count=sum(map(len, batches)), max_state_error=0.,
                              max_flux_error=0., max_amount_error=0., max_balance_error=0.)
    write_json(directory / "receipt.json", receipt)
    np.savez_compressed(directory / "increments.npz", native_face_amounts=accumulate(batches, order),
                        state_amounts=(final - initial) * VOLUME)


def rewrite_states(directory, order, fields):
    initial, predictor, final = fields
    with np.load(directory / "initial.npz", allow_pickle=False) as f:
        data = {key: f[key].copy() for key in f.files}
    data.update(initial=initial, requested_initial=initial, bound_input=initial[list(order)])
    np.savez_compressed(directory / "initial.npz", **data)
    with np.load(directory / "accepted.npz", allow_pickle=False) as f:
        data = {key: f[key].copy() for key in f.files}
    data.update(initial=initial, final=final, state_increment=final - initial,
                oracle_predictor=predictor, oracle_final=final)
    np.savez_compressed(directory / "accepted.npz", **data)


def harness_reseal(root, identity_sha):
    """Unit harness seals altered copies. This is never an execution owner seal."""
    manifest = read_json(root / "manifest.json")
    assert manifest["external_identity_sha256"] == identity_sha
    manifest["files"] = {name: sha(root / name) for name in manifest["files"]}
    write_json(root / "manifest.json", manifest)
    return sha(root / "manifest.json")


@pytest.mark.parametrize("kind", ("serial", "mpi2"))
def test_actual_source_native_junit_and_original_equation(kind):
    root, identity, identity_sha, manifest_sha = actual(kind)
    report = receive(root, identity, identity_sha, manifest_sha, ROOT)
    assert report["status"] == "received" and report["verified_source_files"] == 1109
    assert report["historical_git_checked"] is True
    assert [r["records"] for r in report["cases"].values()] == [3456, 3456]
    for case in report["cases"].values():
        assert case["state_error"] < 5.e-16 and case["balance_error"] < 2.e-17


@pytest.mark.parametrize("kind", ("serial", "mpi2"))
@pytest.mark.parametrize("attack", ("wrong_axis", "transpose_B", "forget_occurrence", "stale_predictor", "constant_initial_shift", "rank_duplicate_equation"))
def test_balanced_artificial_equations_are_refused_after_complete_harness_reseal(tmp_path, kind, attack):
    donor, identity_path, identity_sha, _ = actual(kind)
    root = tmp_path / "HARNESSONLY-NOT-NATIVE"
    shutil.copytree(donor, root)
    identity = read_json(identity_path)
    options, matrix, spacing, weights = {}, B, H, (.5, 1.5)
    if attack == "wrong_axis":
        spacing = tuple(reversed(H))
        options["spacing"] = spacing
    elif attack == "transpose_B":
        matrix = B.T
        options["matrix"] = matrix
    elif attack == "forget_occurrence":
        weights = (0., 1.5)
        options["occurrence_sum"] = sum(weights)
    elif attack in ("stale_predictor", "rank_duplicate_equation"):
        options["stale"] = True
    else:
        options["shift"] = (.125, -.25)
    fields = modal_states(**options)
    for token in ("01", "10"):
        order = tuple(map(int, token))
        directory = root / ("coupled-gradient-dim3-order-" + token)
        rewrite_states(directory, order, fields)
        stage1 = fields[0] if attack in ("stale_predictor", "rank_duplicate_equation") else fields[1]
        rows = rows_from_modal(order, identity["cases"][token], fields[0], stage1,
                               matrix=matrix, spacing=spacing, weights=weights)
        if attack == "rank_duplicate_equation":
            # Duplicate all stage-0 contributions, omitting stage 1. The fake
            # final Euler equation and amount ledger remain mutually balanced.
            halves = [rows[:1728], rows[:1728]]
            batches = [sum(halves, [])] if identity["ranks"] == 1 else halves
        else:
            batches = [rows] + [[] for _ in range(identity["ranks"] - 1)]
        rewrite_rows(directory, batches, order, fields[2], fields[0])
        # The counterfeit satisfies its own discrete equation and conservation.
        # Bound the subtraction of stored binary64 states after a constant shift;
        # this harness bound never changes either scientific verifier's criteria.
        roundoff = 3 * np.finfo(float).eps * VOLUME * float(np.max(np.abs(fields[2]) + np.abs(fields[0])))
        np.testing.assert_allclose(accumulate(batches, order), (fields[2] - fields[0]) * VOLUME, rtol=0, atol=roundoff)
        np.testing.assert_allclose(accumulate(batches, order).sum(axis=(1, 2, 3)), 0., rtol=0, atol=4.e-17)
        assert np.sum(fields[2]**2) < np.sum(fields[0]**2) - 1.e-9
    manifest_sha = harness_reseal(root, identity_sha)
    reason = "prescribed initial" if attack == "constant_initial_shift" else "original SSPRK2 equation"
    with pytest.raises(Rejected, match=reason):
        receive(root, identity_path, identity_sha, manifest_sha)
    with pytest.raises(SUBJECT.Rejected, match="science: (bound native initial|SSPRK2 final state)"):
        SUBJECT.verify(root, identity_path, manifest_sha, identity_sha)


def test_rank_duplicate_resealed_wire_cannot_be_a_unique_owned_incidence(tmp_path):
    donor, identity_path, identity_sha, _ = actual("mpi2")
    root = tmp_path / "HARNESSONLY-RANKDUP"
    shutil.copytree(donor, root)
    for token in ("01", "10"):
        directory = root / ("coupled-gradient-dim3-order-" + token)
        batches = read_json(directory / "native-ledger.json")
        # Keep total count 3456, make two coherent rank batches, duplicate a half
        # and omit the other half. Global ownership is still a union, not a sum
        # of equal rank-local observations.
        batches = [batches[0][:1728], batches[0][:1728]]
        with np.load(directory / "accepted.npz", allow_pickle=False) as f:
            initial, final = f["initial"].copy(), f["final"].copy()
        rewrite_rows(directory, batches, tuple(map(int, token)), final, initial)
    manifest_sha = harness_reseal(root, identity_sha)
    with pytest.raises(Rejected, match="original stage/occurrence/incidence identity"):
        receive(root, identity_path, identity_sha, manifest_sha)
    with pytest.raises(SUBJECT.Rejected, match="duplicate incidence"):
        SUBJECT.verify(root, identity_path, manifest_sha, identity_sha)


@pytest.mark.parametrize("kind", ("serial", "mpi2"))
def test_payload_reseal_does_not_replace_owner_identity_pin(tmp_path, kind):
    donor, identity_path, identity_sha, manifest_sha = actual(kind)
    path = tmp_path / "HARNESSONLY-identity.json"
    identity = read_json(identity_path)
    identity["native_source_commit"] = "0" * 40
    write_json(path, identity)
    with pytest.raises(Rejected, match="external owner seal"):
        receive(donor, path, identity_sha, manifest_sha)


@pytest.mark.parametrize("attack", ("failed_case", "foreign_rank", "foreign_dimension", "foreign_artifact", "duplicate_rank", "resealed_source_blob"))
def test_harness_resealed_execution_links_fail_their_original_authority(tmp_path, attack):
    donor, identity_path, _, _ = actual("mpi2")
    root = tmp_path / "HARNESSONLY-PROVENANCE"
    shutil.copytree(donor, root)
    identity = read_json(identity_path)
    if attack == "resealed_source_blob":
        sources = read_json(identity["authenticated_source_manifest"]["path"])
        name = "python/pops/__init__.py"
        sources[name] = "0" * 64
        source_path = tmp_path / "HARNESSONLY-source-files.json"
        write_json(source_path, sources)
        identity["authenticated_source_manifest"].update(path=str(source_path), sha256=sha(source_path))
        launch = read_json(identity["native_launch_identity"]["path"])
        launch["source_files_sha256"] = sha(source_path)
        launch_path = tmp_path / "HARNESSONLY-launch.json"
        write_json(launch_path, launch)
        identity["native_launch_identity"].update(path=str(launch_path), sha256=sha(launch_path))
        reason = "historical source mismatch"
    elif attack == "duplicate_rank":
        identity["junit_by_rank"][1] = dict(identity["junit_by_rank"][0], rank=1)
        result = read_json(identity["native_launch_result"]["path"])
        result["rank_results"][1]["xml_sha256"] = identity["junit_by_rank"][0]["sha256"]
        result_path = tmp_path / "HARNESSONLY-result.json"
        write_json(result_path, result)
        for rank in range(2):
            original = Path(identity["native_launch_result"]["path"]).parent / f"rank{rank}.log"
            shutil.copyfile(original, tmp_path / f"rank{rank}.log")
        identity["native_launch_result"].update(path=str(result_path), sha256=sha(result_path))
        reason = "all-rank JUnit inventory"
    else:
        entry = identity["junit_by_rank"][1]
        xml = ET.parse(entry["path"])
        case = next(xml.iter("testcase"))
        if attack == "failed_case":
            ET.SubElement(case, "failure", message="HARNESSONLY failed execution")
            reason = "JUnit true passed case"
        else:
            key, value = {"foreign_rank": ("mpi_rank", "0"), "foreign_dimension": ("native_dimension", "2"),
                          "foreign_artifact": ("artifact_identity", "HARNESSONLY-foreign")}[attack]
            next(p for p in case.iter("property") if p.get("name") == key).set("value", value)
            reason = "JUnit runtime identity"
        path = tmp_path / "HARNESSONLY-rank1.xml"
        xml.write(path)
        entry.update(path=str(path), sha256=sha(path))
        # Keep result/XML digests mutually coherent so the actual link guard is
        # reached; these are explicitly harness-owned external files.
        result = read_json(identity["native_launch_result"]["path"])
        result["rank_results"][1]["xml_sha256"] = sha(path)
        # Copy rank logs alongside the harness result so its hashes remain
        # mutually coherent; no authentic launcher file is ever overwritten.
        result_path = tmp_path / "HARNESSONLY-result.json"
        write_json(result_path, result)
        for rank in range(2):
            original = Path(identity["native_launch_result"]["path"]).parent / f"rank{rank}.log"
            shutil.copyfile(original, tmp_path / f"rank{rank}.log")
        identity["native_launch_result"].update(path=str(result_path), sha256=sha(result_path))
    path = tmp_path / "HARNESSONLY-identity.json"
    write_json(path, identity)
    manifest = read_json(root / "manifest.json")
    manifest["external_identity_sha256"] = sha(path)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(Rejected, match=reason):
        receive(root, path, sha(path), sha(root / "manifest.json"), ROOT)


def test_modal_anchors_and_mixed_matrix_are_not_an_isotropic_component_zero_norm():
    eigenvalues = [-4 * sum(np.sin(np.pi * f / n)**2 / h**2 for f, n, h in zip(mode, (6, 4, 3), H, strict=True))
                   for mode in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))]
    np.testing.assert_allclose(eigenvalues, (-36., -8., -3., -47.), rtol=0, atol=2.e-14)
    assert not np.array_equal(B, B.T)
    initial, predictor, final = modal_states()
    for axis in (1, 2, 3):
        assert np.max(np.abs(np.diff(initial, axis=axis))) > .01
    assert np.max(np.abs(predictor - initial)) > 1.e-6
    assert np.max(np.abs(final - initial)) > 1.e-6
