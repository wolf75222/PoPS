"""Synthetic unit controls for the independent offline checker; never native evidence."""
import importlib.util
import itertools
import json
from pathlib import Path
import struct

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("independent_dim3_checker", ROOT / "docs/development/api_040/check_coupled_gradient_dim3_reception.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def wire_image(rows):
    data = bytearray(b"POPSEX01")
    def word(value):
        data.extend(struct.pack("<Q", value % (1 << 64)))
    def text(value):
        encoded = value.encode()
        word(len(encoded))
        data.extend(encoded)
    def real(value):
        data.extend(struct.pack("<d", value))
    word(len(rows))
    for row in rows:
        for name in ("operation_identity", "occurrence_identity", "evaluation_context", "quadrature_identity"):
            text(row[name])
        word(row["orientation"])
        for name in ("face_measure", "numerical_flux", "temporal_weight"):
            real(row[name])
        word(row["multiplicity"])
    return bytes(data)


def synthetic_rows(order, initial, predictor, case):
    rows = []
    for stage, values in enumerate((initial, predictor)):
        potential = checker.B @ values.reshape(2, -1)
        for occurrence, k, j, i, axis, side, component in itertools.product(
                range(2), range(3), range(4), range(6), range(3), range(2), range(2)):
            left, right = [i, j, k], [i, j, k]
            left[axis] = (left[axis] - (side == 0)) % checker.CELLS[axis]
            right[axis] = (right[axis] + (side == 1)) % checker.CELLS[axis]
            def flat(p):
                return (p[2] * 4 + p[1]) * 6 + p[0]
            flux = float((potential[order[component], flat(right)] - potential[order[component], flat(left)]) / checker.H[axis])
            row = dict(operation_identity=case["operation_identity"],
                       occurrence_identity=case["operation_identity"] + "/occurrence:" + str(occurrence),
                       evaluation_context=case["stage_contexts"][str(stage)],
                       quadrature_identity=f"cell:{i}:{j}:{k}/axis:{axis}/side:{side}/component:{component}",
                       orientation=2 * side - 1, face_measure=float(np.prod([checker.H[a] for a in range(3) if a != axis])),
                       numerical_flux=flux, temporal_weight=checker.DT / 2 * checker.WEIGHTS[occurrence], multiplicity=1)
            row["integrated_amount"] = row["orientation"] * row["face_measure"] * flux * row["temporal_weight"]
            rows.append(row)
    return rows


def face_amounts(rows, order):
    result = np.zeros((2, 3, 4, 6))
    for row in rows:
        match = checker.re.fullmatch(r"cell:(\d+):(\d+):(\d+)/axis:([012])/side:([01])/component:([01])", row["quadrature_identity"])
        i, j, k, _, _, component = map(int, match.groups())
        result[order[component], k, j, i] += row["integrated_amount"]
    return result


def reseal_rows(directory, rows, order):
    write_json(directory / "native-ledger.json", [rows])
    wire = directory / "ledger-rank-0000.bin"
    wire.write_bytes(wire_image(rows))
    receipt = checker.strict_json(directory / "receipt.json")
    receipt["ledger_sha256_by_rank"] = [checker.digest(wire)]
    receipt["records_by_rank"] = [len(rows)]
    write_json(directory / "receipt.json", receipt)
    with np.load(directory / "increments.npz") as saved:
        amounts = saved["state_amounts"].copy()
    np.savez_compressed(directory / "increments.npz", native_face_amounts=face_amounts(rows, order), state_amounts=amounts)


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "synthetic-unit-evidence"
    root.mkdir()
    identity_path = tmp_path / "external-identity.json"
    cases = {order: dict(artifact_identity="SYNTHETIC-UNIT-" + order, platform_manifest={"synthetic_unit": True},
                        operation_identity="synthetic-unit.operation." + order,
                        stage_contexts={str(stage): f"StagePoint(name='ssprk2_stage_{stage}', synthetic_unit=True)" for stage in range(2)})
             for order in ("01", "10")}
    identity = dict(contract=checker.IDENTITY_CONTRACT, evidence_kind="synthetic-unit", native_dimension=3, ranks=1,
                    native_sha256="0" * 64, native_abi="SYNTHETIC-NOT-NATIVE", fixture_sha256="1" * 64,
                    native_path="SYNTHETIC-NOT-NATIVE", package_path="SYNTHETIC-NOT-INSTALLED", cases=cases)
    write_json(identity_path, identity)
    initial, predictor, final = checker.independent_solution()
    for token in ("01", "10"):
        order = tuple(map(int, token))
        directory = root / ("coupled-gradient-dim3-order-" + token)
        directory.mkdir()
        np.savez_compressed(directory / "initial.npz", initial=initial, requested_initial=initial,
                            bound_input=initial[list(order)], order=order, cells=checker.CELLS, lengths=checker.LENGTHS)
        np.savez_compressed(directory / "accepted.npz", initial=initial, final=final, state_increment=final-initial,
                            oracle_predictor=predictor, oracle_final=final, order=order, dt=checker.DT,
                            cells=checker.CELLS, lengths=checker.LENGTHS, spacing=checker.H, cell_volume=checker.VOLUME,
                            dissipative=checker.D, reversible=checker.R, occurrence_weights=checker.WEIGHTS)
        rows = synthetic_rows(order, initial, predictor, cases[token])
        write_json(directory / "native-ledger.json", [rows])
        wire = directory / "ledger-rank-0000.bin"
        wire.write_bytes(wire_image(rows))
        np.savez_compressed(directory / "increments.npz", native_face_amounts=face_amounts(rows, order),
                            state_amounts=(final-initial)*checker.VOLUME)
        receipt = dict((key, identity[key]) for key in ("native_dimension", "ranks", "native_sha256", "native_abi", "fixture_sha256", "native_path", "package_path"))
        receipt.update(cases[token])
        receipt.update(component_order=list(order), storage="canonical physical components,z,y,x",
                       criteria=checker.CRITERIA, native_time=checker.DT, native_macro_step=1,
                       cells=list(checker.CELLS), lengths=list(checker.LENGTHS), dt=checker.DT,
                       dissipative=checker.D.tolist(), reversible=checker.R.tolist(), occurrence_weights=list(checker.WEIGHTS),
                       records_by_rank=[len(rows)], ledger_sha256_by_rank=[checker.digest(wire)])
        write_json(directory / "receipt.json", receipt)
    return root, identity_path, checker.seal(root, identity_path), checker.digest(identity_path)


def update_npz(path, transform):
    with np.load(path) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    transform(values)
    np.savez_compressed(path, **values)


def test_analytic_modes_and_connectivity_are_independent():
    matrix = checker.periodic_connectivity_laplacian()
    nx, ny, nz = checker.CELLS
    xyz = np.meshgrid((np.arange(nx)+.5)/nx, (np.arange(ny)+.5)/ny, (np.arange(nz)+.5)/nz, indexing="ij")
    for frequencies, eigenvalue in (((1,0,0), -36.), ((0,1,0), -8.), ((0,0,1), -3.), ((1,1,1), -47.)):
        mode = np.prod([np.exp(2j*np.pi*f*x) for f,x in zip(frequencies,xyz,strict=True)], axis=0).transpose(2,1,0).ravel()
        np.testing.assert_allclose(matrix @ mode, eigenvalue * mode, rtol=0, atol=7.e-14)
    initial = checker.prescribed_cell_means()
    ax = 3/np.pi
    ay = 2*np.sqrt(2)/np.pi
    az = 3*np.sqrt(3)/(2*np.pi)
    expected = 1 + .1*ax*ay*az*np.cos(3*np.pi/4) + .03*ay*np.cos(np.pi/4) + .02*az*np.sin(np.pi/3)
    assert abs(initial[0,0,0,0]-expected) < 5.e-16


def test_synthetic_baseline_never_claims_native(evidence):
    report = checker.verify(*evidence)
    assert report["evidence_kind"] == "synthetic-unit"
    assert all(case["records"] == 3456 for case in report["cases"].values())


@pytest.mark.parametrize("attack,reason", [("payload", "payload digest"), ("foreign_native", "provenance: native_sha256"),
    ("wire_json", "JSON/wire discrepancy"), ("extra", "payload file inventory"), ("missing_clock", "native clock")])
def test_integrity_and_provenance_attacks(evidence, attack, reason):
    root, identity, manifest, identity_sha = evidence
    directory = root / "coupled-gradient-dim3-order-01"
    if attack == "payload":
        with (directory / "accepted.npz").open("ab") as stream:
            stream.write(b"corruption")
    elif attack == "extra":
        (directory / "foreign.npz").write_bytes(b"undeclared")
    else:
        if attack == "wire_json":
            rows = checker.strict_json(directory / "native-ledger.json")
            rows[0][0]["numerical_flux"] += .01
            write_json(directory / "native-ledger.json", rows)
        else:
            receipt = checker.strict_json(directory / "receipt.json")
            if attack == "foreign_native":
                receipt["native_sha256"] = "f" * 64
            else:
                del receipt["native_time"]
            write_json(directory / "receipt.json", receipt)
        manifest = checker.seal(root, identity)
    with pytest.raises(checker.Rejected, match=reason):
        checker.verify(root, identity, manifest, identity_sha)


@pytest.mark.parametrize("attack,reason", [("final", "SSPRK2 final state"), ("constant_initial_shift", "bound native initial"),
    ("flux", "face flux"), ("weight", "quadrature duration"), ("permutation", "face flux"), ("duplicate", "duplicate incidence")])
def test_fully_resealed_scientific_attacks(evidence, attack, reason):
    root, identity, _, identity_sha = evidence
    token = "10" if attack == "permutation" else "01"
    order = tuple(map(int, token))
    directory = root / ("coupled-gradient-dim3-order-" + token)
    if attack == "final":
        def corrupt(data):
            data["final"][0,0,0,0] += 1.e-4
            data["state_increment"] = data["final"] - data["initial"]
        update_npz(directory / "accepted.npz", corrupt)
        with np.load(directory / "accepted.npz") as saved:
            state_amount = (saved["final"]-saved["initial"])*checker.VOLUME
        update_npz(directory / "increments.npz", lambda data: data.update(state_amounts=state_amount))
    elif attack == "constant_initial_shift":
        update_npz(directory / "initial.npz", lambda data: data.update({key:data[key]+.02 for key in ("initial","requested_initial","bound_input")}))
        update_npz(directory / "accepted.npz", lambda data: data.update({key:data[key]+.02 for key in ("initial","final","oracle_predictor","oracle_final")}))
    else:
        rows = checker.strict_json(directory / "native-ledger.json")[0]
        if attack == "duplicate":
            rows[1] = rows[0].copy()
        elif attack == "permutation":
            # Correct fields, but native component 0 receives physical flux 0, not permuted flux 1.
            rows[0]["numerical_flux"] = rows[1]["numerical_flux"]
        elif attack == "flux":
            rows[0]["numerical_flux"] += .01
        else:
            rows[0]["temporal_weight"] *= .5
        for row in rows:
            row["integrated_amount"] = row["orientation"]*row["face_measure"]*row["numerical_flux"]*row["temporal_weight"]*row["multiplicity"]
        reseal_rows(directory, rows, order)
    manifest = checker.seal(root, identity)
    with pytest.raises(checker.Rejected, match=reason):
        checker.verify(root, identity, manifest, identity_sha)


@pytest.mark.parametrize("fault", ("failure", "skipped", "rank_spoof", "foreign_artifact"))
def test_junit_links_refuse_failed_or_foreign_rank(evidence, fault):
    root, identity_path, _, _ = evidence
    identity = checker.strict_json(identity_path)
    identity.update(evidence_kind="native-reception", native_exit_status=0, native_source_commit="2"*40)
    xml = root.parent / "synthetic-unit-junit.xml"
    parts = ["<testsuite>"]
    for order in ("01", "10"):
        case = identity["cases"][order]
        props = dict(mpi_rank="1" if fault == "rank_spoof" else "0", mpi_size="1", native_dimension="3",
                     artifact_identity="foreign" if fault == "foreign_artifact" else case["artifact_identity"],
                     coupled_gradient_dim3_receipts="coupled-gradient-dim3-order-"+order)
        parts.append(f'<testcase name="unit-{order}"><properties>')
        parts += [f'<property name="{name}" value="{value}"/>' for name,value in props.items()]
        parts.append("</properties>")
        if fault in ("failure", "skipped"):
            parts.append(f"<{fault}/>")
        parts.append("</testcase>")
    parts.append("</testsuite>")
    xml.write_text("".join(parts))
    identity["junit_by_rank"] = [dict(rank=0, path=str(xml), sha256=checker.digest(xml),
                                     testcases={order:"unit-"+order for order in ("01", "10")})]
    write_json(identity_path, identity)
    with pytest.raises(checker.Rejected, match="provenance: JUnit"):
        checker.external_identity(identity_path)


@pytest.mark.parametrize("fault", ("nonfinite_json", "nonfinite_npz"))
def test_nonfinite_payloads_cannot_hide_from_maxima(evidence, fault):
    root, identity, _, identity_sha = evidence
    directory = root / "coupled-gradient-dim3-order-01"
    if fault == "nonfinite_json":
        receipt = directory / "receipt.json"
        receipt.write_text(receipt.read_text().replace('"native_time": 1e-05', '"native_time": NaN'))
    else:
        def corrupt(data):
            data["final"][0,0,0,0] = np.nan
        update_npz(directory / "accepted.npz", corrupt)
    manifest = checker.seal(root, identity)
    with pytest.raises(checker.Rejected, match="nonfinite"):
        checker.verify(root, identity, manifest, identity_sha)


def test_draft_identity_is_not_an_execution_pin(evidence):
    root, identity, _, _ = evidence
    draft = checker.draft_identity(root)
    assert draft["evidence_kind"] == "pending-owner-pin"
    write_json(identity, draft)
    with pytest.raises(checker.Rejected, match="evidence kind"):
        checker.external_identity(identity)


def test_external_identity_has_a_separate_caller_pin(evidence):
    root, identity, _, identity_sha = evidence
    data = checker.strict_json(identity)
    data["native_sha256"] = "a" * 64
    write_json(identity, data)
    manifest = checker.seal(root, identity)
    with pytest.raises(checker.Rejected, match="caller-pinned identity digest"):
        checker.verify(root, identity, manifest, identity_sha)
