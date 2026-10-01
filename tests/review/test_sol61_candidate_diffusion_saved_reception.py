"""Explicitly synthetic mathematics/protocol controls, never Native evidence."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import re
import xml.etree.ElementTree as ET
import sys

import numpy as np
import pytest

PATH = Path(__file__).with_name("sol61_candidate_diffusion_saved_reception.py")
spec = importlib.util.spec_from_file_location("candidate_diffusion_synthetic_tests", PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def vector_action(
    q, alpha, *, frozen=False, wrong_column=False, transpose=False, harmonic=False, cubic=True
):
    """Second mathematical form: endpoint coefficient arrays and divergence.

    This uses NumPy rolls, unlike the received oracle's unique-face/Fraction
    loops. It manufactures only labelled synthetic unit-test inputs.
    """
    width, ny, nx = q.shape
    d, reaction = (np.array(matrix, dtype=float) for matrix in r.matrices(width))
    if transpose:
        d = d.T
    cell = d[:, :, None, None] * (1 + alpha)
    if not frozen:
        cell = cell * (1 + 3 * (q[:, None] if wrong_column else q[None]) ** 2)
    spatial = np.zeros_like(q)
    for axis, cells in ((1, ny), (2, nx)):
        next_d, previous_d = np.roll(cell, -1, axis + 1), np.roll(cell, 1, axis + 1)
        if harmonic:
            high = np.divide(
                2 * cell * next_d, cell + next_d, out=np.zeros_like(cell), where=cell + next_d != 0
            )
            low = np.divide(
                2 * cell * previous_d,
                cell + previous_d,
                out=np.zeros_like(cell),
                where=cell + previous_d != 0,
            )
        else:
            high, low = (cell + next_d) / 2, (cell + previous_d) / 2
        for row in range(width):
            spatial[row] -= (
                cells
                * cells
                * np.sum(
                    high[row] * (np.roll(q, -1, axis) - q) - low[row] * (q - np.roll(q, 1, axis)),
                    axis=0,
                )
            )
    return spatial + np.einsum("rc,cyx->ryx", reaction, q) + (0.2 * q**3 if cubic else 0), spatial


def synthetic_math(width, n=8):
    q, alpha = r.prescribed(n, width)
    forcing, _ = vector_action(q, alpha)
    initial = dict(response=np.zeros_like(q), forcing=forcing, material=alpha[None], target=q)
    states = {
        phase: dict(
            response=time * q,
            forcing=forcing.copy(),
            material=alpha[None].copy(),
            solution=q.copy(),
            time=np.array(time),
            step=np.array(step),
        )
        for phase, (time, step) in r.CLOCKS.items()
    }
    return initial, states


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("cells", [8, 16])
def test_synthetic_original_independent_forms_and_unique_face_conservation(width, cells):
    initial, states = synthetic_math(width, cells)
    q, alpha = initial["target"], initial["material"][0]
    expected, _ = vector_action(q, alpha)
    actual, spatial = r.original(q, alpha)
    exact_action, _ = r.original(q, alpha, rational=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(exact_action, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(spatial.sum(axis=(1, 2)), 0, rtol=0, atol=2e-14)
    report = r.science(initial, states, width, cells)
    assert max(v["original_residual_relative_l2"] for v in report.values()) < 2e-13


def test_signed_nonsymmetric_singular_D_is_preserved_and_reaction_closes_row():
    d, reaction = (np.array(matrix, dtype=float) for matrix in r.matrices(3))
    assert np.linalg.det(d) == 0 and d[1, 0] < 0 and not np.array_equal(d, d.T)
    assert np.linalg.det(reaction) != 0 and d[2, 0] != 0 and not np.any(d[:, 2])


@pytest.mark.parametrize(
    "substitute", ["frozen", "wrong_column", "transpose", "harmonic", "no_cubic"]
)
def test_synthetic_resealed_wrong_original_equation_load_is_refused(substitute, tmp_path):
    initial, states = synthetic_math(3)
    kwargs = {substitute: True} if substitute != "no_cubic" else {"cubic": False}
    false_load, _ = vector_action(initial["target"], initial["material"][0], **kwargs)
    initial["forcing"] = false_load
    for image in states.values():
        image["forcing"] = false_load.copy()
    # A fresh NPZ hash is a countermodel reseal, not an externally approved receipt.
    file = tmp_path / "synthetic-wrong-equation.npz"
    np.savez(file, **initial)
    assert r.digest(file.read_bytes())
    with pytest.raises(ValueError, match="original candidate-D manufactured load differs"):
        r.science(initial, states, 3, 8)


@pytest.mark.parametrize(
    "attack",
    ["alpha_bit", "forcing_bit", "clock", "bool_step", "consumer", "replay_bit", "solution", "nan"],
)
def test_synthetic_readonly_clock_replay_and_physical_guards(attack):
    initial, states = synthetic_math(3)
    bad = states["replay"]
    if attack in ("alpha_bit", "forcing_bit", "replay_bit"):
        key = {"alpha_bit": "material", "forcing_bit": "forcing", "replay_bit": "solution"}[attack]
        bad[key].flat[0] = np.nextafter(bad[key].flat[0], np.inf)
    elif attack == "clock":
        bad["time"] = np.array(np.nextafter(0.02, np.inf))
    elif attack == "bool_step":
        bad["step"] = np.array(True)
    elif attack == "consumer":
        bad["response"] *= 2
    elif attack == "solution":
        bad["solution"] *= 1.001
    else:
        bad["solution"].flat[0] = np.nan
    with pytest.raises(ValueError):
        r.science(initial, states, 3, 8)


def diag_payload(records, ranks=1):
    images = []
    for rank in range(ranks):
        body = b"".join(
            struct.pack("<Q", len(name)) + name + bits for name, bits in sorted(records.items())
        )
        images.append(b"POPSDIA1" + struct.pack("<QQQQ", 64, rank, ranks, len(records)) + body)
    offsets = np.cumsum([0] + list(map(len, images)), dtype=np.int64)
    return dict(
        program_diagnostics_state=np.frombuffer(b"".join(images), dtype=np.uint8).copy(),
        program_diagnostics_offsets=offsets,
    )


def synthetic_diagnostics():
    values = dict(
        residual_norm=1e-12,
        reference_residual_norm=1.0,
        rel_residual=1e-12,
        full_residual_evaluations=20.0,
        finite_difference_jvps=5.0,
    )
    names = {key: "field_residual_7." + key for key in values}
    records = {names[key].encode(): struct.pack("<d", value) for key, value in values.items()}
    records[b"\x00\xffopaque"] = bytes.fromhex("420000000000f87f")
    return diag_payload(records, 2), names


def test_synthetic_diagnostic_codec_opaque_bits_and_original_guard():
    payload, names = synthetic_diagnostics()
    tables = r.diagnostic_images(payload, 2)
    r.diagnostic_science(tables, names)
    assert tables[0][b"\x00\xffopaque"] == bytes.fromhex("420000000000f87f")


@pytest.mark.parametrize(
    "attack",
    [
        "rank",
        "width",
        "count",
        "offset",
        "trailing",
        "missing_record",
        "relative",
        "ratio",
        "evaluations",
    ],
)
def test_synthetic_diagnostic_negatives(attack):
    payload, names = synthetic_diagnostics()
    raw = payload["program_diagnostics_state"]
    if attack in ("rank", "width", "count"):
        offset, value = {"rank": (16, 1), "width": (8, 32), "count": (32, 2**64 - 1)}[attack]
        raw[offset : offset + 8] = np.frombuffer(struct.pack("<Q", value), dtype=np.uint8)
    elif attack == "offset":
        payload["program_diagnostics_offsets"][1] = 0
    elif attack == "trailing":
        payload["program_diagnostics_state"] = np.append(raw, np.uint8(0))
        payload["program_diagnostics_offsets"][-1] += 1
    else:
        tables = r.diagnostic_images(payload, 2)
        key = {
            "missing_record": "rel_residual",
            "relative": "rel_residual",
            "ratio": "rel_residual",
            "evaluations": "full_residual_evaluations",
        }[attack]
        if attack == "missing_record":
            del tables[0][names[key].encode()]
        else:
            value = {"relative": 1e-5, "ratio": 1e-13, "evaluations": 4.0}[attack]
            tables[0][names[key].encode()] = struct.pack("<d", value)
        with pytest.raises(ValueError):
            r.diagnostic_science(tables, names)
        return
    with pytest.raises(ValueError):
        r.diagnostic_images(payload, 2)


@pytest.mark.parametrize("step", [1, 2])
def test_synthetic_history_duration_and_occurrence(step):
    name = "q2"
    raw = b"POPSHID1" + struct.pack("<Q", 2) + b"q2" + struct.pack("<qQ", -1, 2)
    raw += b"".join(
        struct.pack(
            "<QQQQ",
            2,
            int.from_bytes(struct.pack("<d", s), "little"),
            int.from_bytes(struct.pack("<d", 0.01), "little"),
            1,
        )
        for s in (0.0, (step - 1) * 0.01)
    )
    r.history_identity(raw, name, step)
    corrupt = bytearray(raw)
    corrupt[-1] ^= 1
    with pytest.raises(ValueError):
        r.history_identity(bytes(corrupt), name, step)


def synthetic_cpp():
    order = [2, 0, 1]
    unknowns = [{"local_id": "q%d" % c} for c in order]
    captures = [{"local_id": "forcing"}, {"local_id": "material"}]

    def literal(v):
        return ["literal", {"kind": "binary64", "value": float(v).hex()}]

    def q(c):
        i = order.index(c)
        return ["unknown", i, unknowns[i]]

    def product(a, b):
        return ["mul", a, b]

    d, reaction = r.matrices(3)
    alpha = ["add", literal(1), ["input", 1, 0, captures[1]]]
    diffusion = [
        product(
            literal(d[row][col]),
            product(alpha, ["add", literal(1), product(literal(3), product(q(col), q(col)))]),
        )
        for row in order
        for col in order
    ]
    local = []
    for row in order:
        expression = ["neg", ["input", 0, row, captures[0]]]
        for col in range(3):
            expression = ["add", expression, product(literal(reaction[row][col]), q(col))]
        local.append(
            ["add", expression, product(literal(0.2), product(q(row), product(q(row), q(row))))]
        )
    source = dict(
        coefficient_evaluation=r.POLICY,
        linear_residual_verification=r.CRITERION,
        coefficient_face_policy="pops.field.face-mean.arithmetic@1",
        unknown_components=unknowns,
        diffusion=diffusion,
        local_expressions=local,
    )
    attrs = dict(
        contract="pops.spatial-field-residual@3",
        ncomp=3,
        coefficient_evaluation=r.POLICY,
        linear_residual_verification=r.CRITERION,
        source_contract=source,
        newton_controls={
            k: (
                {"scalar": {"kind": "integer", "value": str(v)}}
                if type(v) is int
                else {"kind": "binary64", "value": v.hex()}
            )
            for k, v in r.CONTROLS.items()
        },
        finite_difference_step={"kind": "binary64", "value": (1e-6).hex()},
        seed_index=None,
        problem_kind="original_field_equations",
        capture_count=2,
    )
    attrs["solver_identity"] = r.components.identity(
        "prepared-spatial-newton", attrs["newton_controls"]
    )
    clock = {"synthetic": True}
    point = dict(schema_version=1, clock=clock, step=0, offset=dict(kind="integer", value="0"))
    ir = dict(
        version=11,
        name="captured-diffusion-accepted-response",
        clock=clock,
        nodes=[
            dict(id=1, op="state", point=point, state=captures[0]),
            dict(id=2, op="state", point=point, state=captures[1]),
            dict(
                id=3,
                op="solve_spatial_field",
                attrs=attrs,
                inputs=[0, 0, 1, 2],
                point=dict(name="frozen-original-coefficients", partitions={"main": point}),
            ),
        ],
    )
    header = "/*\n" + r.components.MARKER + "\n" + json.dumps(ir) + "\n\nlowering provenance\n*/\n"
    body = "\n".join(
        (
            "pops::elliptic::nd::prepare_general_field_coefficients<pops::kNativeDimension, 3, 9, false, true>(*a,*b);",
            'throw std::logic_error("candidate coefficient point/attempt/lane authority changed");',
            'pops::collectively_rethrow_exception(e,*lane,"candidate diffusion local evaluation");',
            'throw std::logic_error("nonfinite_original_field_residual");',
            "auto workspace=std::make_shared<pops::runtime::program::PreparedSpatialResidual<pops::kNativeDimension>>(q, pops::FieldNewtonOptions{.tolerance = 1e-10, .max_iterations = 20, .linear_tolerance = 1e-08, .linear_max_iterations = 240, .restart = 60, .armijo = 0.0001, .minimum_step = 0.0009765625}, 1e-6, true, &ctx.prepared_execution_lane(), true);",
            *(
                f'ctx.record_scalar("field_residual_7.{member}", v);'
                for member in (
                    "residual_norm",
                    "reference_residual_norm",
                    "rel_residual",
                    "full_residual_evaluations",
                    "finite_difference_jvps",
                )
            ),
        )
    )
    return (header + body).encode()


@pytest.mark.parametrize(
    "attack", [None, "frozen", "version", "criterion", "face", "extra_stage", "true_option"]
)
def test_synthetic_cpp_policy_metadata_and_route(attack):
    cpp = synthetic_cpp()
    if attack is None:
        names, _ = r.cpp_contract(cpp, 3)
        assert len(names) == 5
        return
    replacements = {
        "frozen": (r.POLICY, "pops.field.coefficients.frozen@1"),
        "version": ('"version": 11', '"version": 10'),
        "criterion": (r.CRITERION, "old-projected-criterion"),
        "face": ("pops.field.face-mean.arithmetic@1", "harmonic"),
        "extra_stage": ('"diffusion":', '"temporal_tau": {}, "diffusion":'),
        "true_option": (
            "&ctx.prepared_execution_lane(), true);",
            "&ctx.prepared_execution_lane(), false);",
        ),
    }
    a, b = replacements[attack]
    with pytest.raises(ValueError):
        r.cpp_contract(cpp.replace(a.encode(), b.encode()), 3)


@pytest.mark.parametrize("attack", ["control", "fd", "capture_next", "stage", "seed"])
def test_synthetic_ir_control_capture_and_stage_authority(attack):
    ir, body = r.components.ir_from_cpp(synthetic_cpp().decode())
    attrs = ir["nodes"][2]["attrs"]
    if attack == "control":
        attrs["newton_controls"]["max_iterations"] = {"scalar": {"kind": "integer", "value": "21"}}
    elif attack == "fd":
        attrs["finite_difference_step"]["value"] = (1e-5).hex()
    elif attack == "capture_next":
        ir["nodes"][0]["point"]["step"] = 1
    elif attack == "stage":
        ir["nodes"][2]["point"]["partitions"]["main"]["offset"]["value"] = "1"
    else:
        attrs["seed_index"] = 4
    cpp = (
        "/*\n" + r.components.MARKER + "\n" + json.dumps(ir) + "\n\nlowering provenance\n*/" + body
    ).encode()
    with pytest.raises(ValueError):
        r.cpp_contract(cpp, 3)


@pytest.mark.parametrize("attack", ["file_symlink", "parent_symlink", "foreign", "drift"])
def test_pinned_origin_aliases_and_changed_bytes_refuse(tmp_path, attack):
    allowed = tmp_path / "approved"
    allowed.mkdir()
    path = allowed / "source.cpp"
    path.write_bytes(b"explicitly synthetic original")
    pin = r.leaf(path)
    if attack == "file_symlink":
        alias = allowed / "alias.cpp"
        alias.symlink_to(path)
        pin["path"] = str(alias)
    elif attack == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(allowed, target_is_directory=True)
        pin["path"] = str(alias / "source.cpp")
    elif attack == "foreign":
        other = tmp_path / "outside.cpp"
        other.write_bytes(path.read_bytes())
        pin = r.leaf(other)
    else:
        path.write_bytes(b"different bytes with unchanged pin")
    with pytest.raises(ValueError):
        r.pinned(pin, [str(allowed)])


def test_component_byte_identity_is_recomputed_without_compiler_execution():
    binary = b"synthetic identity unit control, not a DSO"
    identity = r.components.identity
    spec = r.components.identity_data("pops.artifact-spec.v1:sha256:" + "b" * 64, "artifact-spec")
    token = identity(
        "binary",
        dict(algorithm="sha256", content_digest=bytes.fromhex(r.digest(binary)), size=len(binary)),
    )
    binary_data = r.components.identity_data(token, "binary")
    sidecar = dict(
        protocol="pops.artifact-sidecar.v1",
        binary_identity=token,
        artifact_spec_identity="pops.artifact-spec.v1:sha256:" + "b" * 64,
        semantic_identity="pops.semantic.v3:sha256:" + "a" * 64,
        artifact_identity=identity("artifact", dict(spec=spec, binary=binary_data)),
    )
    r.components.component(binary, json.dumps(sidecar).encode())
    with pytest.raises(ValueError):
        r.components.component(binary + b"changed", json.dumps(sidecar).encode())


def test_missing_external_approval_refuses_before_any_data_access():
    with pytest.raises(ValueError, match="two external seals"):
        r.receive("must-not-be-opened", None, None, None)


def test_reviewed_source_fingerprints_and_old_reader_unchanged():
    root = PATH.resolve().parents[2]
    for role, (relative, sha) in r.SOURCE_FINGERPRINTS.items():
        data = (
            subprocess.check_output(["git", "show", "a1eb496f:" + relative], cwd=root)
            if role == "request_contract"
            else (root / relative).read_bytes()
        )
        assert r.digest(data) == sha
    relative = "tests/review/sol61_captured_diffusion_saved_reception.py"
    parent = subprocess.check_output(["git", "show", r.SOURCE + ":" + relative], cwd=root)
    assert (root / relative).read_bytes() == parent


def test_offline_reader_has_no_pops_import_in_fresh_isolated_process(tmp_path):
    code = """import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pops" or fullname.startswith("pops."):
            raise RuntimeError("forbidden PoPS import: " + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    run = subprocess.run(
        [sys.executable, "-I", "-c", code, str(PATH.resolve())],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "pending inventory only" in run.stdout


@pytest.mark.parametrize("attack", ["beta", "reaction", "column", "forcing", "unknown_order"])
def test_synthetic_documentary_original_polynomial_is_exact(attack):
    ir, body = r.components.ir_from_cpp(synthetic_cpp().decode())
    source = ir["nodes"][2]["attrs"]["source_contract"]
    if attack == "beta":
        # factor multiplying q squared; the altered polynomial is reserialized.
        source["diffusion"][1][2][2][2][1][1]["value"] = (2.0).hex()
    elif attack == "reaction":
        source["local_expressions"][0] = [
            "add",
            source["local_expressions"][0],
            ["literal", {"kind": "integer", "value": "1"}],
        ]
    elif attack == "column":
        source["diffusion"][1], source["diffusion"][2] = (
            source["diffusion"][2],
            source["diffusion"][1],
        )
    elif attack == "forcing":
        ir["nodes"][0]["state"]["local_id"] = "foreign-input"
    else:
        source["unknown_components"] = list(reversed(source["unknown_components"]))
    raw = (
        "/*\n" + r.components.MARKER + "\n" + json.dumps(ir) + "\n\nlowering provenance\n*/" + body
    ).encode()
    with pytest.raises(ValueError):
        r.cpp_contract(raw, 3)


def test_no_documentary_ir_cannot_claim_checkpoint_scope():
    _, body = r.components.ir_from_cpp(synthetic_cpp().decode())
    with pytest.raises(ValueError, match="IR was not retained"):
        r.cpp_contract(body.encode(), 3)
    names, ir = r.cpp_contract(body.encode(), 3, require_ir=False)
    assert len(names) == 5 and ir is None


def test_raw_junit_preserves_outside_failure_without_masking_selected_failure():
    cases = {
        key: {
            "receipt": {"path": key + "/receipt.json"},
            "directory": key,
            "artifact": key + "-artifact",
        }
        for key in r.CASES
    }
    tests = []
    for key, (width, _) in r.CASES.items():
        props = {
            "captured_diffusion_receipt": key + "/receipt.json",
            "evidence_path": key,
            "artifact_identity": key + "-artifact",
            "rank": "0",
            "size": "1",
            "dimension": "2",
        }
        test = ET.Element(
            "testcase",
            classname="tests.test_public_captured_diffusion",
            name="test_public_candidate_diffusion_nonconstant_saved_and_exact_replay["
            + ("1-order0" if width == 1 else "3-order1")
            + "]",
        )
        ps = ET.SubElement(test, "properties")
        for k, v in props.items():
            ET.SubElement(ps, "property", name=k, value=v)
        tests.append(test)
    outside = ET.Element("testcase", name="outside-diagnostics")
    ET.SubElement(outside, "failure")
    tests.append(outside)
    suite = ET.Element("testsuite", tests="3", failures="1", errors="0", skipped="0")
    suite.extend(tests)
    assert r.junit(ET.tostring(suite), 0, 1, cases) == dict(
        total=3, selected=2, failures=1, errors=0, skipped=0
    )
    ET.SubElement(tests[0], "failure")
    suite.set("failures", "2")
    with pytest.raises(ValueError, match="selected candidate"):
        r.junit(ET.tostring(suite), 0, 1, cases)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_iterations", "21"),
        ("linear_max_iterations", "241"),
        ("restart", "80"),
        ("tolerance", "1e-9"),
        ("armijo", "0.001"),
    ],
)
def test_retained_cpp_controls_cannot_be_changed_without_ir(field, value):
    _, body = r.components.ir_from_cpp(synthetic_cpp().decode())
    altered = re.sub(r"\." + field + r" = [^,}]+", "." + field + " = " + value, body)
    with pytest.raises(ValueError, match="controls differ"):
        r.cpp_contract(altered.encode(), 3, require_ir=False)
