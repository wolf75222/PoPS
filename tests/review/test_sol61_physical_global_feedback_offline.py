"""Independent source/protocol controls; no manufactured positive scientific state."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("physical_true_reader", HERE / "sol61_physical_global_feedback_offline.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
CPP_PIN = "df50a7d1dd56edd3414e1f3b9360f4c4e93e81c5192d282580133a5dc376ea66"
QUANTITY = ("pops.integral.v2/pops.semantic.v3:sha256:"
            "6cf6627f3d7ccb78826188ff4757410f8eea7dab1e58a20163f334cfce7faf4f/"
            "50d7e8526f8cb8b2de50552eab4bcf98fc9428c66e6d6fc7487cd8ad0fdf81fd/q")


@pytest.fixture
def actual():
    raw = base64.b64decode((HERE / "fixtures/t5_sdk375f_physical_true_n8.cpp.b64").read_bytes().rstrip(b"\n"),
                          validate=True)
    assert hashlib.sha256(raw).hexdigest() == CPP_PIN
    ir, code = reader.ir_from_cpp(raw.decode())
    return ir, code


def test_exact_retained_cpp_is_received_without_native_execution(actual):
    ir, code = actual
    assert reader.source_ir(ir)["version"] == 8
    evidence = reader.cpp_source(code, QUANTITY)
    assert evidence == dict(capture_outside_cell_loop=True, physical_capture_evaluations=1,
                            extra_q_multiplier=False, balance_rate_coefficient=1, candidate_dt_multipliers=1)


@pytest.mark.parametrize("old,new", [
    ("((-0.3 * physical_global_2_1) * mass)", "((-0.3 * physical_global_2_1) * mass * physical_global_2_1)"),
    ("((-0.3 * physical_global_2_1) * mass)", "(-0.3 * mass)"),
    ("((-0.3 * physical_global_2_1) * mass)", "((0.3 * physical_global_2_1) * mass)"),
    ("const pops::Real mass = u0A(index, 0);", "const pops::Real mass = u3A(index, 0);"),
    ("const pops::Real cse3_ = (cse1_ * cse2_);", "const pops::Real cse3_ = (cse1_ * cse1_ * cse2_);"),
    ("ctx.axpy(u3, static_cast<pops::Real>(pops::Real(1)), r2, dt, {{0, 1, 1}});",
     "ctx.axpy(u3, static_cast<pops::Real>(dt), r2, dt, {{0, 1, 1}});"),
    ("const pops::Real cse2_ = u3A(index, 0);", "const pops::Real cse2_ = u0A(index, 0);"),
])
def test_executable_physics_countermodels_are_refused(actual, old, new):
    _, code = actual
    assert code.count(old) == 1
    with pytest.raises(ValueError):
        reader.cpp_source(code.replace(old, new), QUANTITY)


def test_foreign_quantity_is_not_accepted_from_documentary_ir(actual):
    _, code = actual
    with pytest.raises(ValueError, match="foreign candidate quantity"):
        reader.cpp_source(code, QUANTITY + "foreign")


@pytest.mark.parametrize("change", [
    lambda ir: ir.__setitem__("version", 6),
    lambda ir: ir["nodes"][1]["attrs"].__setitem__("scope", "accepted"),
    lambda ir: ir["nodes"][1]["attrs"].__setitem__("integral", "foreign"),
    lambda ir: ir["nodes"][1]["attrs"].__setitem__("units", "dimensionless"),
    lambda ir: ir["nodes"][2].__setitem__("inputs", [0]),
    lambda ir: ir["nodes"][2]["attrs"]["physical_global_inputs_v1"][0].__setitem__("input", 0),
    lambda ir: ir["nodes"][2]["attrs"]["physical_global_inputs_v1"][0]["port"]["handle"].__setitem__("kind", "parameter"),
    lambda ir: ir["nodes"][2]["point"].__setitem__("step", 1),
    lambda ir: ir["nodes"][3].__setitem__("inputs", [1]),
    lambda ir: ir["nodes"][4]["attrs"]["expression_nodes"].append(["mul", 4, 4]),
    lambda ir: ir["nodes"][6].__setitem__("inputs", [0, 5]),
    lambda ir: ir["external_trace_transfers"][0].__setitem__("scale", 1.0),
    lambda ir: ir["external_trace_transfers"][0].__setitem__("axis", 1),
    lambda ir: ir["external_trace_transfers"][0].__setitem__("component", 1),
    lambda ir: ir["external_trace_transfers"][0].__setitem__("rate", 2),
    lambda ir: ir["integral_states"][0].__setitem__("initial", 0.8),
])
def test_actual_ir_countermodels_are_refused(actual, change):
    ir, _ = actual
    mutated = copy.deepcopy(ir)
    change(mutated)
    with pytest.raises(ValueError):
        reader.source_ir(mutated)


def test_embedded_ir_duplicate_keys_are_refused(actual):
    ir, _ = actual
    raw = json.dumps(ir)
    raw = raw.replace('"version": 8', '"version": 8, "version": 8')
    with pytest.raises(ValueError, match="duplicate"):
        reader.ir_from_cpp("/*\n" + reader.MARKER + "\n" + raw + "\n\nlowering provenance\n*/")


def protocol_sidecar(raw):
    binary = reader.identity("binary", dict(algorithm="sha256", content_digest=hashlib.sha256(raw).digest(), size=len(raw)))
    spec = "pops.artifact-spec.v1:sha256:" + "a" * 64
    return dict(protocol="pops.artifact-sidecar.v1", binary_identity=binary, artifact_spec_identity=spec,
                semantic_identity="pops.semantic.v3:sha256:" + "b" * 64,
                artifact_identity=reader.identity("artifact", dict(spec=reader.identity_data(spec, "artifact-spec"),
                                                                   binary=reader.identity_data(binary, "binary"))))


def test_binary_component_recomposition_is_protocol_only():
    raw = b"opaque protocol bytes, not a native executable"
    sidecar = protocol_sidecar(raw)
    assert reader.component(raw, json.dumps(sidecar).encode()) == sidecar
    with pytest.raises(ValueError, match="actual SO"):
        reader.component(raw + b"changed", json.dumps(sidecar).encode())


@pytest.mark.parametrize("key", ["binary_identity", "artifact_identity", "artifact_spec_identity", "semantic_identity", "protocol"])
def test_sidecar_foreign_components_cannot_pass(key):
    raw = b"opaque protocol bytes"
    sidecar = protocol_sidecar(raw)
    sidecar[key] += "foreign"
    with pytest.raises(ValueError):
        reader.component(raw, json.dumps(sidecar).encode())


def test_canonical_binary_codec_vectors():
    assert reader.binary_cbor(b"\x00\xff").hex() == "4200ff"
    assert reader.binary_cbor({"b": b"x", "a": 1}).hex() == "a261610161624178"
    for value in (1.0, 1 << 63):
        with pytest.raises(ValueError):
            reader.binary_cbor(value)


def test_external_seal_required_and_paths_refuse_aliases(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="external ROOT seal differs"):
        reader.receive(path, "0" * 64)
    link = tmp_path / "alias"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="noncanonical"):
        reader.leaf(dict(path=str(link), sha256=reader.digest(path.read_bytes())))
    with pytest.raises(ValueError, match="noncanonical"):
        reader.canonical_path(str(tmp_path / ".." / tmp_path.name / "input.json"))


def test_reader_and_dependencies_do_not_import_pops_in_isolated_process(tmp_path):
    program = """import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pops' or fullname.startswith('pops.'):
            raise AssertionError('PoPS import forbidden: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program,
                             str(HERE / "sol61_physical_global_feedback_offline.py")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
