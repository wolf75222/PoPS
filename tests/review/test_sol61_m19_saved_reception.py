"""Synthetic protocol/math controls only. No native execution or positive NPZ artifact."""
import copy
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("m19_saved_review_test", Path(__file__).with_name("sol61_m19_saved_reception.py"))
reader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reader
spec.loader.exec_module(reader)


def synthetic_states(nx, nv, width):
    # Inputs deliberately differ from the author affine fixture helper.
    population = np.random.default_rng(501 + nx + nv + width).integers(-7, 9, (width, nx, nv)).astype("float64") / 8
    initial = dict(population=population, integral=np.zeros((width, 1, nx)),
                   weighted=np.zeros((width, 1, nx)), extended=np.zeros((width, nx, nv)))
    uniform = np.zeros((width, 1, nx))
    signed = np.zeros((width, 1, nx))
    for c in range(width):
        for x in range(nx):
            uniform[c, 0, x] = float(sum((Fraction(float(population[c, x, j])) * Fraction(4, nv) for j in range(nv)), Fraction()))
            signed[c, 0, x] = float(sum((Fraction(float(population[c, x, j])) * ((-1) ** j * (j + 1)) for j in range(nv)), Fraction()))
    accepted = dict(population=population.copy(), integral=uniform, weighted=signed,
                    extended=np.repeat(signed[:, 0, :, None], nv, axis=2))
    return initial, [copy.deepcopy(accepted) for _ in range(3)]


@pytest.mark.parametrize("nx,nv,width,reverse", sorted(reader.CASES))
def test_fraction_original_from_saved_input_not_author_oracle(nx, nv, width, reverse):
    initial, phases = synthetic_states(nx, nv, width)
    assert max(reader.original(initial, phases, nx, nv, width).values()) == 0


@pytest.mark.parametrize("attack", ("signed_abs", "normalize", "component", "uniform_measure", "lift", "state_nan", "shape", "restart_bits"))
def test_scientific_mutations_still_fail_when_integrity_hashes_are_recomputed(attack):
    nx, nv, width = 4, 3, 3
    initial, phases = synthetic_states(nx, nv, width)
    if attack == "signed_abs":
        phases[0]["weighted"] = reader.oracle.reduce_exact(initial["population"], ("x", "v"), ("x",), {"v": (1, 2, 3)})[:, None, :]
    elif attack == "normalize":
        phases[0]["weighted"] /= 2
    elif attack == "component":
        phases[0]["integral"] = phases[0]["integral"][::-1].copy()
    elif attack == "uniform_measure":
        phases[0]["integral"] *= .75
    elif attack == "lift":
        phases[0]["extended"][:] = phases[0]["extended"][0]
    elif attack == "state_nan":
        phases[0]["population"][0, 0, 0] = np.nan
    elif attack == "shape":
        phases[0]["extended"] = phases[0]["extended"].transpose(0, 2, 1)
    else:
        phases[1]["population"][0, 0, 0] = np.nextafter(phases[1]["population"][0, 0, 0], np.inf)
    # All integrity digests can be recomputed by an owner
    # equations/bit replay
    # are a separate gate, demonstrated here without writing a native NPZ.
    resealed = {name: reader.typed_array(value) for name, value in phases[0].items()}
    assert all(len(row["content_sha256"]) == 64 for row in resealed.values())
    with pytest.raises(ValueError, match="(?:equation|component|finitude|bit-identical)"):
        reader.original(initial, phases, nx, nv, width)


def synthetic_boxes(nx=4, nv=3, ranks=3, replicated=False):
    rows = [dict.fromkeys(reader.NAMES) for _ in range(ranks)]
    for rank in range(ranks):
        for name in reader.NAMES:
            shape = (nv, nx) if name in ("population", "extended") else (nx, 1)
            rows[rank][name] = [[[0, 0], list(shape)]] if replicated or rank == 0 else []
    return rows


@pytest.mark.parametrize("ranks,replicated", [(1, False), (2, False), (3, False), (3, True)])
def test_actual_ownership_native_axes_allow_empty_rank_or_true_replica(ranks, replicated):
    result = reader.ownership(synthetic_boxes(ranks=ranks, replicated=replicated), ranks, 4, 3)
    assert set(result.values()) == {"replicated" if replicated and ranks > 1 else "distributed"}


@pytest.mark.parametrize("attack", ("duplicate", "missing", "swapped_axes", "foreign", "bool_index", "rank_missing"))
def test_ownership_forgery_refused(attack):
    rows = synthetic_boxes()
    if attack == "duplicate":
        rows[1]["weighted"] = copy.deepcopy(rows[0]["weighted"])
    elif attack == "missing":
        rows[0]["population"] = []
    elif attack == "swapped_axes":
        rows[0]["population"][0][1] = [4, 3]
    elif attack == "foreign":
        rows[0]["foreign"] = []
    elif attack == "bool_index":
        rows[0]["weighted"][0][0][0] = False
    else:
        rows.pop()
    with pytest.raises(ValueError, match="(?:owner|rank|box|support)"):
        reader.ownership(rows, 3, 4, 3)


def protocol_directory(tmp_path):
    directory = tmp_path / "phases"
    directory.mkdir()
    (directory / "provenance.json").write_text('{}')
    for phase in reader.PHASES:
        state = directory / (phase + "-state.npz")
        cp = directory / phase
        state.write_bytes(b"OPAQUE SYNTHETIC INVENTORY, NOT NPZ OR NATIVE EVIDENCE")
        cp.write_bytes(b"OPAQUE SYNTHETIC CHECKPOINT, NOT NATIVE EVIDENCE")
        row = dict(checkpoint=phase, saved_state_sha256=reader.leaf(state)["sha256"], checkpoint_sha256=reader.leaf(cp)["sha256"])
        (directory / (phase + "-receipt.json")).write_text(json.dumps(row))
    return directory


@pytest.mark.parametrize("attack", ("extra", "missing", "escape", "cp_reuse", "hash", "duplicate_json", "symlink"))
def test_closed13_protocol_refuses_manifest_resealed_forgery(tmp_path, attack):
    directory = protocol_directory(tmp_path)
    assert len(reader.closed_phases(directory)) == 5
    receipt = directory / "accepted-receipt.json"
    if attack == "extra":
        (directory / "foreign-phase").write_text('not an accepted phase')
    elif attack == "missing":
        (directory / "initial-state.npz").unlink()
    elif attack == "escape":
        data = json.loads(receipt.read_text())
        data["checkpoint"] = "../initial"
        receipt.write_text(json.dumps(data))
    elif attack == "cp_reuse":
        data = json.loads(receipt.read_text())
        data["checkpoint"] = "initial"
        receipt.write_text(json.dumps(data))
    elif attack == "hash":
        (directory / "accepted-state.npz").write_bytes(b"MODIFIED WITHOUT NEW RECEIPT")
    elif attack == "duplicate_json":
        receipt.write_text('{"checkpoint":"accepted","checkpoint":"initial"}')
    else:
        target = directory / "replayed"
        target.unlink()
        target.symlink_to(directory / "initial")
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.closed_phases(directory)


def junit_bytes(ranks=2, rank=0):
    cases = {reader.case_id(*case): dict(directory="/actual-realm/" + reader.case_id(*case), native_sha256="a" * 64) for case in reader.CASES}
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="6", failures="0", errors="0", skipped="0")
    for key, case in cases.items():
        test = ET.SubElement(suite, "testcase", name="test_native_product_reduce_lift_restart[" + key + "]",
                             classname="tests.python.integration.runtime.test_m19_product_support_runtime")
        props = ET.SubElement(test, "properties")
        for name, value in dict(native_dimension=2, mpi_rank=rank, mpi_size=ranks,
                                native_sha256=case["native_sha256"], saved_receipts=case["directory"]).items():
            ET.SubElement(props, "property", name=name, value=str(value))
    return root, cases


@pytest.mark.parametrize("attack", ("failure", "skip", "error", "rank", "realm", "counter", "duplicate", "native", "foreign", "dimension"))
def test_exact_six_junit_rank_and_realm_refuse_mutation(attack):
    root, cases = junit_bytes()
    reader.junit(ET.tostring(root), 0, 2, cases)
    suite = root.find("testsuite")
    test = suite.find("testcase")
    if attack in ("failure", "skip", "error"):
        ET.SubElement(test, "skipped" if attack == "skip" else attack)
    elif attack == "counter":
        suite.set("tests", "5")
    elif attack == "duplicate":
        tests = list(suite)
        tests[1].set("name", tests[0].get("name"))
    elif attack == "foreign":
        test.set("name", "test_other")
    else:
        name = {"rank": "mpi_rank", "realm": "saved_receipts", "native": "native_sha256", "dimension": "native_dimension"}[attack]
        test.find("properties").find("property[@name='" + name + "']").set("value", "FOREIGN")
    with pytest.raises(ValueError, match="JUnit"):
        reader.junit(ET.tostring(root), 0, 2, cases)


@pytest.mark.parametrize("bad", ({"bytes_hex": "AB"}, {"bytes_hex": "a"}, {"bytes_hex": "aa", "extra": 1}, {"bytes_hex": 12}))
def test_bytes_encoding_is_exact_and_fail_closed(bad):
    with pytest.raises(ValueError, match="bytes"):
        reader.metadata_decode(bad)


def test_bytes_cbor_and_aggregate_are_independent_and_sensitive():
    assert reader.identity_cbor(b"\x00\xff") == b"\x42\x00\xff"
    assert reader.metadata_decode({"x": [{"bytes_hex": "00ff"}]}) == {"x": [b"\x00\xff"]}
    value = dict(compiled_plan=dict(target="system", resolved_dimension=2, nested=b"authentic-payload-protocol",
                                          resolved_plan_identity=dict(domain="resolved-plan", schema_version=1,
                                                                      algorithm="sha256", digest=b"r" * 32)),
                 platform=dict(example="synthetic protocol"), compiled_components=dict(example="synthetic components"))
    plan_digest = (b"r" * 32).hex()
    payload = dict(schema_version=2, plan_identity=dict(domain="resolved-plan", schema_version=1, algorithm="sha256", digest=bytes.fromhex(plan_digest)),
                   target="system", platform_manifest=value["platform"], components=value["compiled_components"])
    value["artifact_identity"] = "pops.artifact.v1:sha256:" + reader.identity_hash("artifact", payload)
    assert reader.artifact_payload(value) == plan_digest
    value["compiled_components"]["example"] = "changed and independently redigested source file"
    with pytest.raises(ValueError, match="aggregate"):
        reader.artifact_payload(value)


def pack_synthetic(payload):
    out = io.BytesIO()
    np.savez_compressed(out, **payload)
    return out.getvalue()


def synthetic_envelope():
    # This in-memory archive is only an envelope protocol, never a phase result.
    arrays = dict(t=np.array(0.), macro_step=np.array(0), abi_key=np.array("SYNTHETIC-NOT-NATIVE"),
                  labelled_protocol=np.array("NOT A NATIVE RECEPTION"))
    def identity(domain):
        return dict(domain=domain, schema_version=1, algorithm="sha256", hexdigest="0" * 64)
    manifest = dict(schema_version=2, runtime_kind="synthetic_protocol_only", semantic_identity=identity("semantic"),
                    artifact_identity=identity("artifact"), bind_identity=identity("bind"), run_identity=None,
                    clock=dict(time=0..hex(), macro_step=0), origin=dict(schema_version=1, kind="bound_initial"),
                    arrays={name: reader.typed_array(value) for name, value in arrays.items()})
    payload = dict(protocol="pops.identity", domain="restart", schema_version=1, payload=manifest)
    restart = identity("restart")
    restart["hexdigest"] = reader.digest(reader.protocol.cbor(payload))
    manifest["restart_identity"] = restart
    arrays["pops_checkpoint_manifest"] = np.array(json.dumps(manifest))
    arrays["pops_restart_identity"] = np.array(reader.protocol.identity_token(restart, "restart"))
    return arrays


@pytest.mark.parametrize("attack", ("array", "clock", "restart", "extra", "origin", "member"))
def test_synthetic_envelope_digest_and_typed_members_refuse_forgery(attack):
    arrays = synthetic_envelope()
    reader.envelope(pack_synthetic(arrays), "initial", "SYNTHETIC-NOT-NATIVE")
    if attack == "array":
        arrays["labelled_protocol"] = np.array("changed unsealed bytes")
    elif attack == "member":
        arrays.pop("labelled_protocol")
    elif attack == "extra":
        arrays["foreign"] = np.array(1)
    else:
        data = json.loads(arrays["pops_checkpoint_manifest"].item())
        if attack == "clock":
            data["clock"]["time"] = .01.hex()
        elif attack == "restart":
            data["restart_identity"]["hexdigest"] = "a" * 64
        else:
            data["origin"]["kind"] = "invented_run"
        arrays["pops_checkpoint_manifest"] = np.array(json.dumps(data))
    with pytest.raises(ValueError, match="(?:clock|digest|inventory|provenance)"):
        reader.envelope(pack_synthetic(arrays), "initial", "SYNTHETIC-NOT-NATIVE")


def test_no_external_root_approval_no_receive(tmp_path):
    pins, approval = tmp_path / "pending.json", tmp_path / "approval.json"
    pins.write_text('{}')
    approval.write_text(json.dumps(dict(schema="sol61.m19-root-approval@1", approved_by="helper", pins_sha256=reader.digest(pins.read_bytes()))))
    with pytest.raises(ValueError, match="ROOT has not approved"):
        reader.receive(pins, reader.digest(pins.read_bytes()), approval, reader.digest(approval.read_bytes()))
    with pytest.raises(ValueError, match="external seal"):
        reader.receive(pins, "0" * 64, approval, reader.digest(approval.read_bytes()))


def synthetic_origins(tmp_path):
    roots = {name: tmp_path / name for name in ("installation", "runtime", "source")}
    for path in roots.values():
        path.mkdir()
    def pin(root, name):
        path = roots[root] / name
        path.write_bytes(b"SYNTHETIC ORIGIN PROTOCOL, NEVER NATIVE EVIDENCE")
        return reader.leaf(path)
    keys = {reader.case_id(*case): {} for case in reader.CASES}
    value = dict(schema="sol61.m19-execution-owner@1", source_commit="a" * 40, native_build_source_commit=None,
                 roots={name: str(path) for name, path in roots.items()},
                 python_package=pin("installation", "entry.py"), native=pin("installation", "native.so"),
                 sdk=pin("installation", "pops_headers.manifest"), common_lowering=pin("installation", "physical_support_transfer.hpp"),
                 sources=[pin("source", "fixture.py"), pin("source", "generic.py")],
                 system_packages={}, provider_sources={}, generated_cpp={}, cpp_dso_links={})
    for key in keys:
        value["system_packages"][key] = [pin("runtime", key + str(i) + ".so") for i in range(4)]
        providers = [pin("runtime", key + str(i) + ".cpp") for i in range(3)]
        for leaf in providers:
            Path(leaf["path"]).write_text('#include <pops/runtime/dynamic/physical_support_transfer.hpp>\n'
                                         'return pops::component::apply_physical_support_transfer(descriptor, request, status);')
        value["provider_sources"][key] = [reader.leaf(row["path"]) for row in providers]
        value["generated_cpp"][key] = None
        value["cpp_dso_links"][key] = None
    return value, keys


@pytest.mark.parametrize("attack", ("missing_origin", "foreign_case", "alias", "sourcehash", "duplicate_cpp", "source_route", "invented_link", "build_type"))
def test_exact_origin_records_fail_closed_without_inferring_cpp_or_native_build(tmp_path, attack):
    value, cases = synthetic_origins(tmp_path)
    reader.origin_records(value, cases)
    key = next(iter(cases))
    if attack == "missing_origin":
        value.pop("native")
    elif attack == "foreign_case":
        value["generated_cpp"]["foreign"] = None
    elif attack == "alias":
        original = Path(value["native"]["path"])
        alias = original.with_name("alias.so")
        alias.symlink_to(original)
        value["native"]["path"] = str(alias)
    elif attack == "sourcehash":
        Path(value["sources"][0]["path"]).write_bytes(b"changed actual source")
    elif attack == "duplicate_cpp":
        row = value["provider_sources"][key][0]
        value["generated_cpp"][key] = [row, row]
    elif attack == "source_route":
        row = value["provider_sources"][key][0]
        Path(row["path"]).write_text('for(int j=0;j<3;++j) output[j]=input[j]; // private scalar runtime')
        value["provider_sources"][key][0] = reader.leaf(row["path"])
    elif attack == "invented_link":
        value["cpp_dso_links"][key] = [{"cpp": value["provider_sources"][key][0], "dso": value["system_packages"][key][0],
                                       "build_receipt": value["sdk"]}]
    else:
        value["native_build_source_commit"] = True
    with pytest.raises(ValueError):
        reader.origin_records(value, cases)


def test_physical_axes_units_and_moments_bound_to_actual_payload():
    nv = 5
    phase = dict(kind="physical_support", coordinates=[["velocity", "velocity-interval"], ["position", "periodic-position"]])
    physical = dict(kind="physical_support", coordinates=[["position", "periodic-position"]])
    unit = dict(kind="physical_dimension", powers=[])
    def reducing(weights):
        return dict(kind="support-reduction@2", source_support=phase, target_support=physical,
                    reductions=[dict(axis=0, domain=dict(lower=[-2, 1], upper=[2, 1], cells=nv),
                                     rule="explicit-cell-average-weighted-sum@1", weights=weights, dimension=unit)],
                    storage=dict(native_dimension=2, source_axes=[0, 1], target_axes=[0], source_to_target=[-1, 0],
                                 hidden_cells=1, hidden_measure=1, extension="constant", inverse_closure=False))
    maps = [reducing([[4, 5]] * nv), reducing([[(-1) ** j * (j + 1), 1] for j in range(nv)]),
            dict(kind="support-extension@2", source_support=physical, target_support=phase, reductions=[],
                 storage=dict(native_dimension=2, source_axes=[0], target_axes=[0, 1], source_to_target=[1, -1],
                              hidden_cells=1, hidden_measure=1, extension="constant", inverse_closure=False))]
    reader.physical_maps(dict(nested=maps), nv)
    for target, field, bad in ((0, "source_axes", [1, 0]), (2, "inverse_closure", True), (0, "hidden_measure", 2)):
        changed = copy.deepcopy(maps)
        changed[target]["storage"][field] = bad
        with pytest.raises(ValueError, match="physical map"):
            reader.physical_maps(dict(nested=changed), nv)
    changed = copy.deepcopy(maps)
    changed[1]["reductions"][0]["weights"][1] = [2, 1]
    with pytest.raises(ValueError, match="physical map"):
        reader.physical_maps(dict(nested=changed), nv)
