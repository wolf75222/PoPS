"""Root-authorized installed-native reception of the six accepted static Inputs.

Preparation does not execute this file. The representative must pass before remaining5.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
import traceback

SOURCE = "4017d29d9adc066e2ab64d2eed3b99f1caa8f63c"
NATIVE = "939d331c0f28150429266516e0e6b57fe84986f48d336290ec613a7c4dd85941"
SDK = "acf0eeca77739e7d4dd8a7da161af75667a3f898959e7464961ea243db9107fa"
CORE = "743bc354455285b4828e6d80039949317c31f63ac6fc3c5432ae2897487594a8"
KINDS = ("variable", "diagonal_linear", "diagonal_smooth")
CASES = tuple((kind, method) for kind in KINDS for method in ("euler", "ssprk2"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


BUILD_ADMISSION_SHA = "05f27f6c68525b852d517cfff476c715ca3aa78fbc7013311896510ca64beb68"
BUILD_RESULT_SHA = "14b56ec61fc729a4092379652cbae21373279225e5510adbec2bac63095d5420"


def validate_build_admission(document, status):
    """The actual schema2 admission has no flat status: command exit is separate."""
    assert type(document["schema"]) is int and document["schema"] == 2
    assert document["Source"] == status["Source"] == SOURCE
    assert document["SDK"] == status["SDK"] == SDK
    assert document["core_fingerprint"] == CORE
    assert type(document["core_files"]) is int and document["core_files"] > 0
    assert type(status["exit"]) is int and status["exit"] == 0
    assert document["total_object_graph_count"] == len(document["objects"]) > 0
    row = document["installed_native"]
    identity = document["identity"]
    packaging = document["packaging"]
    assert row["sha256"] == identity["native_sha256"] == NATIVE
    assert identity["source_commit"] == SOURCE and "headers=" + SDK in identity["abi_key"]
    native_path = Path(row["path"]).resolve()
    package_file = Path(packaging["package_file"]).resolve()
    sites = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
    assert any(native_path.is_relative_to(site / "pops/_native/dim2") for site in sites)
    assert any(package_file == site / "pops/__init__.py" for site in sites)
    assert native_path.is_relative_to(Path(sys.prefix).resolve())
    assert Path(identity["native_file"]).resolve() == native_path
    assert Path(identity["package_file"]).resolve() == package_file
    assert Path(packaging["distribution_root"]).resolve() in sites
    variants = [r for r in packaging["native_variants"] if r["dimension"] == 2]
    assert len(variants) == 1 and variants[0]["sha256"] == NATIVE
    assert Path(variants[0]["extension"]).resolve() == native_path


def checked_build_admission(build_receipt):
    assert sha(build_receipt) == BUILD_ADMISSION_SHA, "Exact Root physical admission required"
    result_path = build_receipt.with_name("build-result.json")
    assert sha(result_path) == BUILD_RESULT_SHA, "Exact Root actual build result required"
    document = json.loads(build_receipt.read_text())
    status = json.loads(result_path.read_text())
    validate_build_admission(document, status)
    row = document["installed_native"]
    assert Path(row["path"]).stat().st_size == row["bytes"] and sha(row["path"]) == NATIVE
    return document, result_path, status


def authenticate(checkout, build_receipt):
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    assert actual == SOURCE
    subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD"], check=True)
    assert build_receipt.is_file(), "Root physical build/install receipt required"
    build_admission, build_result_path, build_status = checked_build_admission(build_receipt)
    core_paths = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "--",
                                         "python/pops/", "python/bindings/", "include/", "src/"], text=True).splitlines()
    core_rows = {name: {"bytes": (checkout / name).stat().st_size, "sha256": sha(checkout / name)}
                 for name in core_paths}
    core_hash = hashlib.sha256(json.dumps(core_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert core_hash == CORE
    assert build_admission['core_files'] == len(core_rows)
    import pops
    from pops._native_selector import select_native_dimension
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature
    from scripts.check_packaging_manifest import read_manifest, PYTHON_SOURCE_SUFFIXES
    from scripts.verify_installed_native import verify_installed_native

    package = Path(pops.__file__).resolve().parent
    assert any(package.is_relative_to(Path(sysconfig.get_path(key)).resolve()) for key in ("purelib", "platlib"))
    assert package.is_relative_to(Path(sys.prefix).resolve()) and not package.is_relative_to(checkout)
    native = select_native_dimension(2)
    assert sha(native.__file__) == NATIVE
    assert Path(native.__file__).resolve() == Path(build_admission['installed_native']['path']).resolve()
    assert package / '__init__.py' == Path(build_admission['packaging']['package_file']).resolve()
    assert module_header_signature() == SDK == pops_header_signature(package / "include")
    origin = verify_installed_native(expect_dimension=2, expect_mpi=bool(native.__has_mpi__),
                                    expect_parallel_hdf5=bool(native.__has_parallel_hdf5__))
    files = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "--", "python/pops"], text=True).splitlines()
    files = [name for name in files if Path(name).suffix in PYTHON_SOURCE_SUFFIXES]
    files += ["include/" + str(path) for path in read_manifest(checkout).installed_headers]
    files.append("include/pops_headers.manifest")
    for name in files:
        installed = package / (name.removeprefix("python/pops/") if name.startswith("python/pops/") else name)
        assert sha(installed) == sha(checkout / name), name
    world = native.mpi_world() if native.__has_mpi__ else None
    assert world is None or int(world.size) == 1, "This representative is explicitly world1; MPI reception is separate"
    return {"Source": actual, "Native": sha(native.__file__), "SDK": module_header_signature(),
            "core_fingerprint": core_hash, "tracked_core_rows": len(core_rows),
            "package": str(package), "native_origin": str(origin), "native_path": native.__file__,
            "authenticated_payload_files": len(files), "build_receipt": str(build_receipt),
            "build_receipt_sha256": sha(build_receipt), "build_result_sha256": sha(build_result_path),
            "actual_build_exit": build_status['exit'], "actual_runtime_environment": native.runtime_environment_report()}


def run_case(kind, method, out):
    import numpy as np
    import pops
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_accepted_ssp import prove_accepted_update_ssp
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
    from pops.model.provider_pack import ProviderPack
    from pops.time._program.detach import detach_compiled_program
    from tests.python.integration.runtime.test_public_diffusion_matrix import _build
    from tests.python.support.native_execution_context import artifact_execution_context
    from references import authored_update, inputs

    n, dt = 16, 1e-4
    case_dir = out / (kind + "-" + method)
    case_dir.mkdir(mode=0o700)
    stages = 1 if method == "euler" else 2
    initial, fields = inputs(kind, n)
    expected = authored_update(kind, method, initial, fields, dt)
    np.savez(case_dir / "independent-input-and-reference.npz", initial=initial, expected=expected, **fields)
    case, layout, declared = _build(kind, n, dt, method=method)
    assert set(declared) == set(fields)
    plan = pops.resolve(pops.validate(case), layout=layout)
    program = detach_compiled_program(plan.time)
    authority = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    before = program._serialize()
    certificate, reason = prove_accepted_update_ssp(program, model_authority=authority)
    weights = (Fraction(1),) if method == "euler" else (Fraction(1, 2), Fraction(1, 2))
    assert certificate is not None and not reason and certificate.coefficient == 1
    assert certificate.b == weights
    assert [row for _, row in accepted_diffusive_quadrature(program, model_authority=authority)] == [{1: w} for w in weights]
    assert program._serialize() == before
    assert all(value.block._instance_registry is None for value in program._values if value.block is not None)
    proof = {"kind": kind, "method": method, "n": n, "dt": dt, "coefficient": str(certificate.coefficient),
             "accepted_weights": [str(value) for value in certificate.b], "authored_serialization_unchanged": True,
             "program_serialization_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()}
    write_json(case_dir / "precompile-ssp-proof.json", proof)
    artifact = pops.compile(plan)
    operation_data = artifact.plan.blocks[0].resolved_operations.to_data()
    pack = ProviderPack.from_data(operation_data["provider_evidence"]["auxiliary"])
    keys = {key.component: key for key in pack if pack.declared_entry(key).producer == "runtime_input"}
    expected_slots = {"diffusivity": 0, "source_factor": 1} if kind == "variable" else {"a_x": 0, "a_y": 1, "forcing": 2}
    assert set(keys) == set(fields) == set(expected_slots)
    assert {name: pack.declared_entry(key).slot for name, key in keys.items()} == expected_slots
    publications = operation_data["provider_evidence"].get("program_field_publications", ())
    assert not publications, "Static Input construction must retain zero field publications"
    context = artifact_execution_context(artifact)
    runtime = pops.bind(artifact, initial_state={"heat": np.ascontiguousarray(initial[None])},
                        aux={keys[name]: np.ascontiguousarray(value) for name, value in fields.items()},
                        resources={"execution_context": context})
    report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert report.accepted_steps == 1 and runtime.macro_step() == 1
    assert abs(runtime.time() - dt) < 2e-13
    actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
    np.savez(case_dir / "actual-state.npz", actual=actual)
    # Existing diffusion exchange cohort tolerances, unchanged; independent discrete reference.
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    exchanges = runtime._executor._program_exchange_records()
    assert len(exchanges) == stages * 4 * n * n
    assert len({row["evaluation_context"] for row in exchanges}) == stages
    assert all(abs(row["temporal_weight"] - dt / stages) < 2e-16 for row in exchanges)
    write_json(case_dir / "actual-exchanges.json", exchanges)
    jit_paths = {"program": artifact.so_path, **artifact.layout_program_paths,
                 **{"model:" + block.name: block.model.so_path for block in artifact.blocks}}
    jit_receipts = {name: {"path": str(path), "sha256": sha(path)} for name, path in jit_paths.items()}
    result = {**proof, "result": "PASS", "accepted_steps": report.accepted_steps,
              "time": runtime.time(), "Linf_discrete_defect": float(np.max(np.abs(actual - expected))),
              "rtol": 2e-12, "atol": 2e-13, "input_slots": expected_slots,
              "field_publication_count": 0, "actual_exchange_count": len(exchanges),
              "artifact_identity": artifact.artifact_identity.token, "ABI": artifact.abi_key,
              "compiled_programs": jit_receipts, "platform_manifest": artifact.platform_manifest.to_data(),
              "execution_backend": context.backend.to_data(),
              "scope": "World1 installed CPU numerical reception of one static Input authored composition; not convergence, AMR, MPI2, or GPU"}
    write_json(case_dir / "case-receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--stage", choices=("representative", "remaining5"), required=True)
    parser.add_argument("--representative-receipt", type=Path)
    parser.add_argument("--authorize-native-execution", action="store_true")
    args = parser.parse_args()
    assert args.authorize_native_execution, "Root must authorize the actual Native/JIT/run phase"
    checkout, out = args.checkout.resolve(), args.out.resolve()
    assert not out.exists() and not out.is_relative_to(checkout)
    out.mkdir(mode=0o700)
    for name in ("PYTHONPATH", "PYTHONOPTIMIZE", "POPS_INCLUDE", "POPS_REQUIRE_MPI_TESTS"):
        os.environ.pop(name, None)
    os.environ.update(POPS_CACHE_DIR=str(out / "jit"), POPS_NATIVE_CACHE_DIR=str(out / "jit"),
                      TMPDIR=str(out / "tmp"), PYTHONDONTWRITEBYTECODE="1", POPS_NATIVE_DIM="2")
    (out / "tmp").mkdir(mode=0o700)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(checkout), str(checkout / "scripts"), str(Path(__file__).resolve().parent)]
    assert str(checkout / "python") not in sys.path
    script_hash = sha(__file__)
    reference_hash = sha(Path(__file__).with_name("references.py"))
    receipt = {"Source": SOURCE, "Native": NATIVE, "SDK": SDK, "stage": args.stage,
               "script_sha256": script_hash, "reference_sha256": reference_hash, "cases": [],
               "scope": "Author native-test preparation/run; independent reader reception remains separate"}
    start = time.monotonic()
    try:
        receipt["admission"] = authenticate(checkout, args.build_receipt.resolve())
        if args.stage == "remaining5":
            assert args.representative_receipt is not None
            prior = json.loads(args.representative_receipt.read_text())
            assert all(prior[key] == receipt[key] for key in ("Source", "Native", "SDK", "script_sha256", "reference_sha256"))
            assert prior["stage"] == "representative" and prior["exit"] == 0
            assert [(row["kind"], row["method"], row["result"]) for row in prior["cases"]] == [("variable", "euler", "PASS")]
            receipt["representative_receipt"] = {"path": str(args.representative_receipt), "sha256": sha(args.representative_receipt)}
        cases = CASES[:1] if args.stage == "representative" else CASES[1:]
        for kind, method in cases:
            receipt["current_case"] = {"kind": kind, "method": method}
            receipt["cases"].append(run_case(kind, method, out))
        receipt["exit"] = 0
    except BaseException:
        receipt["exit"] = 1
        receipt["exception"] = traceback.format_exc()
        raise
    finally:
        receipt["seconds"] = time.monotonic() - start
        write_json(out / "run-receipt.json", receipt)
    print(json.dumps({"stage": args.stage, "passed": len(receipt["cases"]), "exit": receipt["exit"]}))


if __name__ == "__main__":
    main()
