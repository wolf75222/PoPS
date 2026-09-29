"""Read-only consistency check for the reviewed API 0.4 registry and receipts.

This verifies inventory/provenance, not mathematical claims or native execution.
Only the standard library is required; no PoPS import or native build occurs.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pytest_counts(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    return {"tests": len(cases), **{
        name: sum(case.find(tag) is not None for case in cases)
        for name, tag in (("failures", "failure"), ("errors", "error"),
                          ("skipped", "skipped"))
    }}


def main():
    from check_local_affine_evidence import verify
    from check_finite_amr_converged import verify_bundle
    from check_m23_m05_converged import verify_bundle as verify_component_bundle

    verify(HERE / "evidence/local-affine-converged")
    finite = verify_bundle(HERE / "evidence/finite-amr-converged")
    with (HERE / "contracts.csv").open(newline="") as source:
        contracts = list(csv.DictReader(source))
    corpus = read_json(HERE / "corpus.json")
    component = verify_component_bundle(HERE / "evidence/m23-m05-converged")
    component_registry = corpus["supplemental_receptions"]["m23_m05_converged"]
    require(component_registry["manifest_sha256"] == hashlib.sha256(
                (HERE / component_registry["manifest"]).read_bytes()).hexdigest()
            and component_registry["saved_states"] == component["npz"] == 26
            and component_registry["raw_ledgers"] == component["raw_ledgers"] == 10
            and component_registry["retained_failed_testcases"]
                == component["retained_failed_testcases"] == 4,
            "M23/M05 archived receipt binding changed")
    finite_registry = corpus["supplemental_receptions"]["finite_amr_converged"]
    require(finite_registry["manifest_sha256"] == hashlib.sha256(
                (HERE / finite_registry["manifest"]).read_bytes()).hexdigest()
            and finite_registry["saved_m09_states"] == finite["saved_m09_states"] == 32,
            "finite/AMR evidence binding changed")
    require(finite_registry["serial_m09_passed"] == 10
            and finite_registry["mpi_m09_passed_per_rank"] == 10
            and finite_registry["amr_passed_serial_and_per_mpi_rank"] == 3
            and finite_registry["w12_ctest_passed"] == 3,
            "finite/AMR reception counts changed")
    for rows, key, prefix, count in (
        (contracts, "contract_id", "C", 40),
        (corpus["models"], "id", "M", 28),
        (corpus["witnesses"], "id", "W", 12),
    ):
        ids = [row[key] for row in rows]
        require(len(ids) == count and set(ids) == {
            f"{prefix}{index:02}" for index in range(1, count + 1)
        }, f"incomplete or duplicate {prefix} inventory")
        for row in rows:
            symbols = row.get("upstream_symbols", row.get("production_candidate_symbols", ""))
            for symbol in symbols.split(";"):
                require((ROOT / symbol.split(":", 1)[0]).exists(), f"missing source: {symbol}")
            evidence = row.get("production_evidence", [])
            if isinstance(evidence, str):
                evidence = list(filter(None, evidence.split(";")))
            for path in evidence:
                require((HERE / path).exists(), f"missing evidence: {path}")
    require(all(row["current_head"] == corpus["current_head"] for row in contracts),
            "mixed mapping revisions")
    manifest_path = HERE / corpus["evidence_manifest"]
    manifest = read_json(manifest_path)
    require(manifest["mapping_source_commit"] == corpus["mapping_baseline_head"],
            "historical mapping/evidence mismatch")
    evidence_root = manifest_path.parent
    for receipt in manifest["receipts"]:
        files = {Path(item["path"]).name: item for item in receipt["files"]}
        for item in receipt["files"]:
            path = evidence_root / item["path"]
            require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                    f"receipt bytes changed: {path}")
        result = read_json(evidence_root / files["result.json"]["path"])
        identity_path = evidence_root / files["identity.json"]["path"]
        identity = read_json(identity_path)
        require(result["status"] == receipt["status"], "receipt status mismatch")
        for key, value in receipt["identity"].items():
            require(identity[key] == value, f"identity mismatch: {receipt['id']} {key}")
        if "identity_sha256" in result:
            require(hashlib.sha256(identity_path.read_bytes()).hexdigest() == result["identity_sha256"],
                    f"result authenticates different identity: {receipt['id']}")
        if "pytest.xml" in files:
            tree = ET.parse(evidence_root / files["pytest.xml"]["path"])
            all_cases = list(tree.getroot().iter("testcase"))
            # Pytest interrupted by fail-fast may leave an empty, unnamed node.
            # Retain and declare it, but never count it as an executed pass.
            cases = [case for case in all_cases if case.get("name") is not None]
            require(len(all_cases) - len(cases) == receipt.get("incomplete_xml_cases", 0),
                    f"undeclared incomplete XML case: {receipt['id']}")
            counts = {"tests": len(cases), **{
                name: sum(case.find(tag) is not None for case in cases)
                for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
            }}
            require(counts == receipt["counts"] == result["counts"],
                    f"XML/result count mismatch: {receipt['id']}")
            if receipt["status"] == "passed":
                require(not any(counts[key] for key in ("failures", "errors", "skipped")),
                        f"nonpass promoted to success: {receipt['id']}")
        if "scientific-receipt.json" in files:
            science_path = evidence_root / files["scientific-receipt.json"]["path"]
            if "scientific_receipt_sha256" in result:
                require(hashlib.sha256(science_path.read_bytes()).hexdigest()
                        == result["scientific_receipt_sha256"], "scientific receipt changed")
            science = read_json(science_path)
            require(science["native_sha256"] == identity["native_sha256"], "scientific native mismatch")
    for item in manifest["ctest"]["files"] + manifest.get("source_reviews", []):
        require(hashlib.sha256((evidence_root / item["path"]).read_bytes()).hexdigest()
                == item["sha256"], f"review/CTest bytes changed: {item['path']}")
    ctest = manifest["ctest"]
    cases = ET.parse(evidence_root / ctest["files"][0]["path"]).getroot().findall("testcase")
    require(len(cases) == ctest["total"] and dict(Counter(case.get("status") for case in cases))
            == ctest["counts"], "CTest inventory mismatch")
    require(corpus["production_evidence_schema"] == 5, "unsupported reception registry schema")
    expected_scopes = {
        "selected_amr": ("pops.api040.selected-native-evidence/v1", "production_commit"),
        "scientific": ("pops.api040.scientific-reception-evidence/v1", "production_source"),
    }
    for scope, (schema, source_key) in expected_scopes.items():
        scoped_path = HERE / corpus["scoped_receptions"][scope]["manifest"]
        scoped = read_json(scoped_path)
        require(scoped["schema"] == schema and scoped[source_key] == "eac92bb",
                f"wrong {scope} reception identity")
        files = {item["path"]: item for item in scoped["files"]}
        for relative, item in files.items():
            path = scoped_path.parent / relative
            require(path.is_file() and path.stat().st_size == item["bytes"]
                    and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                    f"{scope} receipt bytes changed: {relative}")
        statuses = {}
        for receipt in scoped["receipts"]:
            result_path = f"{receipt['id']}/result.json"
            require(result_path in files, f"{scope} missing receipt result: {result_path}")
            result = read_json(scoped_path.parent / result_path)
            status = receipt.get("status", receipt.get("runner_status"))
            require(result["status"] == status, f"{scope} receipt status changed: {receipt['id']}")
            require(result.get("native_sha256", scoped.get("native_sha256"))
                    == corpus["scoped_receptions"][scope]["native_sha256"],
                    f"{scope} native identity changed: {receipt['id']}")
            if scope == "selected_amr" and receipt["counts"] is not None:
                require(result["counts"] == receipt["counts"],
                        f"AMR counts changed: {receipt['id']}")
                require(pytest_counts(scoped_path.parent / f"{receipt['id']}/pytest.xml")
                        == receipt["counts"], f"AMR XML counts changed: {receipt['id']}")
            elif scope == "selected_amr":
                require([rank["counts"] for rank in result["rank_results"]]
                        == receipt["per_rank_counts"],
                        f"AMR MPI rank counts changed: {receipt['id']}")
                for rank, counts in enumerate(receipt["per_rank_counts"]):
                    require(pytest_counts(scoped_path.parent
                                          / f"{receipt['id']}/rank{rank}.xml") == counts,
                            f"AMR MPI XML counts changed: {receipt['id']} rank{rank}")
            if scope == "scientific":
                science_path = f"{receipt['id']}/{receipt['scientific_receipt']}"
                if status == "passed":
                    require(science_path in files, f"missing scientific result: {science_path}")
                if science_path in files:
                    science = read_json(scoped_path.parent / science_path)
                    require(science["status"] == status,
                            f"scientific status changed: {receipt['id']}")
                    require(result["scientific_receipt_sha256"] == files[science_path]["sha256"],
                            f"scientific receipt identity changed: {receipt['id']}")
            statuses[receipt["id"]] = status
        require(statuses == corpus["scoped_receptions"][scope]["statuses"],
                f"{scope} receipt selection changed")
    require(corpus["scoped_receptions"]["selected_amr"]["native_sha256"]
            == corpus["scoped_receptions"]["scientific"]["native_sha256"],
            "selected receptions use different native artifacts")
    ctest_path = (HERE / corpus["scoped_receptions"]["selected_amr"]["manifest"]).parent
    ctest_path /= "native-user-amr-c38-owner.xml"
    ctest_cases = list(ET.parse(ctest_path).getroot().iter("testcase"))
    # The selected-AMR manifest is inspected directly because `scoped` now
    # refers to the scientific manifest after the loop.
    amr_manifest = read_json(HERE / corpus["scoped_receptions"]["selected_amr"]["manifest"])
    scoped_ctest = amr_manifest["ctest"]
    require(len(ctest_cases) == scoped_ctest["rows"]
            and dict(Counter(case.get("status") for case in ctest_cases))
            == scoped_ctest["status_counts"], "selected AMR CTest inventory changed")
    supplemental = corpus["supplemental_receptions"]["after_388c323"]
    bundle_path = HERE / supplemental["manifest"]
    bundle_root = bundle_path.parent
    bundle = read_json(bundle_path)
    require(bundle["schema"] == "pops.api040.scoped-reception-evidence/v1"
            and bundle["mapping_source_commit"] == "cfef849",
            "wrong supplemental receipt identity")
    require(bundle["native_sha256"] == supplemental["native_sha256"],
            "supplemental native identities changed")
    bundle_files = {item["path"]: item for item in bundle["files"]}
    require(len(bundle_files) == len(bundle["files"]), "duplicate bundled path")
    for relative, item in bundle_files.items():
        path = bundle_root / relative
        require(path.is_file() and path.stat().st_size == item["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                f"supplemental receipt bytes changed: {relative}")
    checksum_lines = (bundle_root / "SHA256SUMS").read_text().splitlines()
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    expected_paths = set(bundle_files) | {"manifest.json", "README.md"}
    require(len(checksum_lines) == len(expected_paths) and set(checksums) == expected_paths,
            "incomplete or duplicate supplemental SHA256SUMS")
    for relative, digest in checksums.items():
        require(hashlib.sha256((bundle_root / relative).read_bytes()).hexdigest() == digest,
                f"supplemental checksum changed: {relative}")
    for model, expected_resolutions, expected_time, expected_rejections in (
        ("M07", [40, 80, 160], 1.0, None),
        ("M15", [32, 64, 128], 0.02, 6),
    ):
        science_meta = bundle["scientific"][model]
        science = read_json(bundle_root / science_meta["path"])
        result_dir = science_meta["path"].split("/", 1)[0]
        result = read_json(bundle_root / result_dir / "result.json")
        before = read_json(bundle_root / result_dir / "before/identity.json")
        after = read_json(bundle_root / result_dir / "after/identity.json")
        science_hash = bundle_files[science_meta["path"]]["sha256"]
        require(science["status"] == result["status"] == science_meta["status"] == "passed"
                and result["scientific_receipt_sha256"] == science_hash
                and result["correct_backend"] is True
                and result["same_native"] is True
                and result["same_shipped_sources"] is True
                and result["example_sources_unchanged"] is True
                and result["authentication_before"] == result["authentication_after"] == 0,
                f"{model} result/science identity mismatch")
        require(result["native_sha256"] == science["native_sha256"]
                == before["native_sha256"] == after["native_sha256"]
                == bundle["native_sha256"]["dim1"]
                and before["source_files_sha256"] == after["source_files_sha256"],
                f"{model} installed native/source identity mismatch")
        require(len(science["records"]) == science_meta["records"] == 6
                and sorted({record["cells"][0] for record in science["records"]})
                == science_meta["resolutions"] == expected_resolutions
                and len({tuple(record["order"]) for record in science["records"]}) == 2
                and science_meta["time"] == expected_time,
                f"{model} scientific case inventory changed")
        if expected_rejections is not None:
            require(len(science["inadmissible_initial_rejections"])
                    == science_meta["inadmissible_rejections"] == expected_rejections,
                    "M15 refusal inventory changed")
    for key, dirname, expected in (
        ("products_and_boundaries", "installed-388c323-products-and-boundaries",
         {"tests": 15, "failures": 8, "errors": 0, "skipped": 0}),
        ("c17_stagnation", "installed-3d06cab-storage-newton",
         {"tests": 12, "failures": 6, "errors": 0, "skipped": 0}),
    ):
        receipt = bundle["installed"][key]
        result = read_json(bundle_root / receipt["path"])
        identity_path = bundle_root / dirname / "identity.json"
        require(result["status"] == receipt["status"] == "failed"
                and result["counts"] == receipt["counts"] == expected
                and pytest_counts(bundle_root / dirname / "pytest.xml") == expected
                and result["identity_sha256"]
                == hashlib.sha256(identity_path.read_bytes()).hexdigest(),
                f"failed installed receipt promoted or changed: {key}")
    c17_cases = list(ET.parse(bundle_root / "installed-3d06cab-storage-newton/pytest.xml")
                     .getroot().iter("testcase"))
    c17_passes = [case for case in c17_cases
                  if "stagnation" in case.get("classname", "")
                  and all(case.find(tag) is None for tag in ("failure", "error", "skipped"))]
    require(len(c17_passes) == bundle["installed"]["c17_stagnation"]["passing_stagnation_cases"]
            == 2, "C17 stagnation witness changed")
    new_ctest = bundle["ctest"]
    new_cases = list(ET.parse(bundle_root / new_ctest["path"]).getroot().iter("testcase"))
    require(len(new_cases) == new_ctest["entries"] == 95
            and dict(Counter(case.get("status") for case in new_cases))
            == new_ctest["statuses"] == {"run": 92, "notrun": 3}
            and not any(case.find("failure") is not None for case in new_cases),
            "boundary/AND9 CTest inventory changed")
    mpi_passes = [case.get("name") for case in new_cases
                  if case.get("name", "").endswith("_np2") and case.get("status") == "run"]
    require(mpi_passes == new_ctest["mpi2_aggregates"] and len(mpi_passes) == 4,
            "selected MPI2 aggregates changed")
    resource = read_json(bundle_root / bundle["resource"]["path"])
    timing_root = HERE / "evidence/performance-t2-v1-2"
    timing_manifest = read_json(timing_root / "manifest.json")
    timing_rel = "performance-t2-v1-2-run/result.json"
    timing_item = next(item for item in timing_manifest["files"]
                       if item["path"] == timing_rel)
    timing_path = timing_root / timing_rel
    require(hashlib.sha256(timing_path.read_bytes()).hexdigest() == timing_item["sha256"],
            "timing v1.2 receipt changed")
    timing = read_json(timing_path)
    require(timing["status"] == "measured"
            and timing["numerical_equivalence"]["passed"] is True
            and timing["step_median_s"]
            == {"baseline": 0.0531579375, "candidate": 0.0519979795}
            and abs(timing["candidate_over_baseline_median"] - 0.9781790254747939) < 1e-15,
            "timing v1.2 scope changed")
    for lane in ("baseline", "candidate"):
        require(all(resource["identities"][lane][key] == timing["identities"][lane][key]
                    for key in ("native_sha256", "source_commit")),
                f"resource/timing {lane} identity mismatch")
    counters = resource["counter_comparison"]
    require(resource["status"] == bundle["resource"]["status"] == "observed"
            and resource["numerical_equivalence"]["passed"] is True
            and counters["kernels"]["unit"] == "program_kernel_operations_or_batches"
            and counters["kernels"]["values"]
            == {"baseline": [36, 36, 36], "candidate": [36, 36, 36]}
            and all(not entry["available"] for name, entry in counters.items()
                    if name != "kernels"), "resource observation overstated or changed")
    cfef_registry = corpus["supplemental_receptions"]["after_cfef849"]
    cfef_path = HERE / cfef_registry["manifest"]
    cfef_root = cfef_path.parent
    cfef = read_json(cfef_path)
    require(cfef["schema"] == "pops.api040.cfef849-scoped-evidence/v1"
            and cfef["mapping_source_commit"] == "4bb0639e33598b6c0aa0d1198ba74e5b5d5200db",
            "wrong cfef849 bundle schema/source")
    cfef_files = {item["path"]: item for item in cfef["files"]}
    require(len(cfef_files) == len(cfef["files"]) == 31, "cfef849 file inventory changed")
    for relative, item in cfef_files.items():
        path = cfef_root / relative
        require(path.is_file() and path.stat().st_size == item["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                f"cfef849 copied bytes changed: {relative}")
    cfef_lines = (cfef_root / "SHA256SUMS").read_text().splitlines()
    cfef_sums = dict(line.split("  ", 1)[::-1] for line in cfef_lines)
    require(len(cfef_lines) == 33
            and set(cfef_sums) == set(cfef_files) | {"README.md", "manifest.json"},
            "cfef849 checksum inventory changed")
    for relative, digest_value in cfef_sums.items():
        require(hashlib.sha256((cfef_root / relative).read_bytes()).hexdigest() == digest_value,
                f"cfef849 checksum changed: {relative}")
    for model, expected_id, expected_dimension, expected_ranks, expected_cells in (
        ("M08", "m08-cfef849-stage-reviewed", 2, 1, [(32, 32), (64, 64)]),
        ("M07", "m07-cfef849-dim1-mpi2", 1, 2, [(40,), (80,), (160,)]),
    ):
        info = cfef["scientific"][model]
        require(info["id"] == expected_id and info["dimension"] == expected_dimension
                and info["ranks"] == expected_ranks,
                f"{model} scientific dimension/ranks changed")
        result = read_json(cfef_root / expected_id / "result.json")
        science_path = f"{expected_id}/states/receipt.json"
        science = read_json(cfef_root / science_path)
        before = read_json(cfef_root / expected_id / "before/identity.json")
        after = read_json(cfef_root / expected_id / "after/identity.json")
        require(result["status"] == science["status"] == info["status"] == "passed"
                and result["scientific_receipt_sha256"] == cfef_files[science_path]["sha256"]
                and result["correct_backend"] is True
                and all(result[key] is True for key in
                        ("same_native", "same_shipped_sources", "example_sources_unchanged"))
                and result["authentication_before"] == result["authentication_after"] == 0,
                f"{model} scientific receipt status/authentication changed")
        require(before["source_commit"] == after["source_commit"]
                == info["source_commit"] == "cfef849e8e65aa4aa7d0d682084e1134688dcbbc"
                and before["native_sha256"] == after["native_sha256"]
                == info["native_sha256"] == result["native_sha256"]
                == cfef_registry["native_sha256"][f"dim{expected_dimension}"]
                and before["source_files_sha256"] == after["source_files_sha256"],
                f"{model} native/source identity changed")
        require(len(science["records"]) == info["records"]
                and sorted({tuple(row["cells"]) for row in science["records"]})
                == [tuple(row) for row in info["resolutions"]] == expected_cells
                and all(row["mpi_ranks"] == expected_ranks for row in science["records"]),
                f"{model} scientific inventory changed")
        if model == "M08":
            require(len(science["records"]) == 2
                    and all(row["accepted_steps"] == 2 and row["time"] == 0.04
                            and row["final_step_stage0_charge_max_error"] <= 1.4e-17
                            and row["independent_fv_max_error"] <= 3e-11
                            and row["stale_transport_margin"] >= 20
                            and row["physical_stage_field_separation"] >= 1e-8
                            for row in science["records"]), "M08 stage witness changed")
        else:
            require(len(science["records"]) == 6
                    and len({tuple(row["order"]) for row in science["records"]}) == 2
                    and all(row["run_report"]["final_time"] == 1.0
                            for row in science["records"])
                    and all(row["equilibrium_error"] <= 1e-12 for row in science["records"]),
                    "M07 MPI2 equilibrium witness changed")
    expected_installed = {
        "installed-cfef849-dim1-m10": ("passed", 1, 2, 0),
        "installed-cfef849-integrated-unit": ("passed", 2, 163, 0),
        "installed-cfef849-local-and-frontier": ("failed", 2, 19, 8),
    }
    for name, (status, dimension, tests, failures) in expected_installed.items():
        info = cfef["installed"][name]
        result = read_json(cfef_root / name / "result.json")
        identity_path = cfef_root / name / "identity.json"
        identity = read_json(identity_path)
        expected_counts = {"tests": tests, "failures": failures, "errors": 0, "skipped": 0}
        require(info["status"] == result["status"] == status
                and info["counts"] == result["counts"] == expected_counts
                and pytest_counts(cfef_root / name / "pytest.xml") == expected_counts
                and result["identity_sha256"] == hashlib.sha256(identity_path.read_bytes()).hexdigest()
                and info["dimension"] == dimension
                and identity["source_commit"] == info["source_commit"]
                == "cfef849e8e65aa4aa7d0d682084e1134688dcbbc"
                and identity["native_sha256"] == info["native_sha256"]
                == cfef_registry["native_sha256"][f"dim{dimension}"],
                f"installed cfef849 receipt changed: {name}")
        require(cfef_registry["statuses"][name] == status, f"registry status changed: {name}")
    unit_cases = list(ET.parse(cfef_root / "installed-cfef849-integrated-unit/pytest.xml")
                      .getroot().iter("testcase"))
    c22_names = [case.get("name") for case in unit_cases
                 if case.get("classname", "").endswith("test_external_grid_frontier")]
    require(c22_names == cfef["c22_source_host_nodes"] and len(c22_names) == 24,
            "C22 source/host test selection changed")
    require(cfef_registry["statuses"]["m08-cfef849-stage-reviewed"]
            == cfef["scientific"]["M08"]["status"] == "passed"
            and cfef_registry["statuses"]["m07-cfef849-dim1-mpi2"]
            == cfef["scientific"]["M07"]["status"] == "passed",
            "scientific status promotion")
    historical_registry = corpus["supplemental_receptions"]["after_4bb0639"]
    historical_path = HERE / historical_registry["manifest"]
    historical_root = historical_path.parent
    historical = read_json(historical_path)
    require(historical["schema"] == "pops.api040.4bb0639-final-scoped-evidence/v1"
            and historical["mapping_source_commit"]
            == "4bb0639e33598b6c0aa0d1198ba74e5b5d5200db"
            and historical["native_sha256"] == historical_registry["native_sha256"],
            "wrong 4bb0639 bundle identity")
    historical_files = {item["path"]: item for item in historical["files"]}
    require(len(historical_files) == len(historical["files"]) == 38,
            "4bb0639 copied file inventory changed")
    for relative, item in historical_files.items():
        path = historical_root / relative
        require(path.is_file() and path.stat().st_size == item["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                f"4bb0639 copied bytes changed: {relative}")
    checksum_lines = (historical_root / "SHA256SUMS").read_text().splitlines()
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    require(len(checksum_lines) == 40
            and set(checksums) == set(historical_files) | {"README.md", "manifest.json"},
            "4bb0639 checksum inventory changed")
    for relative, digest_value in checksums.items():
        require(hashlib.sha256((historical_root / relative).read_bytes()).hexdigest()
                == digest_value, f"4bb0639 checksum changed: {relative}")
    source_commit = historical["mapping_source_commit"]
    product_id = "installed-4bb0639-products-operators-frontier"
    product = read_json(historical_root / product_id / "result.json")
    product_identity_path = historical_root / product_id / "identity.json"
    product_identity = read_json(product_identity_path)
    product_counts = {"tests": 21, "failures": 10, "errors": 0, "skipped": 0}
    require(product["status"] == historical_registry["statuses"][product_id] == "failed"
            and product["counts"] == historical["installed"][product_id]["counts"]
            == pytest_counts(historical_root / product_id / "pytest.xml") == product_counts
            and historical["installed"][product_id]["passing"] == 11
            and product["identity_sha256"]
            == hashlib.sha256(product_identity_path.read_bytes()).hexdigest()
            and product_identity["source_commit"] == source_commit
            and product_identity["native_sha256"] == historical["native_sha256"]["dim2"],
            "4bb0639 installed product/frontier result promoted or changed")
    product_cases = list(ET.parse(historical_root / product_id / "pytest.xml")
                         .getroot().iter("testcase"))
    failure_messages = [case.find("failure").get("message", "") for case in product_cases
                        if case.find("failure") is not None]
    require(sum("no exact ghost depth" in message for message in failure_messages) == 8
            and sum("conflicting parameter metadata for 'gain'" in message
                    for message in failure_messages) == 2
            and sum(case.find("failure") is None and case.find("error") is None
                    and case.find("skipped") is None for case in product_cases) == 11,
            "4bb0639 installed failure groups changed")
    require(sum(case.get("classname", "").endswith("test_api040_c17_stagnation_runtime")
                and case.find("failure") is None for case in product_cases) == 2
            and sum(case.get("classname", "").endswith("test_external_grid_frontier_runtime")
                    and case.find("failure") is None for case in product_cases) == 1,
            "4bb0639 C17/C22 scoped passes changed")
    m10_id = "installed-4bb0639-dim1-m10-mpi2"
    m10 = read_json(historical_root / m10_id / "result.json")
    m10_before = read_json(historical_root / m10_id / "before/identity.json")
    m10_after = read_json(historical_root / m10_id / "after/identity.json")
    require(m10["status"] == historical_registry["statuses"][m10_id] == "passed"
            and m10["dimension"] == 1 and m10["ranks"] == 2
            and m10["returncode"] == 0 and m10["timeout"] is False
            and m10["authentication_before"] == m10["authentication_after"] == 0
            and all(m10[key] is True for key in
                    ("same_installation", "test_sources_unchanged", "rank_test_parity"))
            and m10_before["source_commit"] == m10_after["source_commit"] == source_commit
            and m10_before["source_files_sha256"] == m10_after["source_files_sha256"]
            and m10_before["native_sha256"] == m10_after["native_sha256"]
            == m10["native_sha256"] == historical["native_sha256"]["dim1"],
            "4bb0639 M10 MPI2 provenance changed")
    m10_counts = {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}
    m10_nodes = [
        ["tests.python.integration.runtime.test_m10_self_consistent_sg",
         "test_m10_source_has_one_joint_face_construction_and_live_poisson_dependencies"],
        ["tests.python.integration.runtime.test_m10_self_consistent_sg",
         "test_m10_native_self_consistent_stage_field_and_joint_flux"],
    ]
    require(len(m10["rank_results"]) == 2, "4bb0639 M10 rank inventory changed")
    for rank, row in enumerate(m10["rank_results"]):
        prefix = historical_root / m10_id / f"rank{rank}"
        identity = read_json(prefix.with_suffix(".identity.json"))
        require(row["rank"] == rank and row["counts"] == m10_counts
                and pytest_counts(prefix.with_suffix(".xml")) == m10_counts
                and row["nodes"] == m10_nodes
                and row["xml_sha256"]
                == hashlib.sha256(prefix.with_suffix(".xml").read_bytes()).hexdigest()
                and row["log_sha256"]
                == hashlib.sha256(prefix.with_suffix(".log").read_bytes()).hexdigest()
                and identity["native_sha256"] == historical["native_sha256"]["dim1"],
                f"4bb0639 M10 rank {rank} result changed")
    m15_id = "m15-4bb0639-dim1-mpi2"
    m15 = read_json(historical_root / m15_id / "result.json")
    interrupted = read_json(historical_root / m15_id / "interruption.json")
    m15_before = read_json(historical_root / m15_id / "before/identity.json")
    m15_after = read_json(historical_root / m15_id / "after/identity.json")
    require(m15["status"] == historical_registry["statuses"][m15_id] == "failed"
            and m15["returncode"] == 15 and m15["dimension"] == 1 and m15["ranks"] == 2
            and m15["correct_backend"] is False
            and m15["scientific_receipt_sha256"] is None
            and not (historical_root / m15_id / "states/receipt.json").exists()
            and interrupted["status"] == "interrupted_after_diagnosed_collective_mismatch"
            and m15_before["source_commit"] == m15_after["source_commit"] == source_commit
            and m15_before["source_files_sha256"] == m15_after["source_files_sha256"]
            and m15_before["native_sha256"] == m15_after["native_sha256"]
            == m15["native_sha256"] == historical["native_sha256"]["dim1"],
            "4bb0639 interrupted M15 promoted or changed")
    for sample, digest_value in interrupted["samples"].items():
        relative = f"{m15_id}/{sample}"
        require(historical_files[relative]["sha256"] == digest_value
                == historical["scientific_interrupted"]["sample_sha256"][sample],
                f"4bb0639 M15 sample changed: {sample}")
    require("WorldCommunicator::allgather_bytes" in
            (historical_root / m15_id / "rank0.sample.txt").read_text()
            and "System<1>::step" in
            (historical_root / m15_id / "rank1.sample.txt").read_text()
            and "collective_step_rejection_phase" in
            (historical_root / m15_id / "rank1.sample.txt").read_text(),
            "4bb0639 M15 collective mismatch evidence changed")
    partial = historical_root / historical["scientific_interrupted"]["saved_partial_state"]
    with zipfile.ZipFile(partial) as archive:
        require(archive.testzip() is None
                and archive.namelist()
                == [f"{name}.npy" for name in
                    ("initial", "final", "oracle", "order", "time", "dt", "cells")],
                "4bb0639 partial M15 state archive changed")
    diagnostics = historical["source_host_diagnostics"]
    require("BlockRegistry' object is not subscriptable" in
            (historical_root / diagnostics["initial_fixture_error"]).read_text()
            and "conflicting parameter metadata for 'gain'" in
            (historical_root / diagnostics["repaired_fixture_real_failure"]).read_text()
            and diagnostics["status"] == "failed",
            "4bb0639 qualified-parameter diagnostic overstated")
    periodic_registry = corpus["supplemental_receptions"]["after_7b35ffd_periodic"]
    periodic_root = (HERE / periodic_registry["manifest"]).parent
    periodic = read_json(periodic_root / "manifest.json")
    require(periodic["schema"] == "pops.api040.post-7b35ffd-scoped-evidence/v1"
            and len(periodic["files"]) == 78
            and set(periodic["statuses"]) == set(periodic_registry["statuses"]),
            "post-7b35ffd bundle inventory changed")
    periodic_files = {item["path"]: item for item in periodic["files"]}
    require(len(periodic_files) == len(periodic["files"]),
            "duplicate post-7b35ffd copied file")
    for relative, item in periodic_files.items():
        path = periodic_root / relative
        require(path.is_file() and path.stat().st_size == item["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                f"post-7b35ffd copied bytes changed: {relative}")
    checksum_lines = (periodic_root / "SHA256SUMS").read_text().splitlines()
    checksums = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    require(len(checksum_lines) == len(periodic_files) + 3
            and set(checksums) == set(periodic_files) |
            {"README.md", "manifest.json", "recomputed_m15_metrics.json"},
            "post-7b35ffd checksum inventory changed")
    for relative, digest_value in checksums.items():
        require(hashlib.sha256((periodic_root / relative).read_bytes()).hexdigest()
                == digest_value, f"post-7b35ffd checksum changed: {relative}")
    native_dim1 = "cd45279c320533825f10a82433f6039fe4003c547f52546542bf314e176d8b01"
    native_dim2 = "00b38fe09ba42678a5a2beda64a3767be9aa130455ecdfcf864a7182fc519cc9"
    sdk_header = "02723ae9a5d36640fb5ad31d3e89c9b7b3a4a097c92059aac57d2857a3b4020e"
    counts = {
        "installed-7b35ffd-integrated-unit": (95, 0, "passed"),
        "installed-fc0d6f4c-native-smoke": (4, 1, "failed"),
        "installed-readonly-m11-m18-reception": (9, 1, "failed"),
        "installed-affine-consumers-native": (8, 2, "failed"),
        "installed-m18-rectangular-repaired": (6, 0, "passed"),
    }
    for name, (tests, failures, status) in counts.items():
        record = read_json(periodic_root / name / "result.json")
        identity_path = periodic_root / name / "identity.json"
        identity = read_json(identity_path)
        expected_counts = {"tests": tests, "failures": failures,
                           "errors": 0, "skipped": 0}
        require(record["status"] == periodic_registry["statuses"][name] == status
                and record["counts"] == pytest_counts(periodic_root / name / "pytest.xml")
                == expected_counts
                and record["identity_sha256"]
                == hashlib.sha256(identity_path.read_bytes()).hexdigest()
                and identity["source_commit"] == periodic["statuses"][name]["source_commit"]
                and identity["native_sha256"] == native_dim2
                and periodic["statuses"][name]["sdk_header_sha256"] == sdk_header,
                f"post-7b35ffd installed scope changed: {name}")
    readonly_cases = list(ET.parse(periodic_root /
                          "installed-readonly-m11-m18-reception/pytest.xml")
                          .getroot().iter("testcase"))
    require(sum("test_m11_w10_constrained_runtime" in case.get("classname", "")
                and case.find("failure") is None for case in readonly_cases) == 2
            and sum("test_m18_discrete_entropy_runtime" in case.get("classname", "")
                    and case.find("failure") is not None for case in readonly_cases) == 1,
            "M11/W10 pass or historical M18 shape failure changed")
    m18_cases = list(ET.parse(periodic_root /
                     "installed-m18-rectangular-repaired/pytest.xml")
                     .getroot().iter("testcase"))
    require(sum("test_m18_discrete_entropy_runtime" in case.get("classname", "")
                and case.find("failure") is None for case in m18_cases) == 1,
            "M18 repaired fixture native witness changed")
    ctest = ET.parse(periodic_root / "native-periodic-core-ctest.xml").getroot()
    require({key: int(ctest.get(key)) for key in ("tests", "failures", "skipped")}
            == periodic["ctest"] == {"tests": 95, "failures": 0, "skipped": 3}
            and "redefinition of 'selected'" in
            (periodic_root / "build-m15-periodic-core-dim1.log").read_text(),
            "periodic core CTest or prior red build changed")
    for name, dimension, native_hash, tests_per_rank in (
        ("installed-7b35ffd-dim1-m15-refusal-mpi2", 1, native_dim1, 1),
        ("installed-c22-frontier-mpi2", 2, native_dim2, 1),
    ):
        record = read_json(periodic_root / name / "result.json")
        before = read_json(periodic_root / name / "before/identity.json")
        after = read_json(periodic_root / name / "after/identity.json")
        require(record["status"] == periodic_registry["statuses"][name] == "passed"
                and record["dimension"] == dimension and record["ranks"] == 2
                and record["authentication_before"] == record["authentication_after"] == 0
                and record["rank_test_parity"] is True
                and before["native_sha256"] == after["native_sha256"]
                == record["native_sha256"] == native_hash,
                f"post-7b35ffd MPI provenance changed: {name}")
        for rank, row in enumerate(record["rank_results"]):
            xml = periodic_root / name / f"rank{rank}.xml"
            require(row["rank"] == rank and row["counts"]
                    == pytest_counts(xml)
                    == {"tests": tests_per_rank, "failures": 0, "errors": 0,
                        "skipped": 0}
                    and row["xml_sha256"] == hashlib.sha256(xml.read_bytes()).hexdigest(),
                    f"post-7b35ffd MPI rank result changed: {name}/{rank}")
    m15_name = "m15-periodic-core-dim1-mpi2"
    m15 = read_json(periodic_root / m15_name / "result.json")
    m15_receipt_path = periodic_root / m15_name / "states/receipt.json"
    m15_receipt = read_json(m15_receipt_path)
    m15_metrics = read_json(periodic_root / "recomputed_m15_metrics.json")
    require(m15["status"] == periodic_registry["statuses"][m15_name] == "passed"
            and m15["dimension"] == 1 and m15["ranks"] == 2
            and m15["authentication_before"] == m15["authentication_after"] == 0
            and all(m15[key] is True for key in
                    ("same_native", "same_shipped_sources",
                     "example_sources_unchanged", "correct_backend"))
            and m15["native_sha256"] == m15_receipt["native_sha256"] == native_dim1
            and m15["scientific_receipt_sha256"]
            == hashlib.sha256(m15_receipt_path.read_bytes()).hexdigest()
            and len(m15_receipt["records"]) == len(m15_metrics["records"]) == 6
            and len(m15_receipt["inadmissible_initial_rejections"]) == 6
            and m15_receipt["resolutions"] == [32, 64, 128]
            and m15_receipt["t_end"] == 0.02,
            "M15 six-run scientific receipt changed")
    for record, metrics in zip(m15_receipt["records"], m15_metrics["records"], strict=True):
        saved = periodic_root / m15_name / "states" / record["saved_state"]
        with zipfile.ZipFile(saved) as archive:
            require(archive.testzip() is None
                    and {"initial.npy", "final.npy", "oracle.npy", "order.npy",
                         "time.npy", "dt.npy", "cells.npy"} == set(archive.namelist()),
                    "M15 saved NPZ changed")
        require(record["saved_state_sha256"] == metrics["npz_sha256"]
                == hashlib.sha256(saved.read_bytes()).hexdigest()
                and record["cells"] == [metrics["cells"]]
                and record["order"] == metrics["order"]
                and record["accepted_steps"] == 2 * metrics["cells"]
                and record["mpi_ranks"] == 2
                and all(abs(record[key] - metrics[key]) <= 1e-18 for key in
                        ("state_max_error", "moment_integral_error",
                         "time_error", "permutation_max_error")),
                "M15 saved-state metric provenance changed")
    total_scoped = sum(len(read_json(HERE / data["manifest"])["receipts"])
                       for scope, data in corpus["scoped_receptions"].items()
                       if scope in expected_scopes)
    print(f"40 contracts, 28 models, 12 witnesses; {len(manifest['receipts'])} historical, "
          f"{total_scoped} earlier scoped, 5 cfef849, 3 final-4bb0639 "
          "and 8 post-7b35ffd "
          "selected receipt snapshots consistent")
    print("No corpus-wide native, numerical, MPI, or global acceptance implied.")


if __name__ == "__main__":
    main()
