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
    with (HERE / "contracts.csv").open(newline="") as source:
        contracts = list(csv.DictReader(source))
    corpus = read_json(HERE / "corpus.json")
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
    require(corpus["production_evidence_schema"] == 4, "unsupported reception registry schema")
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
    total_scoped = sum(len(read_json(HERE / data["manifest"])["receipts"])
                       for scope, data in corpus["scoped_receptions"].items()
                       if scope in expected_scopes)
    print(f"40 contracts, 28 models, 12 witnesses; {len(manifest['receipts'])} historical and {total_scoped} scoped receipt snapshots consistent")
    print("No native, numerical, MPI, or global acceptance implied.")


if __name__ == "__main__":
    main()
