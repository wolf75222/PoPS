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
    require(corpus["production_evidence_schema"] == 3, "unsupported reception registry schema")
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
    total_scoped = sum(len(read_json(HERE / data["manifest"])["receipts"])
                       for scope, data in corpus["scoped_receptions"].items()
                       if scope in expected_scopes)
    print(f"40 contracts, 28 models, 12 witnesses; {len(manifest['receipts'])} historical and {total_scoped} scoped receipt snapshots consistent")
    print("No native, numerical, MPI, or global acceptance implied.")


if __name__ == "__main__":
    main()
