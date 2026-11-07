"""Verify SDK623 M23/M05 archives offline, without importing or executing PoPS."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HEADERS = "62398f3c13c193eb48db07518735fe755e6d108821adffc37acccd8fcea290eb"
NATIVE = {
    1: "42436a0ccef7c5504273c9e459f4bd946652597aa6ee025bc484ac595ad7ca02",
    2: "1afb920397c7fca0d683444c8c21b3fbc908847bdd7e8906457e6dd384c44c27",
}
SERIAL = {
    "installed-m23-source": (74, 0, 1),
    "installed-m23-native-serial": (3, 3, 1),
    "installed-m23-native-bind-repaired": (3, 0, 1),
    "installed-m23-diffusion-nonregression": (5, 1, 2),
    "installed-m05-periodic-shear": (14, 0, 1),
    "installed-m05-reviewed-serial": (19, 0, 1),
    "installed-implicit-state-reviewed-unit": (33, 0, 2),
    "installed-implicit-state-native-repaired": (5, 0, 2),
}
MPI = {"installed-m23-native-mpi2": 3, "installed-m05-periodic-shear-mpi2": 1,
       "installed-m05-reviewed-mpi2": 1}
PAYLOADS = {
    "installed-m23-native-bind-repaired": (7, 2), "installed-m23-native-mpi2": (7, 2),
    "installed-m05-periodic-shear": (3, 0), "installed-m05-periodic-shear-mpi2": (3, 0),
    "installed-m05-reviewed-serial": (3, 3), "installed-m05-reviewed-mpi2": (3, 3),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON numeric value: " + value)


def load(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def identity_token(value, domain):
    require(value["domain"] == domain and value["schema_version"] == 1 and value["algorithm"] == "sha256", "run report identity schema mismatch")
    digest = value["digest"]
    require(digest["encoding"] == "hex" and len(digest["bytes"]) == 64 and all(c in "0123456789abcdef" for c in digest["bytes"]), "run report identity digest malformed")
    return f"pops.{domain}.v1:sha256:" + digest["bytes"]


def xml_counts(path):
    root = ET.parse(path).getroot()
    rows = list(root.iter("testcase"))
    counts = {"tests": len(rows), **{key: sum(row.find(tag) is not None for row in rows)
              for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}}
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    require(all(sum(int(s.get(k, "0")) for s in suites) == v for k, v in counts.items()), "XML counters disagree with testcase elements")
    return rows, counts


def verify_bundle(bundle, *, recompute=False):
    bundle = Path(bundle).resolve()
    inventory = load(bundle / "manifest.json")["files"]
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    require(actual == set(inventory) | {"manifest.json"}, "archive inventory mismatch")
    for name, row in inventory.items():
        path = bundle / name
        require(not path.is_symlink() and path.resolve().is_relative_to(bundle), "unsafe archive path")
        require(path.suffix not in (".so", ".dylib", ".o"), "binary cache payload forbidden")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "archive hash mismatch: " + name)
    summary = load(bundle / "summary.json")
    sources = load(bundle / "source/run-source-commits.json")
    snapshots = load(bundle / "source/snapshots.json")
    observed = {}
    for name, (total, failures, dimension) in SERIAL.items():
        folder = bundle / "raw" / name
        identity, result = load(folder / "identity.json"), load(folder / "result.json")
        _, counts = xml_counts(folder / "pytest.xml")
        require(counts == dict(tests=total, failures=failures, errors=0, skipped=0), "unexpected archived status: " + name)
        require(result["counts"] == counts and result["returncode"] == int(failures != 0), "runner/XML disagreement")
        require(identity["native_sha256"] == NATIVE[dimension] and "headers=" + HEADERS in identity["abi_key"], "native/SDK mismatch")
        require("dim=" + str(dimension) in identity["abi_key"], "native dimension mismatch")
        require(identity["source_commit"] == sources[name], "wrong receipt source commit")
        require(identity["source_diff_sha256"] == hashlib.sha256(b"").hexdigest(), "uncommitted production source")
        require(identity["verified_source_files"] == len(load(folder / "source-files.json")) == 1089, "source inventory cardinality mismatch")
        require(identity["source_files_sha256"] == sha(folder / "source-files.json"), "installed source manifest hash mismatch")
        require(result["identity_sha256"] == sha(folder / "identity.json") and result["log_sha256"] == sha(folder / "pytest.log"), "raw runner receipt hash mismatch")
        require(all(row[0] is True for row in identity["doctor"].values()), "doctor failed")
        observed["raw/" + name + "/pytest.xml"] = counts
    for name, total in MPI.items():
        folder = bundle / "raw" / name
        result = load(folder / "result.json")
        require(result["returncode"] == 0 and result["timeout"] is False and result["ranks"] == 2, "MPI launcher failed")
        require(result["authentication_before"] == result["authentication_after"] == 0, "MPI authentication failed")
        require(all(result[k] is True for k in ("same_installation", "test_sources_unchanged", "rank_test_parity")), "MPI authority changed")
        nodes = []
        for rank in (0, 1):
            identity = load(folder / f"rank{rank}.identity.json")
            require(identity["rank"] == rank and identity["ranks"] == 2 and identity["native_sha256"] == NATIVE[1], "MPI rank identity mismatch")
            rows, counts = xml_counts(folder / f"rank{rank}.xml")
            require(counts == dict(tests=total, failures=0, errors=0, skipped=0), "unexpected MPI testcase status")
            receipt = result["rank_results"][rank]
            require(receipt["counts"] == counts and receipt["xml_sha256"] == sha(folder / f"rank{rank}.xml") and receipt["log_sha256"] == sha(folder / f"rank{rank}.log"), "MPI raw receipt mismatch")
            nodes.append([(r.get("classname"), r.get("name")) for r in rows])
            observed[f"raw/{name}/rank{rank}.xml"] = counts
        require(nodes[0] == nodes[1], "MPI testcase parity mismatch")
        for path, digest in load(folder / "test-sources.json").items():
            matches = [r for r in snapshots if r["commit"] == sources[name] and r["path"] == path]
            require(len(matches) == 1 and matches[0]["sha256"] == digest, "MPI executed test snapshot mismatch")
    require(summary["xml_counts"] == observed, "summary/XML mismatch")
    for row in snapshots:
        data = (bundle / row["snapshot"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == row["sha256"], "source snapshot SHA mismatch")
        require(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == row["git_blob"], "source Git blob mismatch")
    source_versions = {}
    for proof_file in ("git-production-proof.json", "git-production-proof-refreshed.json"):
        proof = load(bundle / "source" / proof_file)
        canonical = load(bundle / proof["reference_manifest"])
        require(len(canonical) == len(proof["files"]) == 1089 and {r["path"]: r["sha256"] for r in proof["files"]} == canonical, "git production proof mismatch")
        require(sha(bundle / proof["reference_manifest"]) == proof["manifest_sha256"], "git proof manifest hash mismatch")
        require(all(r["all_exact_blobs_match"] and r["source_files"] == 1089 for r in proof["commits"]), "incomplete source proof")
        for row in proof["commits"]:
            source_versions[row["commit"]] = canonical
    require(set(source_versions) == set(sources.values()), "missing source commit proof")
    for name in SERIAL:
        require(load(bundle / "raw" / name / "source-files.json") == source_versions[sources[name]], "Python source window mismatch")
    before = load(bundle / "raw/installed-m23-source/source-files.json")
    after = load(bundle / "raw/installed-implicit-state-reviewed-unit/source-files.json")
    require([key for key in before if before[key] != after[key]] == ["python/pops/codegen/inspect_compiled.py"], "unexpected Python refresh surface")
    artifact_by_case = {}
    for name, (states, ledgers) in PAYLOADS.items():
        folder = bundle / "raw" / name
        ledger_paths = [p for p in folder.rglob("*.json") if p.name == "accepted_ledger.json" or p.name.startswith("ledger_")]
        require(len(list(folder.rglob("*.npz"))) == states and len(ledger_paths) == ledgers, "saved payload cardinality mismatch")
        for ledger in ledger_paths:
            rows = load(ledger)  # JSON NaN/Inf fail before any max/comparison.
            expected = 384 if ledger.name == "accepted_ledger.json" else 2 * int(ledger.stem.split("_")[1])
            require(len(rows) == expected, "ledger record count mismatch")
        for receipt_path in folder.rglob("receipt.json"):
            receipt = load(receipt_path)
            require(receipt["native_sha256"] == NATIVE[1], "scientific receipt native mismatch")
            for row in receipt["records"]:
                require(sha(receipt_path.parent / row["saved_state"]) == row["saved_state_sha256"], "scientific NPZ hash mismatch")
                reports = [row["run_report"]] if "run_report" in row else [row["first_run_report"], row["last_run_report"]]
                artifacts = []
                for report in reports:
                    for key, domain in (("run_identity", "run"), ("bind_identity", "bind"), ("execution_identity", "execution-context")):
                        identity_token(report[key], domain)
                    artifacts.append(identity_token(report["artifact_identity"], "artifact"))
                require(len(set(artifacts)) == 1, "first/last M05 run artifact mismatch")
                if "artifact_identity" in row:
                    require(artifacts[0] == row["artifact_identity"], "nested/public artifact identity mismatch")
                case_key = (receipt["case"], row["cells"], tuple(row.get("order", ())), row.get("eta_h"))
                require(artifact_by_case.setdefault(case_key, artifacts[0]) == artifacts[0], "serial/MPI scientific artifact identity mismatch")
                if "reviewed" in name:
                    require(sha(receipt_path.parent / row["saved_ledger"]) == row["saved_ledger_sha256"], "scientific ledger hash mismatch")
            if "reviewed" in name:
                for key, filename in (("example_sha256", "api040_m05_periodic_shear.py"), ("oracle_sha256", "api040_m05_shear_oracle.py")):
                    match = [r for r in snapshots if r["commit"] == sources[name] and r["path"].endswith("/" + filename)]
                    require(len(match) == 1 and receipt[key] == match[0]["sha256"], "scientific source provenance mismatch")
    for kind, serial, mpi in (
            ("m23", "installed-m23-native-bind-repaired", "installed-m23-native-mpi2"),
            ("m05", "installed-m05-reviewed-serial", "installed-m05-reviewed-mpi2")):
        script = bundle / f"raw/recheck_{kind}_saved_states.py"
        report = load(bundle / f"raw/independent-{kind}-saved-states.json")
        require(report["script_sha256"] == sha(script), "independent oracle script mismatch")
        for key, run, temp in (("serial", serial, "pytest-tmp"), ("mpi2", mpi, "rank0-tmp")):
            base = bundle / "raw" / run / temp
            groups = report[key] if kind == "m23" else {"shear": report[key]}
            require({k: len(v) for k, v in groups.items()} == ({"hall": 5, "matrix": 2} if kind == "m23" else {"shear": 3}), "oracle row cardinality mismatch")
            for group, rows in groups.items():
                for row in rows:
                    state = base / row["path"]
                    require(sha(state) == row["sha256"], "oracle state hash mismatch")
                    if "ledger_sha256" in row:
                        ledger = state.with_name("accepted_ledger.json" if group == "matrix" else f"ledger_{row['cells']}.json")
                        require(sha(ledger) == row["ledger_sha256"], "oracle ledger hash mismatch")
        if recompute:
            with tempfile.TemporaryDirectory(prefix="m23-m05-check-") as temp:
                output = Path(temp) / "metrics.json"
                subprocess.run([sys.executable, str(script), "--serial", str(bundle / "raw" / serial / "pytest-tmp"),
                                "--mpi", str(bundle / "raw" / mpi / "rank0-tmp"), "--output", str(output)], check=True, capture_output=True, text=True)
                fresh = load(output)
                # The scripts independently enforce original scientific bounds.
                require(fresh["status"] == "passed", "independent scientific recomputation failed")
                for key in ("serial", "mpi2"):
                    require(fresh[key] == report[key], "recomputed metrics differ from recorded NumPy receipt")
    return {"archive_integrity": "verified", "files": len(inventory), "npz": 26,
            "raw_ledgers": 10, "retained_failed_testcases": 4, "recomputed_numpy": recompute,
            "scope": summary["scope"], "xml_counts": observed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent / "evidence/m23-m05-converged")
    parser.add_argument("--recompute", action="store_true", help="rerun both independent NumPy scripts on archived states/ledgers")
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle, recompute=args.recompute), indent=2))
