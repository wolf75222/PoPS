"""Verify the archived finite-M09/AMR/W12 receipt; never import PoPS or run it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

NATIVE = "aed8c1582c3344bd5afb19ab784adec260bcdbd4bf944b26a29242b53889c862"
HEADERS = "396719235acdb96435dba0dd27c5e5579aefbda4936f657262b9b2553d13bb4c"
EXPECTED_XML = {
    "installed-finite-amr-repaired-unit/pytest.xml": (21, 0),
    "installed-finite-amr-repaired-serial/pytest.xml": (13, 2),
    "installed-finite-m09-mpi2/rank0.xml": (10, 0),
    "installed-finite-m09-mpi2/rank1.xml": (10, 0),
    "installed-affine-amr-interface-serial/pytest.xml": (3, 0),
    "installed-affine-amr-interface-mpi2/rank0.xml": (3, 0),
    "installed-affine-amr-interface-mpi2/rank1.xml": (3, 0),
    "native-w12-reviewed-ctest.xml": (3, 1),
    "native-w12-snapshot-repaired-ctest.xml": (3, 0),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def xml_rows(path):
    root = ET.parse(path).getroot()
    rows = list(root.iter("testcase"))
    counts = {"tests": len(rows), **{
        name: sum(row.find(tag) is not None for row in rows)
        for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}}
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    for key, value in counts.items():
        require(sum(int(s.get(key, "0")) for s in suites) == value, f"XML counter mismatch {path}: {key}")
    return rows, counts


def verify_bundle(bundle, *, recompute=False):
    bundle = Path(bundle).resolve()
    inventory = load(bundle / "manifest.json")
    require(inventory["schema_version"] == 1, "unknown manifest schema")
    files = inventory["files"]
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    require(actual == set(files) | {"manifest.json"}, "manifest file inventory mismatch")
    for name, record in files.items():
        path = bundle / name
        require(not path.is_symlink() and path.resolve().is_relative_to(bundle), "unsafe archive path")
        require(path.suffix not in (".so", ".dylib", ".o"), "binary/cache payload forbidden")
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], f"hash mismatch: {name}")
    summary = load(bundle / "summary.json")
    observed = {}
    for name, (total, failed) in EXPECTED_XML.items():
        _, counts = xml_rows(bundle / "raw" / name)
        require(counts == dict(tests=total, failures=failed, errors=0, skipped=0), f"unexpected test status: {name}")
        require(summary["xml_counts"]["raw/" + name] == counts, "summary/XML disagreement")
        observed[name] = counts
    serial = bundle / "raw/installed-finite-amr-repaired-serial"
    rows, _ = xml_rows(serial / "pytest.xml")
    finite = [r for r in rows if ".test_finite_m09" in r.get("classname", "")]
    require(len(finite) == 10 and all(r.find("failure") is None for r in finite), "M09 serial subset mismatch")
    for name in ("installed-finite-amr-repaired-unit", "installed-finite-amr-repaired-serial", "installed-affine-amr-interface-serial"):
        folder = bundle / "raw" / name
        identity, result = load(folder / "identity.json"), load(folder / "result.json")
        require(identity["native_sha256"] == NATIVE and "headers=" + HEADERS in identity["abi_key"], "wrong installed SDK/native identity")
        require(identity["source_diff_sha256"] == hashlib.sha256(b"").hexdigest(), "installed source was modified")
        require(identity["verified_source_files"] == len(load(folder / "source-files.json")) == 1089, "source member count mismatch")
        require(identity["source_files_sha256"] == digest(folder / "source-files.json"), "source manifest digest mismatch")
        require(result["identity_sha256"] == digest(folder / "identity.json"), "identity digest mismatch")
        require(result["log_sha256"] == digest(folder / "pytest.log"), "log digest mismatch")
        require(result["counts"] == observed[name + "/pytest.xml"], "runner/XML disagreement")
        require(all(value[0] is True for value in identity["doctor"].values()), "doctor refusal")
    for name in ("installed-finite-m09-mpi2", "installed-affine-amr-interface-mpi2"):
        folder = bundle / "raw" / name
        result = load(folder / "result.json")
        require(result["ranks"] == 2 and result["returncode"] == 0 and result["timeout"] is False, "MPI launcher did not complete")
        require(result["authentication_before"] == result["authentication_after"] == 0, "MPI authentication failed")
        require(all(result[k] is True for k in ("same_installation", "test_sources_unchanged", "rank_test_parity")), "MPI authority changed")
        nodes = []
        for rank in (0, 1):
            identity = load(folder / f"rank{rank}.identity.json")
            require(identity["rank"] == rank and identity["native_sha256"] == NATIVE, "MPI rank identity mismatch")
            rows, counts = xml_rows(folder / f"rank{rank}.xml")
            nodes.append([(r.get("classname"), r.get("name")) for r in rows])
            receipt = result["rank_results"][rank]
            require(receipt["counts"] == counts, "MPI counts mismatch")
            require(receipt["xml_sha256"] == digest(folder / f"rank{rank}.xml"), "MPI XML digest mismatch")
            require(receipt["log_sha256"] == digest(folder / f"rank{rank}.log"), "MPI log digest mismatch")
        require(nodes[0] == nodes[1], "MPI test node parity mismatch")
    source_manifest = load(bundle / "raw/installed-finite-amr-repaired-unit/source-files.json")
    for proof in load(bundle / "source/git-production-proofs.json"):
        require(proof["count"] == len(proof["files"]) == 1089 and proof["all_match"] is True, "incomplete git production proof")
        require({r["path"]: r["sha256"] for r in proof["files"]} == source_manifest, "git proof/installed source mismatch")
    snapshots = load(bundle / "source/snapshots.json")
    for row in snapshots:
        data = (bundle / row["snapshot"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == row["sha256"], "snapshot SHA mismatch")
        require(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == row["git_blob"], "snapshot Git blob mismatch")
    for name, revision in (("installed-finite-m09-mpi2", "86646b6c"), ("installed-affine-amr-interface-mpi2", "fb017cc4")):
        for path, expected in load(bundle / "raw" / name / "test-sources.json").items():
            matches = [r for r in snapshots if r["path"] == path and r["commit"].startswith(revision)]
            require(len(matches) == 1 and matches[0]["sha256"] == expected, "MPI test snapshot differs from executed source")
    native_log = (bundle / "build/w12-LastTest.log").read_text()
    require(len(re.findall(r"\[  PASSED  \] 10 tests\.", native_log)) == 2, "W12 aggregate lacks both ten-case results")
    require(load(bundle / "build/binary-capture.json")["sha256"] == "87c96448ae2ca84693bfbbdd86f55198e767325512ba2b9a0f5f51d7c5d916e7", "W12 binary hash mismatch")
    metrics = load(bundle / "raw/independent-finite-m09-saved-states.json")
    require(metrics["source_sha256"] == digest(bundle / "raw/recheck_finite_m09_states.py"), "oracle script identity mismatch")
    require(metrics["threshold"] == 1e-11 and metrics["seed"] == 20260928, "oracle criterion changed")
    for key, dirname in (("serial", "installed-finite-amr-repaired-serial"), ("mpi2_rank0", "installed-finite-m09-mpi2")):
        folder = bundle / "raw" / dirname
        row = metrics[key]
        require(row["state_count"] == len(row["records"]) == len(list(folder.rglob("*.npz"))) == 16, "saved-state count mismatch")
        # Root's record paths are relative to pytest-tmp/rank0-tmp.
        base = folder / ("pytest-tmp" if key == "serial" else "rank0-tmp")
        for state in row["records"]:
            require(digest(base / state["state"]) == state["sha256"], "saved-state receipt mismatch")
        for metric in ("max_original_residual", "max_monolithic_error", "max_cn_energy_error"):
            require(0 <= row[metric] < 1e-11, "saved-state metric out of bounds")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="finite-receipt-check-") as temporary:
            output = Path(temporary) / "metrics.json"
            subprocess.run([sys.executable, str(bundle / "raw/recheck_finite_m09_states.py"),
                            "--serial", str(serial / "pytest-tmp"),
                            "--mpi", str(bundle / "raw/installed-finite-m09-mpi2/rank0-tmp"),
                            "--output", str(output)], check=True, capture_output=True, text=True)
            fresh = load(output)
            for key in ("serial", "mpi2_rank0"):
                require(fresh[key]["records"] == metrics[key]["records"], "recomputed oracle differs from receipt")
    return {"archive_integrity": "verified", "files": len(files), "saved_m09_states": 32,
            "retained_failed_xml_rows": 3, "recomputed_numpy": recompute,
            "scope": summary["scope"], "xml_counts": observed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent / "evidence/finite-amr-converged")
    parser.add_argument("--recompute", action="store_true", help="also rerun independent NumPy oracle on all 32 NPZ; no PoPS import")
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle, recompute=args.recompute), indent=2))
