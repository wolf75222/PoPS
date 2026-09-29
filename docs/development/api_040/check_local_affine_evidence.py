"""Read-only revalidation of the frozen local/affine serial and MPI2 bundle."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def verify(base):
    def read(path):
        return json.loads((base / path).read_text())

    def digest(path):
        return hashlib.sha256((base / path).read_bytes()).hexdigest()

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    manifest = read("manifest.json")
    require(manifest["schema"] == "pops.api040.local-affine-converged-evidence/v1", "bundle schema")
    for item in manifest["files"]:
        require(digest(item["path"]) == item["sha256"]
                and (base / item["path"]).stat().st_size == item["bytes"], item["path"])
    for line in (base / "SHA256SUMS").read_text().splitlines():
        expected, path = line.split("  ", 1)
        require(digest(path) == expected, f"SHA256SUMS: {path}")
    inventory = read("inventory.json")
    for path, row in inventory.items():
        cases = list(ET.parse(base / path).getroot().iter("testcase"))
        counts = {"tests": len(cases), **{
            key: sum(case.find(tag) is not None for case in cases)
            for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}}
        require(counts == row["counts"] and not any(counts[key] for key in
                ("failures", "errors", "skipped")), f"XML counts: {path}")
        require([[case.get("classname"), case.get("name")] for case in cases] == row["nodes"]
                and dict(Counter(case.get("classname") for case in cases)) == row["by_module"],
                f"XML node inventory: {path}")
    identity = manifest["identities"]
    serial = "installed-local-affine-serial-converged"
    mpi = "installed-local-affine-m11-m18-mpi2"
    unit = "installed-selected-state-waves-unit"
    commits = {serial: "serial_source_commit", unit: "unit_source_commit",
               mpi + "/before": "mpi_before_source_commit",
               mpi + "/after": "mpi_after_source_commit"}
    for parent in (serial, unit, mpi + "/before", mpi + "/after"):
        row = read(parent + "/identity.json")
        require(row["native_sha256"] == identity["native_sha256"]
                and row["abi_key"] == identity["abi_key"]
                and row["verified_source_files"] == identity["verified_source_files"] == 1086
                and row["source_files_sha256"] == identity["source_files_sha256"]
                == digest(parent + "/source-files.json")
                and len(read(parent + "/source-files.json")) == 1086
                and row["source_commit"] == identity[commits[parent]]
                and row["source_diff_sha256"] == hashlib.sha256(b"").hexdigest()
                and all(check[0] is True for check in row["doctor"].values()),
                f"source/native identity: {parent}")
    for name, total in ((serial, 26), (unit, 28)):
        result = read(name + "/result.json")
        require(result["status"] == "passed" and result["returncode"] == 0
                and result["counts"] == inventory[name + "/pytest.xml"]["counts"]
                and result["counts"]["tests"] == total
                and result["identity_sha256"] == digest(name + "/identity.json")
                and result["log_sha256"] == digest(name + "/pytest.log"), name)
    result = read(mpi + "/result.json")
    require(result["status"] == "passed" and result["returncode"] == 0
            and result["authentication_before"] == result["authentication_after"] == 0
            and result["ranks"] == result["dimension"] == 2 and result["threads"] == 1
            and not result["timeout"]
            and len(result["rank_results"]) == 2
            and result["native_sha256"] == identity["native_sha256"]
            and all(result[key] is True for key in
                    ("same_installation", "test_sources_unchanged", "rank_test_parity")),
            "MPI runner result")
    for rank, result_row in enumerate(result["rank_results"]):
        prefix = f"{mpi}/rank{rank}"
        row = inventory[prefix + ".xml"]
        rank_identity = read(prefix + ".identity.json")
        require(result_row["rank"] == rank and result_row["counts"] == row["counts"]
                and row["counts"]["tests"] == 23 and result_row["nodes"] == row["nodes"]
                and result_row["xml_sha256"] == digest(prefix + ".xml")
                and result_row["log_sha256"] == digest(prefix + ".log")
                and rank_identity["rank"] == rank and rank_identity["ranks"] == 2
                and rank_identity["dimension"] == 2
                and rank_identity["native_sha256"] == identity["native_sha256"], prefix)
    require(inventory[mpi + "/rank0.xml"] == inventory[mpi + "/rank1.xml"], "MPI parity")
    serial_nodes = inventory[serial + "/pytest.xml"]["nodes"]
    native_nodes = [row for row in serial_nodes if ".integration." in row[0]]
    mpi_nodes = inventory[mpi + "/rank0.xml"]["nodes"]
    require(len(native_nodes) == 20 and mpi_nodes[:20] == native_nodes
            and len(mpi_nodes[20:]) == 3
            and all("test_m11_w10_constrained_runtime" in row[0] for row in mpi_nodes[20:22])
            and "test_m18_discrete_entropy_runtime" in mpi_nodes[22][0], "serial/MPI coverage")
    for path, expected in read(mpi + "/test-sources.json").items():
        require(digest("source-snapshot/" + path) == expected, f"test source identity: {path}")
    require(identity["sdk_header_sha256"] in identity["abi_key"], "SDK signature")
    print("Local/affine bundle: 26 serial (6 helper + 20 native), 23 native per MPI rank, "
          "28 installed units; XML, hashes and identities verified.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent / "evidence/local-affine-converged")
