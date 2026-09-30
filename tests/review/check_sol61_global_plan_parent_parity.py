"""Explicit source-only historical comparison; requires the named Git object.

Ordinary unit tests do not call this checker and need no Git history. No fetch,
installation, native bootstrap, build, or provenance normalization is performed.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

PARENT = "368055dbe1f4c1f5fad4a11791508ae2b0520f03"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def compare(repository, candidate):
    repository, candidate = Path(repository).resolve(), Path(candidate).resolve()
    root = Path(__file__).resolve().parents[2]
    control = root / "tests/review/sol61_global_plan_parity_control.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    def run(package, *args):
        result = subprocess.run([sys.executable, str(control), str(package), *args],
                                cwd=root, env=environment, check=True,
                                capture_output=True, text=True, timeout=60)
        return json.loads(result.stdout)
    with tempfile.TemporaryDirectory(prefix="pops-global-parent-parity-") as temporary:
        archive = subprocess.run(["git", "archive", PARENT, "python/pops"],
                                 cwd=repository, check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(temporary, filter="data")
        parent = run(Path(temporary) / "python")
        current = run(candidate)
        # Compare every byte represented by the receipt, including the complete
        # canonical plan payload and actual snapshot provenance. No fields drop.
        if parent != current:
            raise ValueError("fresh parent/candidate source receipts differ")
        changed = run(candidate, ".47")
        for key, expected in parent.items():
            actual = changed[key]
            for field in ("ir", "module_hash", "module_manifest", "plan", "plan_payload_hex",
                          "snapshot_artifact_json", "system", "amr_system"):
                if actual[field] == expected[field]:
                    raise ValueError("changed source escaped " + field)
    report = {}
    for key, value in current.items():
        report[key] = {
            **{field: value[field] for field in ("ir", "module_hash", "manifest_version", "plan", "system", "amr_system")},
            "module_manifest_sha256": digest(json.dumps(value["module_manifest"], sort_keys=True).encode()),
            "full_plan_payload_sha256": digest(bytes.fromhex(value["plan_payload_hex"])),
            "snapshot_artifact_sha256": digest(value["snapshot_artifact_json"].encode()),
        }
    return dict(status="source-parent-parity-received", parent_commit=PARENT,
                candidate_package_root=str(candidate), control_path=str(control),
                control_sha256=digest(control.read_bytes()),
                provenance_normalized=False, full_payload_compared=True,
                changed_source_guards=32, native_execution=False, cases=report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--candidate-package-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.repository, args.candidate_package_root)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "changed_source_guards", "native_execution")}, sort_keys=True))
