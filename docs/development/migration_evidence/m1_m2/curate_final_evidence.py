"""Copy exact migration receipts and validate the declared final gate inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--final", action="store_true")
args = parser.parse_args()
root = Path(__file__).parent
destination = Path("/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-m2/docs/development/migration_evidence/m1_m2")
destination.mkdir(parents=True, exist_ok=True)

def read(name):
    return json.loads((root / name).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

names = {
    "m2-build-sixth.log", "m2-build-seventh.log", "m2-build-seventh-proof.json",
    "candidate6-native-summary.json", "candidate7-validation-report.md",
    "candidate7-mms-primitive.log", "candidate7-mms-primitive.xml",
    "candidate7-auxiliary-public-bind-clean.log", "candidate7-auxiliary-public-bind-clean.xml",
    "aux-outcome-amr-equivalence.json", "m2-effects-followup-repro.json",
    "m2-effects-followup-source.log", "projection-model-free-compatibility.json",
    "roe-qualified-ad-report.md", "roe-qualified-ad-rollback-candidate5.log",
    "roe-qualified-ad-rollback-candidate5.xml", "candidate7-codegen-identity.log",
    "candidate7-codegen-identity.xml",
}
native = read("candidate6-native-summary.json")
names.update(native["evidence_sha256"])
names.update(native["runner_receipts"])
for directory in ("candidate7-native-time", "candidate7-provider-fixtures"):
    names.update(str(p.relative_to(root)) for p in (root / directory).iterdir()
                 if p.is_file() and p.suffix in {".json", ".jsonl", ".log"})
report_names = [
    "candidate5-examples/scalar/report.json",
    "candidate7-examples/other/report.json",
    "candidate7-examples/multiphysics-final/report.json",
]
profiles = []
for name in report_names:
    report = read(name)
    assert not report["dirty"], name
    names.add(name)
    for row in report["profiles"]:
        assert row["returncode"] == 0 and not row["timeout"], row["profile"]
        log = Path(row["log"])
        assert digest(log) == row["log_sha256"], log
        names.add(str(log.relative_to(root)))
        profiles.append({
            "profile": row["profile"], "source_revision": report["revision"],
            "native_sha256": report["installation"]["sha256"],
            "dimension": report["dimension"], "omp_num_threads": report["omp_num_threads"],
            "mpi_ranks": row["mpi_ranks"], "seconds": row["seconds"],
            "report": name, "log_sha256": row["log_sha256"], "status": "passed",
        })
assert {p["profile"] for p in profiles} == {
    "scalar_full", "scalar_tutorial_openmp", "scalar_tutorial_mpi2",
    "multiphysics_full", "imex_amr_full",
}
assert len(profiles) == 5
proof = read("m2-build-seventh-proof.json")
solves = read("candidate7-native-time/results.json")
assert solves["total_passed_assertions"] == 145
assert solves["total_failed_assertions"] == solves["total_skips"] == 0

pending = ["final compiler/identity gate", "paired benchmark measurements"]
compiler = None
benchmark = None
optional = ["scalar-candidate5-to7-equivalence.json", "scalar-candidate5-to7-equivalence.md"]
for name in optional:
    if (root / name).is_file():
        names.add(name)
if args.final:
    name = "candidate7-codegen-identity-final.xml"
    suites = ET.parse(root / name).getroot()
    suites = [suites] if suites.tag == "testsuite" else list(suites.findall("testsuite"))
    compiler = {key: sum(int(s.get(key, 0)) for s in suites)
                for key in ("tests", "failures", "errors", "skipped")}
    assert compiler["tests"] > 1000 and compiler["failures"] == compiler["errors"] == 0
    names.update({name, "candidate7-codegen-identity-final.log",
                  "candidate7-codegen-identity-final-summary.json"})
    benchmark = read("candidate7-benchmark-comparison.json")
    assert benchmark["valid_records"] == 8 and len(benchmark["comparisons"]) == 6
    names.update({"candidate7-benchmark-comparison.json", "candidate7-benchmarks.json",
                  "candidate7-benchmarks-mpi2.json", "compare_final_benchmarks.py",
                  "candidate7-benchmarks.log", "candidate7-benchmarks-mpi2.log",
                  "candidate7-benchmark-configure.log", "candidate7-benchmark-build.log",
                  "candidate7-benchmark-provenance.json", "curate_final_evidence.py"})
    pending = []

files = []
for name in sorted(names):
    source = root / name
    assert source.is_file(), name
    expected = native["evidence_sha256"].get(name) or native["runner_receipts"].get(name)
    if expected:
        assert digest(source) == expected, name
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(source) == digest(target), name
    files.append({"path": name, "sha256": digest(target), "bytes": target.stat().st_size})
index = {
    "schema": "pops.migration.m0-m2.final-evidence.v1",
    "status": "complete" if args.final else "qualification_in_progress",
    "pending": pending,
    "implementation_revision": "bd583faf196f3c1faeedec489e04d24e959ed00b",
    "wheel_sha256": proof["wheel_sha256"],
    "installed_tree_sha256": proof["installed_tree_sha256"],
    "installed_member_count": proof["installed_member_count"],
    "native_sha256": proof["native_variants"][0]["sha256"],
    "profiles": profiles,
    "local_solve_assertions": {"passed": 145, "failed": 0, "skipped": 0},
    "compiler_junit": compiler,
    "benchmark_valid_records": None if benchmark is None else benchmark["valid_records"],
    "limits": [
        "Each profile is attributed to its executed source and native artifact; candidate5 scalar execution is not relabeled candidate7.",
        "The original failed baseline and historical diagnostic runs remain separately recorded.",
        "No GPU or full public Dim1/Dim3 scientific qualification; native contract dimensions are a different evidence level.",
        "No zero-overhead claim, hardware memory-traffic counters, or aggregate MPI-worker RSS measurement.",
        "Large numerical outputs and wheels remain at the exact local paths recorded by their original receipts; small raw receipts are copied byte-for-byte here.",
    ],
    "files": files,
}
(destination / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(json.dumps({"status": index["status"], "copied_files": len(files),
                  "bytes": sum(f["bytes"] for f in files), "profiles_passed": len(profiles)}))
