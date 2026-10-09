#!/usr/bin/env python3
"""Submit one authenticated SDK15 Native reception; keep exact submission proof."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = "/project/r250127/api040-composition15-mpich-20261002-v15"
BASE = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib, json, pathlib, subprocess, sys
request = json.loads(sys.argv[1])
root = pathlib.Path(request["root"])
script = root / "preparation/native-run-v1.sbatch"
actual = hashlib.sha256(script.read_bytes()).hexdigest()
if actual != request["script_sha256"]:
    raise SystemExit("Native script digest mismatch")
source = root / "source"
head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
if head != request["source_freeze"]:
    raise SystemExit("Source freeze mismatch")
dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
if dirty:
    raise SystemExit("Tracked source is dirty")
live = subprocess.check_output(["squeue", "-u", "rmdraux", "-h", "-o", "%i|%T|%j|%N|%M"], text=True)
exports = "ALL," + ",".join(
    key + "=" + str(value) for key, value in {
        "NATIVE_RUN_LABEL": request["label"],
        "NATIVE_WORLD_SIZE": request["world_size"],
        "NATIVE_SOURCE_FREEZE": request["source_freeze"],
        "NATIVE_TEST_NODE": request["test"],
    }.items()
)
command = ["sbatch", "--parsable", "--export=" + exports, "--chdir=" + str(root),
           "--output=" + str(root / ("jobs/native-" + request["label"] + "-%j.out")),
           "--error=" + str(root / ("jobs/native-" + request["label"] + "-%j.err")), str(script)]
job = subprocess.check_output(command, text=True).strip()
print(json.dumps({"request": request, "actual_script_sha256": actual, "source_head": head,
                  "active_jobs_before": live.splitlines(), "submission_argv": command, "job_result": job}, sort_keys=True))
'''

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--world-size", type=int, choices=(1, 2), required=True)
    p.add_argument("--source-freeze", required=True)
    p.add_argument("--test", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.label):
        p.error("Label must contain only lowercase letters, digits and hyphens")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_freeze):
        p.error("Use the exact 40-character source commit")
    if any(c in args.test for c in ",\n\r"):
        p.error("SLURM export requires one comma-free test node")
    script = BASE / "native-run.sbatch"
    request = {"root": ROOT, "label": args.label, "world_size": args.world_size,
               "source_freeze": args.source_freeze, "test": args.test,
               "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    import shlex
    command = ["ssh", "romeo", shlex.join(["/usr/bin/python3", "-", json.dumps(request, separators=(",", ":"))])]
    result = subprocess.run(command, input=REMOTE, text=True, capture_output=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = BASE / ("native-submission-" + args.label + "-" + stamp + ".json")
    body = {"request": request, "ssh_returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "submitted_at_utc": stamp}
    if result.returncode == 0:
        body["remote_result"] = json.loads(result.stdout)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"receipt": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                      "returncode": result.returncode,
                      "job": body.get("remote_result", {}).get("job_result")}))
    if result.returncode:
        sys.stderr.write(result.stderr)
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
