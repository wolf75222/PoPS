#!/usr/bin/env python3
"""Run one launch of the throwaway F7 probe and report its time and peak memory.

usage: measure.py --label L --receipt R.json [--marker REGEX] [--match TEXT] [--junit DIR] -- cmd...

The command runs with the step's environment and no standard input; its output is forwarded line
by line. A sampler reads /proc every 2 s: the memory used on the machine (MemTotal - MemAvailable)
and the largest resident set per process name. --marker names the output line from which the
measured time counts (the start of the pytest session, or the line that ci_python_mpi.py prints
just before the MPI launch). --match keeps the output lines that contain TEXT (pytest durations).
--junit reads the testcase times of every JUnit file under DIR once the command has returned.
Prints one PROBE line and writes the receipt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

KIB_PER_GIB = 1024 * 1024


def used_kib() -> int:
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0])
    return values["MemTotal"] - values["MemAvailable"]


def largest_per_name() -> dict[str, int]:
    best: dict[str, int] = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/status" % pid) as handle:
                name, rss = None, 0
                for line in handle:
                    if line.startswith("Name:"):
                        name = line.split(None, 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
        if name and rss > best.get(name, 0):
            best[name] = rss
    return best


def junit_cases(directory: Path) -> list[dict]:
    cases = []
    for path in sorted(directory.rglob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for case in root.iter("testcase"):
            outcome = next((tag for tag in ("failure", "error", "skipped")
                            if case.find(tag) is not None), "passed")
            cases.append({"file": str(path.relative_to(directory)), "name": case.attrib.get("name"),
                          "time": case.attrib.get("time"), "outcome": outcome})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--marker")
    parser.add_argument("--match", action="append", default=[])
    parser.add_argument("--junit", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")
    marker = re.compile(args.marker) if args.marker else None
    baseline = used_kib()
    peak: dict = {"used": baseline, "used_at": 0.0, "rss": {}}
    done = threading.Event()
    start = time.monotonic()

    def sample() -> None:
        while not done.is_set():
            used = used_kib()
            if used > peak["used"]:
                peak["used"], peak["used_at"] = used, time.monotonic() - start
            for name, rss in largest_per_name().items():
                if rss > peak["rss"].get(name, 0):
                    peak["rss"][name] = rss
            done.wait(2.0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    marked, kept = None, []
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if marked is None and marker is not None and marker.search(line):
            marked = time.monotonic()
        if any(text in line for text in args.match):
            kept.append(line.rstrip("\n"))
    returncode = process.wait()
    end = time.monotonic()
    done.set()
    sampler.join()
    top = dict(sorted(peak["rss"].items(), key=lambda item: -item[1])[:6])
    cases = junit_cases(args.junit) if args.junit and args.junit.is_dir() else []
    receipt = {
        "label": args.label,
        "returncode": returncode,
        "command": command,
        "wall_seconds": round(end - start, 1),
        "marker": args.marker,
        "after_marker_seconds": None if marked is None else round(end - marked, 1),
        "baseline_used_gib": round(baseline / KIB_PER_GIB, 2),
        "peak_used_gib": round(peak["used"] / KIB_PER_GIB, 2),
        "peak_used_at_seconds": round(peak["used_at"], 1),
        "peak_rss_gib": {name: round(rss / KIB_PER_GIB, 2) for name, rss in top.items()},
        "junit_cases": cases,
        "matched_lines": kept[-40:],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("PROBE %s rc=%d wall=%.0fs after_marker=%s peak_used=%.2fGiB baseline=%.2fGiB "
          "cc1plus=%.2fGiB cases=%s top=%s" % (
              args.label, returncode, end - start,
              "-" if marked is None else "%.0fs" % (end - marked),
              peak["used"] / KIB_PER_GIB, baseline / KIB_PER_GIB,
              peak["rss"].get("cc1plus", 0) / KIB_PER_GIB,
              ";".join("%s=%ss/%s" % (case["file"], case["time"], case["outcome"]) for case in cases) or "-",
              ",".join("%s:%.2f" % (name, rss / KIB_PER_GIB) for name, rss in top.items())),
          flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
