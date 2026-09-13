"""CI-only pytest timing receipts and bounded selected-shard budgets.

The append-only phase log and atomic snapshot survive an interrupted pytest session.
Reported phase sums exclude collection and scheduler overhead; only a file whose
collected tests all reach teardown has a complete phase receipt. They are not
silently relabelled as measured end-to-end file wall time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

ORDINARY_BUDGET_SECONDS = 35 * 60
SHARD_TOTAL = 31


def selected_budget(paths: list[str], durations: dict[str, float]) -> int:
    """Leave ten minutes beyond the model; only indivisible long files get more."""
    weights = [durations[path] for path in paths]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("selected Python duration weights must be finite and positive")
    load = sum(weights)
    if load > ORDINARY_BUDGET_SECONDS:
        if len(paths) != 1:
            raise ValueError("Python shard exceeds its ordinary budget without one indivisible file")
        return math.ceil(load / 60) + 10
    return 45


def job_budget(shard: int) -> int:
    """LPT places the four longest indivisible files first, even for subsets."""
    if not 0 <= shard < SHARD_TOTAL:
        raise ValueError("Python shard index is outside the configured matrix")
    return {0: 110, 1: 80, 2: 80, 3: 60}.get(shard, 50)


class TimingReceipts:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.state: dict = {"schema": "pops.pytest-phase-timings.v1", "complete": False,
                            "active_node": None, "active_phase": "collection", "files": {}}
        self.events = (directory / "timings.jsonl").open("w", encoding="utf-8", buffering=1)
        self.record("session_start")

    def record(self, event: str, **fields: object) -> None:
        elapsed = time.monotonic() - self.started
        self.events.write(json.dumps({"event": event, "elapsed_seconds": elapsed, **fields}) + "\n")
        self.events.flush()
        self.state["elapsed_seconds"] = elapsed
        temporary = self.directory / "timings.json.tmp"
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "timings.json")

    def pytest_collection_finish(self, session) -> None:
        for item in session.items:
            path = item.nodeid.split("::", 1)[0]
            row = self.state["files"].setdefault(path, {"collected": 0, "finished": 0,
                "reported_phase_seconds": 0.0, "complete": False})
            row["collected"] += 1
        self.record("collection_finish", collected=len(session.items))

    def pytest_runtest_logstart(self, nodeid, location) -> None:
        self.state["active_node"] = nodeid
        self.state["active_phase"] = "setup"
        self.record("test_start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report) -> None:
        path = report.nodeid.split("::", 1)[0]
        row = self.state["files"][path]
        row["reported_phase_seconds"] += report.duration
        if report.when == "teardown":
            row["finished"] += 1
            row["complete"] = row["finished"] == row["collected"]
            self.state["active_node"] = None
            self.state["active_phase"] = "between_tests"
        else:
            self.state["active_phase"] = "call" if report.when == "setup" else "teardown"
        self.record("test_report", nodeid=report.nodeid, phase=report.when,
                    outcome=report.outcome, seconds=report.duration)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        self.state["complete"] = int(exitstatus) in (0, 1, 5)
        self.state["exit_code"] = int(exitstatus)
        self.record("session_finish", exit_code=int(exitstatus))
        self.events.close()


def pytest_configure(config) -> None:
    directory = os.environ.get("POPS_CI_PYTEST_TIMINGS_DIR")
    if directory:
        config.pluginmanager.register(TimingReceipts(Path(directory)), "pops_ci_timing_receipts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-file", type=Path, required=True)
    parser.add_argument("--durations", type=Path, default=Path("tests/python/test_durations.json"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    paths = [line for line in args.selected_file.read_text().splitlines() if line]
    durations = json.loads(args.durations.read_text())
    minutes = selected_budget(paths, durations)
    if minutes + 5 > job_budget(args.shard_index):
        raise ValueError("selected test budget does not leave artifact/setup margin in this job")
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write(f"test_timeout_minutes={minutes}\n")
    print(f"Python shard {args.shard_index}: {len(paths)} files; modeled {sum(durations[p] for p in paths):.1f}s; test watchdog {minutes}m")


if __name__ == "__main__":
    main()
