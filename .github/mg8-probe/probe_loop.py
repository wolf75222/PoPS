#!/usr/bin/env python3
"""Throwaway probe of the MG8 mpi-dim2 stall: master's reference pass in a loop.

This file lives on the branch n5/mg8-probe-stall only, which has no pull request and is never
merged. It runs master's tutorials of the mpi-dim2 lane one launch at a time with one MG8 runner
(the old one of #701, 5c7256b, or the new one of #706) until its time budget is spent, and prints
in the job log, for each launch, one line `PROBE ...`, and for each failed launch what is needed
to find the cause:

- the runner's own output (the new runner prints its verdicts, stacks and log tails itself; for
  the old runner this script prints the verdicts and log tails);
- before the runner's watchdog, the native and Python stacks of every rank and of mpiexec
  (py-spy, gdb, kernel stacks, thread states and CPU use, open files, the system's locks);
- the processes of the launch still alive after the runner returned, which this script then
  stops by pid, so that every launch is independent.

The runner runs in the foreground with this script's standard input, which is the step's: the old
runner passes it on to mpiexec, as in the MG8 job. No tool of the probe reads it.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

LAUNCHERS = ("mpiexec", "mpirun", "orterun")
LEFTOVER_NAMES = LAUNCHERS + ("orted", "prted", "cc1plus", "collect2", "ld", "as", "g++", "c++")
RANK_MARK = " probe --tutorial "
TAIL_LINES = 60
FULL_DUMP_LAUNCHES = 4  # launches per job whose dumps include gdb; later ones keep the rest


def capture(command: list[str], timeout: float) -> str:
    """Run a diagnostic command with no standard input and return its output, whatever happens."""
    try:
        done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout, check=False)
        text = done.stdout
        if done.stderr.strip():
            text += ("" if text.endswith("\n") or not text else "\n") + done.stderr
        if done.returncode:
            text += "(exit %d)\n" % done.returncode
        return text
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or b""
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
        return out + "\n(timed out after %.0f s)\n" % timeout
    except OSError as error:
        return "(could not run %s: %s)\n" % (command[0], error)


def processes() -> list[dict[str, Any]]:
    out = capture(["ps", "-ww", "-eo", "pid=,ppid=,pgid=,etimes=,stat=,args="], 20)
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "pgid": int(parts[2]),
                     "etimes": int(parts[3]), "stat": parts[4], "args": parts[5]})
    return rows


def role(row: dict[str, Any]) -> str | None:
    """Name the part of a launch that a process is: launcher, rank, or a compiler under a rank."""
    argv0 = os.path.basename(row["args"].split()[0]) if row["args"].split() else ""
    if argv0 in LAUNCHERS:
        return "launcher"
    if RANK_MARK in row["args"]:
        return "rank"
    if argv0 in LEFTOVER_NAMES:
        return argv0
    return None


def _ticks(path: str) -> int | None:
    try:
        fields = Path(path).read_text().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime, stime (fields 14 and 15 of stat)
    except (OSError, IndexError, ValueError):
        return None


def threads_now(pid: int) -> str:
    """State, kernel wait channel and CPU use over two seconds of every thread of a process."""
    try:
        tids = sorted(int(name) for name in os.listdir("/proc/%d/task" % pid))
    except OSError as error:
        return "(no threads: %s)\n" % error
    first = {tid: _ticks("/proc/%d/task/%d/stat" % (pid, tid)) for tid in tids}
    time.sleep(2.0)
    hz = os.sysconf("SC_CLK_TCK")
    lines = ["threads (state, CPU over the last 2 s, wait channel, name):"]
    for tid in tids:
        base = "/proc/%d/task/%d" % (pid, tid)
        second = _ticks(base + "/stat")
        try:
            state = Path(base + "/stat").read_text().rsplit(")", 1)[1].split()[0]
            name = Path(base + "/comm").read_text().strip()
            wchan = Path(base + "/wchan").read_text().strip() or "-"
        except OSError:
            lines.append("  %d: gone" % tid)
            continue
        cpu = "?" if first[tid] is None or second is None else "%.0f%%" % (
            100.0 * (second - first[tid]) / (2.0 * hz))
        lines.append("  %d %s cpu=%s wchan=%s %s" % (tid, state, cpu, wchan, name))
    return "\n".join(lines) + "\n"


class Watcher(threading.Thread):
    """Dump every rank and launcher of the current launch once it is older than `dump_after`."""

    def __init__(self, dump_after: int, py_spy: str | None, gdb: str | None, dumps: Path):
        super().__init__(daemon=True)
        self.dump_after = dump_after
        self.py_spy = py_spy
        self.gdb = gdb
        self.dumps = dumps
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.launch: str | None = None
        self.paths: list[Path] = []
        self.workers: list[threading.Thread] = []
        self.dumped: set[int] = set()
        self.launches_dumped = 0
        self.first = True
        self.full = True

    def begin(self, launch: str) -> None:
        with self.lock:
            self.launch = launch
            self.paths = []
            self.workers = []
            self.first = True
            self.full = self.launches_dumped < FULL_DUMP_LAUNCHES

    def end(self) -> list[Path]:
        with self.lock:
            workers = list(self.workers)
        for worker in workers:
            worker.join(timeout=240)
        with self.lock:
            self.launch = None
            return [path for path in self.paths if path.exists()]

    def run(self) -> None:
        own = os.getpid()
        while not self.stop_event.wait(5.0):
            with self.lock:
                if self.launch is None:
                    continue
            for row in processes():
                kind = role(row)
                if row["pid"] != own and row["etimes"] >= self.dump_after \
                        and kind in ("launcher", "rank"):
                    self.dump(row, kind)

    def dump(self, row: dict[str, Any], kind: str) -> None:
        with self.lock:
            if self.launch is None or row["pid"] in self.dumped:
                return
            self.dumped.add(row["pid"])
            path = self.dumps / ("%s-%s-%d.txt" % (self.launch, kind, row["pid"]))
            self.paths.append(path)
            first, self.first = self.first, False
            if first:
                self.launches_dumped += 1
            worker = threading.Thread(target=self._dump,
                                      args=(row, kind, path, first, self.full), daemon=True)
            self.workers.append(worker)
        worker.start()

    def _dump(self, row: dict[str, Any], kind: str, path: Path, first: bool, full: bool) -> None:
        pid = row["pid"]
        parts = ["==== %s pid %d, %d s old: %s" % (kind, pid, row["etimes"], row["args"])]
        if first:
            parts.append("-- processes (forest):\n" + capture(
                ["ps", "-ww", "-eo", "pid,ppid,pgid,sid,stat,etimes,pcpu,rss,wchan:24,args",
                 "--forest"], 20))
            parts.append("-- /proc/locks:\n" + capture(["cat", "/proc/locks"], 10))
            parts.append("-- /dev/shm:\n" + capture(["ls", "-la", "/dev/shm"], 10))
            parts.append("-- memory and load:\n" + capture(["free", "-m"], 10)
                         + capture(["cat", "/proc/loadavg"], 10))
        parts.append("-- " + threads_now(pid))
        parts.append("-- kernel stacks:\n" + capture(
            ["sudo", "-n", "sh", "-c",
             'for t in /proc/%d/task/*; do echo "thread ${t##*/} ($(cat $t/comm)):"; '
             'cat $t/stack; done' % pid], 30))
        parts.append("-- open files:\n" + capture(["ls", "-l", "/proc/%d/fd" % pid], 10))
        if kind == "rank" and self.py_spy:
            parts.append("-- py-spy dump:\n" + capture(
                ["sudo", "-n", "timeout", "-k", "5", "30", self.py_spy, "dump", "--pid",
                 str(pid)], 45))
            parts.append("-- py-spy dump --native:\n" + capture(
                ["sudo", "-n", "timeout", "-k", "5", "45", self.py_spy, "dump", "--native",
                 "--pid", str(pid)], 60))
        if full and self.gdb:
            parts.append("-- gdb, all threads:\n" + capture(
                ["sudo", "-n", "timeout", "-k", "5", "60", self.gdb, "-p", str(pid), "-batch",
                 "-nx", "-iex", "set debuginfod enabled off", "-ex", "set pagination off",
                 "-ex", "info threads", "-ex", "thread apply all bt 40"], 75))
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def tail(path: Path, count: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
    except OSError as error:
        return ["(unreadable: %s)" % error]


def verdict(record: Path) -> str:
    if not record.is_file():
        return "no record"
    try:
        status = json.loads(record.read_text(encoding="utf-8")).get("status", {})
    except (OSError, ValueError) as error:
        return "unreadable record (%s)" % error
    if status.get("ok"):
        return "ok"
    return "%s: %s" % (status.get("exception"), str(status.get("message"))[:300])


def stop_leftovers(rows: list[dict[str, Any]]) -> list[str]:
    """Stop by pid the processes of a launch that outlived it: SIGTERM, then SIGKILL."""
    notes = []
    for signum, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
        alive = []
        for row in rows:
            try:
                os.kill(row["pid"], signum)
                alive.append(row)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                notes.append("cannot signal %d: %s" % (row["pid"], error))
        deadline = time.monotonic() + wait
        while alive and time.monotonic() < deadline:
            time.sleep(0.2)
            current = {row["pid"]: row for row in processes()}
            alive = [row for row in alive if row["pid"] in current
                     and not current[row["pid"]]["stat"].startswith("Z")]
        notes.append("%s: %d still alive after %.0f s" % (signal.Signals(signum).name,
                                                           len(alive), wait))
        rows = alive
        if not rows:
            break
    return notes


def describe_stdin() -> str:
    try:
        info = os.fstat(0)
    except OSError as error:
        return "closed (%s)" % error
    kinds = [(stat.S_ISFIFO, "pipe"), (stat.S_ISCHR, "character device"),
             (stat.S_ISREG, "regular file"), (stat.S_ISSOCK, "socket"), (stat.S_ISDIR, "dir")]
    kind = next((name for test, name in kinds if test(info.st_mode)), "other")
    try:
        target = os.readlink("/proc/self/fd/0")
    except OSError:
        target = "?"
    poller = select.poll()
    poller.register(0, select.POLLIN | select.POLLHUP | select.POLLERR)
    events = poller.poll(0)
    names = []
    for _, mask in events:
        for flag, name in ((select.POLLIN, "readable"), (select.POLLHUP, "hang-up"),
                           (select.POLLERR, "error"), (select.POLLNVAL, "invalid")):
            if mask & flag:
                names.append(name)
    return "%s -> %s, isatty=%s, poll now: %s" % (kind, target, os.isatty(0),
                                                  ", ".join(names) or "nothing (a read blocks)")


def emit_block(title: str, body: list[str]) -> None:
    token = secrets.token_hex(16)
    print("::group::%s" % title)
    print("::stop-commands::%s" % token)
    for line in body:
        print(line)
    print("::%s::" % token)
    print("::endgroup::", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--copy", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--tutorial", action="append", required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    parser.add_argument("--watchdog", type=int, required=True)
    parser.add_argument("--dump-after", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = [arg for arg in args.runner_args if arg != "--"]
    name = "%s-%s" % (args.label, args.copy)
    output = Path(args.output).resolve() / name
    dumps = output / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    py_spy = shutil.which("py-spy")
    gdb = shutil.which("gdb")

    print("PROBE-ENV runner=%s (%s) contract=%s" % (args.label, args.runner, args.contract))
    print("PROBE-ENV stdin of the step: %s" % describe_stdin())
    print("PROBE-ENV py-spy=%s gdb=%s" % (py_spy, gdb))
    print(capture(["mpiexec", "--version"], 30).strip())
    print(capture(["sh", "-c", "nproc; uname -r; cat /proc/sys/kernel/yama/ptrace_scope; "
                   "free -m; df -h /tmp /dev/shm; ip -brief address; ulimit -a"], 30).strip())
    print("PROBE-ENV OMPI/PRTE/PMIX: %s" % " ".join(
        "%s=%s" % item for item in sorted(os.environ.items())
        if item[0].startswith(("OMPI_", "PRTE_", "PMIX_", "OPAL_"))), flush=True)

    watcher = Watcher(args.dump_after, py_spy, gdb, dumps)
    watcher.start()
    started = time.monotonic()
    budget = args.budget_minutes * 60.0
    results: list[dict[str, Any]] = []
    runner_errors = 0
    iteration = 0
    stop = False
    while not stop:
        iteration += 1
        for tutorial in args.tutorial:
            if time.monotonic() - started + args.watchdog + 90 > budget:
                stop = True
                break
            launch = "i%03d-%s" % (iteration, tutorial)
            out = output / launch
            stdout_path = output / (launch + ".stdout")
            command = [sys.executable, args.runner, "run", "--phase", "reference",
                       "--contract", args.contract, "--tutorial", tutorial,
                       "--watchdog", str(args.watchdog), "--output", str(out), *runner_args]
            before = {row["pid"] for row in processes()}
            watcher.begin(launch)
            clock = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stream:
                try:
                    code: Any = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                               timeout=args.watchdog + 120,
                                               check=False).returncode
                except subprocess.TimeoutExpired:
                    code = "runner outlived its watchdog by 120 s"
            wall = time.monotonic() - clock
            leftovers = [row for row in processes()
                         if row["pid"] not in before and row["pid"] != os.getpid()
                         and role(row) is not None]
            for row in leftovers:
                kind = role(row)
                if kind in ("launcher", "rank"):
                    watcher.dump(row, kind)
            dumped = watcher.end()
            notes = stop_leftovers(leftovers) if leftovers else []
            report_path = out / ("report-mpi-dim2-%s.json" % tutorial)
            entry: dict[str, Any] = {}
            try:
                entry = json.loads(report_path.read_text(encoding="utf-8"))["tutorials"][0]
            except (OSError, ValueError, KeyError, IndexError):
                pass
            ranks = [verdict(out / rank["record"]) if rank.get("record") else "no record"
                     for rank in entry.get("ranks", [])]
            result = {
                "iteration": iteration, "tutorial": tutorial, "runner_exit": code,
                "returncode": entry.get("returncode"), "timeout": entry.get("timeout"),
                "seconds": entry.get("seconds"), "wall": round(wall, 1), "ranks": ranks,
                "leftovers": ["%d %s" % (row["pid"], role(row)) for row in leftovers],
                "dumps": [path.name for path in dumped],
            }
            results.append(result)
            failed = entry.get("returncode") != 0
            if not entry:
                runner_errors += 1
                failed = True
            else:
                runner_errors = 0
            print("PROBE runner=%s copy=%s iteration=%d tutorial=%s exit=%s timeout=%s "
                  "seconds=%s ranks=%s leftovers=%d dumps=%d"
                  % (args.label, args.copy, iteration, tutorial, entry.get("returncode", code),
                     str(entry.get("timeout")).lower(), entry.get("seconds"),
                     "|".join(ranks) or "-", len(leftovers), len(dumped)), flush=True)
            if failed or leftovers or dumped:
                body = ["runner exit: %s, wall %.1f s" % (code, wall), "-- runner output:"]
                body += tail(stdout_path, 400)
                if args.label == "old" and entry:
                    for rank, text in enumerate(ranks):
                        body.append("rank %d: %s" % (rank, text))
                    body.append("-- launcher log (last %d lines):" % TAIL_LINES)
                    body += ["    " + line for line in tail(out / entry["launcher_log"],
                                                            TAIL_LINES)]
                    for rank in entry.get("ranks", []):
                        if rank.get("log"):
                            body.append("-- rank %d log (last %d lines):" % (rank["rank"],
                                                                              TAIL_LINES))
                            body += ["    " + line for line in tail(out / rank["log"],
                                                                    TAIL_LINES)]
                if leftovers:
                    body.append("-- processes of the launch alive after the runner returned:")
                    body += ["    %(pid)d ppid=%(ppid)d pgid=%(pgid)d %(etimes)ds %(stat)s "
                             "%(args)s" % row for row in leftovers]
                    body += ["    " + note for note in notes]
                for path in dumped:
                    body += path.read_text(encoding="utf-8", errors="replace").splitlines()
                emit_block("PROBE details %s %s exit=%s" % (name, launch,
                                                            entry.get("returncode", code)), body)
            else:
                shutil.rmtree(out, ignore_errors=True)
                stdout_path.unlink(missing_ok=True)
            if runner_errors >= 3:
                print("PROBE stop: three launches in a row gave no report", flush=True)
                stop = True
                break
    watcher.stop_event.set()

    summary: dict[str, Any] = {"runner": args.label, "copy": args.copy, "tutorials": {}}
    for tutorial in args.tutorial:
        rows = [row for row in results if row["tutorial"] == tutorial]
        good = [row["seconds"] for row in rows if row["returncode"] == 0]
        summary["tutorials"][tutorial] = {
            "launches": len(rows), "passed": len(good),
            "timeouts": sum(1 for row in rows if row["timeout"]),
            "failed_other": sum(1 for row in rows if row["returncode"] not in (0, None)
                                and not row["timeout"]),
            "no_report": sum(1 for row in rows if row["returncode"] is None),
            "with_leftovers": sum(1 for row in rows if row["leftovers"]),
            "seconds_median": round(statistics.median(good), 1) if good else None,
            "seconds_max": round(max(good), 1) if good else None,
        }
    totals = {key: sum(row[key] for row in summary["tutorials"].values())
              for key in ("launches", "passed", "timeouts", "failed_other", "no_report",
                          "with_leftovers")}
    summary["totals"] = totals
    summary["failures"] = [row for row in results if row["returncode"] != 0]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print("::notice title=MG8 probe, %s runner, copy %s::launches=%d passed=%d timeouts=%d "
          "other failures=%d no report=%d with leftovers=%d"
          % (args.label, args.copy, totals["launches"], totals["passed"], totals["timeouts"],
             totals["failed_other"], totals["no_report"], totals["with_leftovers"]))
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as stream:
            stream.write("### MG8 probe, %s runner, copy %s\n\n" % (args.label, args.copy))
            stream.write("| Tutorial | Launches | Passed | Timeouts | Other failures | "
                         "Leftovers | Median s | Max s |\n|--|--|--|--|--|--|--|--|\n")
            for tutorial, row in summary["tutorials"].items():
                stream.write("| %s | %d | %d | %d | %d | %d | %s | %s |\n" % (
                    tutorial, row["launches"], row["passed"], row["timeouts"],
                    row["failed_other"], row["with_leftovers"], row["seconds_median"],
                    row["seconds_max"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
