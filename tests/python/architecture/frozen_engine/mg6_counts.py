"""MG6 by directory and by issue: the count of the sites left in an exception list.

    python -m tests.python.architecture.frozen_engine.mg6_counts            # CF2, markdown
    python -m tests.python.architecture.frozen_engine.mg6_counts --depth 4
    python -m tests.python.architecture.frozen_engine.mg6_counts --list cf8 --json

MG6 (spec 0.3, review of 11/11, P8): the shrinking of the exception lists is followed by directory
and by issue. An issue closes a whole directory of the core paths, by moving it out of them or
deleting it (``include/pops/physics/``, ``python/pops/moments/``), and the plan shows the count of
sites left for each issue; this script prints that count, which each lot copies into the plan.

A directory is the first ``--depth`` components of a path (3 by default: ``include/pops/physics``,
``python/pops/moments``, ``src/runtime/amr``); a file with fewer directories counts in its own
directory. For each directory the script gives the sites of each issue, and names the issue that
can close it alone when it owns every site there. It reads the list file only.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import posixpath
import sys

from . import ratchet, scan

LISTS = {name: scan.HERE / ("%s_exceptions.txt" % name) for name in ("cf2", "cf0b", "cf7", "cf8")}
DEFAULT_DEPTH = 3


def directory_of(path: str, depth: int = DEFAULT_DEPTH) -> str:
    """The directory of @p path that MG6 follows: its first @p depth components, or its own."""
    parts = posixpath.dirname(path).split("/") if "/" in path else ["."]
    return "/".join(parts[:depth]) + "/"


def counts(entries, depth: int = DEFAULT_DEPTH) -> dict:
    """{"total", "by_issue": {issue: n}, "by_directory": {dir: {"total", "by_issue", "closer"}}},
    issues and directories in decreasing order of sites, ties by name."""
    by_issue = collections.Counter(entry.issue for entry in entries)
    by_directory: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for entry in entries:
        by_directory[directory_of(entry.path, depth)][entry.issue] += 1

    def ordered(counter) -> dict:
        return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))

    directories = {}
    for directory, issues in sorted(by_directory.items(),
                                    key=lambda item: (-sum(item[1].values()), item[0])):
        directories[directory] = {
            "total": sum(issues.values()),
            "by_issue": ordered(issues),
            "closer": next(iter(issues)) if len(issues) == 1 else None,
        }
    return {"total": len(entries), "by_issue": ordered(by_issue), "by_directory": directories}


def markdown(name: str, result: dict, depth: int) -> str:
    lines = ["%s exceptions: %d sites." % (name.upper(), result["total"]), "",
             "| Issue | Sites | Directories it closes alone |", "|---|---:|---|"]
    alone = collections.defaultdict(list)
    for directory, row in result["by_directory"].items():
        if row["closer"]:
            alone[row["closer"]].append("`%s` (%d)" % (directory, row["total"]))
    for issue, count in result["by_issue"].items():
        lines.append("| %s | %d | %s |" % (issue, count, ", ".join(alone[issue]) or "none"))
    lines += ["", "| Directory (depth %d) | Sites | By issue |" % depth, "|---|---:|---|"]
    for directory, row in result["by_directory"].items():
        lines.append("| `%s` | %d | %s |" % (directory, row["total"], ", ".join(
            "%s %d" % item for item in row["by_issue"].items())))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", choices=sorted(LISTS), default="cf2")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args(argv)
    if args.depth < 1:
        parser.error("--depth takes a number of directories >= 1")
    _, entries = ratchet.read_list(pathlib.Path(LISTS[args.list]))
    result = counts(entries, args.depth)
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        sys.stdout.write(markdown(args.list, result, args.depth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
