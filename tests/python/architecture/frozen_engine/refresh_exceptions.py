"""Rewrite an exception list of the engine guards from the current tree, without letting it grow.

    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf2
    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf2 --allow-moves
    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf0b   # also cf7, cf8
    python -m tests.python.architecture.frozen_engine.refresh_exceptions --check-base HEAD^1

The default refuses any site that the list does not hold (same path, term and excerpt, ratchet.py):
it only drops sites that have disappeared and updates the line numbers and owners of the others
(MG6). ``--allow-moves`` also accepts a site that moved to another file (a file renamed or split by
a slice): an entry with the same term and excerpt must disappear from one file, and each move is
printed with the issues of both files. ``--bootstrap`` writes the list from scratch; it was used at
b6e64c1d, again for the terms of the early audit of N2 (29/10), and again for the error-text fix
and the solve algorithms of the independent review of N2 (30/10). A PR that uses it again must say
why (the CF0b, CF7 and CF8 lists were written with it at b6e64c1d in lot 2 of Linear N2, once
more, before they merged, with the sites that the audit of lot 2 found, and again when their
excerpts became the spellings that the ratchet pairs by, for the independent review of lot 2).

``--check-base REV`` writes nothing: it scans the tree of REV (a temporary worktree) and the
current tree with the current scanner of every list, and fails on any site of the current tree that
REV does not hold, unless the site moved from another file with its source line. CI runs it
against the base of each PR, so a list edited by hand cannot let a new site in, and a site removed
by a PR leaves no room for another.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

from . import cf0b, cf2, cf7, cf8, ratchet, scan

# name -> (list file, scanner of a tree: root -> entries)
LISTS = {
    "cf0b": (cf0b.EXCEPTIONS, lambda root: cf0b.current_entries(root)),
    "cf2": (cf2.EXCEPTIONS, lambda root: cf2.current_entries(root)[0]),
    "cf7": (cf7.EXCEPTIONS, lambda root: cf7.current_entries(root)[0]),
    "cf8": (cf8.EXCEPTIONS, lambda root: cf8.current_entries(root)),
}


def refresh(name: str, allow_moves: bool, bootstrap: bool) -> int:
    path, current = LISTS[name]
    header, listed = ratchet.read_list(path)
    found = current(scan.REPO_ROOT)
    if allow_moves:
        pairs, unexplained = ratchet.moves(found, listed)
        if unexplained:
            print("refused: no move explains these sites (MG6); remove the new occurrences among "
                  "them instead:", file=sys.stderr)
            for site in unexplained:
                print("  %s" % site.render(), file=sys.stderr)
            return 1
        for before, after in pairs:
            print("moved: %s %s:%d (%s) -> %s:%d (%s)" % (before.term, before.path, before.line,
                                                      before.issue, after.path, after.line,
                                                      after.issue))
    elif not bootstrap:
        comparison = ratchet.compare(found, listed)
        if comparison.new:
            print("refused: the list does not hold these sites (MG6); remove the occurrences "
                  "instead:", file=sys.stderr)
            for sites, _ in comparison.new.values():
                for site in sites:
                    print("  %s" % site.render(), file=sys.stderr)
            return 1
    ratchet.write_list(path, header, found)
    print("%s: %d entries (was %d)" % (path.name, len(found), len(listed)))
    return 0


def check_base(rev: str, names) -> int:
    root = scan.REPO_ROOT
    failed = False
    with tempfile.TemporaryDirectory(prefix="pops-guard-base-") as scratch:
        base_root = pathlib.Path(scratch) / "base"
        added_worktree = subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach",
                                         "--quiet", str(base_root), rev])
        if added_worktree.returncode:
            print("cannot check out %s; CI fetches the base with fetch-depth 2" % rev,
                  file=sys.stderr)
            return 2
        try:
            for name in names:
                path, current = LISTS[name]
                moved, added = ratchet.new_sites(current(base_root), current(root), base_root, root)
                for before, after in moved:
                    print("%s: moved %s %s:%d -> %s:%d" % (name, before.term, before.path,
                                                          before.line, after.path, after.line))
                if added:
                    failed = True
                    print("%s: %d site(s) that %s does not hold (MG6: a list only shrinks; remove "
                          "the occurrence, do not list it):" % (path.name, len(added), rev),
                          file=sys.stderr)
                    for site in added:
                        print("  %s" % site.render(), file=sys.stderr)
                else:
                    print("%s: no site beyond %s" % (path.name, rev))
        finally:
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force",
                            str(base_root)], check=False)
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("list", nargs="?", choices=sorted(LISTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-moves", action="store_true")
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--check-base", metavar="REV")
    args = parser.parse_args(argv)
    if args.check_base:
        return check_base(args.check_base, [args.list] if args.list else sorted(LISTS))
    if not args.list:
        parser.error("name the list to refresh (or use --check-base REV)")
    return refresh(args.list, args.allow_moves, args.bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
