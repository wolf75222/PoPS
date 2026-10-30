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
b6e64c1d, again for the terms of the early audit of N2 (29/10), again for the error-text fix and
the solve algorithms of the independent review of N2 (30/10), and again for one site per token
occurrence (its follow-up review, 30/10). A PR that uses it again must say why (the CF0b, CF7 and
CF8 lists were written with it at b6e64c1d in lot 2 of Linear N2, once more, before they merged,
with the sites that the audit of lot 2 found, and again for the independent review of lot 2, when
their excerpts became spellings and their sites occurrences).

``--check-base REV`` writes nothing. It scans the tree of REV (a temporary worktree) and the
current tree with the current scanner of every list, and fails on any site of the current tree that
REV does not hold, unless the site moved from another file with its source line. It also scans the
current tree with the scanner of REV (``--sites-json``, which prints ``{list: [[path, line, term,
excerpt], ...]}``) and fails where the current scanner sees fewer sites of a (path, term) than the
scanner of REV, unless the PR declares that narrowing, with its reason, in
``accepted_narrowings.txt``: a change to a guard that hides a site (a wider generic sense, a term
taken out of the vocabulary, a path taken out of CF0) says so in a reviewed place. CI runs it
against the base of each PR, so a list edited by hand cannot let a new site in, a site removed by a
PR leaves no room for another, and a list cannot shrink while the code stays.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
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

# The narrowings that PRs declare, relative to the root of a tree.
NARROWINGS = scan.HERE.relative_to(scan.REPO_ROOT) / "accepted_narrowings.txt"
_NARROWING = re.compile(
    r"^(?P<list>\S+) (?P<path>\S+) (?P<term>\S+) (?P<count>[1-9][0-9]*) \| (?P<reason>\S.*)$")


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


def read_narrowings(path: pathlib.Path) -> list[tuple[str, str, str, int, str]]:
    """(list, path, term, count, reason) of every narrowing that a file declares; none if absent."""
    if not path.exists():
        return []
    declared = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        match = _NARROWING.match(raw)
        if match is None:
            raise ratchet.MalformedList("%s:%d is not '<list> <path> <term> <count> | <reason>': %r"
                                        % (path.name, number, raw))
        declared.append((match["list"], match["path"], match["term"], int(match["count"]),
                         match["reason"]))
    return declared


def narrowings(before, after) -> dict:
    """{(path, term): (sites in @p before, sites in @p after)} where @p after has fewer."""
    counted_before, counted_after = ratchet.counts(before), ratchet.counts(after)
    return {key: (count, counted_after.get(key, 0)) for key, count in sorted(counted_before.items())
            if count > counted_after.get(key, 0)}


def base_scanner_sites(base_root: pathlib.Path, root: pathlib.Path):
    """{list name: sites} that the scanner of the tree at @p base_root finds in the tree at @p root,
    or None when that tree has no scanner that reports them (it predates the narrowing check)."""
    if not (base_root / NARROWINGS.parent / "refresh_exceptions.py").exists():
        return None
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONSAFEPATH")}  # import the scanner of the base
    completed = subprocess.run(
        [sys.executable, "-m", __package__ + ".refresh_exceptions", "--sites-json", str(root)],
        cwd=base_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode == 2:  # argparse: a scanner without --sites-json
        return None
    if completed.returncode:
        raise RuntimeError("the scanner of the base failed on this tree:\n" + completed.stderr)
    return {name: [ratchet.Entry(path, line, term, "-", excerpt)
                   for path, line, term, excerpt in sites]
            for name, sites in json.loads(completed.stdout).items()}


def _declared_in_this_pr(base_root: pathlib.Path, root: pathlib.Path) -> collections.Counter:
    """(list, path, term) -> the number of sites that the lines added to accepted_narrowings.txt
    since the base declare; the lines already in the base record earlier PRs."""
    added = collections.Counter(read_narrowings(root / NARROWINGS))
    added.subtract(collections.Counter(read_narrowings(base_root / NARROWINGS)))
    declared = collections.Counter()
    for (name, path, term, count, _), copies in added.items():
        if copies > 0:
            declared[(name, path, term)] += count * copies
    return declared


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
            try:
                base_scans = base_scanner_sites(base_root, root)
            except RuntimeError as error:
                print(error, file=sys.stderr)
                return 2
            declared = _declared_in_this_pr(base_root, root)
            for name in names:
                path, current = LISTS[name]
                head = current(root)
                moved, added = ratchet.new_sites(current(base_root), head, base_root, root)
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
                failed |= _check_narrowings(name, rev, (base_scans or {}).get(name), head,
                                            declared)
            unchecked = sorted(key for key in declared if key[0] not in names)
            if unchecked:
                failed = True
                print("%s: declared narrowings of lists that were not checked: %s"
                      % (NARROWINGS.name, unchecked), file=sys.stderr)
        finally:
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force",
                            str(base_root)], check=False)
    return 1 if failed else 0


def _check_narrowings(name, rev, base_scan, head, declared) -> bool:
    """True when the narrowings of list @p name differ from those the PR declares."""
    mine = {(path, term): count for (list_name, path, term), count in declared.items()
            if list_name == name}
    if base_scan is None:
        print("%s: the scanner of %s does not report this list; no narrowing to check"
              % (name, rev))
        if mine:
            print("%s: narrowings declared for %s, which %s cannot check: %s"
                  % (NARROWINGS.name, name, rev, sorted(mine)), file=sys.stderr)
        return bool(mine)
    narrowed = narrowings(base_scan, head)
    if not narrowed and not mine:
        print("%s: the scanner of this tree hides no site that the scanner of %s sees"
              % (name, rev))
        return False
    hidden = ratchet.compare(head, base_scan).useless
    failed = False
    for key in sorted(set(narrowed) | set(mine)):
        before, after = narrowed.get(key, (0, 0))
        drop, said = before - after, mine.get(key, 0)
        if drop == said:
            print("%s: declared narrowing %s %s: %d -> %d site(s)" % (name, *key, before, after))
            continue
        failed = True
        if drop < said:
            print("%s: %s declares %d hidden site(s) of %s %s; the scanner of this tree hides %d "
                  "that the scanner of %s sees" % (name, NARROWINGS.name, said, *key, drop, rev),
                  file=sys.stderr)
            continue
        print("%s: the scanner of this tree sees %d site(s) of %s %s where the scanner of %s sees "
              "%d; declare the %d hidden site(s) in %s with the reason, or keep the scanner as it "
              "was" % (name, after, *key, rev, before, drop - said, NARROWINGS.name),
              file=sys.stderr)
        for site in hidden.get(key, ((), 0))[0]:
            print("  %s" % site.render(), file=sys.stderr)
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("list", nargs="?", choices=sorted(LISTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-moves", action="store_true")
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--check-base", metavar="REV")
    mode.add_argument("--sites-json", metavar="ROOT", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.sites_json:
        json.dump({name: [[site.path, site.line, site.term, site.excerpt]
                          for site in current(args.sites_json.resolve())]
                   for name, (_, current) in LISTS.items()}, sys.stdout)
        return 0
    if args.check_base:
        return check_base(args.check_base, [args.list] if args.list else sorted(LISTS))
    if not args.list:
        parser.error("name the list to refresh (or use --check-base REV)")
    return refresh(args.list, args.allow_moves, args.bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
