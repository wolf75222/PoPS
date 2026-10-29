"""Rewrite an exception list of the engine guards from the current tree, without letting it grow.

    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf2
    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf2 --allow-moves

The default refuses any (path, term) whose site count would rise: it only drops sites that have
disappeared and updates the line numbers and excerpts of the others (MG6). ``--allow-moves`` also
accepts a site that moved to another file (a file renamed or split by a slice): the same excerpt
must disappear from one file and appear in the other, and each move is printed with the issues of
both files. ``--bootstrap`` writes the list from scratch; it was used at b6e64c1d and again for the
terms of the early audit of N2 (29/10), and a PR that uses it again must say why.
"""
from __future__ import annotations

import argparse
import sys

from . import cf2, ratchet

LISTS = {
    "cf2": (cf2.EXCEPTIONS, lambda: cf2.current_entries()[0]),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("list", choices=sorted(LISTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-moves", action="store_true")
    mode.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args(argv)

    path, current = LISTS[args.list]
    header, listed = ratchet.read_list(path)
    found = current()
    if args.allow_moves:
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
    elif not args.bootstrap:
        listed_counts = ratchet.counts(listed)
        grown = {key: (count, listed_counts[key])
                 for key, count in ratchet.counts(found).items() if count > listed_counts[key]}
        if grown:
            print("refused: the list would grow (MG6); remove these occurrences instead:",
                  file=sys.stderr)
            for key, (count, before) in sorted(grown.items()):
                print("  %s: %d site(s), %d listed" % (key, count, before), file=sys.stderr)
            return 1
    ratchet.write_list(path, header, found)
    print("%s: %d entries (was %d)" % (path.name, len(found), len(listed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
