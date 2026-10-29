"""Rewrite an exception list of the engine guards from the current tree, without letting it grow.

    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf2
    python -m tests.python.architecture.frozen_engine.refresh_exceptions cf2 --allow-moves

The default refuses any (path, term) whose site count would rise: it only drops sites that have
disappeared and updates the line numbers and excerpts of the others (MG6). ``--allow-moves`` accepts
a site that moved to another file (a file renamed or split by a slice) as long as no term's total
rises. ``--bootstrap`` writes the list from scratch; it was used once, at b6e64c1d, and a PR that
uses it again must say why.
"""
from __future__ import annotations

import argparse
import collections
import sys

from . import cf2, ratchet

LISTS = {
    "cf2": (cf2.EXCEPTIONS, lambda: cf2.current_entries()[0]),
}


def _term_totals(entries) -> collections.Counter:
    return collections.Counter(entry.term for entry in entries)


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
    if not args.bootstrap:
        if args.allow_moves:
            grown = {term: (count, _term_totals(listed)[term])
                     for term, count in _term_totals(found).items()
                     if count > _term_totals(listed)[term]}
        else:
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
