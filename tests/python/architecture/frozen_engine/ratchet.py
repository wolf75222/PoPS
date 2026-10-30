"""Exception lists that only shrink (spec 0.3, MG6), shared by the engine guards of Linear N2.

An exception list is a text file with one site per line::

    <path>:<line> <term> <issue> | <excerpt of the source line>

``<line>`` is 0 when the term is in the file path itself. ``<issue>`` is the Linear issue that
removes the site, or the issues joined by ``+`` when the spec names several (``G04+G07`` for the
analytic-expression VM, section 18.3). Blank lines and lines starting with ``#`` are comments.

A guard compares the sites that its scan finds with the list. Within each ``(path, term)``, sites
and entries pair by excerpt, so the identity of a site is its path, its term and its excerpt (for
CF2 the spelling of the term on its line, ``scan.cf2_sites_in_text``):

* a site that no entry of its ``(path, term)`` matches is a new occurrence, and fails, even when
  a site of the same term left the same file in the same PR;
* an entry that no site matches is an exception that has become useless, and fails until its line
  is removed in the same PR (MG6).

The line numbers locate each site for a reader and for the refresh tool; the comparison does not key
on them, so an edit elsewhere in a file that shifts its lines does not fail the guard of an
unrelated PR. ``refresh_exceptions.py`` rewrites the line numbers and refuses a site the list does
not hold. Since a list is a file that a PR can edit by hand, CI also scans the base of the PR with
the same scanner and fails on any site of the head that the base does not hold (``new_sites``,
``refresh_exceptions.py --check-base``); a vocabulary change adds the same sites to both scans.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import pathlib
import re

# The issues that take over a site of the b6 exception lists (spec 0.3, sections 18.3 and 18.4).
KNOWN_ISSUES = frozenset({"N3", "N4", "N7", "N8", "N9", "N12", "G02", "G04", "G05", "G06", "G07",
                          "G10", "G11", "G12", "G13"})

_ENTRY = re.compile(r"^(?P<path>\S+):(?P<line>\d+) (?P<term>\S+)"
                    r" (?P<issue>[A-Z0-9]+(?:\+[A-Z0-9]+)*) \| ?(?P<excerpt>.*)$")


def issue_known(issue: str) -> bool:
    """Every issue of an entry's ``<issue>`` field (``G04+G07`` names two) is a known issue."""
    return all(part in KNOWN_ISSUES for part in issue.split("+"))


@dataclasses.dataclass(frozen=True, order=True)
class Entry:
    path: str
    line: int
    term: str
    issue: str
    excerpt: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.term

    def render(self) -> str:
        return "%s:%d %s %s | %s" % (self.path, self.line, self.term, self.issue, self.excerpt)


class MalformedList(ValueError):
    pass


def read_list(path: pathlib.Path) -> tuple[list[str], list[Entry]]:
    """(header comment lines, entries) of an exception list."""
    header: list[str] = []
    entries: list[Entry] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            if not entries:
                header.append(raw)
            continue
        match = _ENTRY.match(raw)
        if match is None:
            raise MalformedList("%s:%d is not '<path>:<line> <term> <issue> | <excerpt>': %r"
                                % (path.name, number, raw))
        entries.append(Entry(match["path"], int(match["line"]), match["term"], match["issue"],
                             match["excerpt"]))
    return header, entries


def write_list(path: pathlib.Path, header: list[str], entries) -> None:
    body = "\n".join(entry.render() for entry in sorted(entries))
    path.write_text("\n".join(header).rstrip("\n") + "\n\n" + body + "\n", encoding="utf-8")


def counts(items) -> collections.Counter:
    return collections.Counter(item.key for item in items)


@dataclasses.dataclass
class Comparison:
    new: dict  # (path, term) -> (found sites that no entry matches, listed count)
    useless: dict  # (path, term) -> (listed entries that no site matches, found count)

    @property
    def clean(self) -> bool:
        return not self.new and not self.useless


def _by_key(items) -> dict:
    grouped = collections.defaultdict(list)
    for item in items:
        grouped[item.key].append(item)
    return grouped


def _line(item) -> int:
    return item.line


def _pair(items, others, item_hint, other_hint):
    """(items left over, others left over) once each item is paired with an other of the same
    excerpt, first with one whose hint is equal too (its line, or its source line), so that the
    item left over is the one that is new rather than one that shifted."""
    waiting = collections.defaultdict(list)
    for other in others:
        waiting[(other.excerpt, other_hint(other))].append(other)
    unpaired = []
    for item in items:
        bucket = waiting.get((item.excerpt, item_hint(item)))
        if bucket:
            bucket.pop(0)
        else:
            unpaired.append(item)
    by_excerpt = collections.defaultdict(list)
    for bucket in waiting.values():
        for other in bucket:
            by_excerpt[other.excerpt].append(other)
    items_left = []
    for item in unpaired:
        bucket = by_excerpt.get(item.excerpt)
        if bucket:
            bucket.pop(0)
        else:
            items_left.append(item)
    return sorted(items_left), sorted(other for bucket in by_excerpt.values() for other in bucket)


def compare(found, listed, found_hint=_line, listed_hint=_line) -> Comparison:
    """Compare the sites a scan found with the entries of a list (objects with .key, .excerpt).

    Within each (path, term), a site and an entry pair when their excerpts are equal; a site left
    over is new, an entry left over is useless. A swap (one site of a term removed, another added
    in the same file) therefore fails, while a site whose line only shifted does not. The hints
    choose which of several sites of one spelling is reported as new.
    """
    found_by_key, listed_by_key = _by_key(found), _by_key(listed)
    new, useless = {}, {}
    for key in sorted(set(found_by_key) | set(listed_by_key)):
        found_sites, listed_entries = found_by_key.get(key, []), listed_by_key.get(key, [])
        unlisted, unfound = _pair(found_sites, listed_entries, found_hint, listed_hint)
        if unlisted:
            new[key] = (unlisted, len(listed_entries))
        if unfound:
            useless[key] = (unfound, len(found_sites))
    return Comparison(new, useless)


def moves(found, listed):
    """(moves, unexplained): whether the sites that the list does not hold have only moved.

    A site left over by ``compare`` counts as moved when an entry left over in another file has its
    term and its excerpt: a file renamed or split by a slice. ``moves`` pairs each such entry with
    one site; ``unexplained`` holds the sites left over that no entry explains, among them the new
    occurrences. The refresh tool accepts the moves; CI checks them against the source lines of the
    base (``new_sites``).
    """
    comparison = compare(found, listed)
    pool = collections.defaultdict(list)  # (term, excerpt) -> entries that may have moved
    for entries, _ in comparison.useless.values():
        for entry in entries:
            pool[(entry.term, entry.excerpt)].append(entry)
    pairs, unexplained = [], []
    for sites, _ in comparison.new.values():
        for site in sites:
            candidates = pool.get((site.term, site.excerpt), [])
            source = next((entry for entry in candidates if entry.path != site.path), None)
            if source is None:
                unexplained.append(site)
            else:
                candidates.remove(source)
                pairs.append((source, site))
    return pairs, unexplained


def new_sites(base, head, base_root: pathlib.Path, head_root: pathlib.Path):
    """(moves, new): the sites of the head tree that the base tree does not hold.

    @p base and @p head are the sites (or entries) that one scanner finds under @p base_root and
    @p head_root. They pair within each (path, term) by excerpt, as in ``compare``. A head site left
    over has moved when a base site left over in another file has its term, its excerpt and its
    source line: a file renamed or split keeps its lines. Every other head site left over is new.
    """
    @functools.lru_cache(maxsize=None)
    def lines_of(root: pathlib.Path, path: str) -> tuple[str, ...]:
        try:
            return tuple((root / path).read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            return ()

    def source_line(root: pathlib.Path, site) -> str:
        if site.line == 0:
            return "(path)"
        lines = lines_of(root, site.path)
        return " ".join(lines[site.line - 1].split()) if site.line <= len(lines) else ""

    comparison = compare(head, base, lambda site: source_line(head_root, site),
                         lambda site: source_line(base_root, site))
    pool = collections.defaultdict(list)  # (term, excerpt, source line) -> base sites that went
    for sites, _ in comparison.useless.values():
        for site in sites:
            pool[(site.term, site.excerpt, source_line(base_root, site))].append(site)
    moved, added = [], []
    for sites, _ in comparison.new.values():
        for site in sites:
            candidates = pool.get((site.term, site.excerpt, source_line(head_root, site)), [])
            origin = next((other for other in candidates if other.path != site.path), None)
            if origin is None:
                added.append(site)
            else:
                candidates.remove(origin)
                moved.append((origin, site))
    return moved, added


def describe(comparison: Comparison, list_name: str, rule: str) -> str:
    lines = []
    if comparison.new:
        lines.append("%s: %d new occurrence(s) of knowledge that %s refuses (MG6: the %s list can "
                     "only shrink; remove the occurrence, do not list it):"
                     % (rule, sum(len(sites) for sites, _ in comparison.new.values()), rule,
                        list_name))
        for (path, term), (sites, count) in comparison.new.items():
            lines.append("  %s %s: %d site(s) not in the list (%d listed for this file and term)"
                         % (path, term, len(sites), count))
            for site in sites:
                lines.append("    %s" % site.render())
    if comparison.useless:
        lines.append("%s: exception(s) that have become useless (MG6: remove them from %s in the "
                     "same PR, or run refresh_exceptions.py):" % (rule, list_name))
        for (path, term), (entries, count) in comparison.useless.items():
            lines.append("  %s %s: %d listed entry(ies) match no site (%d site(s) found)"
                         % (path, term, len(entries), count))
            for entry in entries:
                lines.append("    %s" % entry.render())
    return "\n".join(lines)
