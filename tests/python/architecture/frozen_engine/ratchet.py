"""Exception lists that only shrink (spec 0.3, MG6), shared by the engine guards of Linear N2.

An exception list is a text file with one site per line::

    <path>:<line> <term> <issue> | <excerpt of the source line>

``<line>`` is 0 when the term is in the file path itself. ``<issue>`` is the Linear issue that
removes the site, or the issues joined by ``+`` when the spec names several (``G04+G07`` for the
analytic-expression VM, section 18.3). Blank lines and lines starting with ``#`` are comments.

A guard compares the sites that its scan finds with the list, counted per ``(path, term)``:

* more sites than listed is a new occurrence, and fails;
* fewer sites than listed is an exception that has become useless, and fails until its line is
  removed in the same PR (MG6).

The line numbers locate each site for a reader and for the refresh tool; the comparison does not key
on them, so an edit elsewhere in a file that shifts its lines does not fail the guard of an
unrelated PR. ``refresh_exceptions.py`` rewrites the line numbers and excerpts and refuses to raise
any count.
"""
from __future__ import annotations

import collections
import dataclasses
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
    new: dict  # (path, term) -> (found sites, listed count)
    useless: dict  # (path, term) -> (listed entries, found count)

    @property
    def clean(self) -> bool:
        return not self.new and not self.useless


def compare(found, listed) -> Comparison:
    """Compare the sites a scan found (objects with .key) with the entries of a list."""
    found_by_key = collections.defaultdict(list)
    for site in found:
        found_by_key[site.key].append(site)
    listed_by_key = collections.defaultdict(list)
    for entry in listed:
        listed_by_key[entry.key].append(entry)
    new, useless = {}, {}
    for key in sorted(set(found_by_key) | set(listed_by_key)):
        found_sites, listed_entries = found_by_key.get(key, []), listed_by_key.get(key, [])
        if len(found_sites) > len(listed_entries):
            new[key] = (found_sites, len(listed_entries))
        elif len(found_sites) < len(listed_entries):
            useless[key] = (listed_entries, len(found_sites))
    return Comparison(new, useless)


def moves(found, listed):
    """(moves, unexplained): whether a scan that grew some (path, term) counts is only a move.

    A site counts as moved when a listed site of the same term disappears from one file and a site
    with the same excerpt appears in another: a file renamed or split by a slice. ``moves`` pairs
    each such listed entry with the site that replaces it, never more sites from a (path, term)
    than it lost. ``unexplained`` holds, for each (path, term) count that grew beyond what the moves
    cover, its sites whose excerpt is not listed there; the new occurrences are among them.
    """
    found_by_key = collections.defaultdict(list)
    for site in found:
        found_by_key[site.key].append(site)
    listed_by_key = collections.defaultdict(list)
    for entry in listed:
        listed_by_key[entry.key].append(entry)

    def unmatched(sites, others):
        left = collections.Counter(other.excerpt for other in others)
        out = []
        for site in sites:
            if left[site.excerpt]:
                left[site.excerpt] -= 1
            else:
                out.append(site)
        return out

    lost = {key: len(entries) - len(found_by_key.get(key, ()))
            for key, entries in listed_by_key.items()}
    gone = collections.defaultdict(list)  # (term, excerpt) -> listed entries that may have moved
    for key, entries in listed_by_key.items():
        if lost[key] > 0:
            for entry in unmatched(entries, found_by_key.get(key, ())):
                gone[(entry.term, entry.excerpt)].append(entry)
    pairs, unexplained = [], []
    for key, sites in sorted(found_by_key.items()):
        surplus = len(sites) - len(listed_by_key.get(key, ()))
        if surplus <= 0:
            continue
        candidates = unmatched(sites, listed_by_key.get(key, ()))
        for site in candidates:
            pool = gone.get((site.term, site.excerpt), [])
            source = next((entry for entry in pool if lost[entry.key] > 0), None)
            if surplus and source is not None:
                pool.remove(source)
                lost[source.key] -= 1
                pairs.append((source, site))
                surplus -= 1
        if surplus:
            paired = {id(new) for _, new in pairs}
            unexplained.extend(site for site in candidates if id(site) not in paired)
    return pairs, unexplained


def describe(comparison: Comparison, list_name: str, rule: str) -> str:
    lines = []
    if comparison.new:
        lines.append("%s: %d new occurrence(s) of knowledge that %s refuses (MG6: the %s list can "
                     "only shrink; remove the occurrence, do not list it):"
                     % (rule, sum(len(sites) - count for sites, count in comparison.new.values()),
                        rule, list_name))
        for (path, term), (sites, count) in comparison.new.items():
            lines.append("  %s %s: %d site(s), %d listed" % (path, term, len(sites), count))
            for site in sites:
                lines.append("    %s" % site.render())
    if comparison.useless:
        lines.append("%s: exception(s) that have become useless (MG6: remove them from %s in the "
                     "same PR, or run refresh_exceptions.py):" % (rule, list_name))
        for (path, term), (entries, count) in comparison.useless.items():
            lines.append("  %s %s: %d listed, %d site(s) left" % (path, term, len(entries), count))
            for entry in entries:
                lines.append("    %s" % entry.render())
    return "\n".join(lines)
