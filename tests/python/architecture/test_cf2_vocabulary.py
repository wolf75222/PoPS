"""CF2 (spec 0.3, section 16.2; Linear N2, review of 11/11, P8 and P3): the refused vocabulary is a
closed, versioned file of the repository that the test reads, and MG6 is followed by directory.

CF2 is a tripwire. It reads words, so it finds a name, not a formula coded without its name, which
CF4 finds by swapping bodies; the evidence of G3 is CF1, CF3, CF4 and CF6.

- Closed: ``frozen_engine/cf2_vocabulary.toml`` is the vocabulary. ``scan.py`` builds every term,
  exact literal, exact name, case pattern and ``not_after`` rule from it and writes none of its own,
  and each row of the file carries the finding that motivates it.
- Versioned: every change adds a ``[[versions]]`` entry, one more than the last, whose digest is
  the digest of the vocabulary as written; a change without its entry fails here.
- No exception for the name of a quadrature rule or of a base dimension (P3, P10; D0, S3): the core
  knows ``pops.Quadrature`` and the base of the model only.
- MG6 by directory and by issue: ``frozen_engine/mg6_counts.py`` counts the sites of an exception
  list per directory of the core paths and per issue, the count that each lot copies into the plan.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import collections
import copy
import datetime
import json
import re

import pytest

from tests.python.architecture.frozen_engine import mg6_counts, ratchet, scan

VOCABULARY = scan.load_vocabulary()
SECTIONS = {"terms": "words", "literals": "strings", "names": "names"}
# The module-level names of scan.py that hold the vocabulary: each is built from the file.
DERIVED = ("CF2_VOCABULARY", "EXACT_LITERALS", "EXACT_NAMES", "CASE_IDENTIFIER", "NOT_AFTER")
# The spellings of named quadrature rules and of the base dimensions of S3 (P3, P10).
QUADRATURE_WORDS = frozenset({"gauss", "legendre", "hermite", "lobatto", "laguerre", "radau"})
BASE_DIMENSIONS = frozenset({"length", "time", "mass", "charge", "temperature", "amount"})


def _rows():
    for section, field in SECTIONS.items():
        for row in VOCABULARY[section]:
            yield section, row, row[field]
    for section in ("case_identifier", "axis_link"):
        yield section, VOCABULARY[section], None


# --- closed ----------------------------------------------------------------------------------

def test_the_scanner_holds_no_word_of_its_own():
    """Every vocabulary constant of scan.py is computed from the file, never written as a literal,
    and the scanner's terms are exactly the words of the file."""
    tree = ast.parse(scan.pathlib.Path(scan.__file__).read_text(encoding="utf-8"))
    assigned = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in DERIVED:
                    assigned[target.id] = node.value
    assert set(assigned) == set(DERIVED)
    # the only strings in their definitions are the keys of the file's sections and fields
    keys = {" ", "terms", "literals", "names", "not_after", "case_identifier", "category", "rule",
            "words", "strings", "pattern", "term"}
    for name, value in assigned.items():
        assert any(isinstance(sub, ast.Name) and sub.id == "_VOCABULARY_DATA"
                   for sub in ast.walk(value)), "scan.%s is not read from the file" % name
        strings = {sub.value for sub in ast.walk(value)
                   if isinstance(sub, ast.Constant) and isinstance(sub.value, str)}
        assert strings <= keys, "scan.%s writes words of its own: %s" % (name, strings - keys)
    words = [word for row in VOCABULARY["terms"] for word in row["words"]]
    labels = [word.removeprefix("core:").removeprefix("ident:") for word in words]
    assert sorted(term.label for term in scan.VOCABULARY) == sorted(labels)
    literals = {s for row in VOCABULARY["literals"] for s in row["strings"]}
    names = {n for row in VOCABULARY["names"] for n in row["names"]}
    assert {label for label in scan.TERM_INFO if label.startswith('"')} \
        == {'"%s"' % s for s in literals}
    assert {label for label in scan.TERM_INFO if label.startswith("`")} == {"`%s`" % n for n in names}
    assert scan.CASE_IDENTIFIER.pattern == VOCABULARY["case_identifier"]["pattern"]
    assert scan.NOT_AFTER == {row["term"]: frozenset(row["words"]) for row in VOCABULARY["not_after"]}


def test_every_row_carries_its_rule_and_its_finding():
    for section, row, words in _rows():
        assert row["category"].strip() and row["rule"].strip(), (section, row)
        # a finding states what was found, not only the rule: a sentence with a site or a count
        assert len(row["finding"]) >= 60, (section, row["rule"])
        assert re.search(r"\d", row["finding"]), (section, row["rule"])
        if words is not None:
            assert words, (section, row["rule"])
    for row in VOCABULARY["not_after"]:
        assert row["term"] in scan.TERM_INFO and row["words"] and len(row["finding"]) >= 30


def test_a_word_is_in_one_row_only():
    words = collections.Counter()
    for section, field in SECTIONS.items():
        for row in VOCABULARY[section]:
            words.update("%s:%s" % (section, word.removeprefix("core:").removeprefix("ident:"))
                         for word in row[field])
    assert [word for word, count in words.items() if count > 1] == []


def test_every_term_is_well_formed():
    for row in VOCABULARY["terms"]:
        for word in row["words"]:
            label = word.removeprefix("core:").removeprefix("ident:")
            assert re.fullmatch(r"[a-z][a-z0-9]*(-[a-z][a-z0-9]*)*", label), word
            assert scan.terms_in(label.replace("-", "_"), "ident") >= {label} or any(
                label in other.label and other.label != label for other in scan.VOCABULARY), word


# --- versioned -------------------------------------------------------------------------------

def test_the_vocabulary_is_versioned():
    versions = VOCABULARY["versions"]
    assert [entry["version"] for entry in versions] == list(range(1, len(versions) + 1))
    assert VOCABULARY["version"] == versions[-1]["version"] == scan.VOCABULARY_VERSION
    assert versions[-1]["digest"] == scan.vocabulary_digest(VOCABULARY), (
        "cf2_vocabulary.toml changed without a new [[versions]] entry: add one, version %d, with "
        "digest %s and the change and its reason"
        % (VOCABULARY["version"] + 1, scan.vocabulary_digest(VOCABULARY)))
    digests = [entry["digest"] for entry in versions]
    assert len(set(digests)) == len(digests), "two versions with the same vocabulary"
    dates = [datetime.date.fromisoformat(entry["date"]) for entry in versions]
    assert dates == sorted(dates)
    assert all(len(entry["change"]) >= 40 for entry in versions)


@pytest.mark.parametrize("edit", [
    lambda data: data["terms"][0]["words"].append("ident:new-word"),
    lambda data: data["literals"][0]["strings"].pop(),
    lambda data: data["not_after"].append({"term": "hermite", "words": ["gauss"], "finding": "x"}),
    lambda data: data["terms"][3].update(finding=data["terms"][3]["finding"] + " More."),
])
def test_any_change_to_the_vocabulary_changes_its_digest(edit):
    changed = copy.deepcopy(VOCABULARY)
    edit(changed)
    assert scan.vocabulary_digest(changed) != scan.vocabulary_digest(VOCABULARY)
    # the version entries themselves are not part of the digest
    renumbered = copy.deepcopy(VOCABULARY)
    renumbered["versions"].append({"version": 99})
    assert scan.vocabulary_digest(renumbered) == scan.vocabulary_digest(VOCABULARY)


# --- no exception for a quadrature rule or a base dimension (P3, P10) --------------------------

def test_no_rule_of_the_scanner_admits_a_quadrature_or_a_base_dimension():
    for term, after in scan.NOT_AFTER.items():
        assert not (after | {term}) & (QUADRATURE_WORDS | BASE_DIMENSIONS), term
    for path, term, pattern, reason in scan.GENERIC_SENSES:
        assert term not in QUADRATURE_WORDS | BASE_DIMENSIONS, (path, term)
        assert not re.search(r"(?i)quadrature|gauss(?!ian)|dimension", reason), (path, term)
    # the name of a rule is refused in a core path, whatever its compound
    assert scan.terms_in("GaussHermite", "ident") == {"gauss-hermite"}
    assert scan.terms_in("GaussLegendre", "ident") == {"legendre"}
    assert scan.terms_in("Quadrature", "ident") == set()


def test_no_exception_names_a_quadrature_rule():
    _, entries = ratchet.read_list(scan.HERE / "cf2_exceptions.txt")
    rules = [entry for entry in entries if entry.term in {"gauss-hermite", "legendre", "lobatto",
                                                          "laguerre", "radau"}
             or re.search(r"(?i)gauss_?(hermite|legendre)", entry.excerpt)]
    assert rules == []


# --- MG6 by directory and by issue ------------------------------------------------------------

def test_mg6_counts_follow_each_directory_and_each_issue():
    _, entries = ratchet.read_list(scan.HERE / "cf2_exceptions.txt")
    result = mg6_counts.counts(entries)
    assert result["total"] == len(entries) == sum(result["by_issue"].values()) \
        == sum(row["total"] for row in result["by_directory"].values())
    assert set(result["by_issue"]) == {entry.issue for entry in entries}
    for directory, row in result["by_directory"].items():
        assert directory.endswith("/") and row["total"] == sum(row["by_issue"].values())
        assert row["closer"] == (next(iter(row["by_issue"])) if len(row["by_issue"]) == 1 else None)
    # the two directories that the review of 11/11 names as whole directories to close
    for directory in ("include/pops/physics/", "python/pops/moments/"):
        assert directory in result["by_directory"]


def test_mg6_directory_of_a_path():
    assert mg6_counts.directory_of("include/pops/physics/bricks/elliptic.hpp") \
        == "include/pops/physics/"
    assert mg6_counts.directory_of("python/pops/moments/fan_li.py") == "python/pops/moments/"
    assert mg6_counts.directory_of("python/pops/realization.py") == "python/pops/"
    assert mg6_counts.directory_of("pyproject.toml") == "./"
    assert mg6_counts.directory_of("include/pops/numerics/elliptic/mg/geometric_mg.hpp", 4) \
        == "include/pops/numerics/elliptic/"


@pytest.mark.parametrize("name", sorted(mg6_counts.LISTS))
def test_mg6_counts_runs_on_every_exception_list(name, capsys):
    assert mg6_counts.main(["--list", name, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    _, entries = ratchet.read_list(mg6_counts.LISTS[name])
    assert result["total"] == len(entries)
    assert mg6_counts.main(["--list", name]) == 0
    assert capsys.readouterr().out.startswith("%s exceptions: %d sites." % (name.upper(),
                                                                         len(entries)))
