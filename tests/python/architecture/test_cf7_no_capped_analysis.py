"""CF7 (spec 0.3, section 16.2; B4, B6; Linear N2): no constant limit on analysis work.

A source scan of the core paths and of the native component tree (FC6b, ``cf0_paths.toml``) refuses
a named integer limit on analysis work: iterations unrolled, alternatives enumerated, entries
visited, and the program sizes that the interpreters of b6 accept (the analytic-expression VM and
the coupled-source bytecode of C44, the tag-graph VM of C36). The iteration limits of numerical
solvers are method budgets of section 12.1, listed with their reason in ``cf7.METHOD_BUDGETS``; the
hard cap of a generated custom-solver loop is a run-time backstop that no author writes
(``cf7.RUNTIME_BACKSTOPS``, a B1 and B7 gap of G10), not an analysis cap either. The scanner is
``frozen_engine/cf7.py``.

The test starts from the exception list of b6e64c1d (``frozen_engine/cf7_exceptions.txt``) and fails
both on a new cap and on an exception that has become useless (MG6). At the exit of Phase 1 the list
holds only the caps of the two interpreters of C44: with G04 and G07, with G04 alone on the
boundary-state route of the analytic VM, or with G06 (section 18.3); at the exit of Phase 2 it is
empty.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import collections

import pytest

from tests.python.architecture.frozen_engine import cf7, ratchet, scan

# The number of sites of each term in cf7_exceptions.txt at b6e64c1d. A total may only go down
# (MG6); never raise one: remove the cap instead.
B6_TERM_TOTALS = {
    "CS_MAX_PROG": 1, "CS_MAX_REG": 1, "CS_MAX_TERMS": 1, "DEFAULT_MAX_DEPTH": 1,
    "DEFAULT_MAX_NODES": 1, "DEFAULT_MAX_STACK": 1, "POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1": 1,
    "POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1": 1, "_MAX_DEPTH": 1, "_MAX_NODES": 1,
    "kAnalyticMaxDepth": 1, "kAnalyticMaxNodes": 1, "kAnalyticMaxStack": 1, "kCsMaxProg": 1,
    "kCsMaxReg": 1, "kCsMaxStack": 1, "kCsMaxTerms": 1, "max_alternatives": 1, "max_iterations": 1,
    "maximum_instruction_count": 2, "maximum_stencil_terms": 2, "runtime-bound-repetition": 1,
    "kMaxAnalyticBoundaryComponents": 1,
}

# The sites that CF7 names at b6 (section 16.2), all @b6: realization_delivery.py:45,
# program_stencil_budget.py:26-28 (the comment that states the ceiling; the anchored site is the
# condition it explains), analytic/_model.py:21-23, _analytic_expression_lowering.py:22-23 and
# _coupled_abi.py:27-29.
SPEC_B6_SITES = {
    ("python/pops/codegen/realization_delivery.py", "max_alternatives"),
    ("python/pops/codegen/realization_delivery.py", "max_iterations"),
    ("python/pops/codegen/program_stencil_budget.py", "runtime-bound-repetition"),
    ("python/pops/analytic/_model.py", "DEFAULT_MAX_DEPTH"),
    ("python/pops/analytic/_model.py", "DEFAULT_MAX_NODES"),
    ("python/pops/analytic/_model.py", "DEFAULT_MAX_STACK"),
    ("python/pops/codegen/_analytic_expression_lowering.py", "_MAX_DEPTH"),
    ("python/pops/codegen/_analytic_expression_lowering.py", "_MAX_NODES"),
    ("python/pops/physics/_coupled_abi.py", "CS_MAX_REG"),
    ("python/pops/physics/_coupled_abi.py", "CS_MAX_TERMS"),
    ("python/pops/physics/_coupled_abi.py", "CS_MAX_PROG"),
}
# Section 18.3: the issues of the caps that may remain at the exit of Phase 1, those of the two
# interpreters of C44 (G04 alone for the boundary-state route of the analytic VM).
PHASE_2_ISSUES = {"G04+G07", "G04", "G06"}
# The ids of ANCHORED_SITES at b6 (the second one found by the audit of lot 2). Anchors only go;
# never add one.
B6_ANCHORS = {("python/pops/codegen/program_stencil_budget.py", "runtime-bound-repetition"),
              ("include/pops/mesh/boundary/prepared_hyperbolic_boundary.hpp",
               "kMaxAnalyticBoundaryComponents")}


@pytest.fixture(scope="module")
def scanned():
    return cf7.current_entries()


@pytest.fixture(scope="module")
def listed():
    return ratchet.read_list(cf7.EXCEPTIONS)[1]


# ---------------------------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------------------------
def test_frozen_engine_caps_match_the_shrinking_exception_list(scanned, listed):
    found, _, _ = scanned
    comparison = ratchet.compare(found, listed)
    assert comparison.clean, ratchet.describe(comparison, "cf7_exceptions.txt", "CF7")


def test_exception_list_never_exceeds_its_b6_totals(listed):
    totals = collections.Counter(entry.term for entry in listed)
    unknown = sorted(set(totals) - set(B6_TERM_TOTALS))
    grown = {term: (count, B6_TERM_TOTALS[term]) for term, count in totals.items()
             if term in B6_TERM_TOTALS and count > B6_TERM_TOTALS[term]}
    assert not unknown and not grown, (
        "CF7: cf7_exceptions.txt lists caps that b6 did not have (MG6 forbids new exceptions): "
        "unknown %s, grown %s" % (unknown, grown))


def test_every_site_that_cf7_names_is_listed_or_removed(listed):
    # A site of the spec leaves the list only with its cap; it never leaves while its cap stays.
    listed_keys = {entry.key for entry in listed}
    found_keys = {entry.key for entry in cf7.current_entries()[0]}
    assert not (SPEC_B6_SITES & found_keys) - listed_keys


def test_every_exception_names_the_issue_that_removes_it(listed):
    wrong = [entry.render() for entry in listed
             if not ratchet.issue_known(entry.issue)
             or entry.issue != cf7.owner_of(entry.path, entry.term)]
    assert not wrong, (
        "CF7: each exception names the issue of OWNER_RULES (cf7.py):\n  " + "\n  ".join(wrong))


def test_phase_1_exit_leaves_only_the_interpreters_of_c44(listed):
    # Section 18.3 lets only the caps of the two interpreters of C44 outlive Phase 1; every other
    # cap is owned by a Phase 1 issue (N7, N12).
    owners = collections.Counter(entry.issue for entry in listed)
    assert set(owners) <= PHASE_2_ISSUES | {"N7", "N12"}, owners


def test_every_method_budget_still_names_a_limit(scanned):
    _, budgets_used, anchors_used = scanned
    stale = [entry for entry in cf7.METHOD_BUDGETS + cf7.RUNTIME_BACKSTOPS
             if entry not in budgets_used]
    assert not stale, (
        "CF7: METHOD_BUDGETS or RUNTIME_BACKSTOPS (cf7.py) entries that no longer name a limit of "
        "their file; remove them so the tables cannot hide a future cap:\n  "
        + "\n  ".join(map(repr, stale)))
    assert set(anchors_used) == set(cf7.ANCHORED_SITES), (
        "CF7: an anchored site no longer matches its line; when its cap is gone, remove the anchor "
        "and its exception in the same PR")


def test_method_budgets_are_solver_iteration_limits():
    for path, names, reason in cf7.METHOD_BUDGETS + cf7.RUNTIME_BACKSTOPS:
        assert names and reason and not path.endswith("/"), (path, names)
        for name in names:
            words = set(scan.split_words(name))
            assert words & {"iter", "iters", "iteration", "iterations"}, (
                "a method budget of section 12.1 is an iteration limit: %s in %s" % (name, path))


def test_runtime_backstops_name_their_gap_and_its_issue():
    # A run-time backstop is not a method budget (B1, B7): each entry says so, names the issue that
    # replaces it with a budget written in the Program, and is not a method budget as well.
    budgets = {(path, name) for path, names, _ in cf7.METHOD_BUDGETS for name in names}
    for path, names, reason in cf7.RUNTIME_BACKSTOPS:
        issues = {word.strip(",.;()") for word in reason.split()} & ratchet.KNOWN_ISSUES
        assert "B1" in reason and "B7" in reason and issues, (path, reason)
        assert not {(path, name) for name in names} & budgets, (path, names)


def test_anchors_only_go():
    ids = {(path, term) for path, term, _, _ in cf7.ANCHORED_SITES}
    assert ids <= B6_ANCHORS, "CF7: ANCHORED_SITES only shrink (MG6): %s" % sorted(ids - B6_ANCHORS)


def test_exception_list_is_sorted_and_well_formed(listed):
    # An entry may repeat: a line that holds a limit twice holds two sites, and the gate pairs
    # them with multiplicity, so a repeat that no site matches fails it as useless.
    assert listed == sorted(listed), "cf7_exceptions.txt must stay sorted (refresh_exceptions.py)"


# ---------------------------------------------------------------------------------------------
# Controls: the scanner refuses a cap on analysis work, and only that.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("source, expected", [
    ("def frontiers(root, *, max_alternatives=4096, max_iterations=256):\n    pass\n",
     ["max_alternatives", "max_iterations"]),
    ("MAX_UNROLL = 64\n", ["MAX_UNROLL"]),
    ("class Walk:\n    max_visited_entries = 1 << 16\n", ["max_visited_entries"]),
    ("_MAX_DEPTH = 64\n_MAX_NODES = 4096\n", ["_MAX_DEPTH", "_MAX_NODES"]),
    ("result = analyse(graph, alternative_limit=512)\n", ["alternative_limit"]),
    ("ABI = {'maximum_instruction_count': 128}\n", ["maximum_instruction_count"]),
    ("BUDGET: int = 64 * 1024\nNODE_BUDGET: int = 64 * 1024\n", ["NODE_BUDGET"]),
])
def test_scanner_refuses_python_caps(source, expected):
    assert sorted(name for _, name, _ in cf7.python_limits(source)) == expected


@pytest.mark.parametrize("source, expected", [
    ("inline constexpr int kCsMaxProg = 256;  // length of a program", ["kCsMaxProg"]),
    ("#define POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1 128u\n",
     ["POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1"]),
    ("struct Walk { std::size_t max_entries{1 << 20}; };", ["max_entries"]),
    ("void unroll(int depth, int max_unrolled_iterations = 32);", ["max_unrolled_iterations"]),
    ("static constexpr std::size_t kMaxNodes = 0x1000;", ["kMaxNodes"]),
])
def test_scanner_refuses_cpp_caps(source, expected):
    assert sorted(name for _, name, _ in cf7.cpp_limits(source)) == expected


@pytest.mark.parametrize("source", [
    "INT64_MAX = (1 << 63) - 1\n_UINT32_MAX = 4294967295\nEXACT = 1 << 53\n",  # integer ranges
    "max_iterations = 1\nmaximum_terms = 0\n",  # sentinels and accumulators
    "max_cycles = 50\nmax_backtracks = 12\nmax_steps = 1000\n",  # solver budgets of section 12.1
    "max_box_size = 32\nmax_levels = 4\n",  # configuration choices
    "tolerance_max = 1e-8\n",  # not an integer
    '"""A docstring: max_iterations=256."""\n# a comment: MAX_NODES = 4096\n',
])
def test_scanner_ignores_what_is_not_an_analysis_cap_in_python(source):
    assert list(cf7.python_limits(source)) == []


@pytest.mark.parametrize("source", [
    "enum class RecoveryCause : int { kNone = 0, kIterationLimit = 2 };",  # a status code
    "typedef enum S { POPS_SOLVE_ITERATION_LIMIT_V2 = 3 } S;",
    "// constexpr int kMaxNodes = 4096;\n/* int max_depth = 64; */",
    'const char* text = "max_iterations = 256";',
    "if (max_nodes == 4096) { return; }",  # a comparison, not a definition
    "constexpr int kMaxNodes = kDefaultNodes;",  # not a literal
    "constexpr std::int64_t kMaxStack = (1LL << 63) - 1;",  # the range of the type
])
def test_scanner_ignores_what_is_not_an_analysis_cap_in_cpp(source):
    assert list(cf7.cpp_limits(source)) == []


def test_a_cap_is_spelled_by_its_name_and_value(tmp_path, monkeypatch):
    # The ratchet pairs sites with entries by this spelling, counted with multiplicity: an edit
    # elsewhere on the line or a re-wrap keeps the site, while a raised cap is a new site, so a
    # listed cap leaves the list only by removal.
    rel = "include/pops/runtime/probe.hpp"

    def spelled(source):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(source)
        monkeypatch.setattr(scan, "frozen_engine_files", lambda engine, root: [rel])
        return [(site.term, site.excerpt) for site in cf7.scan_engine(tmp_path)[0]]

    listed = spelled("inline constexpr int kMaxProgramNodes = 64;\n")
    assert listed == [("kMaxProgramNodes", "kMaxProgramNodes = 64")]
    assert spelled("inline constexpr std::size_t kMaxProgramNodes = 64;  // nodes\n") == listed
    assert spelled("inline constexpr int\n    kMaxProgramNodes = 64;\n") == listed
    assert spelled("inline constexpr int kMaxProgramNodes = 128;\n") != listed


def test_json_caps_are_read():
    text = '{"tagging": {"maximum_stencil_terms": 16, "version": 4, "maximum_rank": 3}}'
    assert [name for _, name, _ in cf7.json_limits(text)] == ["maximum_stencil_terms"]
