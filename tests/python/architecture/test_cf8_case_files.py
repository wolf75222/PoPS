"""CF8 (spec 0.3, section 16.2; B2, B3, B5, A7; Linear N2): no infrastructure and no native code
in case files.

A source scan of the conformance cases, litmus cases, probes and the final witness (the paths of
``frozen_engine/cases.toml``) refuses the sizes that the plan derives (halo widths, stencil radii,
component counts, history depths; B2), the runtime resources (memory and metadata budgets, byte
capacities, buffer sizes, storage sizes, ``capacity=``; B3, B5), C++ files and C++ source strings,
the C++ entry routes of A7 (gap C37: ``pops.external``, ``NativeFunction``, ``NativeDerivative``,
``PreparedNativeComponent``, ``load_cpp_library``, a brick id passed to ``riemann.User``) and
imports from the native component tree. A method budget (section 12.1, B1) stays in the case. The
scanner is ``frozen_engine/cf8.py``; the library side of A7 is CF0b
(``test_cf0b_library_boundary.py``), so a case cannot reach a C++ entry route through ``pops.lib``
either.

The test starts from the exception list of b6e64c1d (``frozen_engine/cf8_exceptions.txt``) and fails
both on a new site and on an exception that has become useless (MG6).

The test reads the source tree only; it does not import ``pops``, ``_pops`` or a case.
"""
import ast
import collections

import pytest

from tests.python.architecture.frozen_engine import cf8, ratchet, scan

# The number of sites of each term in cf8_exceptions.txt at b6e64c1d. A total may only go down
# (MG6); never raise one: remove the size or the resource from the case instead.
B6_TERM_TOTALS = {
    "basis_capacity": 1, "byte_capacity": 70, "capacity": 83, "cell_capacity": 4, "components": 17,
    "depth": 46, "flux_expression_budget": 1, "ghost_depth": 22, "history_field_metadata_budget": 7,
    "level_capacity": 5, "nnz_capacity": 2, "observation_byte_capacity": 1, "patch_capacity": 4,
    "radius": 16, "storage_bytes": 10, "storage_versions": 10, "transport_byte_capacity": 6,
}
# The case paths of b6 (cases.toml). Entries are added for the litmus cases, probes and the witness;
# a b6 entry is never removed while it holds cases.
B6_CASE_PATHS = ("examples/api_v02/",)
# The counter-examples of the spec, all @b6: the mandatory history metadata budget of M01 (B3,
# section 12) and the capacities written in a regrid case (DM13).
SPEC_COUNTER_EXAMPLES = {
    ("examples/api_v02/models_M01_M14.py", "history_field_metadata_budget"),
    ("examples/api_v02/continuation_regrid_amr.py", "cell_capacity"),
    ("examples/api_v02/continuation_regrid_amr.py", "patch_capacity"),
}


@pytest.fixture(scope="module")
def scanned():
    return cf8.current_entries()


@pytest.fixture(scope="module")
def listed():
    return ratchet.read_list(cf8.EXCEPTIONS)[1]


# ---------------------------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------------------------
def test_case_files_match_the_shrinking_exception_list(scanned, listed):
    comparison = ratchet.compare(scanned, listed)
    assert comparison.clean, ratchet.describe(comparison, "cf8_exceptions.txt", "CF8")


def test_exception_list_never_exceeds_its_b6_totals(listed):
    totals = collections.Counter(entry.term for entry in listed)
    unknown = sorted(set(totals) - set(B6_TERM_TOTALS))
    grown = {term: (count, B6_TERM_TOTALS[term]) for term, count in totals.items()
             if term in B6_TERM_TOTALS and count > B6_TERM_TOTALS[term]}
    assert not unknown and not grown, (
        "CF8: cf8_exceptions.txt lists sites that b6 did not have (MG6 forbids new exceptions): "
        "unknown %s, grown %s" % (unknown, grown))


def test_b6_holds_no_native_code_in_case_files(listed):
    # At b6 no case file holds a C++ file, a C++ string or a C++ entry route of A7: none may appear.
    native = [entry.render() for entry in listed if cf8.kind_of(entry.term).startswith("native")]
    assert not native, "\n".join(native)


def test_the_counter_examples_of_the_spec_are_listed(listed):
    keys = {entry.key for entry in listed}
    found = {entry.key for entry in cf8.current_entries()}
    assert (SPEC_COUNTER_EXAMPLES & found) <= keys


def test_every_exception_names_the_issue_that_removes_it(listed):
    wrong = [entry.render() for entry in listed
             if not ratchet.issue_known(entry.issue) or entry.issue != cf8.owner_of(entry.term)]
    assert not wrong, (
        "CF8: each exception names the issue of OWNER_RULES (cf8.py):\n  "
        + "\n  ".join(wrong[:50]))


def test_exception_list_is_sorted_and_well_formed(listed):
    assert listed == sorted(listed), "cf8_exceptions.txt must stay sorted (refresh_exceptions.py)"
    assert len(set(listed)) == len(listed), "cf8_exceptions.txt has duplicate entries"


def test_case_paths_cover_the_b6_corpus_and_exist():
    paths = cf8.load_case_paths()
    assert set(B6_CASE_PATHS) <= set(paths), "a b6 case path left cases.toml: %s" % (
        sorted(set(B6_CASE_PATHS) - set(paths)))
    tracked = scan.tracked_files()
    missing = [entry for entry in paths if not any(scan.covered(rel, (entry,)) for rel in tracked)]
    assert not missing, "cases.toml names case paths that match no tracked file: %s" % missing
    engine = scan.load_cf0()
    frozen = [entry for entry in paths if engine.is_frozen(entry if not entry.endswith("/")
                                                            else entry + "probe.py")]
    assert not frozen, "a case path lies in the frozen engine (CF0, FC6b): %s" % frozen


# ---------------------------------------------------------------------------------------------
# Controls: the scanner refuses what CF8 refuses, and only that.
# ---------------------------------------------------------------------------------------------
def _sites(source):
    return sorted(name for _, name in cf8.infrastructure_sites(ast.parse(source)))


@pytest.mark.parametrize("source, expected", [
    ("p = pops.Program('m', history_field_metadata_budget=(8 << 20, 16 << 20))\n",
     ["history_field_metadata_budget"]),
    ("domain = p.range_set(n, capacity=8, ownership='replicated')\n", ["capacity"]),
    ("q.map(D, body, byte_capacity=1 << 20)\n", ["byte_capacity"]),
    ("q.map(D, body, **({'byte_capacity': c} if c else {}))\n", ["byte_capacity"]),
    ("Distributed(p, storage_bytes=64 << 20, storage_versions=4096)\n",
     ["storage_bytes", "storage_versions"]),
    ("plan.states.add(U, StateStorage(ghost_depth=max(flux.ghost_extent)))\n", ["ghost_depth"]),
    ("StencilFlux(dimension=2, components=3, radius=1, body=face)\n", ["components", "radius"]),
    ("StencilFlux(dimension=2, components=c, radius=r, body=face)\n", ["radius"]),
    ("p.keep_history(q, depth=2)\np.store_history('rhs', rate, depth=1)\n", ["depth", "depth"]),
    ("def build(count=3, *, capacity=8):\n    pass\n", ["capacity"]),
    ("program.history_field_metadata_budget = (1, 2)\n", ["history_field_metadata_budget"]),
    ("comm.exchange(x, buffer_size=4096)\n", ["buffer_size"]),
    ("Program('m', flux_expression_budget=(64, 4))\n", ["flux_expression_budget"]),
])
def test_scanner_refuses_infrastructure(source, expected):
    assert _sites(source) == expected


@pytest.mark.parametrize("source", [
    "relation(p, inner_budget=4, line_budget=20)\n",  # method budgets (B1, section 12.1)
    "@graph(inputs=Product(data=D, budget=Int64))\ndef f(q):\n    pass\n",
    "solve(problem, max_iterations=20, max_steps=1000)\n",
    "model.state('u', components=('u', 'v'))\n",  # components declared, not counted
    "model.flux('F', state=U, components={a: (U[0],) for a in axes})\n",
    "Disc(center=(.5, .5), radius=.46)\n",  # a geometric radius
    "build_finite_structural_depth(depth=160)\n",  # the nesting the case builds, not a history
    "Rectangle('d', (0., 0.), (1., 1.))\nCartesianGrid(frame=f, cells=(16, 8))\n",  # configuration
    "result = dict(passed=True, prescribed_trace_vs_mirrored_ghost_defect=e)\n",
    "history = p.product(q=p.float64(0))\nmetrics(levels=runtime.n_levels())\n",
])
def test_scanner_accepts_what_the_case_writes(source):
    assert _sites(source) == []


@pytest.fixture
def scan_case(tmp_path, monkeypatch):
    """Scan one case file written into a scratch tree."""
    def scan_source(source, rel="examples/api_v02/probe.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(source)
        monkeypatch.setattr(scan, "tracked_files", lambda root=None: [rel])
        return sorted(site.term for site in cf8.scan_cases(tmp_path, paths=("examples/",)))
    return scan_source


@pytest.mark.parametrize("source, expected", [
    ("from pops.native_calls import NativeFunction, NativeDerivative\n",
     ["NativeDerivative", "NativeFunction"]),
    ("from pops.native_components import PreparedNativeComponent\n", ["PreparedNativeComponent"]),
    ("import pops\npops.lib.load_cpp_library('brick.so')\n", ["load_cpp_library"]),
    ("import pops\nk = pops.external('kernel', signature=s)\n", ["external"]),
    ("from pops.numerics import riemann\nf = riemann.User('my_hllc')\n", ["brick-id"]),
    ("from native.riemann import flux\n", ["native-tree-import"]),
    ("pops.compile(case, source='#include <cmath>\\ndouble f(double x) { return x; }')\n",
     ["cpp-source"]),
    ("k = load('kernel.cpp')\n", ["cpp-file-reference"]),
])
def test_scanner_refuses_native_code_in_case_files(scan_case, source, expected):
    assert scan_case(source) == expected


def test_scanner_accepts_a_body_written_in_python(scan_case):
    source = ("from pops.numerics import riemann\n"
              "from pops.time._methods import tableau\n"  # CF0b's rule, not a case's (A4)
              "f = riemann.User(body=face, state=U, stability=bound)\n")
    assert scan_case(source) == []


def test_scanner_refuses_a_cpp_file_in_a_case_path(scan_case):
    assert scan_case("int f();\n", rel="examples/api_v02/kernel.cpp") == ["cpp-file"]
