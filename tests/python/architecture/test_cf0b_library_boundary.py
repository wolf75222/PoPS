"""CF0b (spec 0.3, section 16.1; A4, A7; Linear N2): the library is outside, and stays honest.

``python/pops/lib/`` imports only the public ``pops`` surface: no module or name whose name
starts with ``_``, nothing from ``pops.codegen``, ``pops.runtime`` or the native bindings
``pops._pops``. It reaches none of the C++ entry routes of A7 (``pops.external``,
``NativeFunction``, ``NativeDerivative``, ``PreparedNativeComponent``, ``load_cpp_library``, a
brick id passed to ``User``), holds no C++ file and no C++ source string, imports nothing from
the native component tree and holds no private module of its own. The scanner is
``frozen_engine/cf0b.py``.

The test starts from the b6 exception list (``frozen_engine/cf0b_exceptions.txt``, which is the list
that CF0b names, site for site) and fails both on a new site and on an exception that has become
useless, so the list can only shrink (MG6). It is empty at the exit of Phase 1 (section 18).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import collections

import pytest

from tests.python.architecture.frozen_engine import cf0b, ratchet, scan

# The number of sites of each term in cf0b_exceptions.txt at b6e64c1d. A total may only go down
# (MG6); never raise one: remove the occurrence instead.
B6_TERM_TOTALS = {"private-import": 11, "private-module": 3}

# CF0b at b6e64c1d, verbatim from section 16.1 (all under python/pops/): lib/time/dirk.py:9,
# euler.py:7, imex.py:7-8, rk.py:9, ssprk.py:8, strang.py:25, lib/amr/__init__.py:374,435,
# lib/initial/__init__.py:306,440 and the private modules lib/time/_factory.py,
# lib/time/_helpers.py, lib/initial/_bootstrap_program.py.
SPEC_B6_SITES = {
    ("python/pops/lib/time/dirk.py", 9), ("python/pops/lib/time/euler.py", 7),
    ("python/pops/lib/time/imex.py", 7), ("python/pops/lib/time/imex.py", 8),
    ("python/pops/lib/time/rk.py", 9), ("python/pops/lib/time/ssprk.py", 8),
    ("python/pops/lib/time/strang.py", 25), ("python/pops/lib/amr/__init__.py", 374),
    ("python/pops/lib/amr/__init__.py", 435), ("python/pops/lib/initial/__init__.py", 306),
    ("python/pops/lib/initial/__init__.py", 440), ("python/pops/lib/time/_factory.py", 0),
    ("python/pops/lib/time/_helpers.py", 0), ("python/pops/lib/initial/_bootstrap_program.py", 0),
}


@pytest.fixture(scope="module")
def scanned():
    return cf0b.current_entries()


@pytest.fixture(scope="module")
def listed():
    return ratchet.read_list(cf0b.EXCEPTIONS)[1]


# ---------------------------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------------------------
def test_library_matches_the_shrinking_exception_list(scanned, listed):
    comparison = ratchet.compare(scanned, listed)
    assert comparison.clean, ratchet.describe(comparison, "cf0b_exceptions.txt", "CF0b")


def test_exception_list_never_exceeds_its_b6_totals(listed):
    totals = collections.Counter(entry.term for entry in listed)
    unknown = sorted(set(totals) - set(B6_TERM_TOTALS))
    grown = {term: (count, B6_TERM_TOTALS[term]) for term, count in totals.items()
             if term in B6_TERM_TOTALS and count > B6_TERM_TOTALS[term]}
    assert not unknown and not grown, (
        "CF0b: cf0b_exceptions.txt lists terms or sites that b6 did not have (MG6 forbids new "
        "exceptions): unknown %s, grown %s" % (unknown, grown))


def test_b6_list_is_the_list_of_the_spec():
    # At b6 the scan finds exactly the sites that CF0b names. A slice removes sites; it never adds.
    listed_sites = {(entry.path, entry.line) for entry in ratchet.read_list(cf0b.EXCEPTIONS)[1]}
    assert listed_sites <= SPEC_B6_SITES, (
        "CF0b: sites that the b6 list of section 16.1 does not name: %s"
        % sorted(listed_sites - SPEC_B6_SITES))


def test_every_exception_names_the_issue_that_removes_it(listed):
    wrong = [entry.render() for entry in listed
             if not ratchet.issue_known(entry.issue) or entry.issue != cf0b.owner_of(entry.path)]
    assert not wrong, (
        "CF0b: each exception names the issue of OWNER_RULES (cf0b.py):\n  " + "\n  ".join(wrong))


def test_exception_list_is_sorted_and_well_formed(listed):
    assert listed == sorted(listed), "cf0b_exceptions.txt must stay sorted (refresh_exceptions.py)"
    assert len(set(listed)) == len(listed), "cf0b_exceptions.txt has duplicate entries"
    assert all(entry.term in cf0b.TERMS for entry in listed)


def test_library_is_the_path_that_cf0_excludes():
    assert cf0b.library_root() == "python/pops/lib/"
    assert any(rel.startswith(cf0b.library_root()) for rel in scan.tracked_files())


# ---------------------------------------------------------------------------------------------
# Controls: the scanner refuses what CF0b refuses, and only that.
# ---------------------------------------------------------------------------------------------
def _terms(source, rel="python/pops/lib/time/probe.py"):
    return sorted({term for _, term in cf0b.python_sites(
        rel, source, library_module="pops.lib", native_tree="native/")})


@pytest.mark.parametrize("source, expected", [
    ("from pops.time._methods.tableau import RungeKuttaTableau\n", ["private-import"]),
    ("def f():\n    from pops.time._evaluation_point import evaluation_partition\n",
     ["private-import"]),
    ("from pops.time import _evaluation_point\n", ["private-import"]),  # a private name
    ("from pops.descriptors import _native\n", ["private-import"]),
    ("import pops.time\nx = pops.time._program\n", ["private-access"]),
    ("from pops import codegen\n", ["codegen-import"]),
    ("from pops.codegen.solvers import dsl\n", ["codegen-import"]),
    ("def f():\n    import pops.runtime\n", ["runtime-import"]),
    ("from pops import _pops\n", ["bindings-import"]),
    ("import importlib\nm = importlib.import_module('pops.runtime.defaults')\n",
     ["runtime-import"]),
    ("from pops.native_calls import NativeFunction\nf = NativeFunction\n", ["NativeFunction"]),
    ("from pops.native_calls import NativeDerivative\n", ["NativeDerivative"]),
    ("from pops.native_components import PreparedNativeComponent\n",
     ["PreparedNativeComponent"]),
    ("import pops.native_calls\n", ["native-route-module"]),
    ("import pops.descriptors as d\nn = d.load_cpp_library('x.so')\n", ["load_cpp_library"]),
    ("import pops\nn = pops.lib.load_cpp_library('x.so')\n", ["load_cpp_library"]),
    ("import pops\nk = pops.external('kernel')\n", ["external"]),
    ("from pops.descriptors import external\n", ["external"]),
    ("from pops.numerics import riemann\nflux = riemann.User('my_hllc')\n", ["brick-id"]),
    ("import native.riemann\n", ["native-tree-import"]),
    ("ROOT = 'native/riemann/flux.hpp'\n", ["cpp-file-reference"]),
    ("ROOT = 'native/riemann/'\n", ["native-tree-path"]),
    ("SRC = '#include <cmath>\\ndouble f(double x) { return x; }'\n", ["cpp-source"]),
    ("SRC = 'KOKKOS_INLINE_FUNCTION double f(double x);'\n", ["cpp-source"]),
    ("HEADER = 'flux_kernel.hpp'\n", ["cpp-file-reference"]),
])
def test_scanner_refuses_what_cf0b_refuses(source, expected):
    assert _terms(source) == expected


@pytest.mark.parametrize("source", [
    "from __future__ import annotations\n",  # a dunder module is not private
    "from pops.time import LocalLinear, Program\nfrom pops.solvers import DenseLU\n",
    "from pops.time.operator_resolution import resolve_operator_handle\n",  # public at b6
    "from ._factory import program_factory\nfrom .rk import _runge_kutta\n",  # the library itself
    "from pops.lib.initial import BindArray\n",
    "class P:\n    def f(self):\n        return self._frozen\n",  # not a pops module
    "from pops.numerics import riemann\nflux = riemann.User(body=f, state=U, stability=s)\n",
    "external = True\nx = external\n",  # the word alone, not pops.external
    "def f():\n    raise ValueError('pops.external(...) and NativeFunction are refused here')\n",
    '"""Docstring naming #include <cmath> and std::vector."""\n',
    "TARGET = 'pops::RusanovFlux'\n",  # a qualified name, not C++ source
])
def test_scanner_accepts_the_public_surface(source):
    assert _terms(source) == []


def test_scanner_reports_private_modules_and_cpp_files(tmp_path, monkeypatch):
    rel_private = "python/pops/lib/time/_probe.py"
    rel_init = "python/pops/lib/time/__init__.py"
    rel_cpp = "python/pops/lib/numerics/kernel.cpp"
    for rel in (rel_private, rel_init, rel_cpp):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n" if rel.endswith(".py") else "int x;\n")
    monkeypatch.setattr(scan, "tracked_files", lambda root=None: [rel_private, rel_init, rel_cpp])
    sites = {(site.path, site.term) for site in cf0b.scan_library(tmp_path)}
    assert sites == {(rel_private, "private-module"), (rel_cpp, "cpp-file")}
