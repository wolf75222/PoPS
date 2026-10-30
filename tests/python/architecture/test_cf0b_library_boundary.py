"""CF0b (spec 0.3, section 16.1; A4, A7; Linear N2): the library is outside, and stays honest.

``python/pops/lib/`` imports only the public ``pops`` surface: no module or name whose name
starts with ``_``, nothing from ``pops.codegen``, ``pops.runtime`` or the native bindings
``pops._pops``. It reaches none of the C++ entry routes of A7 (``pops.external``,
``NativeFunction``, ``NativeDerivative``, ``PreparedNativeComponent``, ``load_cpp_library``, a
brick id passed to ``User``), holds no C++ file and no C++ source string, imports nothing from
the native component tree and holds no private module of its own. It neither reads nor writes a
private attribute of an object that it receives from the core (``model._m``,
``getattr(model, "_frame")``; A4). The scanner is ``frozen_engine/cf0b.py``.

The test starts from the b6 exception list (``frozen_engine/cf0b_exceptions.txt``: the list that
CF0b names, site for site, and the private attributes that the audit of lot 2 found) and fails both
on a new site and on an exception that has become useless, so the list can only shrink (MG6). It is
empty at the exit of Phase 1 (section 18).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import collections
import dataclasses

import pytest

from tests.python.architecture.frozen_engine import cf0b, ratchet, scan

# The number of sites of each term in cf0b_exceptions.txt at b6e64c1d. A total may only go down
# (MG6); never raise one: remove the occurrence instead.
B6_TERM_TOTALS = {"private-import": 11, "private-module": 3, "private-attribute": 6}

# CF0b at b6e64c1d, verbatim from section 16.1 (all under python/pops/): lib/time/dirk.py:9,
# euler.py:7, imex.py:7-8, rk.py:9, ssprk.py:8, strang.py:25, lib/amr/__init__.py:374,435,
# lib/initial/__init__.py:306,440 and the private modules lib/time/_factory.py,
# lib/time/_helpers.py, lib/initial/_bootstrap_program.py. The sites are pinned per (path, term),
# the key of the ratchet: refresh_exceptions.py rewrites line numbers and excerpts, so a pin on
# either would fail on the first slice that shifts or edits a listed line.
SPEC_B6_SITES = collections.Counter({
    ("python/pops/lib/time/dirk.py", "private-import"): 1,
    ("python/pops/lib/time/euler.py", "private-import"): 1,
    ("python/pops/lib/time/imex.py", "private-import"): 2,
    ("python/pops/lib/time/rk.py", "private-import"): 1,
    ("python/pops/lib/time/ssprk.py", "private-import"): 1,
    ("python/pops/lib/time/strang.py", "private-import"): 1,
    ("python/pops/lib/amr/__init__.py", "private-import"): 2,
    ("python/pops/lib/initial/__init__.py", "private-import"): 2,
    ("python/pops/lib/time/_factory.py", "private-module"): 1,
    ("python/pops/lib/time/_helpers.py", "private-module"): 1,
    ("python/pops/lib/initial/_bootstrap_program.py", "private-module"): 1,
})
# The b6 sites that the audit of lot 2 added (Linear N2, ROM-81): the library reads private
# attributes of a core model to find its frame and state (electrostatic_lorentz.py:32,50,52,55),
# and writes the provenance of a core Program (_factory.py:26,34). Section 16.1 names imports; its
# intention covers these reads and writes (A4).
AUDIT_B6_SITES = collections.Counter({
    ("python/pops/lib/models/electrostatic_lorentz.py", "private-attribute"): 4,
    ("python/pops/lib/time/_factory.py", "private-attribute"): 2,
})


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


def _beyond_the_b6_pins(entries):
    """(path, term) -> (listed, pinned) for each count above the b6 sites of the spec and audit."""
    pinned = SPEC_B6_SITES + AUDIT_B6_SITES
    return {key: (count, pinned[key]) for key, count in sorted(ratchet.counts(entries).items())
            if count > pinned[key]}


def test_b6_list_is_the_list_of_the_spec_and_the_audit(listed):
    # At b6 the scan finds exactly the sites that CF0b names and those of the audit. A slice
    # removes sites; it never adds.
    beyond = _beyond_the_b6_pins(listed)
    assert not beyond, (
        "CF0b: sites that neither the b6 list of section 16.1 nor the audit names, as (path, "
        "term): (listed, pinned): %s" % beyond)
    per_term = collections.Counter()
    for (_, term), count in (SPEC_B6_SITES + AUDIT_B6_SITES).items():
        per_term[term] += count
    assert per_term == collections.Counter(B6_TERM_TOTALS)


def test_b6_pins_do_not_depend_on_lines_or_excerpts(listed):
    # refresh_exceptions.py moves a site's line when an edit above it shifts the file, and
    # rewrites its excerpt when the line itself is edited: the pins hold through both.
    edited = [dataclasses.replace(entry, line=entry.line + 1, excerpt=entry.excerpt + " # edited")
              for entry in listed]
    assert not _beyond_the_b6_pins(edited)
    elsewhere = dataclasses.replace(listed[0], path="python/pops/lib/numerics/probe.py")
    assert _beyond_the_b6_pins(listed + [elsewhere])
    assert _beyond_the_b6_pins(listed + [listed[0]])


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
def _spellings(source, rel="python/pops/lib/time/probe.py"):
    return sorted({(term, spelling) for _, term, spelling in cf0b.python_sites(
        rel, source, library_module="pops.lib", native_tree="native/", private_attributes=True)})


def _terms(source):
    return sorted({term for term, _ in _spellings(source)})


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
    ("def f(model):\n    return model._m\n", ["private-attribute"]),
    ("def f(model):\n    return getattr(model, '_frame', None)\n", ["private-attribute"]),
    ("def f(program):\n    program._provenance_context = None\n", ["private-attribute"]),
    ("def f(state):\n    setattr(state, '_cache', {})\n", ["private-attribute"]),
    ("def f(model):\n    return getattr(getattr(model, '_dsl', None), '_m', model)\n",
     ["private-attribute"]),
    ("import pops.time\npops.time._registry = {}\n", ["private-access"]),
    ("import pops.time\nhandle = getattr(pops.time, '_program')\n", ["private-access"]),
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
    # private attributes that the library defines itself, on self or on another of its objects
    "class Box:\n    def __init__(self):\n        self._lo = 0\n"
    "    def __eq__(self, other):\n        return self._lo == other._lo\n",
    "class Ring:\n    __slots__ = ('_items',)\n\ndef size(ring):\n    return len(ring._items)\n",
    "class _Cell:\n    pass\n\ndef cells():\n    return _Cell._instances\n",  # a module class
    "def f(row):\n    return row._replace(a=1)._asdict()\n",  # the public API of a namedtuple
    "import sys\nframe = sys._getframe(0)\n",  # not the pops surface
    "def f(model):\n    return model.__class__, getattr(model, '__name__', None)\n",  # dunders
])
def test_scanner_accepts_the_public_surface(source):
    assert _terms(source) == []


def test_a_site_is_its_spelling_on_its_line():
    # The ratchet pairs a site with its entry by this spelling: an edit elsewhere on a listed line
    # keeps the site, and a new private module, name or attribute is a new site.
    tableau = [("private-import", "pops.time._methods.tableau")]
    assert _spellings("from pops.time._methods.tableau import RungeKuttaTableau\n") == tableau
    assert _spellings("from pops.time._methods.tableau import (ButcherTableau,\n"
                      "                                      RungeKuttaTableau)\n") == tableau
    assert _spellings("from pops.time._program import RungeKuttaTableau\n") != tableau
    assert _spellings("from pops.time import _evaluation_point\n") == [
        ("private-import", "pops.time._evaluation_point")]
    frame = [("private-attribute", "._frame")]
    assert _spellings("def f(model):\n    return getattr(model, '_frame', None)\n") == frame
    assert _spellings("def f(model):\n    return model._frame\n") == frame
    assert _spellings("def f(model):\n    return model._state\n") != frame
    assert _spellings("import pops.time\npops.time._registry = {}\n") == [
        ("private-access", "pops.time._registry")]


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


def test_private_attributes_that_a_module_defines_stay_its_own(tmp_path, monkeypatch):
    # A library class that stores _provenance_context on itself hides no write of the same name
    # on a core Program in another library module.
    files = {
        "python/pops/lib/time/trace.py":
            "class Trace:\n    def __init__(self):\n        self._provenance_context = None\n",
        "python/pops/lib/time/factory.py":
            "def build(program):\n    program._provenance_context = {}\n    return program\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.setattr(scan, "tracked_files", lambda root=None: sorted(files))
    sites = {(site.path, site.line, site.term) for site in cf0b.scan_library(tmp_path)}
    assert sites == {("python/pops/lib/time/factory.py", 2, "private-attribute")}
