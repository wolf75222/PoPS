"""CF2 (spec 0.3, section 16.2; Linear N2): the frozen engine does not know the model.

A source-only scan of the core paths and of the native component tree (FC6b, CF0 in
``frozen_engine/cf0_paths.toml``) refuses, outside comments and user-facing error text, the knowledge
of section 15.3: physical roles and role guessing, model, closure and equation-of-state names,
named numerical methods, schemes, limiters, interpolations, field operators and boundary families,
a primitive selected by its spelling, a physical constant in an ABI, case identifiers, and so any
route table or switch keyed on them (A5, FC4). The scanner and its vocabulary are in
``frozen_engine/scan.py``.

The test starts from the exception list of b6e64c1d (``frozen_engine/cf2_exceptions.txt``, each
entry ``file:line`` with the issue that removes it) and fails both on a new occurrence and on an
exception that has become useless, so the list can only shrink (MG6). G3 requires it to be empty
(section 1.5). The closed list of interfaces of FC6b is asserted by
``test_native_interfaces_closed.py``.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import collections

import pytest

from tests.python.architecture.frozen_engine import cf2, ratchet, scan


# The number of sites of each term in cf2_exceptions.txt at b6e64c1d. A total here may only go down
# (MG6). Listing a new site in cf2_exceptions.txt would need a higher number here; do not raise one:
# remove the occurrence instead. A term that the early audit of N2 adds to the vocabulary enters
# here with its count at that commit, in a PR that says so.
B6_TERM_TOTALS = {
    '"E"': 4, '"density"': 39, '"energy"': 19, '"p"': 17, '"rho"': 10, '"rho_E"': 2, '"rho_u"': 2,
    '"rho_v"': 2, '"rho_w"': 2, '"rhou"': 2, '"rhov"': 2, '"rhow"': 2, '"vx"': 3, '"vy"': 3,
    '"vz"': 2, 'acoustic': 17, 'adiabatic': 8, 'bdf': 7, 'bgk': 5, 'canonical-roles': 5,
    'charge-density': 63, 'conservative-coarse-fine': 3, 'conservative-injection': 9,
    'conservative-linear': 5, 'conservative-polynomial': 3, 'dirichlet': 140, 'dirk': 4,
    'divergence-preserving': 29, 'electric': 25, 'electron': 11, 'enthalpy': 9, 'eos': 2,
    'equation-of-state': 8, 'euler': 152, 'fan-li': 75, 'gamma': 231, 'gaussian': 142,
    'gravitational': 5, 'gravity': 72, 'gummel': 11, 'hermite': 18, 'heun': 1, 'hll': 91,
    'hllc': 166, 'hyqmom': 44, 'ideal-gas': 13, 'imex': 72, 'inflow': 177, 'ion': 10,
    'laplacian': 166, 'linear-time-interpolation': 9, 'lorentz': 48, 'mach': 12, 'magnetic': 118,
    'minmod': 56, 'momentum': 127, 'monotonized': 8, 'muscl': 35, 'name-axis-link': 4,
    'neumann': 85, 'node-multilinear': 18, 'osher': 2, 'outflow': 25, 'poisson': 646,
    'pressure': 204, 'qmom': 12, 'reflective': 7, 'robin': 12, 'roe': 420, 'role-of': 4,
    'roles-for': 32, 'runge-kutta': 31, 'rusanov': 110, 'scharfetter': 11, 'slip': 27,
    'sound-speed': 11, 'ssprk': 39, 'strang': 3, 'superbee': 39, 'temperature': 33, 'upwind': 5,
    'van-leer': 26, 'vanleer': 18, 'variable-role': 55, 'velocity': 235, 'vlasov': 4, 'weno': 171,
}


@pytest.fixture(scope="module")
def scanned():
    return cf2.current_entries()


@pytest.fixture(scope="module")
def listed():
    return ratchet.read_list(cf2.EXCEPTIONS)[1]


# ---------------------------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------------------------
def test_frozen_engine_knowledge_matches_the_shrinking_exception_list(scanned, listed):
    found, _ = scanned
    comparison = ratchet.compare(found, listed)
    assert comparison.clean, ratchet.describe(comparison, "cf2_exceptions.txt", "CF2")


def test_exception_list_never_exceeds_its_b6_totals(listed):
    totals = collections.Counter(entry.term for entry in listed)
    unknown = sorted(set(totals) - set(B6_TERM_TOTALS))
    grown = {term: (count, B6_TERM_TOTALS[term]) for term, count in totals.items()
             if term in B6_TERM_TOTALS and count > B6_TERM_TOTALS[term]}
    assert not unknown, (
        "CF2: cf2_exceptions.txt lists terms that b6 did not have (MG6 forbids new exceptions): %s"
        % unknown)
    assert not grown, (
        "CF2: cf2_exceptions.txt lists more sites than b6 had (MG6 forbids new exceptions): %s"
        % grown)


def test_every_exception_names_the_issue_that_removes_it(listed):
    wrong = [entry.render() for entry in listed
             if entry.issue not in ratchet.KNOWN_ISSUES
             or entry.issue != cf2.owner_of(entry.path, entry.term)]
    assert not wrong, (
        "CF2: each exception names the issue of OWNER_RULES (cf2.py); change the rule, then run "
        "refresh_exceptions.py:\n  " + "\n  ".join(wrong[:50]))


def test_every_generic_sense_still_covers_a_site(scanned):
    _, used = scanned
    stale = [entry for entry in scan.GENERIC_SENSES if entry not in used]
    assert not stale, (
        "CF2: GENERIC_SENSES (scan.py) entries that no longer match any site; remove them so the "
        "table cannot hide a future occurrence:\n  " + "\n  ".join(map(repr, stale)))


def test_exception_list_is_sorted_and_well_formed(listed):
    assert listed == sorted(listed), "cf2_exceptions.txt must stay sorted (refresh_exceptions.py)"
    assert len(set(listed)) == len(listed), "cf2_exceptions.txt has duplicate entries"


# ---------------------------------------------------------------------------------------------
# The scan covers both trees of the frozen engine and every code file in them.
# ---------------------------------------------------------------------------------------------
def test_scan_covers_the_core_paths_and_the_native_tree():
    engine = scan.load_cf0()
    assert engine.is_core("include/pops/numerics/fv/numerical_flux.hpp")
    assert engine.is_core("python/pops/codegen/module_emit_brick.py")
    assert engine.is_core("python/bindings/core/init/init_system.cpp")
    assert engine.is_core("pyproject.toml")
    assert not engine.is_core("python/pops/lib/time/ssprk.py")
    assert not engine.is_core("examples/api_v02/models_M01_M14.py")
    assert engine.is_native(engine.native_tree + "riemann/flux.hpp")
    assert engine.is_frozen(engine.native_tree + "riemann/flux.hpp")
    assert not engine.is_core(engine.native_tree + "riemann/flux.hpp")


def test_every_frozen_engine_file_is_scanned_or_holds_no_code():
    unscanned = [rel for rel in scan.frozen_engine_files()
                 if scan.scanned_kind(rel) is None
                 and pathlib_suffix(rel) not in scan.UNSCANNED_SUFFIXES]
    assert not unscanned, (
        "CF2: frozen-engine files that no lexer reads; add a lexer or justify the suffix in "
        "UNSCANNED_SUFFIXES:\n  " + "\n  ".join(unscanned))


def pathlib_suffix(rel):
    name = rel.rsplit("/", 1)[-1]
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


# ---------------------------------------------------------------------------------------------
# Controls: the scanner refuses what CF2 refuses, and only that.
# ---------------------------------------------------------------------------------------------
def _terms(text, kind, path="include/pops/x.hpp"):
    return sorted({occurrence.term for occurrence, sense in scan.cf2_sites_in_text(path, text, kind)
                   if sense is None})


@pytest.mark.parametrize("source, expected", [
    ("switch (r) { case RiemannRouteId::kRusanov: return RusanovFlux{}; }", ["rusanov"]),
    ("using F = HLLCFlux; using G = HLLFlux;", ["hll", "hllc"]),
    ("const int c = vars.index_of(VariableRole::Density);", ["variable-role"]),
    ('route_table.emplace("muscl_vanleer", 2);', ["muscl", "vanleer"]),
    ("Real weno5z(Real a);", ["weno"]),
    ('extern "C" double pops_compiled_gamma();', ["gamma"]),
    ('if (kernel == "conservative_coarse_fine") {}', ["conservative-coarse-fine"]),
    ("enum class SlopeLimiter { MonotonizedCentral = 1 };", ["monotonized"]),
    ("struct ChargeDensityRhs {};", ["charge-density"]),
    ("int k = PhysicalBoundaryKind::dirichlet;", ["dirichlet"]),
])
def test_scanner_refuses_named_knowledge_in_cpp(source, expected):
    assert _terms(source, "cpp") == expected


@pytest.mark.parametrize("source, expected", [
    ('if "p" in model.prim_defs:\n    pass\n', ['"p"']),
    ('ROLES = {"rho": "density", "E": "energy"}\n', ['"E"', '"density"', '"energy"', '"rho"']),
    ("def role_of(name):\n    return CANONICAL_ROLES.get(name)\n", ["canonical-roles", "role-of"]),
    ('values = [output.name + "_" + axis_name for axis_name in axes]\n', ["name-axis-link"]),
    ("from pops.numerics.scharfetter_gummel import fitted\n", ["gummel", "scharfetter"]),
    ("route = {'rusanov': 0, 'hll': 1}['rusanov']\n", ["hll", "rusanov"]),
    ("profile = 'M03'\n", ["case-identifier"]),
])
def test_scanner_refuses_named_knowledge_in_python(source, expected):
    assert _terms(source, "python", path="python/pops/x.py") == expected


def test_scanner_reads_the_file_path():
    assert _terms("int x;", "cpp", path="include/pops/physics/fluids/euler.hpp") == ["euler"]


@pytest.mark.parametrize("source", [
    "// Rusanov flux, see Toro (2009)\nint x;",
    "/* HLLC and Roe\n   are named here only in a comment */ int y;",
    'throw std::runtime_error("positivity needs the Density role of a Euler model");',
    'POPS_REQUIRE(ok, "Rusanov requires wave speeds");',
    "struct FluxDensity { Real density; };",  # a flux density (spec D6), not a role
    "LoadBalanceStrategy::round_robin;",  # not the Robin boundary family
    "Real g = std::tgamma(x);",  # the Gamma function: tgamma is one word, not the word gamma
])
def test_scanner_ignores_comments_error_text_and_generic_senses_in_cpp(source):
    assert _terms(source, "cpp") == []


def test_scanner_ignores_docstrings_comments_and_error_text_in_python():
    source = (
        '"""Rusanov, HLL and the Euler equations, in a docstring."""\n'
        "# a comment about MUSCL\n"
        "def f(x):\n"
        '    """Poisson docstring."""\n'
        '    raise ValueError("WENO5 needs three ghost cells for %s" % "Roe")\n'
        "def g(ok):\n"
        '    assert ok, "HLLC requires a contact speed"\n'
        '    warnings.warn("Superbee is deprecated")\n'
        "    return x\n")
    assert _terms(source, "python", path="python/pops/x.py") == []


def test_scanner_keeps_the_exception_class_of_a_raise():
    source = "def f():\n    raise RusanovError('x')\n"
    assert _terms(source, "python", path="python/pops/x.py") == ["rusanov"]


def test_generic_sense_entries_are_exact_sites():
    for path, term, pattern, reason in scan.GENERIC_SENSES:
        assert term in scan.TERM_INFO, "GENERIC_SENSES names an unknown term %r" % term
        assert reason and pattern, "a generic sense needs a pattern and a reason"
        assert not path.endswith("/"), "a generic sense names one file, never a directory"


# ---------------------------------------------------------------------------------------------
# The ratchet itself fails on a new site and on a useless exception.
# ---------------------------------------------------------------------------------------------
def test_ratchet_fails_on_a_new_site_and_on_a_useless_exception():
    def entry(path, line, term):
        return ratchet.Entry(path, line, term, "N3", "x")

    listed = [entry("a.hpp", 3, "rusanov"), entry("b.hpp", 9, "hll")]
    moved = [entry("a.hpp", 40, "rusanov"), entry("b.hpp", 1, "hll")]
    assert ratchet.compare(moved, listed).clean  # shifted lines are not new sites
    grown = ratchet.compare(moved + [entry("a.hpp", 41, "rusanov")], listed)
    assert set(grown.new) == {("a.hpp", "rusanov")} and not grown.useless
    shrunk = ratchet.compare([entry("a.hpp", 3, "rusanov")], listed)
    assert set(shrunk.useless) == {("b.hpp", "hll")} and not shrunk.new
