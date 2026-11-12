"""CF2 (spec 0.3, section 16.2; Linear N2): the frozen engine does not know the model.

A source-only scan of the core paths and of the native component tree (FC6b, CF0 in
``frozen_engine/cf0_paths.toml``) refuses, outside comments and user-facing error text, the knowledge
of section 15.3: physical roles and role guessing, model, closure and equation-of-state names,
named numerical methods, schemes, limiters, interpolations, approximations, controllers, field
operators and boundary families, a primitive selected by its spelling, a physical constant in an
ABI, case identifiers, and so any route table or switch keyed on them (A5, FC4). The scanner is
``frozen_engine/scan.py``; its vocabulary is the closed, versioned file
``frozen_engine/cf2_vocabulary.toml`` (review of 11/11, P8; ``test_cf2_vocabulary.py``), which the
early audit of N2 (29/10) completed so that the counter-examples of FC2, FC3, FC4 and C43 are refused
whatever their spelling.

CF2 is a tripwire. It reads words, so it finds a name, not a formula coded without its name, which
CF4 finds by swapping bodies; the evidence of G3 is CF1, CF3, CF4 and CF6.

The test starts from the exception list of b6e64c1d (``frozen_engine/cf2_exceptions.txt``, one
entry ``file:line`` per occurrence of a token that holds a term, with the issue that removes it and
the spelling of the token) and fails both on a new occurrence and on an exception that has become
useless, so the list can only shrink (MG6). Sites pair with entries by path, term and spelling, with
multiplicity, so a site that replaces another of the same term in the same file is new, while a
re-wrapped statement keeps its sites. CI also compares the scan with the scan of the base of each PR
and the scanner with the scanner of the base (``refresh_exceptions.py --check-base``), so an entry
added by hand lets nothing in and a change to the scanner cannot hide a site unannounced. G3
requires the list to be empty (section 1.5). The closed list of interfaces of FC6b is asserted by
``test_native_interfaces_closed.py``.

Two rules of CF2 come from the amendment of 08/11 (review of N15 lot 1, point 4). The words
of ``pops.math`` that 2.6.3 lists (``grad``, ``div``, ``curl``, ``outflow``, ...) are the algebra of
the core: CF2 admits the definition of each word in ``pops.math`` by an exact rule of the scanner,
and refuses its name as a selector, an emitter, a kernel or a route key. A refusal table of the
surface (§21.7) is user-facing error text, so it can name ``laplacian`` or ``cfl_guard``. The third
rule of 08/11, for the compounds that name a quadrature, is gone with P3: the core knows
``pops.Quadrature`` only, so CF2 has no exception for the name of a rule.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import collections
import json
import shutil
import subprocess

import pytest

from tests.python.architecture.frozen_engine import cf2, ratchet, refresh_exceptions, scan


# The number of sites of each term in cf2_exceptions.txt at b6e64c1d. A total here may only go down
# (MG6). Listing a new site in cf2_exceptions.txt would need a higher number here; do not raise one:
# remove the occurrence instead. The terms that the early audit of N2 added on 29/10 entered here
# with their counts at that commit, in the PR that added them; a term added later does the same.
# The independent review of N2 (30/10) corrected three counts of b6 that the error-text rule had hid
# and added the solve algorithms; its follow-up review (30/10) made a site one occurrence of a
# token, which counted every term again. These totals are upper bounds; the slack that a removed
# site leaves is closed by the CI check against the base of each PR (refresh_exceptions.py
# --check-base), not by lowering them.
B6_TERM_TOTALS = {
    '"E"': 4, '"density"': 46, '"energy"': 28, '"mc"': 9, '"outlet"': 1, '"p"': 17, '"rho"': 10,
    '"rho_E"': 2, '"rho_u"': 2, '"rho_v"': 2, '"rho_w"': 2, '"rhou"': 2, '"rhov"': 2, '"rhow"': 2,
    '"vx"': 3, '"vy"': 3, '"vz"': 2, '"wall"': 5, '`B0`': 11, '`Density`': 35, '`Energy`': 19,
    '`MC`': 12, '`cs2`': 50, '`kMc`': 8, '`n0`': 32, '`qom`': 49, '`rho0`': 29, 'acoustic': 19,
    'adiabatic': 8, 'armijo': 66, 'arnoldi': 31, 'ars': 20, 'background-density': 13, 'bdf': 7,
    'bgk': 5, 'bi-cg-stab': 16, 'bicgstab': 27, 'c-rho': 28, 'canonical-roles': 5, 'cfl': 141,
    'cg': 160, 'charge-density': 66, 'chebyshev': 7, 'coarse-fine-ghost-interpolation': 27,
    'component-role': 39, 'compressible': 38, 'conjugate-gradient': 1,
    'conservative-coarse-fine': 3, 'conservative-injection': 9, 'conservative-linear': 5,
    'conservative-polynomial': 3, 'constant-extrapolation': 11, 'constant-injection': 36,
    'courant': 86, 'davis': 17, 'dense-output': 12, 'dirichlet': 155, 'dirk': 4,
    'discrete-gradient-stencil': 17, 'divergence': 167, 'divergence-preserving': 29,
    'drift-diffusion': 7, 'einfeldt': 17, 'electric': 31, 'electron': 15, 'electrostatic': 10,
    'elliptic-route': 7, 'energy-role': 4, 'enthalpy': 11, 'eos': 2, 'equation-of-state': 8,
    'euler': 160, 'ex-b': 15, 'exb': 20, 'fac': 655, 'fan-li': 88, 'fft': 489, 'fftw': 56,
    'field-solver-route': 3, 'foextrap': 10, 'four-pi-g': 46, 'from-semantic': 2, 'gamma': 323,
    'gauss-seidel': 52, 'gaussian': 165, 'geometric-mg': 193, 'gmres': 133, 'gradient': 532,
    'gradient-above': 23, 'gradient-below': 20, 'gravitational': 5, 'gravity': 86, 'gummel': 12,
    'harten': 26, 'hermite': 18, 'heun': 1, 'hll': 103, 'hllc': 192, 'hyqmom': 46, 'ideal-gas': 13,
    'imex': 79, 'imexrk': 19, 'inflow': 183, 'ion': 14, 'ionization': 9, 'isothermal': 60,
    'jacobi': 68, 'laplacian': 168, 'lie': 3, 'limiter-route': 37, 'linear-interpolation': 16,
    'linear-prolongation': 18, 'linear-temporal-interpolation': 14, 'linear-time-interpolation': 9,
    'lorentz': 51, 'lu': 127, 'mach': 15, 'magnetic': 144, 'maxwellian': 6, 'minmod': 69,
    'momentum': 172, 'monotonized': 8, 'multispecies': 10, 'muscl': 40, 'name-axis-link': 4,
    'neumann': 93, 'newton': 1088, 'no-flux': 52, 'node-multilinear': 18, 'osher': 2, 'outflow': 28,
    'parse-role': 17, 'parse-source-route': 1, 'physical-default': 141, 'poisson': 746,
    'potential-force': 21, 'pressure': 247, 'q-over-m': 31, 'qmom': 12, 'recon-route': 29,
    'reflective': 7, 'require-density': 4, 'richardson': 40, 'riemann-route': 18, 'robin': 12,
    'roe': 476, 'role-canonical': 8, 'role-from-name': 5, 'role-index': 8, 'role-key': 12,
    'role-name': 13, 'role-of': 4, 'role-token': 28, 'roles-csv': 7, 'roles-for': 32,
    'runge-kutta': 35, 'rusanov': 125, 'scharfetter': 12, 'select-riemann': 18, 'slip': 29,
    'sound-speed': 15, 'source-route-id': 9, 'source-route-tokens': 1, 'species': 140, 'ssprk': 46,
    'strang': 3, 'superbee': 44, 'temperature': 45, 'thermal': 13, 'time-route': 9,
    'transport-route': 7, 'upwind': 7, 'v-cycle': 7, 'vacuum-floor': 58, 'van-leer': 30,
    'vanleer': 21, 'variable-role': 60, 'variable-roles': 13, 'variable-semantic': 64, 'vcycle': 8,
    'velocity': 300, 'vlasov': 4, 'weno': 220, 'zero-flux': 29, 'zhang': 4,
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


def test_owner_rules_read_the_placement_list():
    """The N9 path rule holds the files of the N9 entries of the CF0 placement list (or the solver
    directory that holds them), and the G06 path rule the files of the G06 entries, except those
    that G11 takes over whole for their named interpolations (C43). The term rules before them keep
    the field operators and boundary families of 18.4 with G10 and G04."""
    placement = scan.load_placement()
    path_rules = {issue: paths for issue, paths, terms, _ in cf2.OWNER_RULES
                  if paths is not None and terms is None}
    n9 = {rel for native in placement if native["issue"] == "N9" for rel in native["files"]}
    uncovered = sorted(rel for rel in n9 if not scan.covered(rel, cf2._NATIVE_SOLVERS))
    unlisted = sorted(entry for entry in cf2._NATIVE_SOLVERS
                      if not entry.endswith("/") and entry not in n9)
    assert path_rules["N9"] == cf2._NATIVE_SOLVERS
    assert not uncovered and not unlisted, (
        "cf2.py _NATIVE_SOLVERS differs from the N9 entries of cf0_native_placement.toml: N9 files "
        "outside it %s, files of it that no N9 entry names %s" % (uncovered, unlisted))
    g06 = {rel for native in placement if native["issue"] == "G06" for rel in native["files"]
           if not scan.covered(rel, path_rules["G11"])}
    uncovered = sorted(rel for rel in g06 if not scan.covered(rel, path_rules["G06"]))
    assert not uncovered, "G06 placement files outside the G06 path rule: %s" % uncovered


def test_every_generic_sense_still_covers_a_site(scanned):
    _, used = scanned
    stale = [entry for entry in scan.GENERIC_SENSES if entry not in used]
    assert not stale, (
        "CF2: GENERIC_SENSES (scan.py) entries that no longer match any site; remove them so the "
        "table cannot hide a future occurrence:\n  " + "\n  ".join(map(repr, stale)))


def test_exception_list_is_sorted_and_well_formed(listed):
    assert listed == sorted(listed), "cf2_exceptions.txt must stay sorted (refresh_exceptions.py)"
    # an entry repeats when its token occurs more than once on its line; it never has an empty
    # spelling
    assert all(entry.excerpt and entry.excerpt == entry.excerpt.strip() for entry in listed)


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
    ("switch (r) { case RiemannRouteId::kRusanov: return RusanovFlux{}; }",
     ["riemann-route", "rusanov"]),
    ("using F = HLLCFlux; using G = HLLFlux;", ["hll", "hllc"]),
    ("const int c = vars.index_of(VariableRole::Density);", ["`Density`", "variable-role"]),
    ('route_table.emplace("muscl_vanleer", 2);', ["muscl", "vanleer"]),
    ("Real weno5z(Real a);", ["weno"]),
    ('extern "C" double pops_compiled_gamma();', ["gamma"]),
    ('if (kernel == "conservative_coarse_fine") {}', ["conservative-coarse-fine"]),
    ("enum class SlopeLimiter { MonotonizedCentral = 1 };", ["monotonized"]),
    ("struct ChargeDensityRhs {};", ["charge-density"]),
    ("int k = PhysicalBoundaryKind::dirichlet;", ["dirichlet"]),
    # The counter-examples of FC2, FC3, FC4 and C43 under their other spellings (early audit).
    ("void gradient(field_type& output, field_type& input) const;", ["gradient"]),
    ("Real divergence = Real(0);", ["divergence"]),
    ("const int c = vars.index_of(VariableSemantic::Density);", ["`Density`", "variable-semantic"]),
    ("return select_riemann<Dim, Variables>(std::move(request), MC{});",
     ["`MC`", "select-riemann"]),
    ("case TransferKind::CoarseFineGhostInterpolation: break;",
     ["coarse-fine-ghost-interpolation"]),
    ("inline constexpr Real kPhysicalDefaultFourPiG = Real(1);", ["four-pi-g", "physical-default"]),
    ("Real qom = Real(1);", ["`qom`"]),
    ('if (token == "foextrap") {}', ["foextrap"]),
    ("double step_cfl(double cfl);", ["cfl"]),
    # A term inside a longer term is the longer term only.
    ("DivergencePreservingFaceProlongation = 6,", ["divergence-preserving"]),
    ("opcode == POPS_TAGGING_GRADIENT_ABOVE_V1", ["gradient-above"]),
    # The error text of a statement ends where its parentheses close.
    ('if (failure(x)) { route = "rusanov"; }', ["rusanov"]),
    # A name opens error text only where it is used as error text (independent review, B1): these
    # names are code, and so is the condition of an assertion.
    ('const bool chosen = (name == "rusanov");', ["rusanov"]),
    ('const bool expected = (name == "rusanov");', ["rusanov"]),
    ('const bool required = route == "hll";', ["hll"]),
    ('auto failed_routes = table.at("weno5");', ["weno"]),
    ('template <class T> requires Flux<T> constexpr auto k = lookup("hllc");', ["hllc"]),
    ('void f(const std::string& expected_identity = {}, const std::string& riemann = "rusanov");',
     ["rusanov"]),
    ('static_assert(transport_n_vars_ct("exb") == 3, "the drift has three variables");', ["exb"]),
    # Solve algorithms (independent review, R2).
    ('if (method == "gmres") { return TensorCoarseGMRES{}; }', ["gmres"]),
    ("using S = BiCGStabSolver; class GeometricMG;", ["bi-cg-stab", "geometric-mg"]),
    ('if (family_route == "fft") { solver = PoissonFFTSolver{}; }', ["fft", "poisson"]),
    ("struct DenseLU {}; NewtonOptions newton;", ["lu", "newton"]),
    ('const char* d = "exact-ranked constant-coefficient conjugate gradient";',
     ["conjugate-gradient"]),
    # Adjacent string literals are one literal, as the compiler joins them (follow-up review, R5).
    ('const char* d = "exact-ranked constant-coefficient conjugate "\n                "gradient";',
     ["conjugate-gradient"]),
    ('auto s = "rusa" /* split */ "nov";', ["rusanov"]),
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
    ("from pops.physics.roles import ComponentRole, native_role_token\n",
     ["component-role", "role-token"]),
    ("PARAMETERS = {'gravity': ['sign', 'four_pi_G', 'rho0']}\n",
     ["`rho0`", "four-pi-g", "gravity"]),
    ("class AdaptiveCFL(StepStrategy):\n    cfl: float\n", ["cfl"]),
    ("history = LinearInterpolation()\n", ["linear-interpolation"]),
    ("GHOSTS = {'mc': 2}\n", ['"mc"']),
    # A failure helper whose result is used is code (independent review, B1).
    ('route = fail_safe_route("rusanov")\n', ["rusanov"]),
    # The <name>_<axis> link in its other spellings (independent review, O2).
    ('name = f"{output.name}_{axis_name}"\n', ["name-axis-link"]),
    ('name = "%s_%s" % (output.name, axis)\n', ["name-axis-link"]),
    ('name = "{}_{}".format(output.name, axis_name)\n', ["name-axis-link"]),
    ('name = "_".join((output.name, axis_name))\n', ["name-axis-link"]),
    # Solve algorithms (independent review, R2); the conjugate gradient is not the gradient.
    ("class GeometricMG(Descriptor):\n    pass\n", ["geometric-mg"]),
    ("def BiCGStab(**options):\n    return CG(**options)\n", ["bi-cg-stab", "cg"]),
    ("solver = FFT() if periodic else Newton()\n", ["fft", "newton"]),
    ("solver = 'conjugate_gradient'\n", ["conjugate-gradient"]),
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
    "std::string source_route_contract;",  # the sampled side of a projection, not a source route
    "auto w = std::make_unique<KrylovWorkspace>(); MultigridCycle c;",  # FC6b interfaces
    # error text whose parentheses hold braces and semicolons (a requires-expression, a lambda)
    'static_assert(requires { Brick::dimension; }, "a source with momentum components must declare '
    'its spatial dimension");',
    'POPS_REQUIRE(std::all_of(v.begin(), v.end(), [](Real s) { return s > 0; }), "Rusanov needs '
    'positive speeds");',
    # the call of a failure helper, an error object that is built or assigned
    'reject(request, "the Rusanov route needs wave speeds");',
    'fail_<ContractError>(lane, "HLLC needs a contact speed");',
    'const auto& lane = p_->require_prepared_engine_lane("prepared AMR block Poisson RHS");',
    'message_error = "Rusanov failed";',
    'auto s = flag ? "rusa" : "nov";',  # two literals, not adjacent: nothing joins them
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


def test_scanner_refuses_generic_solve_algorithms_in_the_core_paths_only(tmp_path):
    """FC6: a qualified substitution in the native tree may name its generic algorithm (GMRES);
    Newton, which FC6 does not admit as a substitution, is refused in both trees."""
    engine = scan.FrozenEngine(core_include=("include/",), core_exclude=(), native_tree="native/")
    for rel in ("include/pops/krylov.hpp", "native/krylov/driver.hpp"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("struct GMRESDriver {}; NewtonOptions options;\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    found = {(o.path, o.term) for o in scan.scan_cf2(tmp_path, engine)[0]}
    assert found == {("include/pops/krylov.hpp", "gmres"), ("include/pops/krylov.hpp", "newton"),
                     ("native/krylov/driver.hpp", "newton")}


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


def test_refresh_accepts_only_a_site_that_left_one_file_for_another():
    def entry(path, line, term, excerpt):
        return ratchet.Entry(path, line, term, "N3", excerpt)

    listed = [entry("a.hpp", 3, "rusanov", "return RusanovFlux{};"),
              entry("a.hpp", 9, "hll", "using H = HLLFlux;")]
    split = [entry("b.hpp", 1, "rusanov", "return RusanovFlux{};"),
             entry("a.hpp", 8, "hll", "using H = HLLFlux;")]
    pairs, unexplained = ratchet.moves(split, listed)
    assert [(old.path, new.path) for old, new in pairs] == [("a.hpp", "b.hpp")] and not unexplained
    # one Rusanov site removed and another added elsewhere is a new site, not a move
    swapped = [entry("b.hpp", 1, "rusanov", "auto f = rusanov_flux(u);"),
               entry("a.hpp", 8, "hll", "using H = HLLFlux;")]
    pairs, unexplained = ratchet.moves(swapped, listed)
    assert not pairs and [site.path for site in unexplained] == ["b.hpp"]
    # a copy that leaves the original in place is not a move either
    copied = listed + [entry("b.hpp", 1, "rusanov", "return RusanovFlux{};")]
    pairs, unexplained = ratchet.moves(copied, listed)
    assert not pairs and [site.path for site in unexplained] == ["b.hpp"]


def test_ratchet_pairs_sites_by_spelling_so_a_swap_in_one_file_fails():
    """The first probe of the independent review: a new "rusanov" added to a file while a listed
    site of the same term is renamed away keeps the count of (file, term), and still fails."""
    def entry(path, line, term, excerpt):
        return ratchet.Entry(path, line, term, "N3", excerpt)

    listed = [entry("a.hpp", 137, "rusanov", "kRusanovInvalidStability"),
              entry("a.hpp", 200, "rusanov", "RusanovFlux")]
    swapped = [entry("a.hpp", 62, "rusanov", '"rusanov"'),
               entry("a.hpp", 201, "rusanov", "RusanovFlux")]
    comparison = ratchet.compare(swapped, listed)
    new, _ = comparison.new[("a.hpp", "rusanov")]
    useless, _ = comparison.useless[("a.hpp", "rusanov")]
    assert [site.excerpt for site in new] == ['"rusanov"']
    assert [entry.excerpt for entry in useless] == ["kRusanovInvalidStability"]


def test_scanner_reports_one_site_per_token_occurrence():
    """A site is one token that holds the term (follow-up review, R5): the rest of its line does not
    enter its spelling, a statement re-wrapped over two lines keeps its sites, and a token repeated
    on one line is two sites."""
    def spelling(source):
        return sorted((occurrence.term, occurrence.excerpt) for occurrence, _
                      in scan.cf2_sites_in_text("include/pops/x.hpp", source, "cpp"))

    expected = [("rusanov", '"rusanov"'), ("rusanov", "RusanovFlux")]
    assert spelling('auto f = RusanovFlux{}; route = "rusanov";') == expected
    assert spelling('const auto g = RusanovFlux{}; next = "rusanov";') == expected
    assert spelling('const auto g = RusanovFlux{};\nnext = "rusanov";') == expected
    repeated = spelling('if (name == "rusanov" || alias == "rusanov") {}')
    assert repeated == [("rusanov", '"rusanov"')] * 2


def test_a_rewrap_or_a_partial_removal_passes_and_a_repeated_token_fails(tmp_path):
    """The probes of the follow-up review (R5), through the ratchet and the base check: clang-format
    re-wraps a statement or breaks a string literal; a slice removes one token of a line, whose
    entry becomes useless (refresh drops it); a token is added beside the same token."""
    base_text = ('static_assert(CompressibleFlux::n_vars == transport_n_vars_ct("compressible"),\n'
                 '              "the compressible flux carries its transport variables");\n'
                 'if (name == "rusanov") { return 1; }\n'
                 'const char* kind = "exact-ranked constant-coefficient conjugate gradient";\n')
    heads = {
        "rewrapped": base_text.replace("== transport", "==\n              transport"),
        "broken literal": base_text.replace('constant-coefficient conjugate gradient"',
                                            'constant-coefficient "\n                   '
                                            '"conjugate gradient"'),
        "partial": base_text.replace("CompressibleFlux::n_vars", "5"),
        "repeated": base_text.replace('(name == "rusanov")',
                                      '(name == "rusanov" || alias == "rusanov")'),
    }

    def sites(root):
        return [occurrence for occurrence, sense in scan.cf2_sites("x.hpp", root) if sense is None]

    def spelled(groups):
        return sorted(site.excerpt for found, _ in groups.values() for site in found)

    base = tmp_path / "base"
    base.mkdir()
    (base / "x.hpp").write_text(base_text)
    results = {}
    for name, text in heads.items():
        head = tmp_path / name
        head.mkdir()
        (head / "x.hpp").write_text(text)
        comparison = ratchet.compare(sites(head), sites(base))
        _, added = ratchet.new_sites(sites(base), sites(head), base, head)
        results[name] = (spelled(comparison.new), spelled(comparison.useless),
                         sorted(site.excerpt for site in added))
    assert results["rewrapped"] == ([], [], [])
    assert results["broken literal"] == ([], [], [])
    assert results["partial"] == ([], ["CompressibleFlux"], [])
    assert results["repeated"] == (['"rusanov"'], [], ['"rusanov"'])


def test_base_check_refuses_a_site_moved_by_hand_and_accepts_a_moved_line(tmp_path):
    """The second and third probes of the independent review: the list is not read at all. A site
    of the head that the base does not hold fails, whatever the list says, unless it moved from
    another file with its source line (a file renamed or split)."""
    def tree(name, files):
        root = tmp_path / name
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        return root

    def check(head):
        def sites(root):
            return [occurrence for rel in ("a.hpp", "b.hpp")
                    for occurrence, sense in scan.cf2_sites(rel, root) if sense is None]
        return ratchet.new_sites(sites(base), sites(head), base, head)

    base = tree("base", {"a.hpp": "int x;\nreturn kRusanovInvalidStability;\n",
                         "b.hpp": "int y;\n"})
    moved, added = check(tree("by_hand", {"a.hpp": "int x;\n",
                                          "b.hpp": 'int y;\nauto f = "rusanov";\n'}))
    assert not moved and [(site.path, site.excerpt) for site in added] == [("b.hpp", '"rusanov"')]
    moved, added = check(tree("split", {"a.hpp": "int x;\n",
                                        "b.hpp": "int y;\n\nreturn kRusanovInvalidStability;\n"}))
    assert [(old.path, new.path) for old, new in moved] == [("a.hpp", "b.hpp")] and not added
    # the same spelling on another line of another file is not a move
    moved, added = check(tree("rewritten", {"a.hpp": "int x;\n",
                                            "b.hpp": "int y;\nf(kRusanovInvalidStability);\n"}))
    assert not moved and [site.path for site in added] == ["b.hpp"]


def test_check_base_scans_the_base_revision_in_a_worktree(tmp_path, monkeypatch):
    """End to end, as CI runs it: a site added to the working tree fails against HEAD, and the
    temporary worktree is removed."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, stdout=subprocess.DEVNULL)

    engine = scan.FrozenEngine(core_include=("include/",), core_exclude=(), native_tree="native/")
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "x.hpp").write_text("return RusanovFlux{};\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    monkeypatch.setattr(scan, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(refresh_exceptions, "LISTS", {
        "t": (tmp_path / "t_exceptions.txt", lambda root: cf2.current_entries(root, engine)[0])})
    assert refresh_exceptions.check_base("HEAD", ["t"]) == 0
    (tmp_path / "include" / "x.hpp").write_text('return RusanovFlux{};\nauto f = "rusanov";\n')
    assert refresh_exceptions.check_base("HEAD", ["t"]) == 1
    worktrees = subprocess.run(["git", "-C", str(tmp_path), "worktree", "list"], check=True,
                               stdout=subprocess.PIPE, text=True).stdout.splitlines()
    assert len(worktrees) == 1


def test_narrowings_are_the_sites_that_a_scanner_change_hides():
    def entry(path, term, excerpt):
        return ratchet.Entry(path, 1, term, "N3", excerpt)

    before = [entry("a.hpp", "cg", '"cartesian_cg"'), entry("a.hpp", "rusanov", "RusanovFlux"),
              entry("b.hpp", "fft", "FFT")]
    after = [entry("a.hpp", "rusanov", "RusanovFlux"), entry("b.hpp", "fft", "FFT"),
             entry("b.hpp", "gmres", "GMRES")]
    assert refresh_exceptions.narrowings(before, after) == {("a.hpp", "cg"): (1, 0)}


def test_check_base_refuses_a_narrowing_that_the_pr_does_not_declare(tmp_path, monkeypatch):
    """O4 of the follow-up review: the base check also scans the tree with the scanner of the base,
    so a change to the scanner that hides a site (here a wider generic sense) fails unless the PR
    declares it in accepted_narrowings.txt, and so does a declared narrowing that the PR does not
    make."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, stdout=subprocess.DEVNULL)

    for rel in ("tests/__init__.py", "tests/python/__init__.py",
                "tests/python/architecture/__init__.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(scan.REPO_ROOT / rel, tmp_path / rel)
    shutil.copytree(scan.HERE, tmp_path / refresh_exceptions.NARROWINGS.parent,
                    ignore=shutil.ignore_patterns("__pycache__", "*.txt"))
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "x.hpp").write_text(
        'return RusanovFlux{};\nauto route = "cartesian_cg";\n')
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "base")

    def widened(root):  # the scanner of the PR, with a generic sense that hides "cartesian_cg"
        return [entry for entry in cf2.current_entries(root)[0]
                if entry.excerpt != '"cartesian_cg"']

    monkeypatch.setattr(scan, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(refresh_exceptions, "LISTS",
                        {"cf2": (tmp_path / "cf2_exceptions.txt", widened)})
    assert refresh_exceptions.check_base("HEAD", ["cf2"]) == 1
    declared = tmp_path / refresh_exceptions.NARROWINGS
    declared.write_text("cf2 include/x.hpp cg 1 | the name of a generic routine\n")
    assert refresh_exceptions.check_base("HEAD", ["cf2"]) == 0
    with declared.open("a") as stream:
        stream.write("cf2 include/x.hpp rusanov 1 | a narrowing that the PR does not make\n")
    assert refresh_exceptions.check_base("HEAD", ["cf2"]) == 1


def test_every_exception_list_is_checked_against_the_base():
    lists = {path.name for path, _ in refresh_exceptions.LISTS.values()}
    files = {path.name for path in scan.HERE.glob("*_exceptions.txt")}
    assert files <= lists, (
        "MG6: every exception list is registered in refresh_exceptions.LISTS, which CI checks "
        "against the base of each PR: %s" % sorted(files - lists))


# ---------------------------------------------------------------------------------------------
# The words of pops.math, the refusal tables of the surface and the quadratures (CF2 amendment of
# 08/11, review of N15 lot 1, point 4).
# ---------------------------------------------------------------------------------------------
_MATH = "python/pops/surface/math.py"


@pytest.mark.parametrize("source", [
    'def curl(e):\n    """The curl of a vector field."""\n'
    '    return Call("curl", (_value(e, "curl"),))\n',
    # R20: outflow, the empty relation, whose NoData form selects nothing
    "def outflow(q):\n    return Relation(q, NoData())\n",
    'outflow = _later("outflow", _TRACES)\n',
    "exp, curl = (_elementary(word) for word in _WORDS)\n",
    '__all__ = ["grad", "div", "curl", "outflow"]\n',
])
def test_pops_math_defines_its_words(source):
    """The definition of a word of 2.6.3 in pops.math: its name, its self-reference in its body
    and in its assigned value, and its entry in __all__."""
    for path in scan.POPS_MATH_MODULES:
        assert _terms(source, "python", path=path) == []


@pytest.mark.parametrize("source, expected", [
    ('if op == "curl":\n    pass\n', ["curl"]),  # a selector
    ('KERNELS = {"curl": 1, "outflow": 2}\n', ["curl", "outflow"]),  # a route key
    ("def curl_kernel(e):\n    pass\n", ["curl"]),  # the name of a kernel
    ("class CurlEmitter:\n    pass\n", ["curl"]),  # the name of an emitter
    ("class OutflowBoundary:\n    pass\n", ["outflow"]),  # a boundary family
    ("def f(e):\n    def curl(v):\n        pass\n", ["curl"]),  # not the module's definition
    ('def grad(e):\n    return Call("curl", (e,))\n', ["curl"]),  # another word's body
    ("def laplacian(e):\n    pass\n", ["laplacian"]),  # not a word of 2.6.3 (div(grad(e)))
    ("from ._vector import curl\n", ["curl"]),  # defined elsewhere, re-exported
    # a selector or a route key inside the definition of the word (review of #748, point 1)
    ('def curl(e, kind=None):\n    if kind == "curl":\n        return e\n'
     '    return Call("curl", (e,))\n', ["curl"]),
    ('def curl(e):\n    return _KERNELS["curl"](e)\n', ["curl"]),
    ('def curl(e):\n    routes = {"curl": k}\n    return routes["curl"](e)\n', ["curl"]),
    ('def curl(e):\n    return _KERNELS.get("curl")(e)\n', ["curl"]),
    ('def outflow(q, form):\n    match form:\n        case "outflow":\n            return q\n',
     ["outflow"]),
    ('class curl:\n    def emit(self, op):\n        return op == "curl"\n', ["curl"]),
    ('def outflow(q):\n    return getattr(_FORMS, "outflow")(q)\n', ["outflow"]),
    ('curl = {"curl": k}\n', ["curl"]),
    ('curl = _ROUTES["curl"]\n', ["curl"]),
    ('curl = _ROUTES.get("curl")\n', ["curl"]),
])
def test_pops_math_refuses_its_words_as_selectors_emitters_kernels_and_route_keys(source, expected):
    assert _terms(source, "python", path=_MATH) == expected


def test_a_word_of_pops_math_is_refused_outside_pops_math():
    source = 'def curl(e):\n    return Call("curl", (e,))\n__all__ = ["curl"]\n'
    assert _terms(source, "python", path="python/pops/surface/_expr.py") == ["curl"]
    assert _terms(source, "python", path="python/pops/codegen/math.py") == ["curl"]


def test_the_words_of_pops_math_that_hold_a_term_are_curl_and_outflow():
    """POPS_MATH_WORDS is the closed list of 2.6.3, the "math" list of the lint of the surface
    when the tree has it; only curl and outflow of these words hold a CF2 term, so the rule admits
    nothing else."""
    held = {word for word in scan.POPS_MATH_WORDS if scan.terms_in(word, "ident")}
    assert held == {"curl", "outflow"}
    assert {"grad", "div", "ddt", "trace", "normal"} <= scan.POPS_MATH_WORDS
    assert not {"laplacian", "Laplacian", "gradient", "divergence"} & scan.POPS_MATH_WORDS
    lint = scan.REPO_ROOT / "scripts" / "surface_lint_words.json"
    if lint.exists():
        assert set(json.loads(lint.read_text(encoding="utf-8"))["math"]) == scan.POPS_MATH_WORDS


def test_the_reads_of_an_entity_that_hold_a_term_are_admitted_only_where_they_are_defined():
    """N3 lot 1: face.normal_gradient and term.gradient_variables are words of 2.6.4 (the "words"
    list of the lint of the surface); the module of the entities defines them, and only the name of
    each definition is admitted."""
    assert {word for word in scan.SURFACE_ENTITY_READS if scan.terms_in(word, "ident")} == \
        scan.SURFACE_ENTITY_READS
    lint = scan.REPO_ROOT / "scripts" / "surface_lint_words.json"
    if lint.exists():
        assert scan.SURFACE_ENTITY_READS <= set(json.loads(lint.read_text(encoding="utf-8"))[
            "words"])
    defined = ("class FaceEntity:\n    def normal_gradient(self, value):\n        return value\n"
               "class TermEntity:\n    @property\n    def gradient_variables(self):\n"
               "        return []\n")
    assert _terms(defined, "python", path=scan.SURFACE_ENTITY_MODULE) == []
    assert _terms(defined, "python", path="python/pops/surface/_elaborate.py") == ["gradient"]
    selected = ('def normal_gradient(value):\n    return value\n'
                'class FaceEntity:\n    def read(self):\n        return self.normal_gradient\n'
                'TABLE = {"gradient_variables": 1}\n')
    assert _terms(selected, "python", path=scan.SURFACE_ENTITY_MODULE) == ["gradient"]
    for source in ('def normal_gradient(value):\n    return value\n',
                   'class FaceEntity:\n    def read(self):\n        return self.normal_gradient\n',
                   'TABLE = {"gradient_variables": 1}\n'):
        assert _terms(source, "python", path=scan.SURFACE_ENTITY_MODULE) == ["gradient"]


@pytest.mark.parametrize("source", [
    '_EARLIER = {\n    "laplacian": SpellingRefusal("div(grad(e))"),\n'
    '    "Q": SpellingRefusal("Fraction(1, 2)"),\n}\n',
    '_EARLIER = {"cfl_guard": "pops.lib.time.cfl_limit(...)", "Rectangle": "pops.Box(x=...)"}\n',
    '_EARLIER_METHODS: dict = {"cfl_guard": SpellingRefusal("a guard of the method")}\n',
    # a table of the reads of 2.6.4 that a later lot elaborates, valued by the sentence of its lot
    '_LATER_FACE = {"gradient": "a later lot of N15 with the spacing (C31)",\n'
    '               "velocity": "a later lot of N15 with the moving configurations"}\n',
])
def test_a_refusal_table_of_the_surface_is_error_text(source):
    assert _terms(source, "python", path="python/pops/surface/__init__.py") == []


@pytest.mark.parametrize("source, path, expected", [
    # a table of the surface whose value is code, not a message
    ('_EARLIER = {"laplacian": laplacian_kernel}\n', "python/pops/surface/math.py", ["laplacian"]),
    # a table under another name: a dictionary of names to names is a route table
    ('ALIASES = {"rusanov": "hll"}\n', "python/pops/surface/_numerics.py", ["hll", "rusanov"]),
    ('ROUTES = {"laplacian": SpellingRefusal("x")}\n', "python/pops/surface/math.py",
     ["laplacian"]),
    # a refusal table outside the surface is code
    ('_EARLIER = {"rusanov": "hll"}\n', "python/pops/codegen/x.py", ["hll", "rusanov"]),
    # a bare or dotted name is not a message (review of #748, point 2)
    ('_EARLIER = {"laplacian": "laplacian_kernel"}\n', "python/pops/surface/__init__.py",
     ["laplacian"]),
    ('_EARLIER = {"rusanov": "hll"}\n', "python/pops/surface/__init__.py", ["hll", "rusanov"]),
    ('_EARLIER_ROUTES = {"laplacian": "pops.lib.laplacian"}\n', "python/pops/surface/math.py",
     ["laplacian"]),
    # a table of later reads whose value is a name or a reference, not a sentence, is a route
    ('_LATER_FACE = {"gradient": "gradient_kernel"}\n', "python/pops/surface/_scheme.py",
     ["gradient"]),
    ('_LATER_FACE = {"gradient": _OWNER}\n', "python/pops/surface/_scheme.py", ["gradient"]),
    # _LATER alone, the table of pops.surface whose values are references, is not a refusal table
    ('_LATER = {"gradient": "a later lot"}\n', "python/pops/surface/_scheme.py", ["gradient"]),
    # outside the surface, a table of later reads is code
    ('_LATER_FACE = {"gradient": "a later lot"}\n', "python/pops/codegen/x.py", ["gradient"]),
])
def test_only_a_refusal_table_of_the_surface_is_error_text(source, path, expected):
    assert _terms(source, "python", path=path) == expected


@pytest.mark.parametrize("source, kind, expected", [
    # a named quadrature rule in a core path is refused (review of 11/11, P3)
    ("class GaussHermite:\n    pass\n", "python", ["gauss-hermite"]),
    ('rule = "gauss_hermite"\n', "python", ["gauss-hermite"]),
    ("GaussLegendre = 1\n", "python", ["legendre"]),
    ("struct GaussHermiteNodes {};", "cpp", ["gauss-hermite"]),
    ("auto lobatto_nodes = 3;", "cpp", ["lobatto"]),
    # the Hermite closure keeps its name
    ("def fan_li15_from_hermite(m):\n    pass\n", "python", ["fan-li", "hermite"]),
    ("hermite_raw_edge = 1\n", "python", ["hermite"]),
    # pops.Quadrature itself names no rule
    ("class Quadrature:\n    pass\n", "python", []),
])
def test_a_named_quadrature_rule_is_refused_in_the_core(source, kind, expected):
    path = "python/pops/x.py" if kind == "python" else "include/pops/x.hpp"
    assert _terms(source, kind, path=path) == expected
