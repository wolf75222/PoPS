"""ADC-618 source-scanning fence over user-visible native numeric constants.

Makes it IMPOSSIBLE to add a new ``inline constexpr`` numeric constant to the enumerated native
headers without CLASSIFYING it in the effective-options report. The test scans the header SOURCE
(so it runs in the architecture tier with plain python3, no ``_pops``), collects every
``inline constexpr <numeric-type> kName = ...`` declaration, and asserts each name appears in
``pops.runtime.defaults._CONSTANT_CLASSIFICATION`` with one of the four allowed classes.

A new unclassified user-visible constant fails this test -> it cannot ship unreported. The C++
``numerical_defaults_report_to_dict`` mirrors the same map; ``test_numerical_defaults_reports.py``
checks the two agree at runtime.
"""

import ast
import math  # noqa: F401  (kept for future tolerance-based value fences)
import re
from pathlib import Path

_ALLOWED_CLASSES = {"public_knob", "internal_default", "diagnostic_only", "hard_limit"}

# The repo root: this file is tests/python/architecture/<here>.
_ROOT = Path(__file__).resolve().parents[3]


def _load_classification() -> dict:
    """Parse ``_CONSTANT_CLASSIFICATION`` from the defaults.py SOURCE without importing ``pops``.

    The architecture tier runs with plain python3 (no ``_pops``); importing ``pops.runtime.defaults``
    would pull in the native extension via the package bootstrap. So we read the literal dict out of
    the source with ``ast`` -- the same single source of truth, no runtime import."""
    src = (_ROOT / "python/pops/runtime/defaults.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        # The assignment carries a ``: dict`` annotation -> ast.AnnAssign (single target); a plain
        # assignment would be ast.Assign. Handle both so the fence is robust to either form.
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_CONSTANT_CLASSIFICATION":
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_CONSTANT_CLASSIFICATION":
                    return ast.literal_eval(node.value)
    raise AssertionError("_CONSTANT_CLASSIFICATION not found in pops/runtime/defaults.py")


_CONSTANT_CLASSIFICATION = _load_classification()

# Headers whose inline constexpr numeric constants are the user-visible native defaults / limits.
_SCANNED_HEADERS = (
    "include/pops/runtime/numerical_defaults.hpp",
    "include/pops/core/foundation/types.hpp",
    "include/pops/runtime/config/runtime_params.hpp",
)

# inline constexpr <numeric type> kName = ... ;  (the numeric-typed defaults; skips const char*, etc.)
_CONSTEXPR_RE = re.compile(
    r"inline\s+constexpr\s+(?:Real|int|double|float|bool|std::size_t|size_t|unsigned)\s+"
    r"(k[A-Za-z0-9_]+)\s*=")


def _scan_constants() -> set:
    names = set()
    for rel in _SCANNED_HEADERS:
        path = _ROOT / rel
        assert path.exists(), "scanned header missing: %s" % path
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue  # a commented-out declaration is not a live constant
            m = _CONSTEXPR_RE.search(line)
            if m:
                names.add(m.group(1))
    return names


def test_every_native_numeric_constant_is_classified():
    found = _scan_constants()
    assert found, "the constant scan found nothing -- the regex or the header paths drifted"
    missing = sorted(n for n in found if n not in _CONSTANT_CLASSIFICATION)
    assert not missing, (
        "unclassified user-visible native numeric constant(s): %s. Add each to "
        "pops.runtime.defaults._CONSTANT_CLASSIFICATION AND numerical_defaults_report_to_dict "
        "(bindings_detail.hpp) with a class in %s (ADC-618 fence)." % (missing, sorted(_ALLOWED_CLASSES)))


def test_classification_values_are_valid():
    bad = {k: v for k, v in _CONSTANT_CLASSIFICATION.items() if v not in _ALLOWED_CLASSES}
    assert not bad, "invalid classification class(es): %s (allowed: %s)" % (bad, sorted(_ALLOWED_CLASSES))


def test_classification_has_no_stale_entries():
    # A classified name that no longer exists in any scanned header is stale -> remove it (keeps the
    # map honest). Names classified but declared elsewhere are not scanned, so this only flags the
    # enumerated headers; every entry in the map is expected to be one of those declarations.
    found = _scan_constants()
    stale = sorted(n for n in _CONSTANT_CLASSIFICATION if n not in found)
    assert not stale, (
        "stale classification entries (no longer declared in the scanned headers): %s. Remove them "
        "from _CONSTANT_CLASSIFICATION (ADC-618 fence)." % stale)


def _load_static_report() -> dict:
    """Parse the dict returned by _static_report() from defaults.py SOURCE (no pops import). The only
    non-literal top-level value is "classification": _CONSTANT_CLASSIFICATION (a Name), skipped here."""
    src = (_ROOT / "python/pops/runtime/defaults.py").read_text()
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_static_report"), None)
    assert fn is not None, "_static_report not found in defaults.py"
    ret = next((n for n in ast.walk(fn) if isinstance(n, ast.Return)), None)
    assert ret is not None and isinstance(ret.value, ast.Dict), "_static_report return dict not found"
    report = {}
    for k_node, v_node in zip(ret.value.keys, ret.value.values, strict=True):
        key = ast.literal_eval(k_node)
        if key == "classification":
            continue
        report[key] = ast.literal_eval(v_node)
    return report


_CONSTEXPR_VALUE_RE = re.compile(
    r"inline\s+constexpr\s+(?:Real|int|double|float|bool|std::size_t|size_t|unsigned)\s+"
    r"(k[A-Za-z0-9_]+)\s*=\s*(.+?);")


def _parse_cpp_value(rhs: str):
    rhs = rhs.strip()
    m = re.fullmatch(r"Real\((.*)\)", rhs)
    if m:
        rhs = m.group(1).strip()
    if rhs in ("true", "false"):
        return rhs == "true"
    try:
        return int(rhs)
    except ValueError:
        pass
    try:
        return float(rhs)
    except ValueError:
        return None  # non-literal RHS (references another constant) -> not value-fenced


def _scan_constant_values() -> dict:
    values = {}
    for rel in _SCANNED_HEADERS:
        for line in (_ROOT / rel).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            m = _CONSTEXPR_VALUE_RE.search(line)
            if m:
                values[m.group(1)] = _parse_cpp_value(m.group(2))
    return values


# Report (section, key) -> the scanned constant whose value it must equal. String-valued keys
# (newton.fail_policy), the unscanned kAmrRefRatio (amr.refinement_ratio, defined in
# amr/hierarchy/refinement_ratio.hpp), and runtime counters (diagnostics.*) are intentionally omitted.
_REPORT_VALUE_TO_CONSTANT = {
    ("newton", "max_iters"): "kNewtonDefaultMaxIters",
    ("newton", "rel_tol"): "kNewtonDefaultRelTol",
    ("newton", "abs_tol"): "kNewtonDefaultAbsTol",
    ("newton", "fd_eps"): "kNewtonDefaultFdEps",
    ("newton", "damping"): "kNewtonDefaultDamping",
    ("newton", "finite_abs_limit"): "kNewtonFiniteAbsLimit",
    ("krylov", "rel_tol"): "kKrylovDefaultRelTol",
    ("krylov", "tensor_max_iters"): "kTensorKrylovDefaultMaxIters",
    ("krylov", "schur_cartesian_max_iters"): "kSchurKrylovCartesianMaxIters",
    ("krylov", "schur_polar_max_iters"): "kSchurKrylovPolarMaxIters",
    ("krylov", "breakdown_tiny"): "kKrylovBreakdownTiny",
    ("mg", "rel_tol"): "kMGDefaultRelTol",
    ("mg", "max_cycles"): "kMGDefaultMaxCycles",
    ("mg", "abs_tol"): "kMGDefaultAbsTol",
    ("mg", "min_coarse"): "kMGDefaultMinCoarse",
    ("mg", "pre_smooth"): "kMGDefaultPreSmooth",
    ("mg", "post_smooth"): "kMGDefaultPostSmooth",
    ("mg", "bottom_sweeps"): "kMGDefaultBottomSweeps",
    ("fac", "max_iters"): "kFACDefaultMaxIters",
    ("fac", "fine_sweeps"): "kFACDefaultFineSweeps",
    ("fac", "rel_tol"): "kFACDefaultRelTol",
    ("fac", "abs_tol"): "kFACDefaultAbsTol",
    ("fac", "coarse_rel_tol"): "kFACInitialCoarseRelTol",
    ("fac", "coarse_abs_tol"): "kFACInitialCoarseAbsTol",
    ("fac", "coarse_cycles"): "kFACInitialCoarseMaxCycles",
    ("fft", "spectral_default"): "kFFTDefaultSpectral",
    ("fft", "zero_mean_gauge"): "kFFTZeroMeanGauge",
    ("fft", "direct_dft_fallback"): "kFFTDirectDftFallback",
    ("eb", "cut_fraction_floor"): "kEbCutFractionFloor",
    ("eb", "face_open_eps"): "kEbFaceOpenEps",
    ("eb", "kappa_min"): "kEbKappaMin",
    ("weno", "epsilon"): "kWenoEpsilon",
    ("performance", "cfl_speed_floor"): "kCflSpeedFloor",
    ("performance", "adaptive_no_evolving_block_sentinel"): "kAdaptiveNoEvolvingBlockSentinel",
    ("amr", "max_levels"): "kAmrDefaultMaxLevels",
    ("amr", "refinement_disabled_threshold"): "kAmrRefinementDisabledThreshold",
    ("amr", "phi_refinement_disabled_threshold"): "kAmrPhiRefinementDisabledThreshold",
    ("runtime", "max_runtime_params"): "kMaxRuntimeParams",
    ("physical", "B0"): "kPhysicalDefaultB0",
    ("physical", "gamma"): "kPhysicalDefaultGamma",
    ("physical", "fluid_state_cs2"): "kPhysicalDefaultFluidStateCs2",
    ("physical", "native_brick_isothermal_cs2"): "kPhysicalDefaultNativeIsothermalCs2",
    ("physical", "vacuum_floor"): "kPhysicalDefaultVacuumFloor",
    ("physical", "qom"): "kPhysicalDefaultQOverM",
    ("physical", "charge_q"): "kPhysicalDefaultChargeQ",
    ("physical", "alpha"): "kPhysicalDefaultAlpha",
    ("physical", "n0"): "kPhysicalDefaultBackgroundN0",
    ("physical", "gravity_sign"): "kPhysicalDefaultGravitySign",
    ("physical", "four_pi_G"): "kPhysicalDefaultFourPiG",
    ("physical", "gravity_rho0"): "kPhysicalDefaultGravityRho0",
}


def test_static_report_values_match_parsed_constants():
    report = _load_static_report()
    values = _scan_constant_values()
    for (section, key), cname in _REPORT_VALUE_TO_CONSTANT.items():
        assert section in report and key in report[section], (
            "report key %s.%s missing -- _static_report drifted from the fence map" % (section, key))
        assert cname in values, (
            "constant %s not found in the scanned headers (renamed/moved?) -- update the fence map"
            % cname)
        con = values[cname]
        assert con is not None, "constant %s RHS is not a numeric literal the fence can parse" % cname
        rep = report[section][key]
        assert rep == con, (
            "value drift: _static_report %s.%s = %r but %s = %r -- single-source the literal via "
            "the header constant (ADC-643 value fence)" % (section, key, rep, cname, con))


def main():
    test_every_native_numeric_constant_is_classified()
    test_classification_values_are_valid()
    test_classification_has_no_stale_entries()
    test_static_report_values_match_parsed_constants()
    print("OK  ADC-618 numeric-constant fence")


if __name__ == "__main__":
    main()
