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


def main():
    test_every_native_numeric_constant_is_classified()
    test_classification_values_are_valid()
    test_classification_has_no_stale_entries()
    print("OK  ADC-618 numeric-constant fence")


if __name__ == "__main__":
    main()
