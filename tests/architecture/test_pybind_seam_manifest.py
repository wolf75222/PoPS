"""ADC-593: source-only gates locking the pybind block-build SEAM TUs to ONE declarative manifest.

The _pops extension used to carry ~20 hand-written .cpp files, one per (side, transport, flux) numeric
combination, and grew by a new file every time a Riemann or reconstruction was added. Those leaves are
now GENERATED from python/bindings/seam_combinations.cmake (configure_file per row, into the build tree),
so the growth strategy is a manifest row, not a hand-written file.

These gates enforce that invariant WITHOUT a build (source tree only, no _pops import):

  1. the hand-written specialization files are GONE from git and the manifest + templates exist;
  2. every manifest row's (transport, flux) is a legal route -- transport in the brick catalog
     (brick_catalog.py), flux in the Riemann registry (routes.py) -- so the manifest cannot invent a
     route and is NOT itself the descriptor registry;
  3. no NEW hand-written .cpp under python/bindings/ carries the seam-leaf signature (build_*_for_make /
     build_amr_*_for_flux + make_block_ / dispatch_amr_): that pattern belongs in the templates only,
     so a new numeric combination cannot sneak back in as a hand-written file.

The runtime sibling (a native add_block of every manifest combo actually advancing) lives in
python/tests/test_seam_combinations.py; this file is the pure-source architecture gate.
"""
import importlib.util
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BINDINGS = REPO_ROOT / "python" / "bindings"
MANIFEST = BINDINGS / "seam_combinations.cmake"
TEMPLATES = BINDINGS / "templates"
CATALOG_PY = REPO_ROOT / "python" / "pops" / "runtime" / "brick_catalog.py"
ROUTES_PY = REPO_ROOT / "python" / "pops" / "runtime" / "routes.py"

# The 19 (transport, flux) leaf TUs that USED to be hand-written and are now generated. They must NOT
# reappear as tracked source files; regenerating them into the source tree would defeat the manifest.
GENERATED_LEAF_PATHS = (
    "system/base/system_exb.cpp",
    "system/isothermal/system_isothermal_rusanov.cpp",
    "system/isothermal/system_isothermal_hll.cpp",
    "system/compressible/system_compressible_rusanov.cpp",
    "system/compressible/system_compressible_hll.cpp",
    "system/compressible/system_compressible_hllc.cpp",
    "system/compressible/system_compressible_roe.cpp",
    "amr/block/base/amr_block_exb.cpp",
    "amr/block/base/amr_block_isothermal.cpp",
    "amr/block/compressible/amr_block_compressible_rusanov.cpp",
    "amr/block/compressible/amr_block_compressible_hll.cpp",
    "amr/block/compressible/amr_block_compressible_hllc.cpp",
    "amr/block/compressible/amr_block_compressible_roe.cpp",
    "amr/compiled/base/amr_compiled_exb.cpp",
    "amr/compiled/base/amr_compiled_isothermal.cpp",
    "amr/compiled/compressible/amr_compiled_compressible_rusanov.cpp",
    "amr/compiled/compressible/amr_compiled_compressible_hll.cpp",
    "amr/compiled/compressible/amr_compiled_compressible_hllc.cpp",
    "amr/compiled/compressible/amr_compiled_compressible_roe.cpp",
)

# The hand-written TUs that STAY (unique shapes, not per-combination growth): the two heavy facades, the
# unique polar visitor body, and the two thin riemann DISPATCHERS (one per transport). They are exempt
# from the "no seam signature outside templates" gate below.
HAND_WRITTEN_KEPT = {
    "python/bindings/system/base/system.cpp",
    "python/bindings/system/base/system_polar.cpp",
    "python/bindings/amr/amr_system.cpp",
    "python/bindings/amr/block/compressible/amr_block_compressible.cpp",
    "python/bindings/amr/compiled/compressible/amr_compiled_compressible.cpp",
}

# A manifest row is a quoted string of exactly 7 non-empty |-separated fields starting with a template
# stem (a word). This deliberately does NOT match helper strings like the "|" separator in string(REPLACE).
_ROW_RE = re.compile(r'"(\w[^"|]*(?:\|[^"|]+){6})"')


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _load_module(path, name):
    """Load an import-free mirror module (brick_catalog.py / routes.py) by path, no pops import."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_rows():
    """Parse the manifest into a list of dicts (one per POPS_SEAM_COMBINATIONS row)."""
    text = MANIFEST.read_text(encoding="utf-8")
    rows = []
    for match in _ROW_RE.finditer(text):
        cols = match.group(1).split("|")
        assert len(cols) == 7, "manifest row must have 7 |-separated fields: %r" % (cols,)
        rows.append({
            "template": cols[0], "side": cols[1], "transport": cols[2], "flux": cols[3],
            "symbol": cols[4], "out_subdir": cols[5], "out_name": cols[6],
        })
    return rows


# --- Gate (a): the hand-written leaves are gone; the manifest + templates exist -------------------
def test_manifest_and_templates_exist():
    assert MANIFEST.is_file(), "the seam combination manifest must exist: %s" % _rel(MANIFEST)
    assert TEMPLATES.is_dir(), "the seam templates dir must exist: %s" % _rel(TEMPLATES)
    templates = {p.name for p in TEMPLATES.glob("*.cpp.in")}
    assert templates, "no seam .cpp.in templates found under %s" % _rel(TEMPLATES)


def test_specialized_leaf_files_are_generated_not_tracked():
    present = [rel for rel in GENERATED_LEAF_PATHS if (BINDINGS / rel).exists()]
    assert not present, (
        "these per-route seam TUs must be GENERATED from seam_combinations.cmake, not hand-written "
        "source files; delete them and add a manifest row instead:\n  "
        + "\n  ".join("python/bindings/" + rel for rel in present)
    )


def test_manifest_covers_every_former_leaf():
    """The manifest must still produce every combination that used to be hand-written (no silent drop)."""
    produced = {"%s/%s" % (row["out_subdir"], row["out_name"]) for row in _manifest_rows()}
    missing = [rel for rel in GENERATED_LEAF_PATHS if rel not in produced]
    assert not missing, (
        "the manifest dropped combinations that used to exist (would remove a working route):\n  "
        + "\n  ".join(missing)
    )


# --- Gate (b): every row is a legal route; the manifest cannot invent one -------------------------
def test_every_manifest_row_is_a_catalog_and_registry_route():
    catalog = _load_module(CATALOG_PY, "_seam_brick_catalog")
    routes = _load_module(ROUTES_PY, "_seam_routes")
    transports = set(catalog.catalog_ids("transport"))
    fluxes = {row[0] for row in routes._TABLES["riemann"]}

    violations = []
    for row in _manifest_rows():
        if row["transport"] not in transports:
            violations.append("row %s: transport %r is not a brick_catalog transport (%s)"
                              % (row["symbol"], row["transport"], "|".join(sorted(transports))))
        if row["flux"] != "-" and row["flux"] not in fluxes:
            violations.append("row %s: flux %r is not a routes.py Riemann id (%s)"
                              % (row["symbol"], row["flux"], "|".join(sorted(fluxes))))
    assert not violations, (
        "the seam manifest is NOT the descriptor registry: every (transport, flux) must already exist "
        "in brick_catalog.py / routes.py:\n  " + "\n  ".join(violations)
    )


def test_manifest_templates_all_exist():
    missing = []
    for row in _manifest_rows():
        tmpl = TEMPLATES / (row["template"] + ".cpp.in")
        if not tmpl.is_file():
            missing.append("row %s -> missing template %s" % (row["symbol"], _rel(tmpl)))
    assert not missing, "manifest rows reference templates that do not exist:\n  " + "\n  ".join(missing)


# --- Gate (c): no NEW hand-written seam leaf may reappear -----------------------------------------
def _has_seam_leaf_signature(text):
    """A seam LEAF instantiates exactly one template product leaf: a build_*_for_make / _for_flux call
    plus a make_block_<flux> / dispatch_amr_*_<flux> maker. The thin transport dispatchers and the
    facades do NOT match (they call the transport-level seam symbols, not the *_for_make/_for_flux
    instantiators)."""
    stripped = "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())
    instantiates = ("build_block_for_make" in stripped
                    or "build_amr_block_for_flux" in stripped
                    or "build_amr_compiled_for_flux" in stripped)
    maker = re.search(r"make_block_[a-z]+\s*\(", stripped) or re.search(
        r"dispatch_amr_(?:block|compiled)_[a-z]+\s*\(", stripped)
    return instantiates and bool(maker)


def test_no_new_hand_written_seam_leaf():
    violations = []
    for path in sorted(BINDINGS.rglob("*.cpp")):
        rel = _rel(path)
        if rel in HAND_WRITTEN_KEPT:
            continue
        if _has_seam_leaf_signature(path.read_text(encoding="utf-8")):
            violations.append(rel)
    assert not violations, (
        "new hand-written pybind seam-leaf TU(s) found; a per-(transport, flux) leaf must be a ROW in "
        "python/bindings/seam_combinations.cmake (generated from a template), never a hand-written "
        "file (ADC-593 acceptance criterion):\n  " + "\n  ".join(violations)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
