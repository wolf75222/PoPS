"""Spec 5 (sec.4 / 5 / 5.15 / 16): the central packages are top-level, not under pops.lib.

Spec 5 homes the generic building blocks in top-level packages and reserves ``pops.lib``
for ready-to-use presets (``lib.time`` / ``lib.models``). These checks assert that end state
structurally (source-only; they do not import ``pops`` / ``_pops``). Criterion 7: ``pops.lib``
holds ONLY presets (no spatial / fields / solver building-block catalogs). The solver
descriptors have ONE public home, ``pops.solvers`` (the ``pops.lib.solvers`` shim was removed).
Criterion 19: a solver-gen DSL, if any, lives in ``pops.codegen.solvers`` and is marked
internal / experimental.
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"

# Spec 5 top-level central packages (sec.4 / 5).
CENTRAL_PACKAGES = (
    "numerics",      # discretisation descriptors (riemann/reconstruction/projections/spatial)
    "moments",       # moment-model toolkit
    "diagnostics",   # reduction catalog
    "mesh",          # mesh/layout/AMR descriptors
    "params",        # typed scalar params
    "output",        # output package (general policy surface removed; wired AMR output in mesh.amr)
    "external",      # compiled-brick references
    "fields",        # typed elliptic field-problem authoring + brick catalog (Spec 5 Phase E)
    "linalg",        # abstract algebra: names A x = b (Spec 5 sec.5.6)
    "solvers",       # linear / nonlinear / Schur / elliptic solver catalog (Spec 5 sec.5.7)
)

# Catalogs Spec 5 moves OUT of pops.lib (no longer their own modules under lib/). Criterion 7
# finishes the Phase A2 carve-out: spatial -> pops.numerics.spatial, fields -> pops.fields.catalog.
MOVED_OUT_OF_LIB = ("riemann", "reconstruction", "moments", "diagnostics", "operators",
                    "spatial", "fields")


def test_central_packages_are_top_level():
    for pkg in CENTRAL_PACKAGES:
        init = POPS / pkg / "__init__.py"
        assert init.is_file(), "Spec 5 central package missing: python/pops/%s/__init__.py" % pkg


def test_shared_descriptor_module_is_top_level():
    assert (POPS / "descriptors.py").is_file(), (
        "Spec 5: the shared BrickDescriptor + Descriptor base must live in "
        "python/pops/descriptors.py")


def test_moved_catalogs_are_gone_from_lib():
    offenders = []
    for name in MOVED_OUT_OF_LIB:
        if (POPS / "lib" / name).exists():
            offenders.append("python/pops/lib/%s" % name)
    assert not offenders, (
        "Spec 5 sec.5.15 / criterion 7: these central catalogs must move out of pops.lib:\n  "
        + "\n  ".join(offenders))


def test_lib_keeps_only_presets():
    # Criterion 7: pops.lib holds ONLY presets -- time / models. The solver descriptors are NOT a
    # preset: they live in the ONE public home pops.solvers (the pops.lib.solvers shim was removed,
    # no back-compat path). The spatial / fields catalogs and the solver-gen DSL are not presets
    # either and must not live under lib.
    allowed = {"time", "models", "__init__.py", "__pycache__"}
    unexpected = []
    for child in (POPS / "lib").iterdir():
        if child.name not in allowed and not child.name.endswith(".pyc"):
            unexpected.append(child.name)
    assert not unexpected, (
        "Spec 5 criterion 7: pops.lib should hold only presets (time / models); unexpected: %s"
        % unexpected)


def test_lib_has_no_solvers_shim():
    # Criterion 4 / 7 / no-soft-compat: the solver descriptors have exactly ONE public home,
    # pops.solvers. The transitional pops.lib.solvers re-export shim is REMOVED -- pops.lib must not
    # be a second public path to the solver catalog (and must not host the solver-generation DSL).
    assert not (POPS / "lib" / "solvers").exists(), (
        "Spec 5 criterion 4: the pops.lib.solvers shim must be gone (one public home: pops.solvers)")


def test_output_policy_surface_removed():
    # C4 (ADC-509): the DECORATIVE general output/checkpoint policy surface is removed (zero codegen,
    # zero runtime wiring -> it could only reject at validate). Source-only assertion: the policy /
    # format modules are deleted and pops.output exports none of the removed symbols. The WIRED
    # narrower AMR-output home (pops.mesh.amr) is untouched.
    out = POPS / "output"
    assert not (out / "policies.py").exists(), (
        "C4 (ADC-509): python/pops/output/policies.py must be removed (decorative API)")
    assert not (out / "formats.py").exists(), (
        "C4 (ADC-509): python/pops/output/formats.py must be removed (no remaining consumer)")
    init_src = (out / "__init__.py").read_text()
    removed = ("OutputPolicy", "CheckpointPolicy", "HDF5", "Plotfile",
               "AllLevels", "CoarseOnly", "SelectedLevels")
    for sym in removed:
        assert '"%s"' % sym not in init_src and "'%s'" % sym not in init_src, (
            "C4 (ADC-509): pops.output must not export %s (decorative API removed)" % sym)
    # The wired AMR-output home stays.
    assert (POPS / "mesh" / "amr" / "__init__.py").is_file(), (
        "the wired AMR-output home (pops.mesh.amr) must be untouched")


def test_solver_gen_dsl_lives_in_codegen_solvers_and_is_internal():
    # Criterion 19: a solver-gen DSL, if any, lives in pops.codegen.solvers, marked internal /
    # experimental. Assert the package + its DSL modules exist and carry the experimental marker
    # (source-only: read the marker token from the files, do not import pops / _pops).
    pkg = POPS / "codegen" / "solvers"
    assert (pkg / "__init__.py").is_file(), (
        "Spec 5 criterion 19: the solver-gen DSL package python/pops/codegen/solvers/ must exist")
    for mod in ("__init__.py", "dsl.py", "solver_cpp.py"):
        path = pkg / mod
        assert path.is_file(), "missing pops/codegen/solvers/%s" % mod
        text = path.read_text()
        assert "__experimental__ = True" in text, (
            "Spec 5 criterion 19: pops/codegen/solvers/%s must mark the DSL internal / "
            "experimental (__experimental__ = True)" % mod)
        assert ("INTERNAL" in text and "EXPERIMENTAL" in text), (
            "Spec 5 criterion 19: pops/codegen/solvers/%s must state it is INTERNAL / "
            "EXPERIMENTAL, not a stable public API" % mod)
