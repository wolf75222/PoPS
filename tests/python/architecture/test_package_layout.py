"""ADC-524: the spec package layout is complete and stays flat.

Complements the existing layout guards (``test_file_sizes.py`` / ``test_no_flat_modules.py`` /
``test_import_graph.py`` / ``test_no_forbidden_paths.py``) with two ADC-524 checks:

* ``pops.lib.presets`` exists as a real package (the one home for ready-to-run compose-and-go
  bundles) and is wired into ``pops.lib``;
* a package and a flat root module of the SAME name never coexist -- once a responsibility is a
  sub-package (``pops.<name>/``), a flat ``pops/<name>.py`` must not be re-created beside it.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"


def test_lib_presets_package_exists():
    """pops.lib.presets is the ADC-524 home for ready-to-run compose-and-go bundles."""
    init = POPS / "lib" / "presets" / "__init__.py"
    assert init.exists(), (
        "python/pops/lib/presets/__init__.py must exist: pops.lib.presets is the single home for "
        "ready-to-run compose-and-go bundles (ADC-524).")
    # It advertises a real responsibility (the Preset type + at least one provided preset), not a
    # bare stub. Read the exported __all__ statically (no import: this is the bare architecture lane).
    tree = ast.parse(init.read_text(), str(init))
    exported = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                exported = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    assert "Preset" in exported, "pops.lib.presets must export the Preset bundle type"
    assert len(exported) >= 2, (
        "pops.lib.presets must ship at least one concrete preset, not just the Preset type "
        "(ADC-524 bans a subfolder with no clarified responsibility); exports=%s" % sorted(exported))


def test_lib_wires_presets():
    """pops.lib re-exports the presets sub-package (so pops.lib.presets is reachable)."""
    src = (POPS / "lib" / "__init__.py").read_text()
    assert "from . import presets" in src, "pops.lib must import its presets sub-package"
    assert '"presets"' in src, "pops.lib.__all__ must list presets"


def test_no_package_shadowed_by_a_flat_root_module():
    """A sub-package and a flat root module of the same name must not coexist.

    Once ``pops.<name>/`` is a package, a flat ``pops/<name>.py`` beside it would re-create the
    monolith the restructure removed and split the namespace across two homes. Guard every existing
    package name, so this rule extends automatically as new packages are added (ADC-524).
    """
    package_names = sorted(
        child.name for child in POPS.iterdir()
        if child.is_dir() and (child / "__init__.py").exists())
    offenders = [name for name in package_names if (POPS / ("%s.py" % name)).exists()]
    assert not offenders, (
        "a package and a flat root module share a name (re-created monolith): %s "
        "(ADC-524: one home per responsibility)" % offenders)
