"""The canonical package layout stays flat and has one home per responsibility."""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"


def test_lib_wires_only_ordinary_implementation_families():
    src = (POPS / "lib" / "__init__.py").read_text()
    for family in ("amr", "initial", "models", "time"):
        assert "from . import %s" % family in src
        assert '"%s"' % family in src
    assert not (POPS / "lib" / "presets" / "__init__.py").exists()


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
