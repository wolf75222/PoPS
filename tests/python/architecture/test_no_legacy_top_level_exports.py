"""ADC-545: the retired top-level exports must stay off the pops root.

Phase 7 removed eight names from the ``pops`` package surface -- the runtime engines
(``System`` / ``AmrSystem``), their config PODs (``SystemConfig`` / ``AmrSystemConfig``),
the compiled-Program time policy (``CompiledTime``) and the brick-library manifest trio
(``compile_library`` / ``read_library_manifest`` / ``LibraryManifest``). Each is reachable
only through a clearly-advanced path now, and ``pops.<name>`` raises a targeted
``AttributeError`` naming that path.

These checks are SOURCE-ONLY (they parse ``python/pops/__init__.py``; they never
``import pops`` / ``_pops``), so they are the regression trap that runs in a bare
interpreter -- if a future change re-adds any of these to ``__all__`` or to a root import,
or drops a migration message, the gate fails and names the offender. The runtime companion
(``import pops`` + the AttributeError / advanced-seam assertions) lives in
``test_public_imports.py`` and skips cleanly without a build.
"""
import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INIT = REPO_ROOT / "python" / "pops" / "__init__.py"

# The eight names retired from the pops root, mapped to the advanced-path substring each
# migration AttributeError must name (so a user is always pointed at the real home).
RETIRED = {
    "System": "pops.runtime.system",
    "AmrSystem": "pops.runtime.system",
    "SystemConfig": "pops._bootstrap",
    "AmrSystemConfig": "pops._bootstrap",
    "CompiledTime": "pops.time",
    "compile_library": "pops.codegen",
    "read_library_manifest": "pops.codegen",
    "LibraryManifest": "pops.codegen",
}


def _read():
    return INIT.read_text(encoding="utf-8")


def _tree():
    return ast.parse(_read(), filename=str(INIT))


def _all_names():
    """The literal string entries of the module-level ``__all__`` assignment."""
    for node in _tree().body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                assert isinstance(node.value, (ast.List, ast.Tuple)), "__all__ must be a literal list"
                return [elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
    raise AssertionError("pops/__init__.py has no module-level __all__")


def test_retired_names_are_absent_from_dunder_all():
    exported = set(_all_names())
    present = sorted(name for name in RETIRED if name in exported)
    assert not present, (
        "ADC-545 retired these from pops.__all__; they must not return:\n  " + "\n  ".join(present))


def test_no_module_scope_import_binds_a_retired_name_at_the_root():
    # A module-scope (col 0) import that binds any retired name back onto the pops root would make
    # pops.<name> resolve again, bypassing the __getattr__ refusal. Function-local / lazy imports
    # inside __getattr__ are fine; only col-0 bindings are the regression.
    violations = []
    for node in _tree().body:  # module body only -> col 0 statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in RETIRED:
                    violations.append("%s:%d binds %s" % (INIT.name, node.lineno, bound))
    assert not violations, (
        "pops/__init__.py must not re-bind a retired name at module scope (ADC-545):\n  "
        + "\n  ".join(violations))


def test_getattr_refuses_each_retired_name_with_its_advanced_path():
    # The source of __getattr__ (plus any module-scope table it consults) must, for every retired
    # name, carry the ADC-545 tag AND the advanced-path substring. We check the whole __init__
    # source since the messages live in a small module-level table the refusal formats.
    src = _read()
    assert 'ADC-545' in src, "the migration messages must cite ADC-545"
    missing = []
    for name, home in RETIRED.items():
        # the name and its advanced home must both appear (the message names both).
        if name not in src or home not in src:
            missing.append("%s -> %s" % (name, home))
    assert not missing, (
        "each retired name's refusal must name its advanced path (ADC-545); missing:\n  "
        + "\n  ".join(missing))


def test_pops_init_stays_a_slim_facade():
    # ADC-545 keeps the package facade a re-export hub (test_file_sizes caps it at 120); assert the
    # tighter contract here too so the removal cannot bloat it back into an implementation module.
    line_count = sum(1 for _ in INIT.open("rb"))
    assert line_count <= 120, "pops/__init__.py must stay <= 120 lines (got %d)" % line_count


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
