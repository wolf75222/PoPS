"""Final source-tree hygiene: declarative facade and canonical tracked modules.

The historical restructure used a temporary 500-line punch-list. The final architecture does not
make physical line count a correctness contract: cohesive implementations can legitimately exceed
that threshold, while a short module can still duplicate ownership. These source-only gates protect
the actual invariants instead:

* the root facade only re-exports uniquely owned public names and has no implementation body;
* every tracked Python source has one canonical, importable module path, so editor/sync copies can
  never become production modules.

Untracked cache/copy artifacts do not affect the result. The test does not import ``pops`` or
``_pops``.
"""
import ast
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

FACADE = POPS / "__init__.py"


def _tracked_python_sources():
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "python/pops"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(
        REPO_ROOT / rel.decode()
        for rel in completed.stdout.split(b"\0")
        if rel.endswith(b".py")
    )


def test_tracked_sources_have_canonical_module_paths():
    """A committed source must be importable and cannot be an editor/sync duplicate copy."""
    invalid = []
    module_owners = {}
    duplicates = []
    for path in _tracked_python_sources():
        rel = path.relative_to(POPS).with_suffix("")
        if not all(part.isascii() and part.isidentifier() for part in rel.parts):
            invalid.append(path.relative_to(REPO_ROOT).as_posix())
            continue
        module_parts = rel.parts[:-1] if rel.parts[-1] == "__init__" else rel.parts
        module = ".".join(("pops", *module_parts)).casefold()
        previous = module_owners.setdefault(module, path)
        if previous != path:
            duplicates.append(
                "%s and %s" % (
                    previous.relative_to(REPO_ROOT), path.relative_to(REPO_ROOT)))
    assert not invalid, (
        "tracked Python copy/cache paths are not canonical modules:\n  "
        + "\n  ".join(invalid))
    assert not duplicates, (
        "multiple tracked files own the same Python module:\n  " + "\n  ".join(duplicates))


def test_root_facade_only_reexports_uniquely_owned_names():
    """Keep implementation in thematic owners, with one exact owner per root public name."""
    tree = ast.parse(FACADE.read_text(), str(FACADE))
    owners = {}
    exported = None
    violations = []
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) \
                and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    violations.append("star import from %s" % node.module)
                    continue
                public_name = alias.asname or alias.name
                owners.setdefault(public_name, []).append(node.module)
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "__all__":
            try:
                exported = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                violations.append("__all__ must be a literal list or tuple")
            continue
        violations.append("%s at line %d" % (type(node).__name__, node.lineno))

    assert not violations, (
        "pops.__init__ must remain a declarative re-export facade:\n  "
        + "\n  ".join(violations))
    assert isinstance(exported, (list, tuple)) \
        and all(isinstance(name, str) for name in exported), \
        "pops.__all__ must be a literal sequence of public names"
    duplicate_exports = sorted(name for name in set(exported) if exported.count(name) > 1)
    duplicate_owners = sorted(name for name, modules in owners.items() if len(modules) > 1)
    assert not duplicate_exports, "pops.__all__ repeats public names: %s" % duplicate_exports
    assert not duplicate_owners, (
        "root public names have multiple implementation owners: %s" % duplicate_owners)
    assert set(exported) == set(owners), (
        "pops.__all__ and its imported owners differ: exports=%s imports=%s"
        % (sorted(exported), sorted(owners)))
