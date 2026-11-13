"""IR5 (spec 0.3; G02 lot L, ROM-20): no construct has a private list of admitted operations.

Every Program operation has one row in the effect table, ``python/pops/_ir/effects.py``.
``tests/gates/ir5_private_lists.toml`` lists each module-level set literal of three or more names
that all have a row and that still stands in ``python/pops``. The test checks the registry against
the tree in both directions: an unlisted set fails, and so does an entry whose set is gone.

Each rule fails on a tree written for it, so that a check that sees nothing fails here. The test
reads the source tree and loads the effect table by its file path; it does not import ``pops``.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "python" / "pops"
REGISTRY = ROOT / "tests" / "gates" / "ir5_private_lists.toml"
KINDS = frozenset(("dispatch", "structural", "admission"))


def _operations():
    name = "_ir5_gate_effect_table"
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "_ir" / "effects.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module of a class through sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return frozenset(module.EFFECTS)


def _strings(node):
    """The names of a set literal, or of a set/frozenset call over a literal; None otherwise."""
    if isinstance(node, ast.Set):
        elements = node.elts
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id in ("frozenset", "set") and len(node.args) == 1
          and isinstance(node.args[0], (ast.Tuple, ast.List, ast.Set))):
        elements = node.args[0].elts
    else:
        return None
    if all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in elements):
        return [item.value for item in elements]
    return None


def _literals(expression):
    """The set literals of a module-level value, through ``|`` and ``-``."""
    pending = [expression]
    while pending:
        node = pending.pop()
        found = _strings(node)
        if found is not None:
            yield found
        elif isinstance(node, ast.BinOp):
            pending.extend((node.left, node.right))


def private_lists(package, operations):
    """{(module, name)} of the module-level sets of three or more names that all have a row."""
    found = set()
    for path in sorted(package.rglob("*.py")):
        module = ".".join(("pops", *path.relative_to(package).with_suffix("").parts))
        module = module.removesuffix(".__init__")
        for statement in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets, value = [statement.target], statement.value
            else:
                continue
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if names and any(len(items) >= 3 and set(items) <= operations
                             for items in _literals(value)):
                found.add((module, names[0]))
    return found


def registered(registry):
    data = tomllib.loads(registry.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["gate"] == "ir5-private-lists"
    entries = data.get("list", [])
    for entry in entries:
        assert set(entry) == {"module", "name", "kind", "owner", "removed_by"}, entry
        assert entry["kind"] in KINDS, entry
        assert all(type(entry[key]) is str and entry[key] for key in entry), entry
    keys = [(entry["module"], entry["name"]) for entry in entries]
    assert len(keys) == len(set(keys)), "an entry is listed twice"
    return set(keys)


def check(package, registry, operations):
    """The findings: unlisted private lists and stale entries."""
    found, listed = private_lists(package, operations), registered(registry)
    return ["unlisted private list of operations: %s.%s" % key for key in sorted(found - listed)] + [
        "stale entry, its list is gone: %s.%s" % key for key in sorted(listed - found)]


def test_the_registry_matches_the_tree():
    assert check(PACKAGE, REGISTRY, _operations()) == []


def test_no_list_owned_by_g02_lot_l_remains():
    listed = registered(REGISTRY)
    for module, name in (("pops.graph_functions", "_PURE"),
                         ("pops.graph_functions", "_COLLECTIVE"),
                         ("pops.graph_functions", "_RENDEZVOUS"),
                         ("pops.graph_functions", "_FINITE_CHECKED"),
                         ("pops.graph_functions", "_PARTIAL"),
                         ("pops.graph_functions", "_NATIVE_FIELD_ALGEBRA"),
                         ("pops.codegen.program_emit_structured_control", "_REGION_SAFE_OPS"),
                         ("pops.codegen.program_emit_finite_kokkos", "_OPS"),
                         ("pops.codegen.program_emit_finite_kokkos", "_ORDERED_OPS"),
                         ("pops.codegen.program_emit_finite_kokkos", "_HOST_REGIONS")):
        assert (module, name) not in listed


def _tree(tmp_path, source, entries=""):
    package = tmp_path / "pops"
    (package / "sub").mkdir(parents=True)
    (package / "sub" / "module.py").write_text(textwrap.dedent(source), encoding="utf-8")
    registry = tmp_path / "registry.toml"
    registry.write_text('schema_version = 1\ngate = "ir5-private-lists"\n' + textwrap.dedent(entries),
                        encoding="utf-8")
    return package, registry


OPERATIONS = frozenset(("branch", "reduce", "scalar_op", "compare"))
ENTRY = """
[[list]]
module = "pops.sub.module"
name = "_ADMITTED"
kind = "admission"
owner = "a test"
removed_by = "a test"
"""


def test_an_unlisted_set_literal_fails(tmp_path):
    package, registry = _tree(tmp_path, '_ADMITTED = frozenset({"branch", "reduce", "scalar_op"})\n')
    assert check(package, registry, OPERATIONS) == [
        "unlisted private list of operations: pops.sub.module._ADMITTED"]


def test_a_union_of_literals_and_a_plain_set_fail(tmp_path):
    package, registry = _tree(tmp_path, """
        _A = frozenset(("branch",)) | {"reduce", "scalar_op", "compare"}
        _B: frozenset = {"branch", "reduce", "compare"}
        """)
    assert check(package, registry, OPERATIONS) == [
        "unlisted private list of operations: pops.sub.module._A",
        "unlisted private list of operations: pops.sub.module._B"]


def test_a_stale_entry_fails_and_a_listed_set_passes(tmp_path):
    package, registry = _tree(tmp_path, "_OTHER = 1\n", ENTRY)
    assert check(package, registry, OPERATIONS) == [
        "stale entry, its list is gone: pops.sub.module._ADMITTED"]
    package, registry = _tree(tmp_path / "listed",
                              '_ADMITTED = frozenset({"branch", "reduce", "scalar_op"})\n', ENTRY)
    assert check(package, registry, OPERATIONS) == []


def test_short_sets_and_sets_with_other_names_are_not_lists_of_operations(tmp_path):
    package, registry = _tree(tmp_path, """
        _TWO = frozenset({"branch", "reduce"})
        _MIXED = frozenset({"branch", "reduce", "not_an_operation"})
        def f():
            local = {"branch", "reduce", "scalar_op"}
            return local
        """)
    assert check(package, registry, OPERATIONS) == []
