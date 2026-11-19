"""IR5 (spec 0.3; G02 lot L, ROM-20): no construct has a private list of admitted operations.

Every Program operation has one row in the effect table, ``python/pops/_ir/effects.py``.
``tests/gates/ir5_private_lists.toml`` lists each module-level set literal of three or more names
that all have a row and that still stands in ``python/pops``. The test checks the registry against
the tree in both directions: an unlisted set fails, and so does an entry whose set is gone.

Each rule fails on a tree written for it, so that a check that sees nothing fails here. The test
reads the source tree and loads the effect table by its file path; it does not import ``pops``.
The base ratchet is called by the engine guard base check, including on first introduction.
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


# These three constructors introduce Program nodes. The exceptions below forward an existing
# operation, or a parameter whose caller is scanned, rather than choosing a new operation name.
_FORWARDED_OPS = {
    ("graph_functions.py", "inline", "new_typed", "op"),
    ("time/_program/api.py", "_new", "_new", "op"),
    ("time/_program/distributed_arrays.py", "_new", "_new", "op"),
    ("time/_program/distributed_arrays.py", "_new", "new_typed", "op"),
    ("time/_program/operations.py", "_new", "ProgramValue", "op"),
    ("time/_program/operations.py", "_replace_value", "ProgramValue", "current.op"),
    ("time/_program/rebuild.py", "clone", "ProgramValue", "v.op"),
    ("time/_program/structured_control.py", "new_typed", "_new", "op"),
}


def _operation_argument(call):
    """The operation argument of a Program constructor, including positional wrappers."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name == "_new":
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and (
                func.value.id in ("ScalarExpr", "PredicateExpr")):
            return None  # the separate analytic expression tree
        position = 1 if isinstance(func, ast.Attribute) else 2
    elif name in ("new_typed", "ProgramValue"):
        position = 2 if name == "new_typed" else 3
    else:
        return None
    argument = next((keyword.value for keyword in call.keywords if keyword.arg == "op"), None)
    if argument is None and len(call.args) > position:
        argument = call.args[position]
    assert argument is not None, "no exact operation argument: %s" % ast.unparse(call)
    return name, argument


def _emitted_names(node, constants):
    if isinstance(node, ast.IfExp):
        left, right = _emitted_names(node.body, constants), _emitted_names(node.orelse, constants)
        return left + right if left and right else ()
    if isinstance(node, ast.Name):
        node = constants.get(node.id, node)
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return (node.value,)
    return ()


def missing_operation_rows(package, operations):
    """Check actual node producers, resolving names without importing the production package."""
    missing = []
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in ("_new", "new_typed", "ProgramValue")):
            continue
        relative = path.relative_to(package).as_posix()
        tree = ast.parse(source)
        constants = {
            target.id: statement.value for statement in tree.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets if isinstance(target, ast.Name)
        }
        # Later (nested) functions own their calls rather than inheriting their caller's scope.
        scopes = {child: node.name for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  for child in ast.walk(node) if isinstance(child, ast.Call)}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            producer = _operation_argument(call)
            if producer is None:
                continue
            name, argument = producer
            if (relative, scopes.get(call), name, ast.unparse(argument)) in _FORWARDED_OPS:
                continue
            names = _emitted_names(argument, constants)
            if not names:
                missing.append("%s:%d: unresolved Program operation %s" %
                               (relative, argument.lineno, ast.unparse(argument)))
            for op in names:
                if op not in operations:
                    missing.append("%s:%d: Program operation has no effect row: %s" %
                                   (relative, argument.lineno, op))
    return missing


def test_every_program_operation_emitted_by_a_producer_has_an_effect_row():
    assert missing_operation_rows(PACKAGE, _operations()) == []


def _strings(node, constants):
    """The names of a set literal, or of a set/frozenset call over a literal; None otherwise."""
    if isinstance(node, ast.Set):
        elements = node.elts
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id in ("frozenset", "set") and len(node.args) == 1
          and isinstance(node.args[0], (ast.Tuple, ast.List, ast.Set))):
        elements = node.args[0].elts
    else:
        return None
    elements = [constants.get(item.id, item) if isinstance(item, ast.Name) else item
                for item in elements]
    if all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in elements):
        return [item.value for item in elements]
    return None


def _literals(expression, constants):
    """The set literals of a module-level value, through ``|`` and ``-``."""
    pending = [expression]
    while pending:
        node = pending.pop()
        found = _strings(node, constants)
        if found is not None:
            yield found
        elif isinstance(node, ast.BinOp):
            pending.extend((node.left, node.right))


def _string_constants(statements):
    return {target.id: row.value for row in statements if isinstance(row, ast.Assign)
            and isinstance(row.value, ast.Constant) and isinstance(row.value.value, str)
            for target in row.targets if isinstance(target, ast.Name)}


def private_lists(package, operations):
    """{(module, name)} of the module-level sets of three or more names that all have a row."""
    found = set()
    modules = {".".join(("pops", *path.relative_to(package).with_suffix("").parts))
               .removesuffix(".__init__"): ast.parse(path.read_text(encoding="utf-8")).body
               for path in sorted(package.rglob("*.py"))}
    declarations = {module: _string_constants(body) for module, body in modules.items()}
    for module, body in modules.items():
        constants = declarations[module].copy()
        for row in body:
            if isinstance(row, ast.ImportFrom) and row.level == 0 and row.module in declarations:
                for alias in row.names:
                    imported = declarations[row.module].get(alias.name)
                    if imported is not None:
                        constants[alias.asname or alias.name] = imported
        for statement in body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets, value = [statement.target], statement.value
            else:
                continue
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if names and any(len(items) >= 3 and set(items) <= operations
                             for items in _literals(value, constants)):
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


def registry_growth(base_package, base_registry, registry, operations):
    """One introduction may register lists already present in the base sources."""
    before = (registered(base_registry) if base_registry.is_file()
              else private_lists(base_package, operations))
    return ["new IR5 registry entry: %s.%s" % key for key in sorted(registered(registry) - before)]


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


def test_a_named_producer_operation_remains_visible_in_private_lists(tmp_path):
    package, registry = _tree(tmp_path, """
        from pops.sub.producer import OPERATION as _PRODUCED
        _ADMITTED = frozenset({"branch", "reduce", _PRODUCED})
    """)
    (package / "sub" / "producer.py").write_text('OPERATION = "compare"\n', encoding="utf-8")
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

def test_emitted_names_are_checked_without_using_an_admission_list(tmp_path):
    package, _ = _tree(tmp_path, """
        HISTORY = "history"
        def emit(p, sig, flag):
            new_typed(p, sig, HISTORY, (), {})
            p._new("scalar", "reduce" if flag else "laplacian", (), {}, None, None)
            p._new(vtype="state", op="store_history", inputs=(), attrs={}, name=None, block=None)
    """)
    assert set(missing_operation_rows(package, {"reduce"})) == {
        "sub/module.py:4: Program operation has no effect row: history",
        "sub/module.py:5: Program operation has no effect row: laplacian",
        "sub/module.py:6: Program operation has no effect row: store_history",
    }


def test_an_unresolved_operation_name_is_not_ignored(tmp_path):
    package, _ = _tree(tmp_path, """
        def emit(p, op, flag):
            p._new("scalar", "reduce" if flag else op, (), {}, None, None)
    """)
    assert missing_operation_rows(package, {"reduce"}) == [
        "sub/module.py:3: unresolved Program operation 'reduce' if flag else op"]


def test_an_entry_cannot_hide_a_new_private_list(tmp_path):
    old = '_ADMITTED = {"branch", "reduce", "scalar_op"}'
    base, base_registry = _tree(tmp_path / "base", old, ENTRY)
    new_entry = ENTRY.replace('_ADMITTED', '_EXTRA')
    package, registry = _tree(tmp_path / "head", old + '\n' + old.replace('_ADMITTED', '_EXTRA'),
                              ENTRY + new_entry)
    assert check(package, registry, OPERATIONS) == []
    assert registry_growth(base, base_registry, registry, OPERATIONS) == [
        "new IR5 registry entry: pops.sub.module._EXTRA"]
    base_registry.unlink()  # initial introduction must also reject the extra site
    assert registry_growth(base, base_registry, registry, OPERATIONS) == [
        "new IR5 registry entry: pops.sub.module._EXTRA"]

    _, initial = _tree(tmp_path / "initial", old, ENTRY)
    assert registry_growth(base, base_registry, initial, OPERATIONS) == []


def test_the_registry_can_remove_a_list_after_bootstrap(tmp_path):
    old = '_ADMITTED = {"branch", "reduce", "scalar_op"}'
    base, base_registry = _tree(tmp_path / "base", old, ENTRY)
    package, registry = _tree(tmp_path / "head", "")
    assert check(package, registry, OPERATIONS) == []
    assert registry_growth(base, base_registry, registry, OPERATIONS) == []
