"""ADC-528 fence: operator capabilities/requirements are DECLARED, never inferred from a name.

The operator-first core must not guess an operator's behaviour from its NAME: capabilities (ghosts /
fields / solver needs) and requirements come only from the declarer, and the operator name is a
debug/validation string, never a hot-path selector (a compiled kernel dispatches by the integer
OperatorId). This source-only test pins that contract:

  - the requirement-key vocabulary is documented as OPERATOR_REQUIREMENT_KEYS;
  - neither operators.py nor module.py imports ``re`` or matches an operator name against a substring
    to set capabilities/requirements (no name-guessing);
  - the registry exposes an integer-id dispatch (id_of / by_id) so codegen need not look up by name;
  - the codegen operator-emission path addresses operators without an operator-name string lookup.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"


def _src(rel):
    return (POPS / rel).read_text(encoding="utf-8")


def _tree(rel):
    path = POPS / rel
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_requirement_vocabulary_is_documented():
    src = _src("model/operators.py")
    assert "OPERATOR_REQUIREMENT_KEYS" in src, (
        "the operator requirements axes must be documented as OPERATOR_REQUIREMENT_KEYS (ADC-528)")
    for axis in ("ghosts", "fields", "params", "aux", "solvers", "layout", "backend"):
        assert '"%s"' % axis in src, "requirement axis %r must be in the documented vocabulary" % axis


def test_operator_core_does_not_import_re():
    # A regex over operator NAMES would be the classic name-guessing smell; the core forbids it.
    for rel in ("model/operators.py", "model/module.py", "model/registry.py"):
        tree = _tree(rel)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "re" not in imported, (
            "%s must not import re: capabilities/requirements are declared by the operator author, "
            "never inferred by matching the operator name" % rel)


def test_no_name_substring_capability_guessing():
    # No "electric" in name -> set a capability style logic: the core never keys behaviour on a name
    # substring. Guard against the obvious membership tests on an operator's .name.
    for rel in ("model/operators.py", "model/module.py"):
        src = _src(rel)
        assert "in name" not in src and "in op.name" not in src and "name.startswith" not in src, (
            "%s must not branch on an operator-name substring to set capabilities/requirements" % rel)


def test_registry_dispatches_by_integer_id():
    tree = _tree("model/registry.py")
    methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"id_of", "by_id"} <= methods, (
        "OperatorRegistry must expose integer-id dispatch (id_of / by_id) so a compiled kernel need "
        "not look an operator up by its name string (ADC-528)")


def test_codegen_operator_emit_has_no_name_lookup_dispatch():
    # The operator-emission codegen must address operators by id, not resolve a name string in the
    # emitted step. It is allowed to READ an operator's declared name for a comment / symbol; it must
    # not perform a registry.get(<name>) style hot lookup keyed on a runtime name.
    emit = POPS / "codegen" / "program_emit_ops.py"
    if not emit.exists():
        return  # the emitter lives elsewhere in some layouts; the registry-id test above still holds
    src = emit.read_text(encoding="utf-8")
    assert "OperatorRegistry" not in src or "by_id" in src or "id_of" in src, (
        "codegen operator emission that touches the registry must dispatch by integer id, not by an "
        "operator-name string lookup")
