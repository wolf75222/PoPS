"""ADC-556: the field-solve surface is typed, not free-string.

After ADC-556, ``m.solve_field(...)`` is the single field-solve facade and its result is a typed
``FieldsHandle`` (an ``OperatorHandle`` subtype) with STRUCTURED outputs. This source-only test
pins the target surface:

  - ``field_operator`` SURVIVES as an operator KIND (the compile-time contract) -- it is NOT
    removed; only the free-string authoring/lookup surface is;
  - there is no public ``field_operator(...)`` AUTHORING verb on the board ``Model``;
  - there is no ``.output(<str>)`` free-string output-lookup method on ``FieldsHandle`` (outputs are
    the structured ``FieldOutputs`` attribute object);
  - ``FieldsHandle`` is declared as an ``OperatorHandle`` subclass.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"


def _module(rel):
    path = POPS / rel
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def test_field_operator_survives_as_an_operator_kind():
    src = (POPS / "model" / "operators.py").read_text(encoding="utf-8")
    assert '"field_operator"' in src, (
        "field_operator must remain an operator KIND (the compile-time contract); ADC-556 removes "
        "only the free-string authoring/lookup surface, never the kind")


def test_no_field_operator_authoring_verb_on_board_model():
    tree = _module("physics/board.py")
    model = _class(tree, "Model")
    assert model is not None
    methods = {n.name for n in model.body if isinstance(n, ast.FunctionDef)}
    assert "field_operator" not in methods, (
        "board Model must not expose a public field_operator(...) authoring verb; author a field "
        "solve via m.solve_field(...) / m.field_problem(...)")


def test_fields_handle_has_no_free_string_output_method():
    tree = _module("physics/board_handles.py")
    handle = _class(tree, "FieldsHandle")
    assert handle is not None
    methods = {n.name for n in handle.body if isinstance(n, ast.FunctionDef)}
    assert "output" not in methods, (
        "FieldsHandle must not expose a free-string .output(str) lookup; outputs are the structured "
        "FieldOutputs attribute object (fields.outputs.E)")


def test_fields_handle_is_an_operator_handle_subclass():
    tree = _module("physics/board_handles.py")
    handle = _class(tree, "FieldsHandle")
    assert handle is not None
    base_names = {b.id for b in handle.bases if isinstance(b, ast.Name)}
    assert "OperatorHandle" in base_names, (
        "FieldsHandle must subclass OperatorHandle so a field solve is a typed operator")
