"""ADC-652: symbolic expressions are immutable, non-hashable and non-truthy."""
from __future__ import annotations

from pathlib import Path

import pytest

from pops._ir import (
    Compare, Const, Expr, Partial, SourceLocation, SymbolicTruthValueError, Unknown,
    ValueExpr, Var, diff,
)
from pops._ir.visitors import _children, _key
from pops.model import Handle, OwnerKind, OwnerPath, UnresolvedOwnershipError


def test_symbolic_operators_build_graph_nodes_without_python_evaluation():
    u = Var("u", "cons")

    predicate = (u + 1) > 0
    equality = u == Var("v", "cons")

    assert isinstance(predicate, Compare)
    assert predicate.comparison == "gt"
    assert predicate.a.to_cpp() == "(u + pops::Real(1))"
    assert isinstance(equality, Compare)
    assert equality.comparison == "eq"


def test_expr_has_no_hash_and_cannot_be_mutated():
    expr = Var("u", "cons") + Const(1)

    with pytest.raises(TypeError, match="unhashable"):
        hash(expr)
    with pytest.raises(AttributeError, match="immutable"):
        expr.a = Const(2)
    with pytest.raises(AttributeError, match="immutable"):
        del expr.b


def test_bool_reports_a_stable_code_user_provenance_and_actionable_alternatives():
    expr = Var("u", "cons") > 0

    with pytest.raises(SymbolicTruthValueError) as raised:
        bool(expr)

    error = raised.value
    assert error.code == "symbolic_truth_value"
    assert isinstance(error.location, SourceLocation)
    assert Path(error.location.file).name == Path(__file__).name
    assert error.location.line > 0
    assert "where(...)" in error.suggestions
    assert "T.branch(...)" in error.suggestions
    assert "T.while_(...)" in error.suggestions
    assert "[symbolic_truth_value]" in str(error)


def test_if_expr_fails_at_the_python_control_flow_boundary():
    expr = Var("u", "cons") > 0

    with pytest.raises(SymbolicTruthValueError, match="has no Python truth value"):
        if expr:
            pytest.fail("a symbolic expression must never enter this branch")


def test_chained_comparison_fails_instead_of_building_a_partial_graph():
    u = Var("u", "cons")

    with pytest.raises(SymbolicTruthValueError) as raised:
        _ = 0 < u < 1

    assert raised.value.code == "symbolic_truth_value"
    assert Path(raised.value.location.file).name == Path(__file__).name


def test_value_expr_uses_the_generic_traversal_cse_and_diff_protocols():
    owner = OwnerPath.model("transport")
    tracer = Handle("u", kind="state", owner=owner)
    same = Handle("u", kind="state", owner=owner)
    foreign = Handle("u", kind="state", owner=OwnerPath.model("other"))
    value = ValueExpr(tracer)

    assert _children(value) == ()
    assert _key(value) == _key(ValueExpr(same))
    assert _key(value) != _key(ValueExpr(foreign))
    assert diff(value, tracer).value == 1
    assert diff(value, foreign).value == 0
    with pytest.raises(TypeError, match="owner-aware binding"):
        value.to_cpp()


def test_value_expr_cse_keeps_distinct_live_authoring_owners():
    left = Handle(
        "u", kind="state", owner=OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "same"))
    right = Handle(
        "u", kind="state", owner=OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "same"))

    with pytest.raises(UnresolvedOwnershipError, match="definition fingerprint"):
        left.owner_path.canonical()
    with pytest.raises(UnresolvedOwnershipError, match="definition fingerprint"):
        right.owner_path.canonical()
    assert left != right
    with pytest.raises(UnresolvedOwnershipError, match="authoring-owned"):
        left.canonical_identity()
    assert _key(ValueExpr(left)) != _key(ValueExpr(right))
    assert diff(ValueExpr(left), right).value == 0


def test_external_expr_can_define_small_protocols_and_parameterized_new():
    class Shift(Expr):
        def __new__(cls, child, amount):
            instance = super().__new__(cls)
            object.__setattr__(instance, "allocated_with", amount)
            return instance

        def __init__(self, child, amount):
            self.child = child
            self.amount = amount

        def __pops_ir_children__(self):
            return (self.child,)

        def __pops_ir_key__(self, recurse):
            return ("shift", self.amount, recurse(self.child))

        def __pops_ir_diff__(self, *, recurse, target, definitions):
            return recurse(self.child)

        def eval(self, env):
            return self.child.eval(env) + self.amount

        def to_cpp(self):
            return "(%s + %s)" % (self.child.to_cpp(), self.amount)

    node = Shift(Var("u", "cons"), 2)

    assert node.allocated_with == 2
    assert _children(node) == (node.child,)
    assert _key(node) == ("shift", 2, ("var", "cons", "u"))
    assert diff(node, Var("u", "cons")).value == 1
    with pytest.raises(AttributeError, match="immutable"):
        node.amount = 3


@pytest.mark.parametrize(
    ("name", "kind"),
    [("", "cons"), (object(), "cons"), ("u", ""), ("u", 3)],
)
def test_var_rejects_unstable_identity_metadata(name, kind):
    with pytest.raises(TypeError, match="non-empty string"):
        Var(name, kind)


@pytest.mark.parametrize("axis", [True, -1, 2, "0"])
def test_partial_rejects_implicit_or_out_of_range_axes(axis):
    with pytest.raises(ValueError, match="integer 0 or 1"):
        Partial(Unknown("phi"), axis)


@pytest.mark.parametrize("name", ["", 3, object()])
def test_unknown_rejects_implicit_stringification(name):
    with pytest.raises(TypeError, match="non-empty name or a declared field Handle"):
        Unknown(name)


def test_symbolic_metadata_is_transitively_frozen_and_opaque_leaves_fail_loud():
    class Payload(Expr):
        def __init__(self, items):
            self.items = items

    original = [Var("u", "cons")]
    node = Payload(original)
    original.append(Var("v", "cons"))
    assert len(node.items) == 1 and node.items[0].name == "u"

    class Opaque:
        pass

    with pytest.raises(TypeError, match="mutable or opaque"):
        Payload(Opaque())

    class MutableKey:
        __hash__ = object.__hash__

    with pytest.raises(TypeError, match="mutable or opaque"):
        Payload({MutableKey(): "value"})
