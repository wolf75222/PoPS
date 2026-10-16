"""Automatic AMR ceilings include consumed implicit face publications."""
import pytest
import pops
from examples.api_v02.method_path_hierarchy_composition import build_path_hierarchy_composition
from pops.codegen.program_emit_amr import _flux_expression_budgets


@pytest.mark.parametrize("mode,count", [("straight", 1), ("branch", 1), ("unrolled", 2), ("fold", 2)])
@pytest.mark.parametrize("implicit", [False, True])
def test_nested_path_budgets_the_consumed_diffusion_once(mode, count, implicit):
    spec = build_path_hierarchy_composition(mode=mode, implicit=implicit)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert _flux_expression_budgets(plan.time) == ((count, 1), (1, 1) if implicit else (0, 0))


def test_temporal_polynomial_comes_from_the_exact_residual():
    from fractions import Fraction
    from types import SimpleNamespace
    from pops.codegen.program_emit_amr import _flux_expression_budgets
    # Use a real prepared residual and change only its authenticated dt coefficient
    # in a private test view. The counter checks the polynomial, not a solver name.
    spec = build_path_hierarchy_composition(mode="straight")
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    solve = next(node for node in plan.time._values if node.op == "solve_spatial_nonlinear")
    rate = next(node for node in solve.attrs["residual_block"] if node.op == "diffusive_rhs")
    residual = solve.attrs["residual"]
    coefficients = tuple({0: -Fraction(1, 3), 1: -Fraction(1, 5)} if node is rate else coefficient
        for node, coefficient in zip(residual.inputs, residual.attrs["coeffs"], strict=True))
    residual_view = SimpleNamespace(op=residual.op, inputs=residual.inputs,
        attrs={**residual.attrs, "coeffs": coefficients})
    solve_view = SimpleNamespace(**{key:getattr(solve,key) for key in ("id","op","inputs","block","name")},
        attrs={**solve.attrs, "residual": residual_view})
    program_view = SimpleNamespace(_values=tuple(solve_view if node is solve else node for node in plan.time._values),
        _block_indices=plan.time._block_indices, flux_expression_budget=None,
        temporal_manifest=plan.time.temporal_manifest)
    assert _flux_expression_budgets(program_view) == ((1, 1), (1, 2))


def test_dynamic_fold_with_implicit_only_requires_explicit_resource_capacity():
    from types import SimpleNamespace

    spec = build_path_hierarchy_composition(mode="straight")
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    solve = next(node for node in plan.time._values if node.op == "solve_spatial_nonlinear")
    # Real prepared solve/residual, placed under a native dynamic-capacity region.
    # No direct RHS outside its residual can trigger the guard accidentally.
    dynamic_bound = SimpleNamespace(op="int64_binary")
    fold = SimpleNamespace(op="structured_fold", inputs=(solve.inputs[0], dynamic_bound, dynamic_bound),
        attrs={"body_block": (solve,), "body_argument": solve.inputs[0], "body": solve},
        block=solve.block, id=-1, name="implicit_native_fold")
    view = SimpleNamespace(_values=(fold,), _block_indices=plan.time._block_indices,
        flux_expression_budget=None, temporal_manifest=plan.time.temporal_manifest)
    with pytest.raises(ValueError, match="dynamic fold flux storage requires an explicit"):
        _flux_expression_budgets(view)
    view.flux_expression_budget = (11, 7)
    assert _flux_expression_budgets(view) == ((11, 7), (11, 7))


def test_product_fold_keeps_the_exact_flux_polynomial_of_each_field():
    from pops.numerics.terms import Flux
    from tests.python.support.typed_program import program_states, synthetic_module

    program = pops.Program("nested_product_flux_budget")
    _, states = program_states(program, synthetic_module(program.name + "_model", components=("u",)),
                               ("blk",))
    state = states["blk"].n
    rhs = program.rhs(state=state, terms=[Flux()])
    initial_field = program.value("initial_flux", state+program.dt*rhs, at=state.point)
    initial = program.product(nested=program.product(rate=initial_field), count=program.int64(0))
    result = program.fold(initial, 2, lambda p, carried, index: p.product(
        nested=p.product(rate=p.value("polynomial", (1+p.dt)*carried["nested"]["rate"], at=carried["nested"]["rate"].point)),
        count=carried["count"] + 1))
    assert result.block is None
    # One actual flux evaluation, initially weighted dt, then multiplied twice by (1+dt).
    assert _flux_expression_budgets(program) == ((1, 3),)


def _budget_fixture():
    from types import SimpleNamespace
    from itertools import count

    ids = count()
    def node(op, inputs=(), **attrs):
        return SimpleNamespace(id=next(ids), op=op, inputs=inputs, attrs=attrs,
                               block="u", name=op, region=0)
    state = node("state")
    one = node("int64_constant", value=1)
    def program(values, histories=None):
        histories = {} if histories is None else histories
        return SimpleNamespace(_values=values, _block_indices=lambda: {"u": 0},
            flux_expression_budget=None, _histories=histories,
            temporal_manifest=lambda: {"histories": [
                {"name": name, "interpolation": {"kind": "hold"}} for name in histories]})
    return node, state, one, program


@pytest.mark.parametrize("kind", ["fold", "branch"])
def test_flux_budget_deep_control_preserves_paths_and_exact_multiplicity(kind):
    node, state, one, program = _budget_fixture()
    result = node("rhs")
    for _ in range(1200):
        if kind == "fold":
            result = node("structured_fold", (state, one, one), body_block=(result,),
                          body_argument=state, body=result)
        else:
            result = node("branch", true_block=(result,), true_result=result,
                          false_block=(), false_result=state)
    assert _flux_expression_budgets(program((state, result))) == ((1, 1),)


def test_flux_budget_deep_rhs_detection_keeps_dynamic_rejection():
    node, state, one, program = _budget_fixture()
    result = node("rhs")
    for _ in range(1600):
        result = node("region_holder", apply_block=(result,))
    dynamic = node("int64_binary")
    fold = node("structured_fold", (state, one, dynamic), body_block=(result,),
                body_argument=state, body=result)
    with pytest.raises(ValueError, match="dynamic fold flux storage requires an explicit"):
        _flux_expression_budgets(program((fold,)))


def test_flux_budget_deep_history_walk_and_shared_rhs_count_once():
    node, state, one, program = _budget_fixture()
    rhs = node("rhs")
    expression = rhs
    values = [state, rhs]
    for _ in range(1600):
        expression = node("linear_combine", (expression, expression), coeffs=({0: 1}, {0: -1}))
        values.append(expression)
    store = node("store_history", (expression,), history="h")
    # The retention estimate conservatively counts distinct RHS ancestry even
    # when the present exact expression cancels. Sharing must not double it.
    nested = store
    for _ in range(1600):
        nested.region = 1
        nested = node("region_holder", apply_block=(nested,))
    values.extend((nested, node("history", history="h")))
    assert _flux_expression_budgets(program(tuple(values), {"h": 2})) == ((3, 1),)


def test_flux_budget_history_region_order_keeps_last_store():
    node, state, one, program = _budget_fixture()
    a, b = node("rhs"), node("rhs")
    both = node("linear_combine", (a, b), coeffs=({0: 1}, {0: 1}))
    first = node("store_history", (a,), history="h")
    last = node("store_history", (both,), history="h")
    first.region = last.region = 1
    wrapper = node("region_holder", true_block=(first,), false_block=(last,))
    values = (a, b, both, wrapper, node("history", history="h"))
    assert _flux_expression_budgets(program(values, {"h": 2})) == ((6, 1),)
    wrapper.attrs.update(true_block=(last,), false_block=(first,))
    assert _flux_expression_budgets(program(values, {"h": 2})) == ((4, 1),)


def test_flux_budget_shared_region_counts_each_execution_and_rejects_cycle():
    node, state, one, program = _budget_fixture()
    rhs = node("rhs")
    shared = (rhs,)
    two = node("int64_constant", value=2)
    folded = node("structured_fold", (state, one, two), body_block=shared,
                  body_argument=state, body=rhs)
    assert _flux_expression_budgets(program((folded,))) == ((2, 1),)
    folded.attrs["body_block"] = (folded,)
    with pytest.raises(ValueError, match="cyclic region"):
        _flux_expression_budgets(program((folded,)))


def test_deep_region_traversal_preserves_ancestry_order_and_rejects_forgery():
    from pops.time._program.traversal import all_ops_with_ancestry

    node, state, one, program = _budget_fixture()
    leaf = node("rhs")
    root = leaf
    for _ in range(1600):
        root.region = 1
        root = node("region_holder", body_block=(root,))
    found = list(all_ops_with_ancestry(program((root,))))
    assert len(found) == 1601
    assert found[0] == (root, ())
    assert found[-1][0] is leaf
    assert len(found[-1][1]) == 1600
    assert all(key == "body_block" for _, key in found[-1][1])
    root.region = 1
    with pytest.raises(ValueError, match="structured ancestry disagrees"):
        list(all_ops_with_ancestry(program((root,))))
    root.region = 0
    with pytest.raises(ValueError, match="appears in multiple structured regions"):
        list(all_ops_with_ancestry(program((root, root))))
    root.id = True
    with pytest.raises(TypeError, match="non-negative exact integer"):
        list(all_ops_with_ancestry(program((root,))))
