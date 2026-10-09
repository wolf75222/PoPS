"""Independent dependency and mathematical counterexamples for authoring layers."""
import ast
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _finite_program_fixture():
    from pops import Case, Model
    from pops.frames import Cartesian2D
    from pops.time import Program
    model = Model("finite_expr", frame=Cartesian2D())
    state = model.state("U", components=("rho", "m"))
    block = Case("finite_case").block("fluid", model)
    program = Program("finite_program")
    return program, program.state(block[state])


@pytest.mark.parametrize("path", (
    "linalg/finite.py", "numerics/reconstruction/user.py",
    "numerics/reconstruction/joint.py", "numerics/riemann/user.py",
    "_ir/finite_linear.py",
))
def test_direction_includes_function_scoped_imports(path):
    tree = ast.parse((ROOT / "python/pops" / path).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    forbidden = ("pops.",) if path.startswith("linalg/") else \
                ("pops._ir", "pops.physics") if path.startswith("numerics/") else \
                ("pops.linalg", "pops.time", "pops.model", "pops.physics")
    assert not [name for name in imports if name.startswith(forbidden)]


def test_exact_numeric_finite_plans_keep_joint_native_application_and_arithmetic():
    from pops.linalg import FiniteLinearMap, FiniteSupport
    from pops._ir.expr import Const, _wrap
    from pops._ir.finite_linear import lower_finite_scalars, FiniteProjection
    support = FiniteSupport("ordered", ("a", "b"))
    mapping = FiniteLinearMap(support, support, ((1, 2), (3, 4)))
    vector = mapping.apply(support.bind((Fraction(1, 3), Decimal("0.125"))))
    roots = lower_finite_scalars(vector.components)
    assert all(isinstance(root, FiniteProjection) for root in roots)
    assert roots[0].application is roots[1].application
    assert roots[0].application.inputs[0].value == Fraction(1, 3)
    assert roots[0].application.inputs[1].value == Decimal("0.125")
    expression = _wrap((vector[0] + 2 * vector[1]) / 3)
    assert isinstance(expression.b, Const) and expression.b.value == 3
    with pytest.raises(FrozenInstanceError):
        vector[0].operation = "neg"
    with pytest.raises(TypeError, match="truth value"):
        bool(vector[0] > 0)
    # Pure literals retain the same scalar DAG instead of folding in Python.
    literal = support.bind((2**60, Fraction(1, 7)))
    sum_roots = lower_finite_scalars((literal + support.bind((1, 2))).components)
    assert sum_roots[0].a.value == 2**60 and sum_roots[0].b.value == 1


def test_invalid_or_cyclic_scalar_plan_protocol_refuses_conversion():
    from pops._ir.expr import _wrap

    class Invalid:
        def __pops_scalar_plan__(self):
            return ("pops.finite-scalar-plan@1", "projection", (object(), 0))

    class Cyclic:
        def __pops_scalar_plan__(self):
            return ("pops.finite-scalar-plan@1", "neg", (self,))

    with pytest.raises(TypeError, match="declaration protocol"):
        _wrap(Invalid())
    with pytest.raises(ValueError, match="cycle"):
        _wrap(Cyclic())


def test_separate_mixed_expr_arithmetic_keeps_one_joint_application_and_old_program_dag():
    from pops.linalg import FiniteLinearMap, FiniteSupport
    from pops._ir.expr import Const
    from pops._ir.finite_linear import FiniteApplication, FiniteProjection
    from pops.time.expressions import encode_expressions
    support = FiniteSupport("ordered", ("rho", "m"))
    vector = FiniteLinearMap(support, support, ((1, 2), (3, 4))).apply(support.bind((5, 6)))
    expressions = (Const(1) + vector[0], Const(2) + vector[1])
    assert expressions[0].b.application is expressions[1].b.application
    _, nodes, _ = encode_expressions(expressions, None)
    assert len(nodes) == 9
    assert sum(node[0] == "finite_linear_v1" for node in nodes) == 1

    actual, state = _finite_program_fixture()
    actual.value("mixed", (state.n[0] + vector[0], state.n[1] + vector[1]))
    legacy, previous = _finite_program_fixture()
    # The exact previous compiler DAG, authored directly in its owning IR layer.
    application = FiniteApplication("apply", support.contract, support.contract,
                                    ((1, 2), (3, 4)), (Const(5), Const(6)))
    legacy.value("mixed", (previous.n[0] + FiniteProjection(application, 0),
                            previous.n[1] + FiniteProjection(application, 1)))
    assert actual._ir_hash() == legacy._ir_hash()
    assert actual._values[-1].attrs["expression_nodes"] == legacy._values[-1].attrs["expression_nodes"]


def test_scalar_reuse_across_finite_vector_arithmetic_preserves_const_sharing():
    from pops.linalg import FiniteSupport
    from pops._ir.finite_linear import lower_finite_scalars
    support = FiniteSupport("two", ("a", "b"))
    values = support.bind((5, 6))
    combined = values + values
    roots = lower_finite_scalars(combined.components)
    assert roots[0].a is roots[0].b
    assert roots[1].a is roots[1].b


def test_compiler_lowering_cache_does_not_own_math_declarations_or_expressions():
    import gc
    import weakref
    from pops.linalg import FiniteLinearMap, FiniteSupport
    from pops._ir.expr import _wrap
    support = FiniteSupport("one", ("a",))
    vector = FiniteLinearMap(support, support, ((2,),)).apply(support.bind((3,)))
    declaration = vector[0]
    declaration_ref = weakref.ref(declaration)
    expression = _wrap(declaration)
    expression_ref = weakref.ref(expression)
    del vector, declaration
    gc.collect()
    assert declaration_ref() is None
    del expression
    gc.collect()
    assert expression_ref() is None


def test_vector_scalar_arithmetic_preserves_distinct_literal_nodes_and_old_program_hash():
    from pops.linalg import FiniteLinearMap, FiniteSupport
    from pops._ir.expr import Const
    from pops._ir.finite_linear import FiniteApplication, FiniteProjection
    from pops.time.expressions import ProgramExpression
    support = FiniteSupport("ordered", ("rho", "m"))
    actual, state = _finite_program_fixture()
    vector = FiniteLinearMap(support, support, ((1, 2), (3, 4))).apply(support.bind(state.n))
    (vector + vector * Fraction(2, 3)).materialize(actual, "scaled", template=state.n)
    legacy, previous = _finite_program_fixture()
    application = FiniteApplication("apply", support.contract, support.contract,
                                    ((1, 2), (3, 4)), (previous.n[0], previous.n[1]))
    left, right = FiniteProjection(application, 0), FiniteProjection(application, 1)
    legacy._pointwise_expression("scaled", ProgramExpression(
        (left + left * Const(Fraction(2, 3)), right + right * Const(Fraction(2, 3))), previous.n),
        finite_support=support.contract)
    assert actual._ir_hash() == legacy._ir_hash()


def test_registered_exact_literals_and_mutable_hook_values_are_captured_once():
    from pops.identity.scalar import scalar_literal
    from pops.linalg import FiniteSupport
    from pops._ir.finite_linear import lower_finite_scalars
    exact = scalar_literal(Fraction(1, 7), unit="unitless", target="double")

    class Literal:
        value = .2

        def __pops_scalar_literal__(self):
            return {"kind": "binary64", "payload": self.value.hex()}

    supplied = Literal()
    vector = FiniteSupport("literals", ("fraction", "custom")).bind((exact, supplied))
    supplied.value = .9
    roots = lower_finite_scalars(vector.components)
    assert roots[0].literal == exact
    assert roots[1].value == .2


def test_finite_scalar_declarations_remain_legal_model_equation_outputs():
    from pops.linalg import FiniteLinearMap, FiniteSupport
    from pops.model import Module, Signature, Rate
    from pops._ir.finite_linear import FiniteProjection
    support = FiniteSupport("one", ("u",))
    projection = FiniteLinearMap(support, support, ((2.,),)).apply(support.bind((3.,)))[0]
    module = Module("finite_model_capture")
    state = module.state_space("U", ("u",))

    @module.operator(name="constant_rate", signature=Signature((state,), Rate(state)),
                     kind="local_source")
    def constant_rate(_state):
        return projection

    assert isinstance(module.operator_registry().get("constant_rate").body[0], FiniteProjection)


def test_numerical_state_shape_accepts_real_boards_and_registry_rejects_ducks():
    from pops import Model
    from pops.frames import Cartesian2D
    from pops.model.scalar_contract import state_component_count
    model = Model("shape_contract", frame=Cartesian2D())
    state = model.state("U", components=("first", "second"))
    assert state_component_count(state) == 2
    module = model.module
    assert state_component_count(module.state_handle(module.state_spaces()["U"])) == 2

    class Forged:
        state_components = ("first", "second")

    with pytest.raises(TypeError, match="StateHandle"):
        state_component_count(Forged())
    from pops.model.handles import StateShapeHandle
    with pytest.raises(TypeError, match="abstract"):
        StateShapeHandle("U", kind="state", owner=state.owner_path)
