"""Independent source-only lifetime/refusal probes for 1c84af0 reification."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import gc
import weakref

import pytest

from pops._ir.expr import Const, _wrap
from pops._ir.finite_linear import _lowered_finite_declarations, lower_finite_scalars
from pops.linalg import FiniteLinearMap, FiniteSupport


def declaration():
    support = FiniteSupport("lifetime", ("a", "b"))
    mapping = FiniteLinearMap(support, support, ((1, 2), (-3, 4)))
    return mapping.apply(support.bind((5, 6)))


def test_mixed_conversions_share_projection_and_application_while_alive():
    vector = declaration()
    first = Const(1) + vector[0]
    second = Const(2) + vector[1]
    assert first.b.application is second.b.application
    assert _wrap(vector[0]) is first.b
    roots = lower_finite_scalars(vector.components)
    assert roots[0] is first.b and roots[1] is second.b


def test_exact_constant_input_sharing_survives_conversion_and_vector_reuse():
    support = FiniteSupport("literal_reuse", ("a", "b"))
    constant = Const(Fraction(1, 7))
    mapping = FiniteLinearMap(support, support, ((1, 2), (3, 4)))
    vector = mapping.apply(support.bind((constant, constant)))
    roots = lower_finite_scalars((vector + vector).components)
    assert roots[0].a is roots[0].b and roots[1].a is roots[1].b
    assert roots[0].a.application is roots[1].a.application
    assert roots[0].a.application.inputs == (constant, constant)
    assert roots[0].a.application.inputs[0] is roots[0].a.application.inputs[1]


def test_releasing_expression_does_not_invalidate_live_declaration():
    vector = declaration()
    expression = _wrap(vector[0])
    previous = weakref.ref(expression)
    del expression
    gc.collect()
    assert previous() is None
    rebuilt = _wrap(vector[0])
    assert rebuilt.application.coefficients == ((1, 2), (-3, 4))
    assert _wrap(vector[1]).application is rebuilt.application


def test_many_retired_declarations_and_expressions_leave_no_cache_ownership():
    references, keys = [], set()
    for _ in range(100):
        vector = declaration()
        scalar = vector[0]
        application = scalar.arguments[0]
        expression = _wrap(scalar)
        for item in (scalar, application, *application.inputs):
            references.append(weakref.ref(item))
            keys.add(id(item))
        references.append(weakref.ref(expression))
        del vector, scalar, application, expression, item
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not keys.intersection(_lowered_finite_declarations)


@pytest.mark.parametrize("row", (
    ("pops.finite-scalar-plan@2", "number", (1,)),
    ("pops.finite-scalar-plan@1", "unknown", (1,)),
    ("pops.finite-scalar-plan@1", "number", (1, 2)),
    ("pops.finite-scalar-plan@1", "projection", (object(), 0)),
))
def test_malformed_version_operation_arity_or_application_never_enters_cache(row):
    class Plan:
        def __pops_scalar_plan__(self):
            return row

    supplied = Plan()
    with pytest.raises((TypeError, ValueError)):
        _wrap(supplied)
    assert id(supplied) not in _lowered_finite_declarations


def test_cyclic_and_mutation_attempts_remain_explicitly_refused():
    class Cycle:
        def __pops_scalar_plan__(self):
            return ("pops.finite-scalar-plan@1", "neg", (self,))

    with pytest.raises(ValueError, match="cycle"):
        _wrap(Cycle())
    vector = declaration()
    with pytest.raises(FrozenInstanceError):
        vector[0].operation = "number"
    with pytest.raises(TypeError, match="truth value"):
        bool(vector[0] > 0)
