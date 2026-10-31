"""Independent adversarial checks for the authored numerical face contract."""
import math
import operator
import random
import re
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pops._ir.visitors import _key
from pops.codegen.user_riemann_lowering import _concealed_operand_keys, emit_user_face_policy
from pops.numerics import riemann
from pops.numerics.riemann.user import authenticated_user_face
from tests.python.unit.numerics.test_user_face_authoring import state


@pytest.mark.parametrize("width", (1, 2, 5, 17, 257))
def test_common_vector_body_has_no_small_component_catalogue(width):
    _, quantity = state(width)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed: .5 * (fl + fr) - .5 * speed * (right - left),
        stability=lambda left, right, fl, fr, speed: speed,
        state=quantity)
    assert authenticated_user_face(descriptor) is descriptor
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert "density[%d] =" % (width - 1) in source
    assert "canonical_evaluation(*this, physical, left, right, face)" in source
    assert "physical_pair_failed(" in source


@pytest.mark.parametrize("field,value", (
    ("capabilities", {"vector_face": True, "conservative_shared_face": False}),
    ("requirements", {"capabilities": (), "source_compiled": True}),
))
def test_changed_face_permissions_are_rejected_before_native_emission(field, value):
    _, quantity = state(2)
    descriptor = riemann.User(body=lambda left, right, fl, fr, speed: fl,
                              stability=lambda left, right, fl, fr, speed: speed,
                              state=quantity)
    altered = copy(descriptor)
    object.__setattr__(altered, field, value)
    with pytest.raises(ValueError, match="capabilities changed"):
        authenticated_user_face(altered)


def test_lazy_inactive_face_leaf_stays_inside_selected_common_ir_branch():
    from pops.math import where

    _, quantity = state(1)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed:
            (where(left[0] > 0, lambda: 1 / (right[0] - right[0]), lambda: fl[0]),),
        stability=lambda left, right, fl, fr, speed: speed,
        state=quantity)
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    body = source[source.index("State density{};"):]
    assert body.index("?") < body.index("right.state[0]")
    assert "std::isfinite" in body


def test_dissipative_body_can_author_its_numerical_stability_separately():
    """The physical speed alone is not a CFL certificate for arbitrary bodies."""
    _, quantity = state(1)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed: .5 * (fl + fr) - 10 * speed * (right - left),
        stability=lambda left, right, fl, fr, speed: 20 * speed,
        state=quantity)
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert "20" in source
    # With F=0 and physical speed=1, the flux is -10*(R-L).
    # At dt/dx=.5 a unit isolated peak becomes 1 - 2*10*.5 = -9,
    # whereas the authored bound 20 rejects this step (frequency*dt=10).
    dx, dt, diffusion = 1., .5, 10.
    left_face, right_face = -diffusion * (1. - 0.), -diffusion * (0. - 1.)
    assert 1. - dt / dx * (right_face - left_face) == -9.
    assert dt * 1. / dx <= 1.
    assert dt * 20. / dx > 1.


_UNARY = {"Neg": np.negative, "Abs": np.fabs, "Sqrt": np.sqrt, "Exp": np.exp, "Expm1": np.expm1,
          "Log": np.log, "Sin": np.sin, "Cos": np.cos, "Rounded": float}
_BINARY = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide,
           "Pow": np.power, "Minimum": np.fmin, "Maximum": np.fmax}
_COMPARE = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
            "gt": operator.gt, "ge": operator.ge}
_SPECIAL = (0., -0., .5, 1., -1., 2., 1e300, -1e300, math.inf, -math.inf, math.nan)


def _ieee(node, inputs, values):
    """Evaluate ``node`` as the emitted policy does and append each evaluated (node, value).

    Binary64 arithmetic, IEEE fmin/fmax, the branch-free sign, only the selected branch of a
    where and short-circuit Boolean operators.
    """
    kind = type(node).__name__
    if kind == "Const":
        return float(node.value)
    if kind == "Var":
        value = inputs[node.name]
    elif kind == "Where":
        value = _ieee(node.yes if _ieee(node.test, inputs, values) else node.no, inputs, values)
    elif kind in ("BooleanAnd", "BooleanOr"):
        value = _ieee(node.a, inputs, values)
        if value == (kind == "BooleanAnd"):
            value = _ieee(node.b, inputs, values)
    elif kind == "BooleanNot":
        value = not _ieee(node.a, inputs, values)
    elif kind == "Compare":
        value = _COMPARE[node.comparison](_ieee(node.a, inputs, values),
                                          _ieee(node.b, inputs, values))
    elif kind == "Sign":
        operand = _ieee(node.a, inputs, values)
        value = float(operand > 0) - float(operand < 0)
    elif kind in _UNARY:
        value = float(_UNARY[kind](_ieee(node.a, inputs, values)))
    else:
        value = float(_BINARY[kind](_ieee(node.a, inputs, values), _ieee(node.b, inputs, values)))
    values.append((node, value))
    return value


def _verdicts(roots, width, inputs, *rules):
    """The policy verdict for each rule, a predicate on the evaluated nodes that it tests."""
    values = []
    with np.errstate(all="ignore"):
        outputs = [_ieee(root, inputs, values) for root in roots]
    if not math.isfinite(outputs[width]) or outputs[width] < 0:
        return ("kUserInvalidStability",) * len(rules)
    return tuple(
        "ok" if all(math.isfinite(value) for value in
                    [*(value for node, value in values if rule(node)), *outputs[:width]])
        else "kUserNonFiniteFlux" for rule in rules)


def _random_face(seed, width=2):
    """Random face roots over every scalar node kind, sharing nodes between positions."""
    from pops._ir.control_expr import Rounded, Where
    from pops._ir.expr import (Abs, BooleanAnd, BooleanNot, BooleanOr, Compare, Const, Cos, Exp,
                               Expm1, Log, Maximum, Minimum, Neg, Sign, Sin, Sqrt, Var)

    rng = random.Random(seed)
    built = [Var("x%d" % index, "face_input") for index in range(4)]

    def boolean(depth):
        kind = rng.randrange(5) if depth > 0 else 0
        if kind < 2:
            return Compare(rng.choice(tuple(_COMPARE)), real(depth), real(depth))
        if kind == 2:
            return BooleanNot(boolean(depth - 1))
        return (BooleanAnd, BooleanOr)[kind - 3](boolean(depth - 1), boolean(depth - 1))

    def real(depth):
        if depth == 0 or rng.random() < .2:
            return (rng.choice(built) if rng.random() < .8
                    else Const(rng.choice((0., .5, 1., -2.))))
        kind = rng.randrange(18)
        if kind < 7:
            a, b = real(depth - 1), real(depth - 1)
            node = (a + b, a - b, a * b, a / b, a ** b, Minimum(a, b), Maximum(a, b))[kind]
        elif kind < 17:
            node = (Neg, Abs, Sqrt, Exp, Expm1, Log, Sin, Cos, Sign, Rounded)[kind - 7](
                real(depth - 1))
        else:
            node = Where(boolean(depth - 1), real(depth - 1), real(depth - 1))
        built.append(node)
        return node

    return (*(real(4) for _ in range(width)), Abs(real(2)))


def _policy_rule(roots):
    """The nodes whose evaluated values the emitted policy tests besides its outputs."""
    concealed, memo = _concealed_operand_keys(roots)
    return lambda node: _key(node, memo) in concealed


def test_tested_operands_refuse_the_faces_of_every_non_finite_evaluated_value():
    """The policy tests its outputs and the operands that could conceal a non-finite value.

    On random face DAGs and IEEE special inputs, it refuses the same faces, with the same cause,
    as a test of every evaluated value (ROM-100): the refusal set of kUserNonFiniteFlux holds.
    """
    width, concealing = 2, 0
    for seed in range(40):
        roots = _random_face(seed, width)
        policy, rng = _policy_rule(roots), random.Random(seed)
        for _ in range(100):
            inputs = {"x%d" % index: (rng.choice(_SPECIAL) if rng.random() < .35
                                      else rng.uniform(-3., 3.)) for index in range(4)}
            every, tested, outputs = _verdicts(
                roots, width, inputs, lambda node: True, policy, lambda node: False)
            assert tested == every, (seed, inputs)
            concealing += outputs != every
    # Outputs alone would accept these faces: the tested operands are not decorative.
    assert concealing >= 100


def test_policy_tests_only_outputs_of_carried_arithmetic():
    _, quantity = state(2)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed: .5 * (fl + fr) - .5 * speed * (right - left),
        stability=lambda left, right, fl, fr, speed: speed, state=quantity)
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    assert "if (!(std::isfinite(density[0]) && std::isfinite(density[1])))" in source


def test_policy_tests_a_denominator_but_not_the_quotient_it_carries():
    _, quantity = state(1)
    descriptor = riemann.User(
        body=lambda left, right, fl, fr, speed: (fl[0] / (right[0] - left[0]),),
        stability=lambda left, right, fl, fr, speed: speed, state=quantity)
    source = emit_user_face_policy(SimpleNamespace(_user_face=descriptor))
    declared = dict(re.findall(r"const pops::Real (cse\d+_) = (.*);", source))
    finite = next(line for line in source.splitlines() if "if (!(std::isfinite" in line)
    difference = next(name for name, value in declared.items() if " - " in value)
    quotient = next(name for name, value in declared.items() if " / " in value)
    assert "std::isfinite(%s)" % difference in finite
    assert "std::isfinite(%s)" % quotient not in finite
    assert "std::isfinite(density[0])" in finite
