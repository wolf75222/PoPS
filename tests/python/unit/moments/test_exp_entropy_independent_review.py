"""Independent guards, affine-exact moments and entropy optimality witnesses."""
import ctypes
import math
import shutil
import subprocess

import numpy as np
import pops
import pytest
from pops import math as pmath
from pops._ir.expr import Var
from pops._ir.lowering import diff
from pops._ir.visitors import _key
from pops.moments.closures import DiscreteEntropyQuadrature


def test_exp_chain_derivative_and_immutable_public_program_detachment():
    from examples.migration.scientific.api040_m18_entropy import make_case
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    x = Var("x", "cons")
    value = pmath.exp(x*x-2*x)
    derivative = diff(value, x)
    for point in (-.7, 0., 1., 2.3):
        assert derivative.eval({"x": point}) == pytest.approx(
            (2*point-2)*math.exp(point*point-2*point), rel=2e-15, abs=1e-15)
    assert _key(value) != _key(x*x-2*x)
    case, layout, _ = make_case()
    plan = pops.resolve(pops.validate(case), layout=layout)
    detached = detach_compiled_program(plan.time)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    assert detached._ir_hash() == plan.time._ir_hash()
    assert emit_cpp_program(detached, model=graph) == emit_cpp_program(plan.time, model=graph)


def test_seven_node_four_constraint_quadrature_matches_direct_probability_sum():
    nodes = (-1., -.7, -.2, .1, .35, .7, 1.)
    weights = (.1, .2, .15, .1, .2, .15, .1)
    # A permuted, mixed basis: not the witness's hardcoded three moments.
    basis = np.asarray([[v*v*v-.2*v for v in nodes], [1.]*7,
                        [v for v in nodes], [v*v for v in nodes]])
    quadrature = DiscreteEntropyQuadrature(nodes, weights, basis.tolist())
    multiplier = (.05, -.2, .13, -.07)
    probabilities = np.asarray(weights)*np.exp(basis.T @ multiplier)
    symbolic = quadrature.populations(multiplier)
    actual = np.array([value.eval({}) for value in symbolic])
    np.testing.assert_allclose(actual, probabilities, rtol=3e-15, atol=0.)
    moments = np.asarray([value.eval({}) for value in quadrature.moments(symbolic)])
    np.testing.assert_allclose(moments, basis @ probabilities, rtol=3e-15, atol=1e-16)
    # Stationarity lies in row(B), independently of the solver and target recipe.
    gradient = np.log(probabilities / np.asarray(weights))
    _, _, right = np.linalg.svd(basis, full_matrices=True)
    tangent = right[4:].T @ np.asarray((.003, -.002, .001))
    assert abs(gradient @ tangent) < 2e-17
    perturbed = probabilities + tangent
    assert perturbed.min() > 0
    entropy = lambda p: np.sum(p*(np.log(p/np.asarray(weights))-1))
    assert entropy(perturbed) > entropy(probabilities)


def test_exp_bind_expression_retains_exact_instance_parameter_identity():
    from pops.frames import Cartesian2D
    from pops.model._bind_expression import qualified_expression_key, eval_expression_key
    from pops.params import RuntimeParam
    model = pops.Model("shared_entropy_parameter", frame=Cartesian2D())
    model.state("U", components=("u",))
    parameter = model.param(RuntimeParam("gain", default=.2))
    expression = pmath.exp(model.value(parameter))
    case = pops.Case("two_parameter_owners")
    blocks = [case.block(name, model) for name in ("left", "right")]
    keys = [qualified_expression_key(expression.resolve_references(
        lambda declaration, block=block: case.resolve(declaration, block=block)), where="review")
        for block in blocks]
    assert keys[0] != keys[1]
    qids = [case.resolve(block[parameter]).qualified_id for block in blocks]
    env = {qids[0]: .2, qids[1]: -.7}
    assert eval_expression_key(keys[0], env, where="review") == pytest.approx(math.exp(.2))
    assert eval_expression_key(keys[1], env, where="review") == pytest.approx(math.exp(-.7))
    with pytest.raises((ValueError, KeyError), match="depend|missing|unavailable"):
        eval_expression_key(keys[1], {qids[0]: .2}, where="review")


def test_m18_target_cone_has_an_independent_polygon_certificate():
    from examples.migration.scientific.api040_m18_entropy import moderate_multipliers, target_moments
    from tests.python.support.discrete_entropy_oracle import cone_position
    # With mass one, (mean,second moment) is in the polygon through (v,v²).
    # Lower edges interpolate neighboring nodes; the top edge is variance<=1.
    nodes = np.asarray((-1., -.5, 0., .5, 1.))

    def geometric(target):
        mass, first, second = target
        if mass <= 0:
            return "outside"
        x, y = first/mass, second/mass
        if x < -1 or x > 1:
            return "outside"
        k = min(3, max(0, np.searchsorted(nodes, x, side="right")-1))
        a, b = nodes[k:k+2]
        low = (a+b)*x-a*b
        if y < low-1e-12 or y > 1+1e-12:
            return "outside"
        return "boundary" if min(y-low, 1-y) <= 1e-12 else "interior"

    targets = target_moments(moderate_multipliers()).reshape(3, -1).T
    witnesses = [*targets, (1., 0., 1.1), (1., 1., 1.), (1., .25, .125), (1., .9, .2)]
    for target in witnesses:
        assert cone_position(target)[0] == geometric(target)


def test_rank_local_assertion_failure_is_converged_before_following_collective(monkeypatch):
    from tests.python.integration.runtime.test_m18_discrete_entropy_runtime import _all_check
    from pops import _native_collectives
    seen = []

    def gather(world, value):
        seen.append(value)
        return (None, ("AssertionError", "rank 1 rollback clock differs"))

    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    for _rank in (0, 1):
        with pytest.raises(AssertionError, match="rank 1 rollback clock differs"):
            _all_check(object(), lambda: None)
    assert seen == [None, None]


def test_exp_nonfinite_leaf_and_overflow_cannot_be_masked_by_minimum(tmp_path):
    from pops.codegen.cpp_writer import _cse_emit
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler required")
    x = Var("x", "cons")
    roots = [pmath.minimum(pmath.exp(x), 1.),
             pmath.where(x < 0., lambda: 7., lambda: pmath.exp(1000*x))]
    lines, names = _cse_emit(roots, "double", "")
    source = '\n'.join(['#include <cmath>', '#include <limits>',
        'namespace pops {using Real=double;}', 'namespace Kokkos {using std::fmin;}',
        'extern "C" void evaluate(double x,double* out){', *lines,
        *(f'out[{i}]={name};' for i, name in enumerate(names)), '}'])
    cpp, library = tmp_path/'exp_review.cpp', tmp_path/'exp_review.so'
    cpp.write_text(source)
    completed = subprocess.run([compiler,'-std=c++20','-O2','-fno-fast-math','-ffp-contract=off',
        '-shared','-fPIC',str(cpp),'-o',str(library)], capture_output=True,text=True)
    assert completed.returncode == 0, completed.stderr
    function = ctypes.CDLL(str(library)).evaluate
    function.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    values = (ctypes.c_double*2)()
    function(-1., values)
    assert values[0] == pytest.approx(math.exp(-1.)) and values[1] == 7.
    for invalid in (1000., float("nan"), -float("inf")):
        function(invalid, values)
        assert not math.isfinite(values[0])
