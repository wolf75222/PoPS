"""Source/stdlib mathematics only: no synthetic positive native M18 states."""
import ast
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests/review" / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


guard = load("m18_contra_equations", "sol61_m18_original_equation_guard.py")
subject = load("m18_review_subject", "sol61_m18_entropy_offline_oracle.py")


@pytest.fixture(scope="module")
def actual_source_ir():
    import pops
    from examples.migration.scientific.api040_m18_entropy import make_case
    case, layout, _ = make_case()
    plan = pops.resolve(pops.validate(case), layout=layout)
    return plan.time._serialize(include_provenance=False)


def ir_hash(ir):
    return hashlib.sha256(json.dumps(ir, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_actual_public_source_equations_are_exactly_the_original_moments(actual_source_ir):
    guard.verify(actual_source_ir)
    subject.compiled_contract(actual_source_ir, ir_hash(actual_source_ir))


ATTACKS = ("zero_roots", "swapped_moments", "weight", "exponent_basis", "target_component",
           "capture_in_exponent", "unknown_component", "foreign_operator", "reversed_inputs",
           "future_capture", "foreign_clock", "bool_capture_step", "publication_bypass",
           "refusal_bypass", "duplicate_node_id", "hidden_operation")


def counter_ir(original, label):
    ir = copy.deepcopy(original)
    solve = next(node for node in ir["nodes"] if node["op"] == "solve_coupled_implicit")
    attrs, rows = solve["attrs"], solve["attrs"]["expression_nodes"]
    if label == "zero_roots":
        attrs["expressions"] = [0, 0, 0]
    elif label == "swapped_moments":
        attrs["expressions"].reverse()
    elif label == "weight":
        next(row for row in rows if row == ["literal", {"kind": "binary64", "value": .1.hex()}])[1]["value"] = .2.hex()
    elif label == "exponent_basis":
        next(row for row in rows if row == ["literal", {"kind": "binary64", "value": (-1.).hex()}])[1]["value"] = (-.5).hex()
    elif label == "target_component":
        next(row for row in rows if row == ["input", 1, 0])[2] = 1
    elif label == "capture_in_exponent":
        next(row for row in rows if row == ["input", 0, 0])[1] = 1
    elif label == "unknown_component":
        next(row for row in rows if row == ["input", 0, 2])[2] = 1
    elif label == "foreign_operator":
        next(row for row in rows if row[0] == "exp")[0] = "log"
    elif label == "reversed_inputs":
        solve["inputs"].reverse()
    elif label in {"future_capture", "foreign_clock", "bool_capture_step"}:
        capture = next(node for node in ir["nodes"] if node["id"] == solve["inputs"][1])
        if label == "future_capture":
            capture["point"]["step"] = 1
        elif label == "bool_capture_step":
            capture["point"]["step"] = False
        else:
            capture["point"]["clock"]["owner"]["nodes"][0]["name"] = "foreign_program"
    elif label == "publication_bypass":
        ir["commits"][0]["value"] = solve["inputs"][0]
    elif label == "refusal_bypass":
        next(node for node in ir["nodes"] if node["op"] == "solve_outcome")["attrs"]["action"]["kind"] = "continue"
    elif label == "duplicate_node_id":
        ir["nodes"].append(copy.deepcopy(ir["nodes"][0]))
    else:
        ir["nodes"].append(dict(id=1000, op="integral_candidate", inputs=[]))
    return ir


@pytest.mark.parametrize("label", ATTACKS)
def test_redigested_compiled_counterequations_and_routes_refused(actual_source_ir, label):
    forged = counter_ir(actual_source_ir, label)
    assert ir_hash(forged) != ir_hash(actual_source_ir)
    # Rehash intentionally: refusal must be semantic, never merely a stale hash.
    with pytest.raises(ValueError, match="M18 original equation"):
        subject.compiled_contract(forged, ir_hash(forged))


@pytest.mark.parametrize("target,label", [
    ((1., 0., .5), "interior"), ((1., 0., 1.1), "outside"),
    ((0., 0., 0.), "boundary"), ((-1., 0., 0.), "outside"),
    ((0., 1., 0.), "outside"), ((1., 0., 0.), "boundary"),
    ((1., .25, .125), "boundary"), ((1., .25, .124), "outside"),
    ((1., -.75, .625), "boundary"), ((1., -.75, .624), "outside"),
    ((1., 1., 1.), "boundary"), ((1., -1., 1.), "boundary"),
    ((1., 1.01, 1.), "outside"), ((1., 0., 1.), "boundary"),
])
def test_exact_homogeneous_cone_facets_and_positive_scale(target, label):
    assert guard.exact_cone(target) == label
    for scale in (2.**-20, 2.**20):
        assert guard.exact_cone(tuple(scale * x for x in target)) == label
    assert subject.cone(target) == label


@pytest.mark.parametrize("velocity", guard.VELOCITIES)
def test_every_discrete_extreme_ray_is_boundary(velocity):
    target = tuple(float(x) for x in (Fraction(1), velocity, velocity * velocity))
    assert guard.exact_cone(target) == "boundary"


def determinant(matrix):
    a, b, c = matrix
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def test_twenty_mathematical_vectors_have_spd_hessian_and_global_entropy_certificate():
    # Pure finite-dimensional witnesses; these are not NPZ/native states.
    directions = ((1, -3, 3, -1, 0), (0, 1, -3, 3, -1))
    for direction in directions:
        assert all(sum(v**k * d for v, d in zip(guard.VELOCITIES, direction, strict=True)) == 0 for k in range(3))
    for index in range(20):
        multiplier = (-.18 + .018 * index, .14 * math.sin(.23 * index), -.12 + .01 * index)
        p = guard.populations(multiplier)
        target = guard.moments(p)
        assert guard.exact_cone(target) == "interior"
        # Exact rational Gram determinants of the computed positive binary64 p.
        h = [[sum(Fraction(value) * v**(i + j) for v, value in zip(guard.VELOCITIES, p, strict=True))
              for j in range(3)] for i in range(3)]
        assert h[0][0] > 0 and h[0][0] * h[1][1] - h[0][1]**2 > 0 and determinant(h) > 0
        for direction in directions:
            delta = [d / 1024 for d in direction]
            q = [a + b for a, b in zip(p, delta, strict=True)]
            moment_error = max(abs(a - b) for a, b in zip(guard.moments(q), target, strict=True))
            assert min(q) > 0 and moment_error < 5.e-16
            # H(q)-H(p) = gradient dot delta + positive relative entropy.
            gradient_term = math.fsum(math.log(a / float(w)) * d for a, w, d in zip(p, guard.WEIGHTS, delta, strict=True))
            divergence = math.fsum(b * math.log(b / a) - b + a for a, b in zip(p, q, strict=True))
            curvature_bound = math.fsum(d*d / (2 * max(a, b)) for a, b, d in zip(p, q, delta, strict=True))
            gap = guard.entropy(q) - guard.entropy(p)
            assert abs(gradient_term) < 5.e-18 and divergence >= curvature_bound - 2.e-16
            assert gap > 1.e-7 and abs(gap - gradient_term - divergence) < 5.e-16


def test_fixture_phase_structure_keeps_same_outside_runtime_and_fresh_safe_bind():
    tree = ast.parse((ROOT / "tests/python/integration/runtime/test_m18_discrete_entropy_runtime.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "test_twenty_interior_targets_and_outside_cone_refusal")
    statements = function.body
    binds = {node.targets[0].id: node.value for node in statements if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "bound"}
    assert ast.unparse(binds["failed"]) == "bound(outside)" and ast.unparse(binds["safe"]) == "bound(targets)"
    loop = next(node for node in statements if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(2)")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bound" for node in ast.walk(loop))
    runs = [node for node in ast.walk(loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"]
    assert len(runs) == 1 and ast.unparse(runs[0].args[0]) == "failed"
    labels = [node.args[4].value for node in ast.walk(function) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "save_entropy_snapshot"
              and isinstance(node.args[4], ast.Constant)]
    assert set(labels) == set(subject.PHASES) - {"outside_rejected_0", "outside_rejected_1"}
    assert len(labels) == 8
