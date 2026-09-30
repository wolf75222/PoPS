"""Exact mathematical M18 DAG guard; stdlib only, no PoPS or author oracle.

Expand the serialized scalar arithmetic over rational coefficients and opaque
exponential atoms. This verifies equations, rather than sampling residuals at a
solution or trusting a freshly redigested IR. It never creates native receipts.
"""
from fractions import Fraction
import math


VELOCITIES = tuple(map(Fraction, ("-1", "-1/2", "0", "1/2", "1")))
WEIGHTS = tuple(Fraction(value) for value in (.1, .2, .4, .2, .1))


def need(condition, message):
    if not condition:
        raise ValueError("M18 original equation: " + message)


def add(left, right, sign=1):
    result = dict(left)
    for term, coefficient in right.items():
        result[term] = result.get(term, Fraction(0)) + sign * coefficient
        if not result[term]:
            del result[term]
    return result


def multiply(left, right):
    result = {}
    for lterm, lvalue in left.items():
        for rterm, rvalue in right.items():
            term = tuple(sorted(lterm + rterm, key=repr))
            result = add(result, {term: lvalue * rvalue})
    return result


def literal(data):
    need(type(data) is dict and set(data) == {"kind", "value"}, "literal shape")
    need(type(data["value"]) is str, "literal spelling")
    if data["kind"] == "integer":
        value = int(data["value"])
        need(str(value) == data["value"], "noncanonical integer")
        return Fraction(value)
    need(data["kind"] == "binary64", "unexpected literal kind")
    value = float.fromhex(data["value"])
    need(math.isfinite(value) and value.hex() == data["value"], "noncanonical/nonfinite real")
    return Fraction(value)


def expanded_residuals(attrs):
    rows = attrs["expression_nodes"]
    need(type(rows) is list, "missing scalar DAG")
    values = []
    def previous(index):
        need(type(index) is int and 0 <= index < len(values), "non-topological DAG reference")
        return values[index]
    for row in rows:
        need(type(row) is list and row, "scalar DAG row shape")
        op = row[0]
        if op == "literal":
            need(len(row) == 2, "literal arity")
            value = literal(row[1])
            result = {(): value} if value else {}
        elif op == "input":
            need(len(row) == 3 and all(type(x) is int for x in row[1:])
                 and row[1] in (0, 1) and 0 <= row[2] < 3, "foreign product input")
            result = {(("input", row[1], row[2]),): Fraction(1)}
        elif op in ("add", "sub", "mul"):
            need(len(row) == 3, "binary arithmetic arity")
            left, right = previous(row[1]), previous(row[2])
            result = multiply(left, right) if op == "mul" else add(left, right, -1 if op == "sub" else 1)
        elif op == "exp":
            need(len(row) == 2, "exponential arity")
            affine = previous(row[1])
            need(all(term == () or (len(term) == 1 and term[0][0] == "input" and term[0][1] == 0)
                     for term in affine), "nonlinear/captured exponential exponent")
            coefficients = tuple(affine.get((("input", 0, i),), Fraction(0)) for i in range(3))
            result = {(("exp", affine.get((), Fraction(0)), coefficients),): Fraction(1)}
        else:
            raise ValueError("M18 original equation: unexpected scalar op " + str(op))
        values.append(result)
    roots = attrs["expressions"]
    need(type(roots) is list and len(roots) == 3, "residual product shape")
    return [previous(index) for index in roots]


def expected_residuals():
    result = []
    for moment in range(3):
        equation = {(("input", 1, moment),): Fraction(-1)}
        for velocity, weight in zip(VELOCITIES, WEIGHTS, strict=True):
            atom = ("exp", Fraction(0), (Fraction(1), velocity, velocity * velocity))
            equation = add(equation, {(atom,): weight * velocity ** moment})
        result.append(equation)
    return result


def verify(ir):
    nodes = ir["nodes"]
    need(type(nodes) is list, "node inventory")
    indexed = {node["id"]: node for node in nodes}
    need(len(indexed) == len(nodes) and all(type(key) is int for key in indexed), "duplicate/invalid node id")
    need(sorted(node["op"] for node in nodes) == sorted(("state", "state", "solve_coupled_implicit",
         "solve_outcome", "solve_outcome_component")), "hidden operations in the witness")
    solves = [node for node in nodes if node["op"] == "solve_coupled_implicit"]
    need(len(solves) == 1, "solve count")
    solve = solves[0]
    need(expanded_residuals(solve["attrs"]) == expected_residuals(), "five-velocity moment residual differs")
    need(type(solve["inputs"]) is list and len(solve["inputs"]) == 2, "seed/capture routes")
    expected_routes = (("dual", "multipliers", 1), ("target", "moments", 0))
    for input_id, (block, state, step) in zip(solve["inputs"], expected_routes, strict=True):
        need(type(input_id) is int and input_id in indexed, "foreign input id")
        node = indexed[input_id]
        need(node["op"] == "state" and node["inputs"] == [] and node["block"]["local_id"] == block
             and node["state"]["local_id"] == state, "seed/capture state route")
        point = node["point"]
        need(type(point["step"]) is int and point["step"] == step
             and point["offset"] == {"kind": "integer", "value": "0"}, "capture/seed timepoint")
        need(point["clock"] == solve["point"]["clock"], "capture/seed clock owner")
        need(node["attrs"]["state"]["handle"] == node["state"], "state handle aliases")
        need(node["state"]["block_ref"] == node["block"], "state block owner")
    need(type(solve["point"]["step"]) is int and solve["point"]["step"] == 1, "solve timepoint")
    outcomes = [node for node in nodes if node["op"] == "solve_outcome"]
    need(len(outcomes) == 1 and outcomes[0]["inputs"] == [solve["id"]]
         and outcomes[0]["attrs"]["action"]["kind"] == "fail_run", "refusal consumer")
    commits = ir["commits"]
    need(len(commits) == 1 and commits[0]["value"] in indexed, "publication route")
    output = indexed[commits[0]["value"]]
    need(output["op"] == "solve_outcome_component" and output["inputs"] == [outcomes[0]["id"]]
         and type(output["attrs"]["index"]) is int and output["attrs"]["index"] == 0
         and output["state"] == commits[0]["state"] and output["block"] == commits[0]["block"],
         "publication bypass/foreign output")


def exact_cone(target):
    """Homogeneous polygon facets, exact over the provided finite binary64."""
    need(len(target) == 3 and all(math.isfinite(value) for value in target), "invalid moment target")
    mass, first, second = map(Fraction, target)
    if mass < 0 or (mass == 0 and (first != 0 or second != 0)):
        return "outside"
    if mass == 0:
        return "boundary"
    facets = [mass - first, mass + first, mass - second]
    for left, right in zip(VELOCITIES[:-1], VELOCITIES[1:], strict=True):
        facets.append(second - (left + right) * first + left * right * mass)
    if min(facets) < 0:
        return "outside"
    return "boundary" if min(facets) == 0 else "interior"


def populations(multiplier):
    need(len(multiplier) == 3 and all(math.isfinite(value) for value in multiplier), "invalid multiplier")
    result = [float(weight) * math.exp(multiplier[0] + float(v) * multiplier[1] + float(v * v) * multiplier[2])
              for v, weight in zip(VELOCITIES, WEIGHTS, strict=True)]
    need(all(math.isfinite(value) and value > 0 for value in result), "nonfinite/nonpositive population")
    return result


def moments(population):
    return [math.fsum(value * float(velocity ** k) for velocity, value in zip(VELOCITIES, population, strict=True))
            for k in range(3)]


def entropy(population):
    need(len(population) == 5 and all(math.isfinite(value) and value > 0 for value in population), "entropy domain")
    result = math.fsum(p * (math.log(p / float(w)) - 1) for p, w in zip(population, WEIGHTS, strict=True))
    need(math.isfinite(result), "entropy overflow")
    return result
