"""Proof-carrying, exact properties derived from temporal method tableaux."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pops.time._methods.coefficients import exact_fraction


@dataclass(frozen=True, slots=True)
class UnknownOrder:
    """Explicit result when the implemented exact order conditions cannot certify an order."""

    reason: str


@dataclass(frozen=True, slots=True)
class SSPCertificate:
    coefficient: Fraction
    source: str


@dataclass(frozen=True, slots=True)
class MethodProperties:
    order: int | UnknownOrder
    abscissae: tuple[Fraction, ...]
    stability_polynomial: tuple[Fraction, ...]
    flux_weights: tuple[Fraction, ...]
    ssp: SSPCertificate | None = None


@dataclass(frozen=True, slots=True)
class MethodCertificate:
    """Immutable evidence bundle; labels are deliberately absent from semantic identity."""

    A: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]
    c: tuple[Fraction, ...]
    properties: MethodProperties


@dataclass(frozen=True, slots=True)
class AdditiveMethodProperties:
    order: int | UnknownOrder
    explicit_abscissae: tuple[Fraction, ...]
    implicit_abscissae: tuple[Fraction, ...]
    flux_weights: tuple[tuple[str, tuple[Fraction, ...]], ...]


@dataclass(frozen=True, slots=True)
class AdditiveMethodCertificate:
    explicit: MethodCertificate
    implicit_A: tuple[tuple[Fraction, ...], ...]
    implicit_b: tuple[Fraction, ...]
    implicit_c: tuple[Fraction, ...]
    properties: AdditiveMethodProperties


@dataclass(frozen=True, slots=True)
class ProgramMethodCertificate:
    """Method evidence reconstructed exclusively from one normalized ProgramGraph."""

    graph_hash: str
    tableau: MethodCertificate | None
    properties: MethodProperties


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction())


def _matvec(A: tuple[tuple[Fraction, ...], ...], x: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(sum((row[j] * x[j] for j in range(len(row))), Fraction()) for row in A)


def _dense(tableau: Any) -> tuple[tuple[Fraction, ...], ...]:
    stages = tableau.stages
    return tuple(tuple(
        exact_fraction(tableau.A[i][j], "tableau.A") if j < len(tableau.A[i]) else Fraction()
        for j in range(stages)) for i in range(stages))


def _proved_order(A: tuple[tuple[Fraction, ...], ...], b: tuple[Fraction, ...],
                  c: tuple[Fraction, ...]) -> int | UnknownOrder:
    if sum(b, Fraction()) != 1:
        return UnknownOrder("first-order consistency condition failed")
    order = 1
    if _dot(b, c) != Fraction(1, 2):
        return order
    order = 2
    Ac = _matvec(A, c)
    c2 = tuple(x * x for x in c)
    if _dot(b, c2) != Fraction(1, 3) or _dot(b, Ac) != Fraction(1, 6):
        return order
    order = 3
    c3 = tuple(x * x * x for x in c)
    Ac2 = _matvec(A, c2)
    AAc = _matvec(A, Ac)
    if (_dot(b, c3) != Fraction(1, 4)
            or _dot(b, tuple(c[i] * Ac[i] for i in range(len(c)))) != Fraction(1, 8)
            or _dot(b, Ac2) != Fraction(1, 12)
            or _dot(b, AAc) != Fraction(1, 24)):
        return order
    return 4


def _stability(A: tuple[tuple[Fraction, ...], ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Coefficients of R(z)=1+sum_k z^k b^T A^(k-1) 1 for an explicit tableau."""
    vector = tuple(Fraction(1) for _ in b)
    result = [Fraction(1)]
    for _ in range(len(b)):
        result.append(_dot(b, vector))
        vector = _matvec(A, vector)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


_SSPRK2_KEY = (
    ((0, 0), (1, 0)), (Fraction(1, 2), Fraction(1, 2)), (0, 1),
)
_SSPRK3_KEY = (
    ((0, 0, 0), (1, 0, 0), (Fraction(1, 4), Fraction(1, 4), 0)),
    (Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)), (0, 1, Fraction(1, 2)),
)
_KNOWN_SSP = {
    (((0,),), (1,), (0,)): SSPCertificate(Fraction(1), "one forward Euler step"),
    _SSPRK2_KEY: SSPCertificate(Fraction(1), "exact Shu-Osher convex decomposition"),
    _SSPRK3_KEY: SSPCertificate(Fraction(1), "exact Shu-Osher convex decomposition"),
}


def analyze_runge_kutta(tableau: Any) -> MethodProperties:
    A = _dense(tableau)
    b = tuple(exact_fraction(x, "tableau.b") for x in tableau.b)
    c = tuple(exact_fraction(x, "tableau.c") for x in tableau.c)
    return MethodProperties(
        order=_proved_order(A, b, c),
        abscissae=c,
        stability_polynomial=_stability(A, b),
        flux_weights=b,
        ssp=_KNOWN_SSP.get((A, b, c)),
    )


def _certificate_from_coefficients(
    A: tuple[tuple[Fraction, ...], ...],
    b: tuple[Fraction, ...],
    c: tuple[Fraction, ...],
) -> MethodCertificate:
    """Build one certificate from exact arrays without re-entering the tableau layer."""
    properties = MethodProperties(
        order=_proved_order(A, b, c),
        abscissae=c,
        stability_polynomial=_stability(A, b),
        flux_weights=b,
        ssp=_KNOWN_SSP.get((A, b, c)),
    )
    return MethodCertificate(A, b, c, properties)


def certify_runge_kutta(tableau: Any) -> MethodCertificate:
    A = _dense(tableau)
    b = tuple(exact_fraction(x, "tableau.b") for x in tableau.b)
    c = tuple(exact_fraction(x, "tableau.c") for x in tableau.c)
    return _certificate_from_coefficients(A, b, c)


@dataclass(frozen=True, slots=True)
class ImplicitMethodProperties:
    order: int | UnknownOrder
    abscissae: tuple[Fraction, ...]
    flux_weights: tuple[Fraction, ...]
    # An implicit stability function is generally rational. Never publish the
    # truncated explicit polynomial as a stability certificate.
    stability_status: str = "unverified"


@dataclass(frozen=True, slots=True)
class ImplicitMethodCertificate:
    A: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]
    c: tuple[Fraction, ...]
    properties: ImplicitMethodProperties


def certify_implicit_runge_kutta(tableau: Any) -> ImplicitMethodCertificate:
    A = _dense(tableau)
    b = tuple(exact_fraction(x, "tableau.b") for x in tableau.b)
    c = tuple(exact_fraction(x, "tableau.c") for x in tableau.c)
    return ImplicitMethodCertificate(A, b, c, ImplicitMethodProperties(
        _proved_order(A, b, c), c, b))


def certify_additive_runge_kutta(tableau: Any) -> AdditiveMethodCertificate:
    explicit = certify_runge_kutta(tableau.explicit)
    stages = tableau.stages
    implicit_A = tuple(tuple(
        exact_fraction(tableau.implicit_A[i][j], "implicit_A")
        if j < len(tableau.implicit_A[i]) else Fraction()
        for j in range(stages)) for i in range(stages))
    implicit_b = tuple(exact_fraction(x, "implicit_b") for x in tableau.implicit_b)
    implicit_c = tuple(exact_fraction(x, "implicit_c") for x in tableau.implicit_c)
    explicit_c = explicit.properties.abscissae
    order: int | UnknownOrder = 1
    # All four coloured order-two trees must agree; otherwise only consistency is certified.
    weights = (explicit.properties.flux_weights, implicit_b)
    nodes = (explicit_c, implicit_c)
    if all(_dot(b, c) == Fraction(1, 2) for b in weights for c in nodes):
        order = 2
    properties = AdditiveMethodProperties(
        order, explicit_c, implicit_c,
        (("explicit", weights[0]), ("implicit", weights[1])),
    )
    return AdditiveMethodCertificate(
        explicit, implicit_A, implicit_b, implicit_c, properties)


def _literal(data: Any) -> Fraction:
    """Decode canonical ScalarLiteral data without depending on its authoring Python domain."""
    if isinstance(data, dict) and set(data) == {"scalar"}:
        data = data["scalar"]
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind == "integer":
        return Fraction(int(data["value"]))
    if kind == "rational":
        return Fraction(int(data["numerator"]), int(data["denominator"]))
    if kind == "decimal":
        return Fraction(data["value"])
    if kind == "binary64":
        return Fraction.from_float(float.fromhex(data["hex"]))
    raise ValueError("unsupported canonical scalar literal")


def _polynomial(data: Any) -> dict[int, Fraction]:
    result: dict[int, Fraction] = {}
    for power, coefficient in data:
        exact_power = _literal(power)
        if exact_power.denominator != 1 or exact_power < 0:
            raise ValueError("RK coefficient requires a non-negative integer dt power")
        degree = int(exact_power)
        if degree in result:
            raise ValueError("RK coefficient repeats a canonical dt power")
        result[degree] = _literal(coefficient)
    return result


def _point_offset(point: Any) -> Fraction:
    if hasattr(point, "time"):
        point = point.time
    return Fraction(point.step) + _literal(point.offset.to_data())


def _unknown_graph(graph: Any, reason: str, abscissae: Any = ()) -> ProgramMethodCertificate:
    properties = MethodProperties(
        UnknownOrder(reason), tuple(abscissae), (), (), None)
    return ProgramMethodCertificate(graph.graph_hash, None, properties)


def _is_explicit_rate_call(node: Any) -> bool:
    """Return whether *node* is one authenticated explicit rate evaluation.

    Normalized graphs retain primitive ``rhs`` nodes for the internal lowering route and expose
    public operator calls as :class:`OperatorCall` nodes.  The latter must be classified from their
    canonical typed handle and lowering metadata; names and debug labels are deliberately ignored.
    """
    if node.kind == "program_value":
        return node.op in {"rhs", "diffusive_rhs"}
    if node.kind != "operator_call":
        return False
    operator = node.operator.to_data()
    handle = operator.get("handle", {})
    lowering = operator.get("lowering", {})
    return (
        handle.get("kind") in {"grid_operator", "local_rate"}
        and lowering.get("op") in {"rhs", "diffusive_rhs"}
        and lowering.get("value_type") == "rhs"
    )


def certify_program_graph(graph: Any) -> ProgramMethodCertificate:
    """Reconstruct an explicit RK certificate from normalized graph semantics.

    Debug labels and preset provenance are never inspected.  Graphs outside the affine, single-state
    explicit-RK language remain valid executable Programs and receive :class:`UnknownOrder`.
    """
    from pops.time._graph.program import ProgramGraph

    if type(graph) is not ProgramGraph:
        raise TypeError("certify_program_graph requires an exact normalized ProgramGraph")
    nodes = {node.node_id: node for node in graph.nodes}
    states = [node for node in graph.nodes if node.kind == "state_read"]
    rhs = [node for node in graph.nodes if _is_explicit_rate_call(node)]
    commits = [node for node in graph.nodes if node.kind == "commit"]
    try:
        abscissae = tuple(_point_offset(node.point) for node in rhs)
    except (TypeError, ValueError) as exc:
        return _unknown_graph(graph, str(exc))
    if len(states) != 1 or not rhs or len(commits) != 1:
        return _unknown_graph(graph, "graph is not a single-state explicit RK step", abscissae)
    state_id = states[0].node_id
    rhs_index = {node.node_id: i for i, node in enumerate(rhs)}
    clock = states[0].clock
    if not graph.cadence.is_default or any(node.clock != clock for node in graph.nodes):
        return _unknown_graph(graph, "graph has a different clock or macro-step cadence", abscissae)
    identities = tuple(node.operator.to_data() if node.kind == "operator_call" else
                       {"op": node.op, "attrs": node.attrs.to_data()} for node in rhs)
    if any(identity != identities[0] for identity in identities[1:]):
        return _unknown_graph(graph, "RK stages call different rate operators or effects", abscissae)
    for node in rhs:
        payload = (node.operator.to_data()["lowering"].get("attrs", {})
                   if node.kind == "operator_call" else node.attrs.to_data())
        data = payload.get("attrs", payload)
        if data.get("schedule") is not None:
            return _unknown_graph(graph, "RK rate has a nontrivial scheduled evaluation effect", abscissae)

    # Affine SSA expressions may materialize named stage/rate values before a
    # later combination reads them. Trace their exact polynomial weights; never
    # treat an arbitrary primitive or its debug label as a transparent alias.
    cache: dict[tuple[int, int], tuple[dict[int, Fraction], ...]] = {}

    def add_product(target: dict[int, Fraction], factor: dict[int, Fraction],
                    expression: dict[int, Fraction]) -> None:
        for left, coefficient in factor.items():
            for right, value in expression.items():
                degree = left + right
                target[degree] = target.get(degree, Fraction()) + coefficient * value

    def expression(node_id: int, *, stage: int) -> tuple[dict[int, Fraction], ...]:
        key = (node_id, stage)
        if key in cache:
            return cache[key]
        result = tuple({} for _ in range(stage + 1))
        if node_id == state_id:
            result[0][0] = Fraction(1)
        elif node_id in rhs_index:
            index = rhs_index[node_id]
            if index >= stage:
                raise ValueError("stage reads a non-previous rate")
            result[index + 1][0] = Fraction(1)
        else:
            node = nodes[node_id]
            if node.kind != "program_value" or node.op != "linear_combine":
                raise ValueError("RK stage reads an opaque or non-affine primitive")
            if node.value_type not in {"state", "rhs"}:
                raise ValueError("RK affine value has an unknown state/rate representation")
            payload = node.attrs.to_data()
            data = payload.get("attrs", payload)
            if set(data) != {"coeffs"}:
                raise ValueError("RK affine value has an unknown evaluation effect")
            for ref, encoded in zip(node.references(), data["coeffs"], strict=True):
                polynomial = _polynomial(encoded)
                # Visit even zero-weight inputs: an unknown effect or unavailable
                # rate is not made safe by algebraic cancellation of its value.
                terms = expression(ref.node_id, stage=stage)
                for total, term in zip(result, terms, strict=True):
                    add_product(total, polynomial, term)
        cache[key] = result
        return result

    def affine(node_id: int, *, stage: int) -> tuple[Fraction, tuple[Fraction, ...]]:
        terms = tuple({power: value for power, value in term.items() if value}
                      for term in expression(node_id, stage=stage))
        if set(terms[0]) - {0}:
            raise ValueError("base state has a non-constant coefficient")
        if any(set(term) - {1} for term in terms[1:]):
            raise ValueError("rate has a non-dt coefficient")
        return terms[0].get(0, Fraction()), tuple(term.get(1, Fraction()) for term in terms[1:])

    try:
        # Effects are properties of the whole executed graph, not just the
        # endpoint's value dependencies. An unused state-mutating primitive can
        # still invalidate every stage expression traced below.
        for node in graph.nodes:
            if node.kind in {"state_read", "commit"} or _is_explicit_rate_call(node):
                continue
            if node.kind != "program_value" or node.op != "linear_combine":
                raise ValueError("graph contains an opaque or non-affine evaluation effect")
            if node.value_type not in {"state", "rhs"}:
                raise ValueError("graph contains an unknown affine representation")
            payload = node.attrs.to_data()
            data = payload.get("attrs", payload)
            if set(data) != {"coeffs"}:
                raise ValueError("graph contains an unknown affine evaluation effect")
            for _, encoded in zip(node.references(), data["coeffs"], strict=True):
                _polynomial(encoded)
        if _point_offset(states[0].point) != 0 or _point_offset(commits[0].point) != 1:
            raise ValueError("RK graph does not span its initial-to-next step window")
        A = []
        for i, rate in enumerate(rhs):
            if len(rate.references()) != 1:
                raise ValueError("RK rate has an unknown additional input effect")
            state_ref = rate.references()[0]
            base, row = affine(state_ref.node_id, stage=i)
            if base != 1:
                raise ValueError("RK stage does not preserve the base state")
            A.append(row)
        final_ref = commits[0].references()[0]
        base, b = affine(final_ref.node_id, stage=len(rhs))
        if base != 1 or sum(b, Fraction()) != 1:
            raise ValueError("RK endpoint is not a consistent affine rate combination")
        if tuple(sum(row, Fraction()) for row in A) != abscissae:
            raise ValueError("stage point abscissae differ from reconstructed row sums")
    except (KeyError, TypeError, ValueError) as exc:
        return _unknown_graph(graph, str(exc), abscissae)

    stages = len(b)
    dense_A = tuple(
        tuple(row[j] if j < len(row) else Fraction() for j in range(stages))
        for row in A
    )
    certificate = _certificate_from_coefficients(dense_A, tuple(b), tuple(abscissae))
    return ProgramMethodCertificate(graph.graph_hash, certificate, certificate.properties)


__all__ = [
    "AdditiveMethodCertificate", "AdditiveMethodProperties", "MethodCertificate",
    "MethodProperties", "SSPCertificate", "UnknownOrder", "analyze_runge_kutta",
    "ProgramMethodCertificate", "certify_additive_runge_kutta", "certify_program_graph",
    "certify_runge_kutta",
]
