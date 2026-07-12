"""Proof-carrying, exact properties derived from temporal method tableaux."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pops.time.method_tableau import exact_fraction


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


def certify_runge_kutta(tableau: Any) -> MethodCertificate:
    A = _dense(tableau)
    b = tuple(exact_fraction(x, "tableau.b") for x in tableau.b)
    c = tuple(exact_fraction(x, "tableau.c") for x in tableau.c)
    return MethodCertificate(A, b, c, analyze_runge_kutta(tableau))


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
        result[int(_literal(power))] = _literal(coefficient)
    return result


def _point_offset(point: Any) -> Fraction:
    if hasattr(point, "time"):
        point = point.time
    return _literal(point.offset.to_data())


def _unknown_graph(graph: Any, reason: str, abscissae: Any = ()) -> ProgramMethodCertificate:
    properties = MethodProperties(
        UnknownOrder(reason), tuple(abscissae), (), (), None)
    return ProgramMethodCertificate(graph.graph_hash, None, properties)


def certify_program_graph(graph: Any) -> ProgramMethodCertificate:
    """Reconstruct an explicit RK certificate from normalized graph semantics.

    Debug labels and preset provenance are never inspected.  Graphs outside the affine, single-state
    explicit-RK language remain valid executable Programs and receive :class:`UnknownOrder`.
    """
    from pops.time.graph import ProgramGraph

    if type(graph) is not ProgramGraph:
        raise TypeError("certify_program_graph requires an exact normalized ProgramGraph")
    nodes = {node.node_id: node for node in graph.nodes}
    states = [node for node in graph.nodes if node.kind == "state_read"]
    rhs = [node for node in graph.nodes
           if node.kind == "program_value" and node.op == "rhs"]
    commits = [node for node in graph.nodes if node.kind == "commit"]
    abscissae = tuple(_point_offset(node.point) for node in rhs)
    if len(states) != 1 or not rhs or len(commits) != 1:
        return _unknown_graph(graph, "graph is not a single-state explicit RK step", abscissae)
    state_id = states[0].node_id
    rhs_index = {node.node_id: i for i, node in enumerate(rhs)}

    def affine(node_id: int, *, stage: int) -> tuple[Fraction, tuple[Fraction, ...]]:
        if node_id == state_id:
            return Fraction(1), tuple(Fraction() for _ in range(stage))
        node = nodes[node_id]
        if node.kind != "program_value" or node.op != "linear_combine":
            raise ValueError("RK stage is not an affine state expression")
        data = node.attrs.to_data()["attrs"]["coeffs"]
        refs = node.references()
        base = Fraction()
        row = [Fraction() for _ in range(stage)]
        for ref, encoded in zip(refs, data, strict=True):
            polynomial = _polynomial(encoded)
            if ref.node_id == state_id:
                if set(polynomial) - {0}:
                    raise ValueError("base state has a non-constant coefficient")
                base += polynomial.get(0, Fraction())
                continue
            index = rhs_index.get(ref.node_id)
            if index is None or index >= stage or set(polynomial) - {1}:
                raise ValueError("stage reads a non-previous rate or non-dt coefficient")
            row[index] += polynomial.get(1, Fraction())
        return base, tuple(row)

    try:
        A = []
        for i, rate in enumerate(rhs):
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

    # Re-enter the common exact proof engine only after graph reconstruction is complete.
    from pops.time.method_tableau import RungeKuttaTableau
    tableau = RungeKuttaTableau(A=A, b=b, c=abscissae)
    certificate = certify_runge_kutta(tableau)
    return ProgramMethodCertificate(graph.graph_hash, certificate, certificate.properties)


__all__ = [
    "AdditiveMethodCertificate", "AdditiveMethodProperties", "MethodCertificate",
    "MethodProperties", "SSPCertificate", "UnknownOrder", "analyze_runge_kutta",
    "ProgramMethodCertificate", "certify_additive_runge_kutta", "certify_program_graph",
    "certify_runge_kutta",
]
