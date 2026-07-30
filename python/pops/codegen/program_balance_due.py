"""Compile-time fusion of accepted Balance consumers into Program scalar producers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any

from pops._balance_contract import BALANCE_TERM_NAMES
from pops.identity import Identity
from pops._balance_due_contract import BalanceDueContract
from pops.time.values import ProgramValue


@dataclass(frozen=True, slots=True)
class BalanceDueLowering:
    """Immutable lowering facts for one Program and exact ConsumerGraph contract."""

    contract: Identity
    route_periods: Mapping[str, tuple[int, ...]]
    guarded_values: Mapping[int, tuple[str, ...]]
    record_routes: Mapping[int, str]

    def __post_init__(self) -> None:
        if (
            type(self.contract) is not Identity
            or self.contract.domain != "balance-due-contract"
            or self.contract.schema_version != 1
        ):
            raise TypeError(
                "BalanceDueLowering.contract must be a version-1 balance-due-contract Identity"
            )
        object.__setattr__(
            self,
            "route_periods",
            MappingProxyType(dict(self.route_periods)),
        )
        object.__setattr__(
            self,
            "guarded_values",
            MappingProxyType(dict(self.guarded_values)),
        )
        object.__setattr__(
            self,
            "record_routes",
            MappingProxyType(dict(self.record_routes)),
        )


def _attribute_sources(value: ProgramValue) -> tuple[ProgramValue, ...]:
    sources = []
    for key in (
        "true_result",
        "false_result",
        "body",
        "residual",
        "apply_result",
    ):
        candidate = value.attrs.get(key)
        if isinstance(candidate, ProgramValue):
            sources.append(candidate)
    return tuple(sources)


def _program_balance_records(
    program: Any,
) -> tuple[
    tuple[ProgramValue, ...],
    dict[int, str],
    dict[str, dict[str, ProgramValue]],
]:
    from pops.codegen.program_lowerability import all_ops

    operations = tuple(all_ops(program))
    ids = [value.id for value in operations]
    if len(ids) != len(set(ids)):
        raise ValueError("Program balance due lowering requires globally unique SSA ids")
    record_routes: dict[int, str] = {}
    terms: dict[str, dict[str, ProgramValue]] = {}
    for value in operations:
        if value.op != "record_balance_term":
            continue
        route = Identity.from_token(value.attrs.get("route"))
        if (
            route.domain != "balance-ledger-route"
            or route.schema_version != 1
            or value.attrs.get("term") not in BALANCE_TERM_NAMES
        ):
            raise ValueError(
                "record_balance_term requires one canonical route and five-term name"
            )
        term = value.attrs["term"]
        by_term = terms.setdefault(route.token, {})
        if term in by_term:
            raise ValueError(
                "Program records balance route %s term %s more than once"
                % (route.token, term)
            )
        by_term[term] = value
        record_routes[value.id] = route.token
    expected = set(BALANCE_TERM_NAMES)
    for route, by_term in terms.items():
        if set(by_term) != expected:
            missing = sorted(expected.difference(by_term))
            extra = sorted(set(by_term).difference(expected))
            raise ValueError(
                "Program balance route %s must record exactly five terms; missing=%s extra=%s"
                % (route, missing, extra)
            )
    return operations, record_routes, terms


def validate_balance_due_contract(program: Any, contract: Any) -> None:
    """Fail before codegen when a Balance consumer has no matching five-term producer."""
    if type(contract) is not BalanceDueContract:
        raise TypeError(
            "balance due validation requires an exact BalanceDueContract"
        )
    _operations, _records, terms = _program_balance_records(program)
    missing = sorted(
        row.route.token for row in contract.routes if row.route.token not in terms
    )
    if missing:
        raise ValueError(
            "ConsumerGraph Balance routes have no Program.record_balance producer: %s"
            % ", ".join(missing)
        )


def prepare_balance_due_lowering(
    program: Any,
    contract: Any,
) -> BalanceDueLowering:
    """Return exclusive balance-producer guards without mutating the Program graph."""
    if type(contract) is not BalanceDueContract:
        raise TypeError(
            "balance due lowering requires an exact BalanceDueContract"
        )
    operations, record_routes, terms = _program_balance_records(program)
    route_periods = {
        route: (
            () if (row := contract.route(route)) is None
            else row.accepted_step_periods()
        )
        for route in terms
    }
    by_id = {value.id: value for value in operations}
    required_routes: dict[int, set[str]] = {}

    def require(value: ProgramValue, route: str) -> None:
        if value.op not in {"reduce", "scalar_op"}:
            raise ValueError(
                "record_balance producer %r is not an additive reduction/scalar chain"
                % value.name
            )
        routes = required_routes.setdefault(value.id, set())
        if route in routes:
            return
        routes.add(route)
        if value.op == "scalar_op":
            for source in value.inputs:
                require(source, route)

    for record_id, route in record_routes.items():
        record = by_id[record_id]
        if len(record.inputs) != 1:
            raise ValueError("record_balance_term must consume one exact scalar")
        require(record.inputs[0], route)

    balance_nodes = set(required_routes).union(record_routes)
    consumers: dict[int, set[int]] = {value_id: set() for value_id in by_id}
    for consumer in operations:
        for source in (*consumer.inputs, *_attribute_sources(consumer)):
            consumers.setdefault(source.id, set()).add(consumer.id)

    # A scalar chain shared with a non-balance use remains unconditional. Propagate that liveness
    # backwards so an upstream reduction cannot be skipped while a downstream ordinary diagnostic
    # still reads it.
    always_required = {
        value_id
        for value_id in balance_nodes
        if any(consumer not in balance_nodes for consumer in consumers.get(value_id, ()))
    }
    pending = list(always_required)
    while pending:
        value = by_id[pending.pop()]
        for source in value.inputs:
            if source.id in balance_nodes and source.id not in always_required:
                always_required.add(source.id)
                pending.append(source.id)

    guarded = {
        value_id: tuple(sorted(routes))
        for value_id, routes in required_routes.items()
        if value_id not in always_required
    }
    return BalanceDueLowering(
        contract.identity,
        route_periods,
        guarded,
        record_routes,
    )


def emit_balance_due_guards(
    lowering: BalanceDueLowering,
    var: dict[Any, Any],
    lines: list[str],
) -> None:
    """Emit one host-side due decision per recorded route before any balance collective."""
    if type(lowering) is not BalanceDueLowering:
        raise TypeError("balance due guard emission requires BalanceDueLowering")
    contract = json.dumps(lowering.contract.token)
    automatic_tokens = []
    for index, (route, periods) in enumerate(sorted(lowering.route_periods.items())):
        if not periods:
            token = "false"
        else:
            calls = [
                "ctx.balance_consumer_is_due(%s, %s, %d)"
                % (contract, json.dumps(route), period)
                for period in periods
            ]
            token = "balance_due_%d" % index
            lines.append("const bool %s = (%s);" % (token, " || ".join(calls)))
            automatic_tokens.append(token)
        var[("balance_due_route", route)] = token
    if automatic_tokens:
        lines.append(
            "ctx.note_automatic_balance_capture_due(%s);"
            % (" || ".join(automatic_tokens))
        )
    var[("balance_guarded_values",)] = lowering.guarded_values
    var[("balance_record_routes",)] = lowering.record_routes


def balance_value_due_expression(var: Mapping[Any, Any], value_id: int) -> str | None:
    routes = var.get(("balance_guarded_values",), {}).get(value_id)
    if routes is None:
        return None
    tokens = tuple(var[("balance_due_route", route)] for route in routes)
    if "true" in tokens:
        return "true"
    tokens = tuple(token for token in tokens if token != "false")
    return "false" if not tokens else "(" + " || ".join(tokens) + ")"


def balance_record_due_expression(var: Mapping[Any, Any], value_id: int) -> str:
    route = var.get(("balance_record_routes",), {}).get(value_id)
    if not isinstance(route, str) or not route:
        raise ValueError("record_balance_term lost its compile-time due route")
    token = var.get(("balance_due_route", route))
    if not isinstance(token, str) or not token:
        raise ValueError("record_balance_term route has no compile-time due decision")
    return token


__all__ = [
    "BalanceDueLowering",
    "balance_record_due_expression",
    "balance_value_due_expression",
    "emit_balance_due_guards",
    "prepare_balance_due_lowering",
    "validate_balance_due_contract",
]
