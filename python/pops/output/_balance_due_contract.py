"""Typed compile-time bridge from one resolved ConsumerGraph to Balance producers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity, make_identity
from pops.time._schedule.api import Always, Every, Schedule, When
from pops.time._schedule.domains import AcceptedStep


_MAX_NATIVE_ACCEPTED_STEP = (1 << 31) - 1


def _identity(value: Any, domain: str, *, where: str) -> Identity:
    if type(value) is not Identity or value.domain != domain or value.schema_version != 1:
        raise TypeError("%s must be an exact version-1 %s Identity" % (where, domain))
    return Identity.from_data(value.to_data())


@dataclass(frozen=True, slots=True)
class BalanceDueConsumer:
    """One exact ConsumerGraph node whose schedule requests a balance route."""

    consumer: Identity
    schedule: Schedule

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "consumer",
            _identity(
                self.consumer,
                "consumer-manifest",
                where="BalanceDueConsumer.consumer",
            ),
        )
        if type(self.schedule) is not Schedule:
            raise TypeError("BalanceDueConsumer.schedule must be an exact Schedule")

    def to_data(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer.to_data(),
            "schedule": self.schedule.to_data(),
        }


@dataclass(frozen=True, slots=True)
class BalanceDueRoute:
    """All consumer schedules that request one exact native balance route."""

    route: Identity
    consumers: tuple[BalanceDueConsumer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route",
            _identity(
                self.route,
                "balance-ledger-route",
                where="BalanceDueRoute.route",
            ),
        )
        if not isinstance(self.consumers, tuple) or any(
            type(value) is not BalanceDueConsumer for value in self.consumers
        ):
            raise TypeError(
                "BalanceDueRoute.consumers must contain exact BalanceDueConsumer values"
            )
        consumers = tuple(
            sorted(self.consumers, key=lambda value: value.consumer.token)
        )
        identities = [value.consumer.token for value in consumers]
        if len(identities) != len(set(identities)):
            raise ValueError("BalanceDueRoute contains a duplicate consumer")
        object.__setattr__(self, "consumers", consumers)

    def to_data(self) -> dict[str, Any]:
        return {
            "route": self.route.to_data(),
            "consumers": [value.to_data() for value in self.consumers],
        }

    def accepted_step_periods(self) -> tuple[int, ...]:
        """Return exact native periods, conservatively using period one when unprovable.

        ``Every(n)`` on the accepted-step domain is the first optimized cutover. ``Always`` and a
        statically true ``When`` are exactly period one; a statically false ``When`` contributes no
        occurrence. Any other domain/trigger remains active every step so this optimization can
        never suppress evidence required by a ConsumerGraph extension or physical-time cadence.
        """
        periods = []
        for row in self.consumers:
            schedule = row.schedule
            if type(schedule.domain) is not AcceptedStep:
                return (1,)
            trigger = schedule.trigger
            if type(trigger) is Every:
                # The native facade's public macro-step is a signed 32-bit ``int`` and rejects
                # overflow before increment. A larger positive period can therefore never fire in
                # any representable run; omit it instead of emitting an implementation-defined C++
                # narrowing conversion.
                if trigger.n <= _MAX_NATIVE_ACCEPTED_STEP:
                    periods.append(trigger.n)
            elif type(trigger) is Always:
                periods.append(1)
            elif type(trigger) is When and type(trigger.condition) is bool:
                if trigger.condition:
                    periods.append(1)
            else:
                return (1,)
        if 1 in periods:
            return (1,)
        return tuple(sorted(set(periods)))


@dataclass(frozen=True, slots=True)
class BalanceDueContract:
    """Immutable ConsumerGraph-derived cadence authority consumed by native codegen."""

    consumer_graph: Identity | None
    routes: tuple[BalanceDueRoute, ...]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if self.consumer_graph is not None:
            object.__setattr__(
                self,
                "consumer_graph",
                _identity(
                    self.consumer_graph,
                    "consumer-graph",
                    where="BalanceDueContract.consumer_graph",
                ),
            )
        if not isinstance(self.routes, tuple) or any(
            type(value) is not BalanceDueRoute for value in self.routes
        ):
            raise TypeError(
                "BalanceDueContract.routes must contain exact BalanceDueRoute values"
            )
        routes = tuple(sorted(self.routes, key=lambda value: value.route.token))
        tokens = [value.route.token for value in routes]
        if len(tokens) != len(set(tokens)):
            raise ValueError("BalanceDueContract contains a duplicate route")
        object.__setattr__(self, "routes", routes)
        object.__setattr__(
            self,
            "identity",
            make_identity("balance-due-contract", self._payload()),
        )

    @classmethod
    def from_consumer_graph(cls, graph: Any) -> BalanceDueContract:
        from pops.output._consumer_contracts import ConsumerGraph

        if graph is None:
            return cls(None, ())
        if type(graph) is not ConsumerGraph or not graph.is_resolved:
            raise TypeError(
                "BalanceDueContract requires an exact resolved ConsumerGraph or None"
            )
        by_route: dict[str, tuple[Identity, list[BalanceDueConsumer]]] = {}
        for manifest in graph.nodes:
            for quantity in manifest.diagnostic_quantities:
                for operation in quantity.execution["operations"]:
                    if operation["reduction"] != "accepted_balance":
                        continue
                    route = Identity.from_token(operation["balance_route"])
                    _identity(
                        route,
                        "balance-ledger-route",
                        where="accepted balance operation route",
                    )
                    existing = by_route.setdefault(route.token, (route, []))
                    existing[1].append(
                        BalanceDueConsumer(manifest.identity, manifest.schedule)
                    )
        return cls(
            graph.identity,
            tuple(
                BalanceDueRoute(route, tuple(consumers))
                for route, consumers in by_route.values()
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "consumer_graph": (
                None if self.consumer_graph is None else self.consumer_graph.to_data()
            ),
            "routes": [value.to_data() for value in self.routes],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}

    def route(self, route: str) -> BalanceDueRoute | None:
        if not isinstance(route, str) or not route:
            raise TypeError("balance due route lookup requires non-empty text")
        return next((value for value in self.routes if value.route.token == route), None)


__all__ = [
    "BalanceDueConsumer",
    "BalanceDueContract",
    "BalanceDueRoute",
]
