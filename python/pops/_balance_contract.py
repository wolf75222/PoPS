"""Core typed identity shared by Program balance evidence and output consumers.

This module deliberately lives outside :mod:`pops.diagnostics`: Program/codegen imports must not
make every PoPS test transitively depend on the public diagnostics package initializer. The public
``pops.diagnostics.BalanceLedger`` name is an alias of this exact class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity, make_identity


BALANCE_TERM_NAMES = (
    "storage_change",
    "outward_boundary_flux",
    "sources",
    "reflux",
    "projection",
)


def _canonical_name(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


@dataclass(frozen=True, slots=True)
class BalanceLedger:
    """Identity joining one Program-authored discrete balance to one consumer.

    The ledger does not contain values. :meth:`Program.record_balance` writes the explicitly
    authored reduced scalars into the current native step-attempt mailbox. A ledger may delegate
    ``reflux`` and/or ``projection`` to exact native operators for one typed component role, while
    :class:`pops.diagnostics.Balance` selects the same identity after that attempt has advanced
    successfully.
    """

    name: str
    role: Any = None
    component: int | None = None
    automatic_terms: tuple[str, ...] = ()
    identity: Identity = field(init=False)
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        name = _canonical_name(self.name, where="BalanceLedger.name")
        role = None
        if self.role is not None:
            from pops.physics.roles import native_role_token

            try:
                role = native_role_token(self.role)
            except TypeError as exc:
                raise TypeError(
                    "BalanceLedger.role must be a typed pops.physics.roles.ComponentRole"
                ) from exc
        if not isinstance(self.automatic_terms, tuple):
            raise TypeError("BalanceLedger.automatic_terms must be a tuple")
        automatic_terms = tuple(sorted(set(self.automatic_terms)))
        if len(automatic_terms) != len(self.automatic_terms):
            raise ValueError("BalanceLedger.automatic_terms must be unique")
        unsupported = set(automatic_terms).difference({"reflux", "projection"})
        if unsupported:
            raise ValueError(
                "BalanceLedger automatic native producers currently support only "
                "reflux and projection; got %s" % sorted(unsupported)
            )
        component = self.component
        if automatic_terms and component is None:
            component = 0
        if component is not None and (type(component) is not int or component < 0):
            raise TypeError("BalanceLedger.component must be a non-negative int or None")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "automatic_terms", automatic_terms)
        payload: dict[str, Any] = {"schema_version": 1, "name": name}
        if role is not None:
            payload["role"] = role
        if component is not None:
            payload["component"] = component
        if automatic_terms:
            payload["automatic_terms"] = list(automatic_terms)
        object.__setattr__(
            self,
            "identity",
            make_identity("balance-ledger", payload),
        )

    def to_data(self) -> dict[str, Any]:
        data = {
            "schema_version": 1,
            "name": self.name,
            "identity": self.identity.to_data(),
        }
        if self.role is not None:
            from pops.physics.roles import native_role_token

            data["role"] = native_role_token(self.role)
        if self.component is not None:
            data["component"] = self.component
        if self.automatic_terms:
            data["automatic_terms"] = list(self.automatic_terms)
        return data

    def route_identity(self, block: Any) -> Identity:
        from pops.problem.handles import BlockHandle

        if not isinstance(block, BlockHandle):
            raise TypeError("balance ledger block must be a BlockHandle")
        return make_identity(
            "balance-ledger-route",
            {
                "schema_version": 1,
                "ledger": self.identity.to_data(),
                # Program records this route before Case resolution. Runtime block names are unique
                # inside one Case/Program; the consumer separately carries the canonical block and
                # state identity.
                "runtime_block": block.local_id,
            },
        )


def balance_record_name(route: Any, term: Any) -> str:
    """Return the reserved native Program diagnostic key for one exact term."""
    if (
        type(route) is not Identity
        or route.domain != "balance-ledger-route"
        or route.schema_version != 1
    ):
        raise TypeError("balance route must be an exact balance-ledger-route Identity")
    if term not in BALANCE_TERM_NAMES:
        raise ValueError("unknown balance term %r" % (term,))
    return "pops.balance-term.v1:%s:%s" % (route.token, term)


__all__ = ["BALANCE_TERM_NAMES", "BalanceLedger", "balance_record_name"]
