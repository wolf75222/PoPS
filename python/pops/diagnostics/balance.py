"""Typed identity shared by native Program balance evidence and output consumers."""
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

    The ledger does not contain values. :meth:`Program.record_balance` writes the five
    reduced scalars into the current native step-attempt mailbox, while
    :class:`pops.diagnostics.Balance` selects the same identity after that attempt has
    advanced successfully.
    """

    name: str
    identity: Identity = field(init=False)
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        name = _canonical_name(self.name, where="BalanceLedger.name")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "identity",
            make_identity("balance-ledger", {"schema_version": 1, "name": name}),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "identity": self.identity.to_data(),
        }

    def route_identity(self, block: Any) -> Identity:
        from pops.problem.handles import BlockHandle

        if not isinstance(block, BlockHandle):
            raise TypeError("balance ledger block must be a BlockHandle")
        return make_identity(
            "balance-ledger-route",
            {
                "schema_version": 1,
                "ledger": self.identity.to_data(),
                # The Program records this route before Case resolution, whereas the
                # consumer is resolved later. Runtime block names are unique inside one
                # Case/Program, and the consumer quantity separately carries the complete
                # canonical block/state identity.
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


__all__ = ["BALANCE_TERM_NAMES", "BalanceLedger"]
