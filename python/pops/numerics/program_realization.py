"""A retained balance realized explicitly by the enclosing native Program."""
from __future__ import annotations

from collections.abc import Mapping

from pops.descriptors import Descriptor
from pops.realization import Use
from .state_storage import StateStorage


class ProgramRealization(Descriptor):
    """Select authored Program equations for exact physical contribution uses.

    This descriptor supplies storage, not a hidden residual or flux. Resolution
    requires every requested Use to have one retained Realize SSA output per
    candidate path reaching the commit of the exact target state. Provisional
    evaluations may be repeated without delivering their inputs a second time.
    Several explicitly named temporal Uses
    may realize one occurrence. Consistency, order and stability of those authored
    equations are numerical obligations; structural coverage does not prove them.
    """

    category = "program_realization"
    formal_order = 1
    requires_program_coverage = True

    def __init__(self, *, uses, ghost_depth=1):
        self.uses = tuple(uses)
        if not self.uses or any(type(use) is not Use for use in self.uses):
            raise TypeError("ProgramRealization requires explicit nonempty Use objects")
        if len({use.key for use in self.uses}) != len(self.uses):
            raise ValueError("ProgramRealization repeats a structural Use")
        if len({use.occurrence.target for use in self.uses}) != 1:
            raise ValueError("ProgramRealization selects exactly one physical target state")
        self.ghost_depth = StateStorage(ghost_depth=ghost_depth).ghost_depth
        self.target = None

    def validate_rate_contract(self, contract):
        if not isinstance(contract, Mapping) or contract.get("state") is None:
            raise TypeError("ProgramRealization requires a retained state balance")
        if any(use.occurrence.target != contract["state"] for use in self.uses):
            raise ValueError("ProgramRealization Uses target another physical state")
        return True

    def validate_balance_view(self, view):
        if view is None:
            raise ValueError("ProgramRealization requires retained physical balance occurrences")
        expected = {row.identity: row.to_data() for row in view.occurrences}
        supplied = {use.occurrence.identity for use in self.uses}
        if supplied != set(expected):
            raise ValueError("ProgramRealization has missing or foreign balance occurrences")
        for use in self.uses:
            if use.occurrence.to_data() != expected[use.occurrence.identity]:
                raise ValueError("ProgramRealization altered a retained physical occurrence")
        return True

    def resolve_references(self, resolver):
        if not callable(resolver):
            raise TypeError("ProgramRealization requires a typed reference resolver")
        result = type(self)(uses=self.uses, ghost_depth=self.ghost_depth)
        result.target = resolver(self.uses[0].occurrence.target)
        return result

    def to_data(self):
        return {"schema_version": 1, "method": "program_realization",
                "uses": tuple(use.to_data() for use in self.uses),
                "target": None if self.target is None else self.target.canonical_identity(),
                "ghost_depth": self.ghost_depth, "formal_order": self.formal_order}

    def runtime_storage_requirements(self):
        return {"ghost_depth": self.ghost_depth}

    def runtime_configuration(self):
        return StateStorage(ghost_depth=self.ghost_depth).runtime_configuration()

    def runtime_spatial(self):
        return StateStorage(ghost_depth=self.ghost_depth).runtime_spatial()


__all__ = ["ProgramRealization"]
