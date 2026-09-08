"""Explicit pointwise numerical realization of qualified joint rate outputs."""
from __future__ import annotations

from pops.descriptors import Descriptor


class JointEvaluation(Descriptor):
    category = "joint_evaluation"
    native_id = "pops::NativeCallResult"
    formal_order = 1
    ghost_depth = 1  # the shared native storage contract reserves one ghost cell

    def __init__(self, state, *, sampling="cell"):
        from pops.model.handles import Handle
        from pops.model.spaces import StateSpace
        if not isinstance(state, Handle) or state.kind != "state" or not isinstance(state.space, StateSpace):
            raise TypeError("JointEvaluation requires the exact evolved StateHandle")
        if sampling != "cell":
            raise ValueError("JointEvaluation implements cell sampling; face laws require a flux construction")
        self.state, self.sampling = state, sampling

    def validate(self, context=None):
        return True

    def validate_rate_contract(self, contract):
        if contract["state"] != self.state or contract.get("flux") not in (None, ()):
            raise ValueError("JointEvaluation requires its exact state and no divergence term")
        return True

    def validate_balance_view(self, view):
        from pops.physics.interactions import joint_balance_supported
        if not joint_balance_supported(view) or view.target != self.state or any(
                occurrence.kind == "flux" for occurrence in view.occurrences):
            raise ValueError("JointEvaluation requires exact joint-source occurrence coverage")
        for occurrence in view.occurrences:
            if occurrence.kind == "projection" and occurrence.payload.application.context.sampling not in (
                    "cell", "unspecified"):
                raise ValueError("joint application sampling differs from its cell realization")
        return True

    def resolve_references(self, resolver):
        return type(self)(resolver(self.state), sampling=self.sampling)

    def options(self):
        return {"state": self.state, "sampling": self.sampling}

    def to_data(self):
        return {"schema_version": 1, "method": "joint_evaluation",
                "state": self.state.canonical_identity(), "sampling": self.sampling,
                "formal_order": self.formal_order, "ghost_depth": self.ghost_depth}

    def runtime_configuration(self):
        from pops.numerics import StateStorage
        return StateStorage().runtime_configuration()

    def runtime_spatial(self):
        from pops.numerics import StateStorage
        return StateStorage().runtime_spatial()
