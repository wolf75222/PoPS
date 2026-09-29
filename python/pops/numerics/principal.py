"""Exact finite-volume groups: physical rows and numerical sampling stay distinct."""
from dataclasses import dataclass, field

from pops.identity import Identity, semantic_identity


@dataclass(frozen=True, slots=True)
class PrincipalGroup:
    states: tuple
    rates: tuple
    methods: tuple
    dimension: int
    identity: Identity = field(init=False)

    def __post_init__(self):
        if not self.states or not (len(self.states) == len(self.rates) == len(self.methods)):
            raise ValueError("principal group requires one physical rate and method per state")
        if len(set(self.states)) != len(self.states):
            raise ValueError("principal group repeats a state instance")
        object.__setattr__(self, "identity", semantic_identity(self._payload()))

    @property
    def component_counts(self):
        return tuple(len((state.declaration_ref or state).space.components) for state in self.states)

    @property
    def component_count(self):
        return sum(self.component_counts)

    def _payload(self):
        return {"schema_version": 1, "kind": "principal_finite_volume_group",
                "states": [state.canonical_identity() for state in self.states],
                "rates": [rate.canonical_identity() for rate in self.rates],
                "methods": [method.to_data() for method in self.methods],
                "dimension": self.dimension, "component_counts": list(self.component_counts),
                "publication": "joint", "flux_rows": "complete_physical_balances"}

    def to_data(self):
        return {**self._payload(), "identity": self.identity.token}


def resolve_principal_groups(case, block):
    """Close explicitly sampled groups over the Case's actual block instances.

    This does not call resolve_for recursively: all methods are resolved from their
    original plans, preserving their physical owners and per-block selections.
    """
    from pops.numerics.spatial import FiniteVolume, _brick_data
    from pops.model.ownership import OwnerKind

    def choice(method):
        # Authored componentwise/row bodies form an explicitly ordered product.
        # Their complete descriptors remain in the group identity; equal display
        # names or equal scalar body widths never merge their captured contexts.
        recon = ("source_stencil" if method.reconstruction.scheme == "source_stencil"
                 else _brick_data(method.reconstruction))
        face = ("source_face" if method.riemann.scheme == "source_face"
                else _brick_data(method.riemann))
        return recon, face, method.variables.scheme, method.positivity_floor

    rows = []
    for block_name, plan in case._numerics_assignments.items():
        owner = case._block_registry.handle(block_name)
        spec = case._block_registry.spec(block_name)
        model, selected_states = spec["model"], spec["states"]

        def resolve(handle):
            if handle.owner_path.nodes[0].kind in (OwnerKind.CASE, OwnerKind.SHARED):
                return case.resolve(handle)
            if handle.kind == "state" and handle not in selected_states:
                return case.resolve(handle)
            return case.resolve(handle, block=owner)

        for rate, method in plan.rates.items():
            if not isinstance(method, FiniteVolume):
                continue
            contract = model.rate_contract(rate)
            if contract["state"] not in selected_states:
                continue
            target = resolve(contract["state"])
            resolved = method.resolve_references(resolve)
            if resolved.sampling:
                rows.append((target, case.resolve(rate, block=owner), resolved, model, contract))
    groups, seen = [], set()
    for target, rate, method, model, contract in rows:
        members = frozenset((target, *method.sampling))
        if members in seen:
            continue
        seen.add(members)
        if not any(state.block_ref == block._resolved() for state in members):
            continue
        selected = [row for row in rows if row[0] in members]
        if len(selected) != len(members) or {row[0] for row in selected} != members:
            raise ValueError("principal sampling group must select every physical flux row exactly once")
        # Sort by the source Model's declaration order; instance ids disambiguate repeated Models.
        declared_order = {state.name: i for i, state in enumerate(model._states.values())}
        selected.sort(key=lambda row: (declared_order[(row[0].declaration_ref or row[0]).local_id],
                                       row[0].qualified_id))
        reference = choice(method)
        spaces = tuple((row[0].declaration_ref or row[0]).space for row in selected)
        for state, row_rate, row_method, row_model, row_contract in selected:
            if row_model is not model:
                raise ValueError("principal group states must belong to one authenticated physical Model")
            if frozenset((state, *row_method.sampling)) != members:
                raise ValueError("principal reconstruction sampling must name the complete same group")
            if choice(row_method) != reference:
                raise ValueError("principal group requires one common reconstruction and numerical flux")
            registry = model.module.operator_registry()
            view = registry.get(row_rate.registered_operator_name).lowering["physical_balance"]
            if any(term.kind != "flux" or term.coefficient != -1 for term in view.occurrences):
                raise ValueError("principal group currently requires conservative -div(flux) rows")
            for occurrence in view.occurrences:
                operator = registry.get(occurrence.payload.reg_name)
                if any(space not in spaces for space in operator.signature.inputs):
                    raise ValueError("principal sampling omits a physical flux dependency")
        first = spaces[0]
        if any((space.frame, space.support, space.sampling) !=
               (first.frame, first.support, first.sampling) for space in spaces):
            raise ValueError("principal group requires the same physical frame, support and sampling")
        groups.append(PrincipalGroup(tuple(row[0] for row in selected),
            tuple(row[1] for row in selected), tuple(row[2] for row in selected), len(model.frame.axes)))
    return tuple(groups)
