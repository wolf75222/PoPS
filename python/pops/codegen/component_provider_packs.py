"""Exact component-provider packs shared by every compiler entry route.

The operator-first :class:`pops.model.Module` is the authority for provider identity.  Kernel
emitters must not rediscover providers from the legacy auxiliary layout: this module resolves the
full pack, every per-operator subset, and the physical-flux subset once and passes that immutable
value through the explicit compiler-emitter protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops.model.provider_pack import (
    ProviderPack,
    build_operator_provider_pack,
    build_provider_pack,
)


@dataclass(frozen=True, slots=True)
class ComponentProviderPacks:
    """One immutable provider resolution for a canonical Module."""

    complete: ProviderPack
    by_operator: Mapping[str, ProviderPack]
    physical_flux: ProviderPack
    auxiliary: ProviderPack
    consumer_plans: Mapping[str, tuple[dict[str, Any], ...]]
    physical_flux_plan: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.complete) is not ProviderPack:
            raise TypeError("ComponentProviderPacks.complete must be an exact ProviderPack")
        rows = dict(self.by_operator)
        if any(not isinstance(name, str) or not name for name in rows):
            raise TypeError(
                "ComponentProviderPacks.by_operator keys must be non-empty strings"
            )
        if any(type(pack) is not ProviderPack for pack in rows.values()):
            raise TypeError(
                "ComponentProviderPacks.by_operator values must be exact ProviderPack values"
            )
        object.__setattr__(self, "by_operator", MappingProxyType(rows))
        if type(self.physical_flux) is not ProviderPack:
            raise TypeError(
                "ComponentProviderPacks.physical_flux must be an exact ProviderPack"
            )
        if type(self.auxiliary) is not ProviderPack:
            raise TypeError(
                "ComponentProviderPacks.auxiliary must be an exact ProviderPack"
            )
        plans = dict(self.consumer_plans)
        if any(not isinstance(name, str) or not name for name in plans):
            raise TypeError("ComponentProviderPacks consumer plan names must be non-empty strings")
        if any(not isinstance(plan, tuple) for plan in plans.values()):
            raise TypeError("ComponentProviderPacks consumer plans must be immutable tuples")
        if not isinstance(self.physical_flux_plan, tuple):
            raise TypeError("ComponentProviderPacks physical flux plan must be an immutable tuple")
        plans = {
            name: tuple(MappingProxyType(dict(row)) for row in plan)
            for name, plan in plans.items()
        }
        object.__setattr__(
            self,
            "physical_flux_plan",
            tuple(MappingProxyType(dict(row)) for row in self.physical_flux_plan),
        )
        object.__setattr__(self, "consumer_plans", MappingProxyType(plans))

    def attach(self, target: Any) -> None:
        """Attach compiler-owned immutable evidence to one emitter carrier.

        Reattachment is idempotent and verifies byte-for-byte logical equality.  This is needed
        because a facade and its private formula carrier are distinct Python objects but emit one
        native package; neither may retain a different provider resolution.
        """
        values = {
            "_component_provider_pack": self.complete,
            "_component_provider_metadata": self.complete.to_data(),
            "_component_operator_provider_packs": self.by_operator,
            "_component_operator_provider_metadata": MappingProxyType({
                name: pack.to_data() for name, pack in self.by_operator.items()
            }),
            "_component_flux_provider_pack": self.physical_flux,
            "_component_flux_provider_metadata": self.physical_flux.to_data(),
            "_auxiliary_provider_pack": self.auxiliary,
            "_auxiliary_provider_metadata": self.auxiliary.to_data(),
            "_component_operator_consumer_plans": self.consumer_plans,
            "_component_flux_consumer_plan": self.physical_flux_plan,
        }

        def canonical(value: Any) -> Any:
            if isinstance(value, ProviderPack):
                return value.to_data()
            if isinstance(value, Mapping):
                return {
                    key: canonical(item)
                    for key, item in value.items()
                }
            return value

        for name, value in values.items():
            previous = getattr(target, name, None)
            if previous is not None and canonical(previous) != canonical(value):
                raise ValueError(
                    "compiler emitter retained a conflicting component-provider pack"
                )
            object.__setattr__(target, name, value)


def resolve_component_provider_packs(module: Any) -> ComponentProviderPacks:
    """Resolve all exact provider packs from one canonical Module authority."""
    complete = build_provider_pack(module)
    by_operator = {
        operator.name: build_operator_provider_pack(module, operator)
        for operator in module.operator_registry()
    }
    flux_requirements = []
    for operator in module.operator_registry():
        if operator.kind == "grid_operator":
            flux_requirements.extend(by_operator[operator.name])
    physical_flux = complete.select(flux_requirements)
    by_operator_plan = {
        name: consumer_provider_plan(pack)
        for name, pack in by_operator.items()
    }
    return ComponentProviderPacks(
        complete=complete,
        by_operator=by_operator,
        physical_flux=physical_flux,
        auxiliary=compact_auxiliary_provider_pack(complete),
        consumer_plans=by_operator_plan,
        physical_flux_plan=consumer_provider_plan(physical_flux),
    )


def compact_auxiliary_provider_pack(pack: Any) -> ProviderPack:
    """Project declared :class:`AuxSpace` values to compact native-channel slots.

    A slot is not inferred from a spelling, an axis, or a physics role.  It is
    assigned once, in declaration order, from the exact owner-qualified
    ``aux`` and ``field`` rows of the complete provider pack.  The empty pack
    is valid: a model without auxiliary inputs or field outputs allocates no
    channel at all.
    """
    if type(pack) is not ProviderPack:
        raise TypeError("auxiliary projection requires an exact ProviderPack")
    rows = []
    declared = {}
    for key in pack:
        if key.space_kind not in {"aux", "field"}:
            continue
        previous = declared.get(key.component)
        if previous is not None:
            raise ValueError(
                "auxiliary component %r is declared by both %s and %s; "
                "declare one owner-qualified storage route"
                % (key.component, previous.space, key.space)
            )
        declared[key.component] = key
        # A FieldSpace may deliberately be provided later by the case-owned
        # field provider.  Its slot is still part of the package ABI, while
        # availability is verified at bind rather than guessed at codegen.
        entry = pack.declared_entry(key)
        rows.append((
            key,
            pack.contract(key),
            type(entry)(entry.producer, entry.available, len(rows)),
        ))
    return ProviderPack(rows, capacity=len(rows))


def auxiliary_component_slot(pack: Any, *, owner_qid: Any, name: Any) -> int:
    """Resolve one source-level auxiliary name through an exact compact pack.

    Bare names never pick a provider globally: matching is constrained to the
    compiler authority's owner and auxiliary-capable (``aux`` or ``field``)
    space. Duplicate local spellings remain an error instead of taking an
    arbitrary storage route.
    """
    if type(pack) is not ProviderPack:
        raise TypeError("auxiliary slot lookup requires an exact ProviderPack")
    if not isinstance(owner_qid, str) or not owner_qid:
        raise ValueError("auxiliary slot owner_qid must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("auxiliary slot name must be a non-empty string")
    matches = [
        key for key in pack
        if key.owner_qid == owner_qid
        and key.space_kind in {"aux", "field"}
        and key.component == name
    ]
    if len(matches) != 1:
        detail = "absent" if not matches else "ambiguous"
        raise ValueError(
            "auxiliary component %r is %s in ProviderPack for owner %r"
            % (name, detail, owner_qid)
        )
    slot = pack.declared_entry(matches[0]).slot
    if slot is None:
        raise AssertionError("usable auxiliary provider has no compact slot")
    return slot


def consumer_provider_plan(pack: Any) -> tuple[Mapping[str, Any], ...]:
    """Return immutable ordered ``consumer_slot -> ComponentKey`` evidence.

    Consumer-local slots deliberately differ from the carrier/storage slots in
    :class:`ProviderEntry`.  The native registry resolves this plan to storage
    components at installation; generated formulas never assume the two orders
    coincide.
    """
    if type(pack) is not ProviderPack:
        raise TypeError("consumer plan requires an exact ProviderPack")
    rows = []
    for consumer_slot, key in enumerate(pack):
        rows.append(MappingProxyType({
            "consumer_slot": consumer_slot,
            "key": key.to_data(),
            "contract": pack.contract(key).to_data(),
            "provider": pack.declared_entry(key).to_data(),
        }))
    return tuple(rows)


__all__ = [
    "ComponentProviderPacks",
    "auxiliary_component_slot",
    "compact_auxiliary_provider_pack",
    "consumer_provider_plan",
    "resolve_component_provider_packs",
]
