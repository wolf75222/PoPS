"""Typed component-provider registry used while lowering a :class:`Module`.

The legacy DSL addresses auxiliary storage by a bare component name.  A Module does not: a
component is owned by a particular typed space.  These small immutable values preserve that exact
identity and the declaration contract until the native component registry can consume it.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class MissingInputProvider(LookupError):
    """An exact component has no usable provider in a :class:`ProviderPack`."""


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % label)
    return value


@dataclass(frozen=True, slots=True, order=True)
class ComponentKey:
    """Exact identity of one component: owner, typed-space kind/name, component."""

    owner_qid: str
    space_kind: str
    space_name: str
    component: str

    def __post_init__(self) -> None:
        _non_empty(self.owner_qid, "ComponentKey owner_qid")
        _non_empty(self.space_kind, "ComponentKey space_kind")
        _non_empty(self.space_name, "ComponentKey space_name")
        _non_empty(self.component, "ComponentKey component")

    @property
    def space(self) -> str:
        """Stable ``kind/name`` spelling used by diagnostics and serialized metadata."""
        return "%s/%s" % (self.space_kind, self.space_name)

    def to_data(self) -> dict[str, str]:
        return {"owner_qid": self.owner_qid, "space_kind": self.space_kind,
                "space_name": self.space_name, "component": self.component}


@dataclass(frozen=True, slots=True)
class ComponentContract:
    """Lowering-relevant physical/storage contract for one component."""

    representation: str
    centering: str
    unit: str | None
    layout: str
    value_kind: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.representation, "ComponentContract representation")
        _non_empty(self.centering, "ComponentContract centering")
        if self.unit is not None:
            _non_empty(self.unit, "ComponentContract unit")
        _non_empty(self.layout, "ComponentContract layout")
        if self.value_kind is not None:
            _non_empty(self.value_kind, "ComponentContract value_kind")

    def to_data(self) -> dict[str, Any]:
        return {"representation": self.representation, "centering": self.centering,
                "unit": self.unit, "layout": self.layout, "value_kind": self.value_kind}


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """The producer and concrete slot for one component.

    ``producer=None`` deliberately means *unset*.  It is retained for inspection but exact lookup
    refuses it, just like an explicitly unavailable route.
    """

    producer: str | None
    availability: bool
    slot: int | None

    def __post_init__(self) -> None:
        if self.producer is not None:
            _non_empty(self.producer, "ProviderEntry producer")
        if not isinstance(self.availability, bool):
            raise TypeError("ProviderEntry availability must be a bool")
        if self.slot is not None and (isinstance(self.slot, bool) or not isinstance(self.slot, int)
                                      or self.slot < 0):
            raise ValueError("ProviderEntry slot must be a non-negative integer or None")

    @property
    def available(self) -> bool:
        return self.availability

    def to_data(self) -> dict[str, Any]:
        return {"producer": self.producer, "availability": self.availability,
                "slot": self.slot}


class ProviderPack:
    """Immutable, capacity-checked exact component lookup.

    Rows may be supplied as ``(key, contract, provider)`` triples or as a mapping from a key to a
    ``(contract, provider)`` pair.  Validation is completed in temporary dictionaries before the
    object publishes any state, so an over-capacity or duplicate construction has no partial pack.
    """

    __slots__ = ("_contracts", "_entries", "_capacity", "_sealed")
    __pops_ir_immutable__ = True

    def __init__(self, rows: Any = (), *, capacity: int | None = None,
                 contracts: Mapping[ComponentKey, ComponentContract] | None = None) -> None:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int)
                                     or capacity < 0):
            raise ValueError("ProviderPack capacity must be a non-negative integer or None")
        pending_contracts: dict[ComponentKey, ComponentContract] = dict(contracts or {})
        pending_entries: dict[ComponentKey, ProviderEntry] = {}
        source = rows.items() if isinstance(rows, Mapping) else rows
        for row in source:
            values = tuple(row)
            if len(values) == 3:
                key, contract, entry = values
            elif len(values) == 2:
                key, value = values
                if isinstance(value, tuple) and len(value) == 2:
                    contract, entry = value
                else:
                    contract, entry = pending_contracts.get(key), value
            else:
                raise TypeError("ProviderPack rows must be (key, contract, provider) triples")
            if not isinstance(key, ComponentKey):
                raise TypeError("ProviderPack keys must be ComponentKey values")
            if not isinstance(contract, ComponentContract):
                raise TypeError("ProviderPack contracts must be ComponentContract values")
            if not isinstance(entry, ProviderEntry):
                raise TypeError("ProviderPack entries must be ProviderEntry values")
            if key in pending_entries:
                raise ValueError("duplicate component provider for %r" % (key,))
            pending_contracts[key] = contract
            pending_entries[key] = entry
        if set(pending_contracts) != set(pending_entries):
            raise ValueError("ProviderPack contracts and provider entries must have identical keys")
        if capacity is not None:
            if len(pending_entries) > capacity:
                raise ValueError("ProviderPack capacity %d exceeded by %d entries"
                                 % (capacity, len(pending_entries)))
            overflow = [(key, entry.slot) for key, entry in pending_entries.items()
                        if entry.slot is not None and entry.slot >= capacity]
            if overflow:
                raise ValueError("ProviderPack capacity %d cannot hold slot(s) %r"
                                 % (capacity, overflow))
        object.__setattr__(self, "_contracts", MappingProxyType(pending_contracts))
        object.__setattr__(self, "_entries", MappingProxyType(pending_entries))
        object.__setattr__(self, "_capacity", capacity)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ProviderPack is immutable")
        object.__setattr__(self, name, value)

    @property
    def capacity(self) -> int | None:
        return self._capacity

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[ComponentKey]:
        return iter(self._entries)

    def keys(self) -> Any:
        return self._entries.keys()

    def contract(self, key: ComponentKey) -> ComponentContract:
        try:
            return self._contracts[key]
        except KeyError:
            raise MissingInputProvider("missing component contract for %r" % (key,)) from None

    def declared_entry(self, key: ComponentKey) -> ProviderEntry:
        """Return an inspection row even when its provider is unset/unavailable."""
        try:
            return self._entries[key]
        except KeyError:
            raise MissingInputProvider("missing input provider for %r" % (key,)) from None

    def lookup(self, key: ComponentKey, contract: ComponentContract | None = None) -> ProviderEntry:
        """Return only an exact, contract-compatible, usable provider."""
        entry = self.declared_entry(key)
        actual = self._contracts[key]
        if contract is not None and contract != actual:
            raise MissingInputProvider(
                "input provider contract mismatch for %r: requested %r, declared %r"
                % (key, contract, actual))
        if entry.producer is None or entry.slot is None:
            raise MissingInputProvider("input provider for %r is unset" % (key,))
        if not entry.available:
            raise MissingInputProvider("input provider for %r is unavailable" % (key,))
        return entry

    __getitem__ = lookup

    def select(self, requirements: Iterable[Any]) -> ProviderPack:
        """Resolve and retain exactly the qualified components a consumer declares.

        A requirement is either a :class:`ComponentKey` or ``(key, contract)``.  Every row is
        validated through :meth:`lookup` before the result is published: missing, unavailable,
        unset, or contract-incompatible inputs fail at resolve time and can never become a neutral
        runtime value.  Producer slots are storage descriptors, not positional argument indices,
        so they retain their original values in the minimal pack.
        """
        rows = []
        seen = set()
        for requirement in requirements:
            if isinstance(requirement, ComponentKey):
                key, expected = requirement, None
            else:
                try:
                    key, expected = tuple(requirement)
                except (TypeError, ValueError):
                    raise TypeError(
                        "ProviderPack requirements must be ComponentKey or (key, contract)"
                    ) from None
                if not isinstance(key, ComponentKey) or not isinstance(expected, ComponentContract):
                    raise TypeError(
                        "ProviderPack requirements must be ComponentKey or (key, contract)"
                    )
            if key in seen:
                continue
            entry = self.lookup(key, expected)
            rows.append((key, self._contracts[key], entry))
            seen.add(key)
        return ProviderPack(rows, capacity=self._capacity)

    def select_spaces(self, *, owner_qid: str,
                      spaces: Iterable[tuple[str, str]]) -> ProviderPack:
        """Select complete typed spaces without falling back to a bare component name."""
        _non_empty(owner_qid, "ProviderPack selection owner_qid")
        requested = set(spaces)
        for row in requested:
            if (not isinstance(row, tuple) or len(row) != 2 or
                    not all(isinstance(value, str) and value for value in row)):
                raise TypeError("ProviderPack spaces must be (space_kind, space_name) pairs")
        keys = [key for key in self if key.owner_qid == owner_qid and
                (key.space_kind, key.space_name) in requested]
        found = {(key.space_kind, key.space_name) for key in keys}
        missing = requested - found
        if missing:
            raise MissingInputProvider(
                "missing typed provider space(s) for owner %r: %r" %
                (owner_qid, sorted(missing)))
        return self.select(keys)

    def to_data(self) -> dict[str, Any]:
        rows = []
        for key in sorted(self._entries):
            rows.append({"key": key.to_data(), "contract": self._contracts[key].to_data(),
                         "provider": self._entries[key].to_data()})
        return {"schema_version": 1, "capacity": self._capacity, "entries": rows}

    @classmethod
    def from_data(cls, data: Any) -> ProviderPack:
        if not isinstance(data, Mapping) or set(data) != {
            "schema_version", "capacity", "entries",
        }:
            raise TypeError("ProviderPack data must contain schema_version, capacity, entries")
        if data["schema_version"] != 1:
            raise ValueError("unsupported ProviderPack schema_version %r" % data["schema_version"])
        if not isinstance(data["entries"], list):
            raise TypeError("ProviderPack entries must be a list")
        rows = []
        for index, row in enumerate(data["entries"]):
            if not isinstance(row, Mapping) or set(row) != {"key", "contract", "provider"}:
                raise TypeError("ProviderPack entries[%d] has an invalid schema" % index)
            key_data, contract_data, provider_data = row["key"], row["contract"], row["provider"]
            if not isinstance(key_data, Mapping) or set(key_data) != {
                "owner_qid", "space_kind", "space_name", "component",
            }:
                raise TypeError("ProviderPack entries[%d].key has an invalid schema" % index)
            if not isinstance(contract_data, Mapping) or set(contract_data) != {
                "representation", "centering", "unit", "layout", "value_kind",
            }:
                raise TypeError("ProviderPack entries[%d].contract has an invalid schema" % index)
            if not isinstance(provider_data, Mapping) or set(provider_data) != {
                "producer", "availability", "slot",
            }:
                raise TypeError("ProviderPack entries[%d].provider has an invalid schema" % index)
            rows.append((
                ComponentKey(**key_data),
                ComponentContract(**contract_data),
                ProviderEntry(**provider_data),
            ))
        result = cls(rows, capacity=data["capacity"])
        if result.to_data() != dict(data):
            raise ValueError("ProviderPack data is not in canonical order")
        return result


def build_provider_pack(module: Any) -> ProviderPack:
    """Build the canonical logical provider pack from one qualified Module authority."""
    canonical_owner = module.owner_path.canonical()
    owner_qid = str(canonical_owner)
    rows = []
    field_producers = {}
    for operator in module.operator_registry():
        output = operator.signature.output
        if getattr(output, "kind", None) == "field":
            field_producers.setdefault(output, []).append(module.operator_handle(operator.name))

    for space_kind, spaces in (("state", module.state_spaces()),
                               ("field", module.field_spaces())):
        for space in spaces.values():
            producers = field_producers.get(space, ()) if space_kind == "field" else ()
            if len(producers) > 1:
                raise ValueError(
                    "typed space %s/%s has multiple component providers %s"
                    % (space_kind, space.name, [p.qualified_id for p in producers]))
            producer = (producers[0]._resolved(canonical_owner).qualified_id if producers else
                        ("initial_state" if space_kind == "state" else None))
            for slot, component in enumerate(space.components):
                rows.append((
                    ComponentKey(owner_qid, space_kind, space.name, component),
                    ComponentContract(
                        space.representation, space.centering, space.units[slot], space.layout),
                    ProviderEntry(producer, producer is not None,
                                  slot if producer is not None else None),
                ))
    for slot, aux in enumerate(module.aux().values()):
        rows.append((
            ComponentKey(owner_qid, "aux", aux.name, aux.name),
            ComponentContract(
                aux.representation, aux.centering, aux.unit, aux.centering, aux.kind),
            ProviderEntry("runtime_input", True, slot),
        ))
    return ProviderPack(rows)


def build_operator_provider_pack(module: Any, operator: Any) -> ProviderPack:
    """Build the minimal exact provider pack consumed by one typed operator.

    State traces are explicit NumericalFlux operands and therefore are not duplicated in the
    provider pack.  Every FieldSpace input is retained with its complete qualified contract.  The
    selection goes through :meth:`ProviderPack.select_spaces`, so a stale signature or missing
    producer is a resolve-time error rather than a runtime zero.
    """
    full = build_provider_pack(module)
    spaces = []
    for input_space in operator.signature.inputs:
        if getattr(input_space, "kind", None) == "field":
            spaces.append(("field", input_space.name))
    if not spaces:
        return ProviderPack(capacity=full.capacity)
    return full.select_spaces(owner_qid=str(module.owner_path.canonical()), spaces=spaces)


__all__ = ["ComponentKey", "ComponentContract", "ProviderEntry", "ProviderPack",
           "MissingInputProvider", "build_provider_pack", "build_operator_provider_pack"]
