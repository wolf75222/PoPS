"""Typed component-provider registry used while lowering a :class:`Module`.

The legacy DSL addresses auxiliary storage by a bare component name.  A Module does not: a
component is owned by a particular typed space.  These small immutable values preserve that exact
identity and the declaration contract until the native component registry can consume it.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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

    def __iter__(self) -> Iterator[ComponentKey]:
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

    def select_components(
        self,
        *,
        owner_qid: str,
        spaces: Iterable[tuple[str, str]],
        components: Iterable[str],
    ) -> ProviderPack:
        """Select exact components from declared spaces without a bare-name fallback.

        Component spelling is only a filter inside the already-qualified owner/space set.  A
        missing component or the same spelling in two selected spaces is rejected rather than
        guessed, so an operator that needs one of two homonymous fields must qualify its input
        space more narrowly.
        """
        _non_empty(owner_qid, "ProviderPack selection owner_qid")
        requested_spaces = set(spaces)
        requested_components = tuple(components)
        if any(not isinstance(name, str) or not name for name in requested_components):
            raise TypeError(
                "ProviderPack components must contain non-empty strings"
            )
        if len(set(requested_components)) != len(requested_components):
            raise ValueError("ProviderPack components contains a duplicate")
        candidates = [
            key for key in self
            if key.owner_qid == owner_qid
            and (key.space_kind, key.space_name) in requested_spaces
        ]
        selected = []
        for component in requested_components:
            matches = [key for key in candidates if key.component == component]
            if not matches:
                raise MissingInputProvider(
                    "missing component %r in qualified provider spaces %r for owner %r"
                    % (component, sorted(requested_spaces), owner_qid)
                )
            if len(matches) != 1:
                raise MissingInputProvider(
                    "ambiguous component %r in qualified provider spaces %r for owner %r"
                    % (component, sorted(requested_spaces), owner_qid)
                )
            selected.append(matches[0])
        return self.select(selected)

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


def _bound_field_projections(
    module: Any,
    *,
    canonical_owner: Any,
    field_spaces: Mapping[str, Any],
) -> tuple[list[tuple[ComponentKey, ComponentContract, ProviderEntry]], set[tuple[str, str]]]:
    """Project explicit physical-field bindings through one declared storage carrier.

    A Board field handle is a scientific declaration, not the legacy aggregate ``FieldSpace`` the
    formula backend uses for storage.  The binding registry is the sole authority joining that
    subject to a field operator.  This deliberately does not try to infer either side from a name,
    an alias, or an overlapping component spelling.
    """
    owner_qid = str(canonical_owner)
    declaration_index = module.declaration_index()
    operator_registry = module.operator_registry()
    operator_index = operator_registry.declaration_index()
    rows = []
    claimed = set()
    projected = set()

    for raw_subject, raw_target in module.operator_bindings().items():
        if getattr(raw_subject, "kind", None) != "field":
            continue
        try:
            subject = declaration_index.authenticate(raw_subject)
            target = operator_index.authenticate(raw_target)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid field operator binding for subject %r" %
                getattr(raw_subject, "local_id", raw_subject)
            ) from exc
        if getattr(target, "kind", None) != "field_operator":
            raise ValueError(
                "invalid field operator binding for subject %r: target is not a field_operator"
                % subject.local_id
            )
        try:
            operator = operator_registry.get(target.registered_operator_name)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid field operator binding for subject %r: target is not registered"
                % subject.local_id
            ) from exc
        output = getattr(getattr(operator, "signature", None), "output", None)
        if getattr(output, "kind", None) != "field":
            raise ValueError(
                "invalid field operator binding for subject %r: target has no FieldSpace output"
                % subject.local_id
            )

        carriers = []
        for name, carrier in field_spaces.items():
            if getattr(carrier, "name", None) != name:
                raise ValueError(
                    "Module field-space registry key %r does not match carrier name %r"
                    % (name, getattr(carrier, "name", None))
                )
            if all(component in carrier.components for component in output.components):
                carriers.append(carrier)
        if not carriers:
            raise ValueError(
                "field operator binding for subject %r has no declared storage carrier for %r"
                % (subject.local_id, output.name)
            )
        if len(carriers) != 1:
            raise ValueError(
                "field operator binding for subject %r has ambiguous storage carriers %r"
                % (subject.local_id, sorted(carrier.name for carrier in carriers))
            )
        carrier = carriers[0]
        producer = target._resolved(canonical_owner).qualified_id
        for output_slot, component in enumerate(output.components):
            carrier_slot = carrier.components.index(component)
            output_contract = ComponentContract(
                output.representation,
                output.centering,
                output.units[output_slot],
                output.layout,
            )
            carrier_contract = ComponentContract(
                carrier.representation,
                carrier.centering,
                carrier.units[carrier_slot],
                carrier.layout,
            )
            if carrier_contract != output_contract:
                raise ValueError(
                    "field operator binding for subject %r has contract mismatch for component %r"
                    % (subject.local_id, component)
                )
            carrier_key = (carrier.name, component)
            if carrier_key in claimed:
                raise ValueError(
                    "field operator bindings claim legacy carrier component %s/%s more than once"
                    % carrier_key
                )
            key = ComponentKey(owner_qid, "field", subject.local_id, component)
            if key in projected:
                raise ValueError(
                    "field operator bindings collide on projected component %s/%s"
                    % (subject.local_id, component)
                )
            claimed.add(carrier_key)
            projected.add(key)
            rows.append((
                key,
                output_contract,
                ProviderEntry(producer, True, carrier_slot),
            ))
    return rows, claimed


def build_provider_pack(module: Any) -> ProviderPack:
    """Build the canonical logical provider pack from one qualified Module authority."""
    canonical_owner = module.owner_path.canonical()
    owner_qid = str(canonical_owner)
    state_spaces = module.state_spaces()
    field_spaces = module.field_spaces()
    projected_rows, claimed_field_components = _bound_field_projections(
        module, canonical_owner=canonical_owner, field_spaces=field_spaces
    )
    rows = []
    field_producers = {}
    for operator in module.operator_registry():
        output = operator.signature.output
        if getattr(output, "kind", None) == "field":
            field_producers.setdefault(output, []).append(module.operator_handle(operator.name))

    for space_kind, spaces in (("state", state_spaces), ("field", field_spaces)):
        for space in spaces.values():
            producers = field_producers.get(space, ()) if space_kind == "field" else ()
            resolved_producers = tuple(sorted(
                producer._resolved(canonical_owner).qualified_id for producer in producers
            ))
            if len(resolved_producers) == 1:
                producer = resolved_producers[0]
            elif resolved_producers:
                # A field RHS is intentionally compositional: several owner-qualified model
                # providers may contribute to one materialized FieldSpace (species charge is the
                # canonical case).  The Module manifest records that complete ordered set as one
                # deterministic logical provider.  Coefficients/measures remain the Case
                # FieldProviderPack's authority and are never guessed here.
                producer = "field_provider_set:[%s]" % ",".join(resolved_producers)
            else:
                # An unproduced FieldSpace is an explicit runtime input, not a
                # missing numerical value.  The case/bind plan supplies or
                # rejects it later; retaining that provider identity lets the
                # compiler allocate its exact auxiliary slot without a name
                # based fallback.
                producer = "initial_state" if space_kind == "state" else "runtime_input"
            for slot, component in enumerate(space.components):
                if space_kind == "field" and (space.name, component) in claimed_field_components:
                    continue
                rows.append((
                    ComponentKey(owner_qid, space_kind, space.name, component),
                    ComponentContract(
                        space.representation, space.centering, space.units[slot], space.layout),
                    ProviderEntry(producer, producer is not None,
                                  slot if producer is not None else None),
                ))
    existing_keys = {key for key, _, _ in rows}
    projection_keys = {key for key, _, _ in projected_rows}
    if existing_keys & projection_keys:
        raise ValueError(
            "field operator bindings collide with declared provider components %r"
            % sorted(existing_keys & projection_keys)
        )
    rows.extend(projected_rows)
    aux_producers = module.aux_providers()
    for slot, aux in enumerate(module.aux().values()):
        declared = aux_producers.get(aux.name)
        if declared is None:
            producer = "runtime_input"
        elif declared.producer_kind == "input":
            producer = "runtime_input"
        elif declared.producer_kind == "derived":
            producer = "derived:%s" % aux.name
        else:
            raise TypeError(
                "auxiliary component %r has unsupported producer descriptor %s"
                % (aux.name, type(declared).__name__)
            )
        rows.append((
            ComponentKey(owner_qid, "aux", aux.name, aux.name),
            ComponentContract(
                aux.representation, aux.centering, aux.unit, aux.centering, aux.kind),
            ProviderEntry(producer, True, slot),
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
    owner_qid = str(module.owner_path.canonical())
    requirements = getattr(operator, "requirements", {})
    required_components = requirements.get("aux", ())
    if required_components:
        selected = []
        for component in required_components:
            matches = [
                key for key in full
                if key.owner_qid == owner_qid
                and key.space_kind in {"aux", "field"}
                and key.component == component
            ]
            if len(matches) != 1:
                detail = "missing" if not matches else "ambiguous"
                raise MissingInputProvider(
                    "%s auxiliary component %r for operator %r; declare one exact "
                    "AuxSpace or FieldSpace route" % (detail, component, operator.name)
                )
            selected.append(matches[0])
        return full.select(selected)
    if not spaces:
        return ProviderPack(capacity=full.capacity)
    return full.select_spaces(owner_qid=owner_qid, spaces=spaces)


__all__ = ["ComponentKey", "ComponentContract", "ProviderEntry", "ProviderPack",
           "MissingInputProvider", "build_provider_pack", "build_operator_provider_pack"]
