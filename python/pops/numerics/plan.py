"""Family-organized numerical authoring and its owner-qualified resolved value."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from typing import Any

from pops.descriptors import Descriptor
from pops.identity import Identity, make_identity, semantic_identity
from pops.model import Handle, OperatorHandle, OwnerKind


_RATE_METHOD_PROTOCOL = (
    "validate", "validate_rate_contract", "resolve_references", "to_data", "freeze",
)


def _require_rate_method(value: Any, where: str) -> Any:
    missing = [name for name in _RATE_METHOD_PROTOCOL if not callable(getattr(value, name, None))]
    if missing:
        raise TypeError(
            "%s must implement the small rate-method protocol; missing=%s"
            % (where, missing)
        )
    return value


def _callable_projection(value: Any, where: str) -> dict[str, Any]:
    for name in ("to_data", "canonical_identity"):
        method = getattr(value, name, None)
        if callable(method):
            result = method()
            if not isinstance(result, dict):
                raise TypeError("%s.%s() must return a mapping" % (where, name))
            return result
    raise TypeError(
        "%s has no canonical data projection; numerical extensions must implement "
        "to_data() or canonical_identity()" % where)


def _is_typed_authority(value: Any) -> bool:
    return isinstance(value, Handle) or any(
        callable(getattr(value, name, None))
        for name in ("to_data", "canonical_identity", "resolve_for_numerics")
    )


def _authority_key(value: Any) -> tuple[Any, ...]:
    """Identity-only duplicate key; never invokes overloaded symbolic equality."""
    if isinstance(value, Handle):
        return ("handle", value)
    return ("authority", type(value), id(value))


def _resolve_value(value: Any, resolver: Any, *, where: str) -> Any:
    """Resolve typed references through one deliberately small extension protocol."""
    if isinstance(value, Handle):
        return resolver(value)
    protocol = getattr(value, "resolve_references", None)
    if callable(protocol):
        return protocol(resolver)
    if isinstance(value, Mapping):
        return {
            key: _resolve_value(item, resolver, where="%s[%r]" % (where, key))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _resolve_value(item, resolver, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _resolve_value(item, resolver, where="%s[%d]" % (where, index))
            for index, item in enumerate(value)
        ]
    if _is_typed_authority(value):
        return value
    raise TypeError(
        "%s is not a typed numerical authority; implement resolve_references(resolver) "
        "and to_data()" % where)


def _projection_sort_key(value: Any, where: str) -> str:
    return json.dumps(
        _callable_projection(value, where), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


class _RateFamily:
    """Unique rate-to-method bindings; expressions can never become mapping keys."""

    def __init__(self) -> None:
        self._rows: list[tuple[OperatorHandle, Any]] | tuple[tuple[OperatorHandle, Any], ...] = []
        self._frozen = False

    def add(self, rate: Any, method: Any) -> None:
        if self._frozen:
            raise RuntimeError("DiscretizationPlan.rates is frozen")
        rows = self._rows
        if not isinstance(rows, list):
            raise RuntimeError("DiscretizationPlan.rates has an inconsistent frozen state")
        if not isinstance(rate, OperatorHandle) or rate.kind not in {"local_rate", "coupled_rate"}:
            raise TypeError("rates.add requires a typed rate OperatorHandle")
        _require_rate_method(method, "rates.add method")
        if any(existing == rate for existing, _ in rows):
            raise ValueError("rate %s already has a numerical method" % rate.qualified_id)
        rows.append((rate, method))

    def items(self) -> tuple[tuple[OperatorHandle, Any], ...]:
        return tuple(self._rows)

    def freeze(self) -> None:
        if self._frozen:
            return
        for _, method in self._rows:
            method.freeze()
        self._rows = tuple(self._rows)
        self._frozen = True


class _PairFamily:
    """Unique physical/numerical descriptor pairs for a named numerical family."""

    def __init__(self, family: str) -> None:
        self.family = family
        self._rows: list[tuple[Any, Any]] | tuple[tuple[Any, Any], ...] = []
        self._frozen = False

    def add(self, subject: Any, method: Any) -> None:
        if self._frozen:
            raise RuntimeError("DiscretizationPlan.%s is frozen" % self.family)
        rows = self._rows
        if not isinstance(rows, list):
            raise RuntimeError(
                "DiscretizationPlan.%s has an inconsistent frozen state" % self.family)
        if not _is_typed_authority(subject) or not _is_typed_authority(method):
            raise TypeError(
                "%s.add requires typed subject/method authorities with canonical projections"
                % self.family)
        subject_key = _authority_key(subject)
        if any(_authority_key(existing) == subject_key
               for existing, _ in rows):
            raise ValueError("%s subject already has a numerical method" % self.family)
        rows.append((subject, method))

    def items(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(self._rows)

    def freeze(self) -> None:
        if self._frozen:
            return
        for subject, method in self._rows:
            for value in (subject, method):
                freeze = getattr(value, "freeze", None)
                if callable(freeze):
                    freeze()
        self._rows = tuple(self._rows)
        self._frozen = True


class _ValueFamily:
    """Register-once descriptors such as boundary/interface authorities."""

    def __init__(self, family: str) -> None:
        self.family = family
        self._rows: list[Any] | tuple[Any, ...] = []
        self._frozen = False

    def preflight_add(self, value: Any) -> None:
        """Validate one registration without mutating the family.

        Multi-owner authoring protocols use this to prove that every destination can accept the
        same immutable authority before committing any of them.
        """
        if self._frozen:
            raise RuntimeError("DiscretizationPlan.%s is frozen" % self.family)
        if not _is_typed_authority(value):
            raise TypeError(
                "%s.add requires a typed authority with a canonical projection or "
                "resolve_for_numerics() protocol"
                % self.family)
        value_key = _authority_key(value)
        if any(_authority_key(existing) == value_key for existing in self._rows):
            raise ValueError("duplicate %s authority" % self.family)

    def add(self, value: Any) -> None:
        self.preflight_add(value)
        rows = self._rows
        if not isinstance(rows, list):
            raise RuntimeError(
                "DiscretizationPlan.%s has an inconsistent frozen state" % self.family)
        rows.append(value)

    def values(self) -> tuple[Any, ...]:
        return tuple(self._rows)

    def freeze(self) -> None:
        if self._frozen:
            return
        for value in self._rows:
            freeze = getattr(value, "freeze", None)
            if callable(freeze):
                freeze()
        self._rows = tuple(self._rows)
        self._frozen = True


@dataclass(frozen=True, slots=True)
class ResolvedRateMethod:
    rate: OperatorHandle
    method: Any

    def __post_init__(self) -> None:
        if not isinstance(self.rate, OperatorHandle) or not self.rate.is_resolved:
            raise TypeError("ResolvedRateMethod.rate must be owner-qualified")
        _require_rate_method(self.method, "ResolvedRateMethod.method")
        data = self.method.to_data()
        if not isinstance(data, dict):
            raise TypeError("ResolvedRateMethod.method.to_data() must return a dict")
        self.method.freeze()

    def to_data(self) -> dict[str, Any]:
        return {"rate": self.rate.canonical_identity(), "method": self.method.to_data()}


@dataclass(frozen=True, slots=True)
class ResolvedNumericalBinding:
    """One canonical physical-subject to numerical-method binding."""

    subject: Any
    method: Any

    def __post_init__(self) -> None:
        _callable_projection(self.subject, "resolved numerical subject")
        _callable_projection(self.method, "resolved numerical method")

    def to_data(self) -> dict[str, Any]:
        return {
            "subject": _callable_projection(self.subject, "resolved numerical subject"),
            "method": _callable_projection(self.method, "resolved numerical method"),
        }


@dataclass(frozen=True, slots=True)
class BoundaryResolutionContext:
    """The small, immutable protocol exposed to numerical boundary authorities.

    Boundary families consume resolved physics and intrinsic stencil requirements without
    importing ``Case`` or branching on concrete boundary classes.  Third-party authorities can
    implement ``resolve_for_numerics(context)`` against this same value.
    """

    owner: Any
    block: Handle
    frame: Any
    rates: tuple[ResolvedRateMethod, ...]
    resolve: Callable[[Handle], Handle]

    def __post_init__(self) -> None:
        if not getattr(self.owner, "is_canonical", False):
            raise TypeError("BoundaryResolutionContext.owner must be a canonical OwnerPath")
        if not isinstance(self.block, Handle) or self.block.kind != "block" \
                or not self.block.is_resolved:
            raise TypeError("BoundaryResolutionContext.block must be a canonical BlockHandle")
        if not isinstance(self.rates, tuple) or not self.rates:
            raise TypeError("BoundaryResolutionContext.rates must contain resolved rate methods")
        if any(not isinstance(row, ResolvedRateMethod) for row in self.rates):
            raise TypeError("BoundaryResolutionContext.rates contains an unresolved rate method")
        if not callable(self.resolve):
            raise TypeError("BoundaryResolutionContext.resolve must be callable")


@dataclass(frozen=True, slots=True)
class ResolvedDiscretizationPlan:
    block: Handle
    rates: tuple[ResolvedRateMethod, ...]
    fields: tuple[ResolvedNumericalBinding, ...]
    boundaries: tuple[Any, ...]
    sources: tuple[ResolvedNumericalBinding, ...]
    interfaces: tuple[Any, ...]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.block, Handle) or self.block.kind != "block" or not self.block.is_resolved:
            raise TypeError("resolved numerical plan requires a canonical BlockHandle")
        if not self.rates:
            raise ValueError("resolved numerical plan has no rate method for its block")
        for name in ("rates", "fields", "boundaries", "sources", "interfaces"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError("ResolvedDiscretizationPlan.%s must be a tuple" % name)
        if any(type(row) is not ResolvedRateMethod for row in self.rates):
            raise TypeError("ResolvedDiscretizationPlan.rates must contain ResolvedRateMethod")
        for family in ("fields", "sources"):
            if any(type(row) is not ResolvedNumericalBinding for row in getattr(self, family)):
                raise TypeError(
                    "ResolvedDiscretizationPlan.%s must contain ResolvedNumericalBinding"
                    % family)
        for family in ("boundaries", "interfaces"):
            for row in getattr(self, family):
                _callable_projection(row, "resolved %s authority" % family)
        object.__setattr__(self, "identity", semantic_identity(self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "block": self.block.canonical_identity(),
            "rates": [row.to_data() for row in self.rates],
            "fields": [row.to_data() for row in self.fields],
            "boundaries": [_callable_projection(row, "resolved boundary") for row in self.boundaries],
            "sources": [row.to_data() for row in self.sources],
            "interfaces": [_callable_projection(row, "resolved interface") for row in self.interfaces],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.token}

    def primary_spatial(self) -> Any:
        """Compatibility projection for the current per-block native spatial ABI.

        The resolved plan retains every per-rate binding. The native engine currently accepts one
        spatial method per block. Rates may name different physical fluxes while selecting the
        same native reconstruction/Riemann/variable configuration; that exact physical ownership
        remains on each resolved rate row. Genuinely different runtime configurations fail
        explicitly instead of picking the first.
        """
        methods = [row.method for row in self.rates]
        configurations = []
        for method in methods:
            provider = getattr(method, "runtime_configuration", None)
            configuration = provider() if callable(provider) else method.to_data()
            if not isinstance(configuration, dict):
                raise TypeError(
                    "rate method runtime_configuration() must return a dict"
                )
            configurations.append(configuration)
        first = configurations[0]
        if any(configuration != first for configuration in configurations[1:]):
            raise ValueError(
                "native runtime requires one finite-volume method per block; resolved rates select "
                "distinct runtime configurations and cannot be lowered without a per-operator "
                "native ABI"
            )
        return methods[0]

    def amr_stencil_requirement(self, *, owner: Any, dimension: int) -> Any:
        """Project the exact spatial methods onto the open AMR nesting protocol."""
        from pops.mesh._amr import NestingRequirementSource

        if isinstance(dimension, bool) or dimension not in (1, 2, 3):
            raise ValueError("AMR stencil dimension must be 1, 2, or 3")
        ghost_depth = max(row.method.ghost_depth for row in self.rates)
        lookahead = max(row.method.formal_order - 1 for row in self.rates)
        evidence = {
            "plan": self.identity.to_data(),
            "dimension": dimension,
            "ghost_depth": ghost_depth,
            "lookahead": lookahead,
        }
        provider = Handle(
            "stencil_%s" % make_identity("amr-stencil-requirement", evidence).token,
            kind="amr_stencil_requirement",
            owner=owner,
        )
        return NestingRequirementSource(
            provider,
            (ghost_depth,) * dimension,
            lookahead,
        )

    def amr_reflux_requirement(self, *, owner: Any, dimension: int) -> Any:
        """Project conservative flux correction needs without inspecting a layout class."""
        from pops.mesh._amr import NestingRequirementSource

        if isinstance(dimension, bool) or dimension not in (1, 2, 3):
            raise ValueError("AMR reflux dimension must be 1, 2, or 3")
        evidence = {
            "plan": self.identity.to_data(),
            "dimension": dimension,
            "rates": [row.rate.canonical_identity() for row in self.rates],
        }
        provider = Handle(
            "reflux_%s" % make_identity("amr-reflux-requirement", evidence).token,
            kind="amr_reflux_requirement",
            owner=owner,
        )
        return NestingRequirementSource(provider, (1,) * dimension, 0)


class DiscretizationPlan(Descriptor):
    """One family-organized numerical authority, reusable across Model instances."""

    category = "discretization_plan"

    def __init__(self) -> None:
        self.rates = _RateFamily()
        self.fields = _PairFamily("fields")
        self.boundaries = _ValueFamily("boundaries")
        self.sources = _PairFamily("sources")
        self.interfaces = _ValueFamily("interfaces")

    def options(self) -> dict[str, Any]:
        return {
            "rates": self.rates.items(),
            "fields": self.fields.items(),
            "boundaries": self.boundaries.values(),
            "sources": self.sources.items(),
            "interfaces": self.interfaces.values(),
        }

    def freeze(self) -> DiscretizationPlan:
        if getattr(self, "_frozen", False):
            return self
        for family in (self.rates, self.fields, self.boundaries, self.sources, self.interfaces):
            family.freeze()
        object.__setattr__(self, "_frozen", True)
        return self

    def validate_for(self, model: Any, *, states: Any = None) -> bool:
        contracts = getattr(model, "_rate_contracts", None)
        if not isinstance(contracts, Mapping):
            raise TypeError("DiscretizationPlan requires a Model exposing typed rate contracts")
        selected = dict(self.rates.items())
        state_set = None if states is None else set(states)
        expected = {
            rate for rate, contract in contracts.items()
            if state_set is None or contract["state"] in state_set
        }
        missing, extra = expected - set(selected), set(selected) - expected
        if missing or extra:
            raise ValueError(
                "DiscretizationPlan rate coverage mismatch for Model %r block states: "
                "missing=%s extra=%s"
                % (model.name, sorted(row.local_id for row in missing),
                   sorted(row.local_id for row in extra)))
        if not selected:
            raise ValueError("DiscretizationPlan has no rate binding for Model %r" % model.name)
        for rate, method in selected.items():
            contract = model.rate_contract(rate)
            method.validate_rate_contract(contract)
            method.validate()
        return True

    def resolve_for(self, case: Any, block: Any) -> ResolvedDiscretizationPlan:
        model = case._block_registry.spec(block.local_id)["model"]
        states = case._block_registry.spec(block.local_id)["states"]
        self.validate_for(model, states=states)

        def resolve_handle(value: Handle) -> Handle:
            if not isinstance(value, Handle):
                raise TypeError("numerical reference resolution requires a typed Handle")
            root_kind = value.owner_path.nodes[0].kind
            if root_kind in (OwnerKind.CASE, OwnerKind.SHARED):
                return case.resolve(value)
            return case.resolve(value, block=block)

        rates = []
        for rate, method in self.rates.items():
            if rate.owner_path != model.owner_path:
                continue
            if model.rate_contract(rate)["state"] not in states:
                continue
            rates.append(ResolvedRateMethod(
                case.resolve(rate, block=block), method.resolve_references(resolve_handle)))
        def resolve_pairs(rows: Any, family: str) -> tuple[ResolvedNumericalBinding, ...]:
            resolved = [
                ResolvedNumericalBinding(
                    _resolve_value(subject, resolve_handle, where="%s subject" % family),
                    _resolve_value(method, resolve_handle, where="%s method" % family),
                )
                for subject, method in rows
            ]
            return tuple(sorted(
                resolved,
                key=lambda row: json.dumps(
                    row.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False),
            ))

        def resolve_values(rows: Any, family: str) -> tuple[Any, ...]:
            resolved = [
                _resolve_value(value, resolve_handle, where="%s authority" % family)
                for value in rows
            ]
            return tuple(sorted(
                resolved,
                key=lambda value: _projection_sort_key(value, "resolved %s" % family),
            ))

        resolved_block = case.resolve(block)
        # Runtime boundary identities belong to the concrete block instance, not merely to the
        # Case root shared by every block.  Geometry boundaries remain frame-level labels; the
        # resolved BoundaryHandles below are executable per-block endpoints and must stay distinct.
        boundary_owner = resolved_block.instance_owner_path
        resolved_rates = tuple(sorted(rates, key=lambda row: row.rate.qualified_id))
        needs_boundary_context = bool(
            self.boundaries.values() or self.interfaces.values()
        )
        boundary_context = None
        if needs_boundary_context:
            frame = getattr(model, "frame", None)
            if frame is None:
                raise TypeError(
                    "numerical boundary/interface authorities require a Model exposing a "
                    "typed frame"
                )
            boundary_context = BoundaryResolutionContext(
                owner=boundary_owner,
                block=resolved_block,
                frame=frame,
                rates=resolved_rates,
                resolve=resolve_handle,
            )
        boundaries = []
        for authority in self.boundaries.values():
            resolve_boundary = getattr(authority, "resolve_for_numerics", None)
            if not callable(resolve_boundary):
                raise TypeError(
                    "numerical boundary authorities must expose "
                    "resolve_for_numerics(BoundaryResolutionContext)"
                )
            resolved_boundary = resolve_boundary(boundary_context)
            _callable_projection(resolved_boundary, "resolved boundary authority")
            boundaries.append(resolved_boundary)
        def resolve_interface(value: Any) -> Any:
            if boundary_context is None:
                raise RuntimeError(
                    "interface resolution lost its required BoundaryResolutionContext")
            protocol = getattr(value, "resolve_for_numerics", None)
            if callable(protocol):
                resolved = protocol(boundary_context)
            else:
                resolved = _resolve_value(
                    value, resolve_handle, where="interfaces authority")
            _callable_projection(resolved, "resolved interfaces authority")
            return resolved

        resolved_interfaces = tuple(sorted(
            (resolve_interface(value) for value in self.interfaces.values()),
            key=lambda value: _projection_sort_key(value, "resolved interfaces"),
        ))
        return ResolvedDiscretizationPlan(
            resolved_block,
            resolved_rates,
            resolve_pairs(self.fields.items(), "fields"),
            tuple(sorted(
                boundaries,
                key=lambda value: _projection_sort_key(value, "resolved boundaries"),
            )),
            resolve_pairs(self.sources.items(), "sources"),
            resolved_interfaces,
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "rates": [
                {"rate": rate.qualified_id, "method": method.inspect()}
                for rate, method in self.rates.items()
            ],
            "fields": len(self.fields.items()),
            "boundaries": len(self.boundaries.values()),
            "sources": len(self.sources.items()),
            "interfaces": len(self.interfaces.values()),
        }


__all__ = [
    "BoundaryResolutionContext", "DiscretizationPlan", "ResolvedDiscretizationPlan",
    "ResolvedNumericalBinding",
    "ResolvedRateMethod",
]
