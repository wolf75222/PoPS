"""Typed, callback-free initial-condition authoring."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from pops.model import Handle, OwnerPath
from ._plan import (
    InitialConditionBinding,
    InitialConditionOptions,
    InitialConditionPlan,
    InitialConditionPlanBuilder,
    InitialConditionSource,
)


def _protocol(value: Any, method: str, *, where: str) -> Any:
    member = getattr(value, method, None)
    if isinstance(value, (str, bytes)) or callable(value) or not callable(member):
        raise TypeError(
            "%s must implement the data-only %s() protocol; strings and callbacks are forbidden"
            % (where, method))
    return member


def _deterministic_data_projection(
    value: Any,
    method: str,
    *,
    where: str,
) -> tuple[dict[str, Any], str]:
    """Authenticate one strict JSON mapping returned by an extension protocol."""
    projection = _protocol(value, method, where=where)
    first, second = projection(), projection()
    encoded = []
    for result in (first, second):
        if type(result) is not dict:
            raise TypeError("%s %s() must return an exact dict" % (where, method))
        try:
            payload = json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "%s %s() must return strict JSON data" % (where, method)) from exc
        if json.loads(payload) != result:
            raise TypeError(
                "%s %s() must return strict JSON data" % (where, method))
        encoded.append(payload)
    if encoded[0] != encoded[1]:
        raise TypeError("%s %s() must be deterministic" % (where, method))
    return first, encoded[0]


def _freeze_captured_json(value: Any) -> Any:
    """Deep-freeze data already authenticated by strict JSON round-tripping."""

    if value is None or type(value) in (bool, int, str):
        return ("scalar", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("captured initial-condition data must be finite")
        return ("float", value.hex())
    if type(value) is list:
        return ("list", tuple(_freeze_captured_json(item) for item in value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(
                "captured initial-condition mappings require exact string keys")
        return (
            "dict",
            tuple(
                (key, _freeze_captured_json(value[key])) for key in sorted(value)
            ),
        )
    raise TypeError("captured initial-condition data must contain only strict JSON values")


def _thaw_captured_json(value: Any) -> Any:
    tag, payload = value
    if tag == "scalar":
        return payload
    if tag == "float":
        return float.fromhex(payload)
    if tag == "list":
        return [_thaw_captured_json(item) for item in payload]
    if tag == "dict":
        return {key: _thaw_captured_json(item) for key, item in payload}
    raise ValueError("invalid captured initial-condition data tag")


def _bootstrap_phase_order(projection: Any) -> tuple[str, ...]:
    phases = getattr(projection, "bootstrap_phases", None)
    if type(phases) is not tuple:
        raise TypeError(
            "InitialCondition.projection must declare exact tuple bootstrap_phases")
    if len(phases) != 3 or any(type(phase) is not str for phase in phases) \
            or set(phases) != {"transfer", "projection", "constraint"}:
        raise ValueError(
            "InitialCondition.projection bootstrap_phases must explicitly order "
            "transfer, projection, constraint")
    return phases


def _contains_authoring_handle(value: Any) -> bool:
    """Return whether strict JSON contains an unresolved Handle inspection payload."""

    stack = [value]
    required = {
        "kind", "local_id", "owner_path", "ownership_phase", "qualified_id",
        "schema_version",
    }
    while stack:
        item = stack.pop()
        if type(item) is dict:
            if required.issubset(item) and item.get("ownership_phase") == "authoring":
                return True
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
    return False


def _captured_resolution_protocol(
    value: Any,
    *,
    value_identity: dict[str, Any],
    source_options: dict[str, Any],
) -> tuple[type[Any] | None, tuple[Handle, ...]]:
    """Capture an optional data-only reference transformer and its immutable authorities."""

    capture = getattr(value, "captured_reference_handles", None)
    transform = getattr(type(value), "resolve_captured_references", None)
    legacy = getattr(value, "resolve_references", None)
    if capture is None and transform is None:
        if legacy is not None:
            raise TypeError(
                "InitialCondition.value resolve_references() is unsafe after capture; providers "
                "with references must implement captured_reference_handles() and the classmethod "
                "resolve_captured_references(...)"
            )
        if _contains_authoring_handle(value_identity) \
                or _contains_authoring_handle(source_options):
            raise TypeError(
                "InitialCondition.value contains authoring Handles but exposes no data-only "
                "captured-reference resolution protocol"
            )
        return None, ()
    if not callable(capture) or not callable(transform):
        raise TypeError(
            "InitialCondition.value captured-reference resolution requires both "
            "captured_reference_handles() and classmethod resolve_captured_references(...)"
        )
    first = capture()
    second = capture()
    if type(first) is not tuple or type(second) is not tuple or first != second:
        raise TypeError(
            "InitialCondition.value captured_reference_handles() must return one deterministic "
            "exact tuple"
        )
    if any(not isinstance(reference, Handle) for reference in first):
        raise TypeError(
            "InitialCondition.value captured references must contain only Handle values"
        )
    return type(value), first


def _resolved_value_snapshot(
    provider_type: type[Any],
    *,
    value_identity: dict[str, Any],
    source_options: dict[str, Any],
    references: tuple[Handle, ...],
    resolver: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one provider class protocol twice over detached data and authenticate its result."""

    transform = getattr(provider_type, "resolve_captured_references", None)
    if not callable(transform):
        raise TypeError(
            "captured initial provider type has no resolve_captured_references class protocol"
        )
    results = []
    frozen_results = []
    for _ in range(2):
        result = transform(
            value_identity=_thaw_captured_json(_freeze_captured_json(value_identity)),
            source_options=_thaw_captured_json(_freeze_captured_json(source_options)),
            references=references,
            resolver=resolver,
        )
        if type(result) is not dict or set(result) != {"value_identity", "source_options"} \
                or type(result["value_identity"]) is not dict \
                or type(result["source_options"]) is not dict:
            raise TypeError(
                "resolve_captured_references() must return exact value_identity/source_options "
                "dicts"
            )
        try:
            frozen = _freeze_captured_json(result)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "resolve_captured_references() must return strict JSON data"
            ) from exc
        results.append(result)
        frozen_results.append(frozen)
    if frozen_results[0] != frozen_results[1]:
        raise TypeError("resolve_captured_references() must be deterministic")
    result = results[0]
    if _contains_authoring_handle(result):
        raise ValueError(
            "resolve_captured_references() left an authoring Handle in resolved data"
        )
    return result["value_identity"], result["source_options"]


@dataclass(frozen=True, slots=True)
class _CapturedInitialValue:
    """Read-only presentation of a provider snapshot on a resolved condition."""

    identity_data: Any
    source_options_data: Any
    reprojectable: bool

    def to_data(self) -> dict[str, Any]:
        return _thaw_captured_json(self.identity_data)

    canonical_identity = to_data
    inspect = to_data

    def initial_source_options(self) -> dict[str, Any]:
        return _thaw_captured_json(self.source_options_data)


@dataclass(frozen=True, slots=True)
class _CapturedInitialProjection:
    """Read-only presentation of a projection snapshot on a resolved condition."""

    identity_data: Any
    options_data: Any
    bootstrap_phases: tuple[str, ...]

    def to_data(self) -> dict[str, Any]:
        return _thaw_captured_json(self.identity_data)

    canonical_identity = to_data
    inspect = to_data

    def initial_projection_options(self) -> dict[str, Any]:
        return _thaw_captured_json(self.options_data)


@dataclass(frozen=True, slots=True)
class InitialCondition:
    """One qualified physical state, one data provider and one projection authority."""

    state: Handle
    value: Any
    projection: Any
    _source_options_data: Any = field(init=False, repr=False, compare=False)
    _projection_options_data: Any = field(init=False, repr=False, compare=False)
    _value_identity_data: Any = field(init=False, repr=False, compare=False)
    _projection_identity_data: Any = field(init=False, repr=False, compare=False)
    _reprojectable: bool = field(init=False, repr=False, compare=False)
    _bootstrap_phases: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _value_snapshot_type: type[Any] | None = field(init=False, repr=False, compare=False)
    _value_reference_handles: tuple[Handle, ...] = field(
        init=False, repr=False, compare=False)
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.state, Handle) or self.state.kind != "state" \
                or not self.state.is_instance:
            raise TypeError(
                "InitialCondition.state must be a block-qualified state Handle")
        validate_value = _protocol(
            self.value, "validate_for", where="InitialCondition.value")
        validate_projection = _protocol(
            self.projection, "validate_for", where="InitialCondition.projection")
        if validate_value(self.state) is not True:
            raise TypeError("InitialCondition.value validate_for() must return exact True")
        if validate_projection(self.state, self.value) is not True:
            raise TypeError("InitialCondition.projection validate_for() must return exact True")
        source_options, _ = _deterministic_data_projection(
            self.value,
            "initial_source_options",
            where="InitialCondition.value",
        )
        projection_options, _ = _deterministic_data_projection(
            self.projection,
            "initial_projection_options",
            where="InitialCondition.projection",
        )
        _value_data, value_data_identity = _deterministic_data_projection(
            self.value, "to_data", where="InitialCondition.value")
        value_canonical, value_canonical_identity = _deterministic_data_projection(
            self.value, "canonical_identity", where="InitialCondition.value")
        _projection_data, projection_data_identity = _deterministic_data_projection(
            self.projection, "to_data", where="InitialCondition.projection")
        projection_canonical, projection_canonical_identity = _deterministic_data_projection(
            self.projection,
            "canonical_identity",
            where="InitialCondition.projection",
        )
        if value_data_identity != value_canonical_identity:
            raise ValueError(
                "InitialCondition.value canonical_identity() must match to_data()")
        if projection_data_identity != projection_canonical_identity:
            raise ValueError(
                "InitialCondition.projection canonical_identity() must match to_data()")
        native_route = source_options.get("native_route")
        if type(native_route) is not str or not native_route \
                or native_route.strip() != native_route:
            raise TypeError(
                "InitialCondition.value initial_source_options() must declare a canonical "
                "native_route")
        reprojectable = getattr(self.value, "reprojectable", None)
        if type(reprojectable) is not bool:
            raise TypeError(
                "InitialCondition.value must declare exact bool reprojectable")
        phases = _bootstrap_phase_order(self.projection)
        snapshot_type, references = _captured_resolution_protocol(
            self.value,
            value_identity=value_canonical,
            source_options=source_options,
        )
        object.__setattr__(self, "_source_options_data", _freeze_captured_json(source_options))
        object.__setattr__(
            self, "_projection_options_data", _freeze_captured_json(projection_options))
        object.__setattr__(self, "_value_identity_data", _freeze_captured_json(value_canonical))
        object.__setattr__(
            self, "_projection_identity_data", _freeze_captured_json(projection_canonical))
        object.__setattr__(self, "_reprojectable", reprojectable)
        object.__setattr__(self, "_bootstrap_phases", phases)
        object.__setattr__(self, "_value_snapshot_type", snapshot_type)
        object.__setattr__(self, "_value_reference_handles", references)

    @classmethod
    def _from_captured_snapshot(
        cls,
        *,
        state: Handle,
        source_options: dict[str, Any],
        projection_options: dict[str, Any],
        value_identity: dict[str, Any],
        projection_identity: dict[str, Any],
        reprojectable: bool,
        bootstrap_phases: tuple[str, ...],
    ) -> InitialCondition:
        """Build a resolved condition solely from already-authenticated detached data."""

        if not isinstance(state, Handle) or state.kind != "state" \
                or not state.is_instance or not state.is_resolved:
            raise TypeError(
                "captured InitialCondition requires a canonical block-qualified state Handle"
            )
        source_frozen = _freeze_captured_json(source_options)
        projection_options_frozen = _freeze_captured_json(projection_options)
        value_frozen = _freeze_captured_json(value_identity)
        projection_frozen = _freeze_captured_json(projection_identity)
        result = object.__new__(cls)
        object.__setattr__(result, "state", state)
        object.__setattr__(result, "value", _CapturedInitialValue(
            value_frozen, source_frozen, reprojectable))
        object.__setattr__(result, "projection", _CapturedInitialProjection(
            projection_frozen, projection_options_frozen, bootstrap_phases))
        object.__setattr__(result, "_source_options_data", source_frozen)
        object.__setattr__(result, "_projection_options_data", projection_options_frozen)
        object.__setattr__(result, "_value_identity_data", value_frozen)
        object.__setattr__(result, "_projection_identity_data", projection_frozen)
        object.__setattr__(result, "_reprojectable", reprojectable)
        object.__setattr__(result, "_bootstrap_phases", bootstrap_phases)
        object.__setattr__(result, "_value_snapshot_type", None)
        object.__setattr__(result, "_value_reference_handles", ())
        return result

    def resolve_references(self, resolver: Any) -> InitialCondition:
        if not callable(resolver):
            raise TypeError("InitialCondition resolver must be callable")
        state = resolver(self.state)
        if not isinstance(state, Handle):
            raise TypeError("InitialCondition resolver must return a state Handle")
        source_options = _thaw_captured_json(self._source_options_data)
        value_identity = _thaw_captured_json(self._value_identity_data)
        if self._value_snapshot_type is not None and self._value_reference_handles:
            value_identity, source_options = _resolved_value_snapshot(
                self._value_snapshot_type,
                value_identity=value_identity,
                source_options=source_options,
                references=self._value_reference_handles,
                resolver=resolver,
            )
        elif _contains_authoring_handle(value_identity) \
                or _contains_authoring_handle(source_options):
            raise ValueError(
                "captured InitialCondition contains unresolved Handles without captured "
                "reference authorities"
            )
        return type(self)._from_captured_snapshot(
            state=state,
            source_options=source_options,
            projection_options=_thaw_captured_json(self._projection_options_data),
            value_identity=value_identity,
            projection_identity=_thaw_captured_json(self._projection_identity_data),
            reprojectable=self._reprojectable,
            bootstrap_phases=self._bootstrap_phases,
        )

    def canonical_identity(self) -> dict[str, Any]:
        if not self.state.is_resolved:
            raise TypeError(
                "InitialCondition canonical identity requires a resolved qualified state")
        return {
            "schema_version": 1,
            "state": self.state.canonical_identity(),
            "value": _thaw_captured_json(self._value_identity_data),
            "projection": _thaw_captured_json(self._projection_identity_data),
        }

    @property
    def qualified_id(self) -> str:
        payload = json.dumps(
            self.canonical_identity(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "pops.initial.v1::sha256:%s" % hashlib.sha256(payload.encode()).hexdigest()

    def source(self, owner: Any) -> Any:
        """Lower a resolved declaration to the layout-generic source contract."""
        if not self.state.is_resolved:
            raise TypeError("InitialCondition.source requires a resolved qualified state")

        source_options = _thaw_captured_json(self._source_options_data)
        projection_options = _thaw_captured_json(self._projection_options_data)
        overlap = set(source_options).intersection(projection_options)
        if overlap:
            raise ValueError(
                "initial value and projection options collide: %s" % sorted(overlap))
        options = {**source_options, **projection_options}
        if not isinstance(options.get("native_route"), str):
            raise TypeError("initial value protocol must declare a native_route")
        provider = Handle(
            "source_%s" % self.qualified_id.rsplit(":", 1)[-1],
            kind="initial_condition_provider",
            owner=OwnerPath.coerce(owner).canonical(),
        )
        return InitialConditionSource(provider, InitialConditionOptions(options))

    def bootstrap_method(self) -> Any:
        if self._reprojectable is True:
            from pops.mesh._amr import AnalyticReprojection

            return AnalyticReprojection()
        if self._reprojectable is False:
            from pops.mesh._amr import ProlongFromParent

            return ProlongFromParent()
        raise TypeError("initial value protocol must declare exact bool reprojectable")

    @property
    def bootstrap_phases(self) -> tuple[str, ...]:
        return self._bootstrap_phases

    def inspect(self) -> dict[str, Any]:
        return {
            "state": self.state.inspect(),
            "value": _thaw_captured_json(self._value_identity_data),
            "projection": _thaw_captured_json(self._projection_identity_data),
        }


@dataclass(frozen=True, slots=True)
class InitialConditionAuthorities:
    """The exact initial and bootstrap authorities generated from one Case registry."""

    initial_condition_plan: Any
    bootstrap_plan: Any

    def __post_init__(self) -> None:
        from pops.mesh._amr import BootstrapPlan

        if type(self.initial_condition_plan) is not InitialConditionPlan:
            raise TypeError("initial_condition_plan must be an exact InitialConditionPlan")
        if type(self.bootstrap_plan) is not BootstrapPlan:
            raise TypeError("bootstrap_plan must be an exact BootstrapPlan")


__all__ = [
    "InitialCondition",
    "InitialConditionAuthorities",
    "InitialConditionBinding",
    "InitialConditionOptions",
    "InitialConditionPlan",
    "InitialConditionPlanBuilder",
    "InitialConditionSource",
]
