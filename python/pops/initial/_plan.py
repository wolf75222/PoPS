"""Layout-generic resolved initial-condition contracts.

This module deliberately knows nothing about AMR transfers or hierarchy bootstrap.  It
authenticates only three facts: the physical subject, its resolved layout and the data-only
provider that initializes it.  Adaptive bootstrap composes this plan with transfer authorities
later, without making either authority part of the other's identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Protocol, cast

from pops.identity import Identity, make_identity
from pops.model import Handle


class _LayoutPlanContract(Protocol):
    """Small immutable layout authority consumed by initial-condition resolution."""

    @property
    def qualified_id(self) -> str: ...

    def canonical_identity(self) -> dict[str, Any]: ...

    def layout_for(self, subject: Any) -> Handle: ...

    def normalized(self, layout: Any) -> Any: ...


def _layout_plan(value: Any) -> _LayoutPlanContract:
    """Authenticate the data and lookup protocol without depending on mesh implementation."""

    qualified_id = getattr(value, "qualified_id", None)
    canonical_identity = getattr(value, "canonical_identity", None)
    layout_for = getattr(value, "layout_for", None)
    normalized = getattr(value, "normalized", None)
    if not isinstance(qualified_id, str) or not qualified_id \
            or not all(callable(member) for member in (canonical_identity, layout_for, normalized)):
        raise TypeError(
            "InitialConditionPlanBuilder requires an immutable layout-plan authority "
            "exposing qualified_id, canonical_identity(), layout_for(), and normalized()"
        )
    plan = cast(_LayoutPlanContract, value)
    identity = plan.canonical_identity()
    if not isinstance(identity, Mapping) \
            or identity.get("report_type") != "layout_plan" \
            or identity.get("qualified_id") != qualified_id:
        raise ValueError(
            "initial-condition layout-plan authority has an unauthenticated canonical identity"
        )
    return plan


def _canonical_handle(value: Any, *, where: str) -> Any:
    """Require and round-trip one real immutable owner-qualified Handle."""

    if not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    data = value.canonical_identity()
    rebuilt = Handle.from_canonical_identity(data)
    if rebuilt.canonical_identity() != data or rebuilt != value or hash(rebuilt) != hash(value):
        raise ValueError("%s Handle canonical identity does not round-trip" % where)
    return value


def _layout_handle(value: Any, *, where: str) -> Handle:
    """Require a canonical layout identity through the generic Handle contract."""

    value = _canonical_handle(value, where=where)
    if value.kind != "layout":
        raise TypeError("%s requires a canonical Handle with kind='layout'" % where)
    return value


def _freeze_options(value: Any, *, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return ("scalar", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite float" % where)
        return ("binary64", value.hex())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return (
            "mapping",
            tuple(
                (key, _freeze_options(value[key], where="%s.%s" % (where, key)))
                for key in sorted(value)
            ),
        )
    if isinstance(value, (tuple, list)):
        return (
            "sequence",
            tuple(_freeze_options(item, where="%s[]" % where) for item in value),
        )
    raise TypeError(
        "%s contains non-data value %s; callbacks and opaque objects are forbidden"
        % (where, type(value).__name__)
    )


def _thaw_options(value: Any) -> Any:
    tag, payload = value
    if tag == "scalar":
        return payload
    if tag == "binary64":
        return {"binary64": payload}
    if tag == "mapping":
        return {key: _thaw_options(item) for key, item in payload}
    if tag == "sequence":
        return [_thaw_options(item) for item in payload]
    raise ValueError("invalid frozen initial-condition option tag")


@dataclass(frozen=True, slots=True, init=False)
class InitialConditionOptions:
    """Deeply immutable data-only options for one initial-condition provider."""

    _data: Any
    __pops_ir_immutable__ = True

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        values = {} if values is None else values
        if not isinstance(values, Mapping):
            raise TypeError("InitialConditionOptions requires a mapping")
        object.__setattr__(
            self,
            "_data",
            _freeze_options(values, where="InitialConditionOptions"),
        )

    def to_data(self) -> dict[str, Any]:
        return _thaw_options(self._data)


@dataclass(frozen=True, slots=True)
class InitialConditionSource:
    provider: Any
    options: InitialConditionOptions = InitialConditionOptions()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops.model import Handle

        provider = _canonical_handle(
            self.provider, where="InitialConditionSource.provider")
        if type(provider) is not Handle or provider.kind != "initial_condition_provider":
            raise TypeError(
                "InitialConditionSource.provider requires "
                "an exact Handle with kind='initial_condition_provider'"
            )
        options = self.options
        if type(options) is not InitialConditionOptions:
            to_data = getattr(options, "to_data", None)
            if not callable(to_data):
                raise TypeError(
                    "InitialConditionSource.options must be InitialConditionOptions or expose "
                    "a data-only to_data() projection"
                )
            options = InitialConditionOptions(
                cast(Mapping[str, Any], to_data())
            )
            object.__setattr__(self, "options", options)

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "provider": self.provider.canonical_identity(),
            "options": self.options.to_data(),
        }


@dataclass(frozen=True, slots=True)
class InitialConditionBinding:
    subject: Any
    layout: Handle
    source: InitialConditionSource
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _canonical_handle(self.subject, where="InitialConditionBinding.subject")
        if self.subject.kind not in {"state", "particle"}:
            raise ValueError(
                "initial conditions may target only physical state/particle Handles")
        _layout_handle(self.layout, where="InitialConditionBinding.layout")
        if type(self.source) is not InitialConditionSource:
            raise TypeError("InitialConditionBinding.source must be InitialConditionSource")

    def to_data(self) -> dict[str, Any]:
        return {
            "subject": self.subject.canonical_identity(),
            "layout": self.layout.canonical_identity(),
            "source": self.source.canonical_identity(),
        }


@dataclass(frozen=True, slots=True)
class InitialConditionPlan:
    layout_plan_id: str
    bindings: tuple[InitialConditionBinding, ...]
    authoring_aliases: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout_plan_id, str) or not self.layout_plan_id:
            raise TypeError("InitialConditionPlan.layout_plan_id must be non-empty")
        bindings = tuple(self.bindings)
        if not bindings or any(type(row) is not InitialConditionBinding for row in bindings):
            raise TypeError("InitialConditionPlan.bindings must contain bindings")
        subjects = [row.subject.qualified_id for row in bindings]
        if len(subjects) != len(set(subjects)):
            raise ValueError("InitialConditionPlan contains duplicate subjects")
        object.__setattr__(self, "bindings", bindings)

        by_id = {row.subject.qualified_id: row.subject for row in bindings}
        if not isinstance(self.authoring_aliases, Mapping):
            raise TypeError("InitialConditionPlan authoring_aliases must be a mapping")
        aliases = {}
        for alias_qid, target in self.authoring_aliases.items():
            if not isinstance(alias_qid, str) or not alias_qid:
                raise TypeError(
                    "InitialConditionPlan authoring alias keys must be non-empty strings")
            expected = by_id.get(getattr(target, "qualified_id", None))
            if expected is None \
                    or target.canonical_identity() != expected.canonical_identity():
                raise ValueError(
                    "InitialConditionPlan authoring alias targets an unknown canonical subject")
            previous = aliases.get(alias_qid)
            if previous is not None and previous != expected:
                raise ValueError(
                    "InitialConditionPlan authoring alias resolves to multiple subjects")
            aliases[alias_qid] = expected
        object.__setattr__(self, "authoring_aliases", MappingProxyType(aliases))

    @property
    def identity(self) -> Identity:
        return make_identity("initial-condition-plan", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "layout_plan_id": self.layout_plan_id,
            "bindings": [row.to_data() for row in self.bindings],
        }

    def canonical_subject(self, handle: Any) -> Any:
        """Authenticate a canonical subject or a live alias issued by the originating Case."""
        from pops.model import Handle

        if not isinstance(handle, Handle) or not handle.is_instance:
            raise TypeError(
                "InitialConditionPlan values require block-qualified Handle keys")
        if handle.is_resolved:
            by_id = {row.subject.qualified_id: row.subject for row in self.bindings}
            subject = by_id.get(handle.qualified_id)
            if subject is not None \
                    and handle.canonical_identity() == subject.canonical_identity():
                return subject
        else:
            subject = self.authoring_aliases.get(handle.qualified_id)
            if subject is not None:
                return subject
        raise KeyError(
            "Handle %s is not an authenticated subject or authoring alias of this "
            "InitialConditionPlan" % handle.qualified_id
        )


class InitialConditionPlanBuilder:
    """Resolve exact initialization coverage against any immutable LayoutPlan."""

    def __init__(
        self, layout_plan: _LayoutPlanContract, expected_subjects: Any
    ) -> None:
        layout_plan = _layout_plan(layout_plan)
        if isinstance(expected_subjects, (str, bytes, Mapping)):
            raise TypeError(
                "InitialConditionPlanBuilder expected_subjects must be a finite Handle iterable")
        try:
            subjects = tuple(expected_subjects)
        except TypeError as exc:
            raise TypeError(
                "InitialConditionPlanBuilder expected_subjects must be a finite Handle iterable"
            ) from exc
        if not subjects:
            raise ValueError("InitialConditionPlan requires at least one physical subject")
        expected = {}
        for subject in subjects:
            subject = _canonical_handle(
                subject, where="InitialConditionPlanBuilder expected subject")
            if subject.kind not in {"state", "particle"}:
                raise ValueError(
                    "initial conditions may target only physical state/particle Handles")
            if subject.qualified_id in expected:
                raise ValueError(
                    "InitialConditionPlan expected_subjects contains duplicate %s"
                    % subject.qualified_id)
            expected[subject.qualified_id] = subject
        self._layout_plan = layout_plan
        self._expected = expected
        self._bindings: dict[str, InitialConditionBinding] = {}
        self._aliases: dict[str, Any] = {}

    def add(
        self,
        subject: Any,
        source: InitialConditionSource,
        *,
        layout: Handle | None = None,
        authoring_alias: Any = None,
    ) -> InitialConditionBinding:
        subject = _canonical_handle(
            subject, where="InitialConditionPlanBuilder.add subject")
        if subject.kind not in {"state", "particle"}:
            raise ValueError(
                "initial conditions may target only physical state/particle Handles")
        expected = self._expected.get(subject.qualified_id)
        if expected is None \
                or subject.canonical_identity() != expected.canonical_identity():
            raise ValueError(
                "initial conditions may target only declared physical plan subjects")
        try:
            assigned_layout = self._layout_plan.layout_for(subject)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "initial subjects require one exact LayoutPlan assignment"
            ) from exc
        if layout is None:
            layout = assigned_layout
        else:
            layout = _layout_handle(
                layout, where="InitialConditionPlanBuilder.add layout")
        if layout != assigned_layout:
            raise ValueError(
                "initial-condition layout differs from the subject's LayoutPlan assignment"
            )
        normalized = self._layout_plan.normalized(layout)
        source_data = source.options.to_data()
        source_frame = source_data.get("frame_id")
        if source_frame is not None:
            if not isinstance(source_frame, str) or not source_frame:
                raise TypeError("initial-condition source frame_id must be non-empty text")
            layout_frame = normalized.geometry.frame_id
            if layout_frame is None:
                raise ValueError(
                    "initial-condition source is frame-bound but its assigned layout does not "
                    "publish a normalized frame identity"
                )
            if layout_frame != source_frame:
                raise ValueError(
                    "initial-condition source frame differs from its assigned layout frame"
                )
        binding = InitialConditionBinding(expected, layout, source)
        if expected.qualified_id in self._bindings:
            raise ValueError("duplicate initial condition for %s" % expected.qualified_id)
        self._bindings[expected.qualified_id] = binding
        if authoring_alias is not None:
            from pops.model import Handle

            if not isinstance(authoring_alias, Handle) or not authoring_alias.is_instance \
                    or authoring_alias.is_resolved:
                raise TypeError(
                    "InitialConditionPlan authoring alias must be an unresolved "
                    "block-qualified Handle")
            alias_qid = authoring_alias.qualified_id
            previous = self._aliases.get(alias_qid)
            if previous is not None and previous != expected:
                raise ValueError(
                    "InitialConditionPlan authoring alias resolves to multiple subjects")
            self._aliases[alias_qid] = expected
        return binding

    def resolve(self) -> InitialConditionPlan:
        missing = sorted(set(self._expected) - set(self._bindings))
        if missing:
            raise ValueError(
                "initial-condition plan is missing physical subjects %s" % missing)
        return InitialConditionPlan(
            self._layout_plan.qualified_id,
            tuple(self._bindings[key] for key in sorted(self._bindings)),
            self._aliases,
        )


__all__ = [
    "InitialConditionBinding",
    "InitialConditionOptions",
    "InitialConditionPlan",
    "InitialConditionPlanBuilder",
    "InitialConditionSource",
]
