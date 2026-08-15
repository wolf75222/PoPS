"""Dimension-aware state-component identities shared by model, physics, and runtime.

This module is the neutral authority for the exact role vocabulary.  Physical authoring keeps its
convenient public re-exports in :mod:`pops.physics.roles`, while lower layers can consume the
identity protocol without importing the physics package.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_AXIS_ROLE = re.compile(r"^(momentum|velocity|axial):(0|[1-9][0-9]*)$")
_SCALAR_FAMILIES = frozenset({"density", "energy", "pressure", "scalar", "temperature"})
_AXIS_FAMILIES = frozenset({"momentum", "velocity", "axial"})
_PHYSICAL_FAMILIES = _SCALAR_FAMILIES | _AXIS_FAMILIES


def _exact_role_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError("%s must be an exact role string" % where)
    if not value or value != value.strip() or "\x00" in value or "," in value:
        raise ValueError(
            "%s must be a non-empty exact role string without commas or NUL bytes" % where
        )
    return value


def _dimension(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError("%s must be one of 1, 2, or 3" % where)
    return value


class ComponentRole:
    """Typed protocol translated to the native role vocabulary at the IR boundary."""

    __slots__ = ()

    @property
    def native_name(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RoleKey(ComponentRole):
    """One parsed physical or user role without conflating the two namespaces."""

    family: str
    axis: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.family == "custom":
            if self.axis is not None or self.label is None:
                raise ValueError("custom StateSchema role has an invalid exact label")
            label = _exact_role_text(self.label, where="custom StateSchema role")
            if label == "custom" or label.split(":", 1)[0] in _PHYSICAL_FAMILIES:
                raise ValueError("custom StateSchema role collides with a physical role token")
        elif (
            self.family not in _PHYSICAL_FAMILIES
            or self.label is not None
            or (self.family in _SCALAR_FAMILIES and self.axis is not None)
            or (
                self.family in _AXIS_FAMILIES
                and (isinstance(self.axis, bool) or not isinstance(self.axis, int) or self.axis < 0)
            )
        ):
            raise ValueError("RoleKey physical family/label contract is invalid")

    @property
    def token(self) -> str:
        if self.family == "custom":
            assert self.label is not None
            return self.label
        return self.family if self.axis is None else "%s:%d" % (self.family, self.axis)

    @property
    def native_name(self) -> str:
        return self.token

    @property
    def physical(self) -> bool:
        return self.family != "custom"


def parse_role(role: Any, *, dimension: Any = None, where: str = "role") -> RoleKey:
    """Parse the sole structured role vocabulary and validate its axis against ``dimension``.

    Physical roles use scalar families (``density``, ``energy``, ``pressure``, ``scalar``,
    ``temperature``) or an exact vector token (``momentum:<axis>``, ``velocity:<axis>``,
    ``axial:<axis>``).  Every other non-empty exact string is a user role label and remains in a
    separate ``custom`` namespace; malformed spellings of reserved physical families are rejected
    instead of being reinterpreted as custom roles.
    """
    role = _exact_role_text(role, where=where)
    if role in _SCALAR_FAMILIES:
        return RoleKey(role)
    if role == "custom":
        raise ValueError("%s cannot use the anonymous 'custom' role" % where)
    match = _AXIS_ROLE.fullmatch(role)
    if match is None:
        reserved = role.split(":", 1)[0]
        if reserved in _PHYSICAL_FAMILIES:
            raise ValueError("%s is a malformed reserved physical role: %r" % (where, role))
        return RoleKey("custom", label=role)
    family, axis_text = match.groups()
    axis = int(axis_text)
    if dimension is not None and axis >= _dimension(dimension, where="%s.dimension" % where):
        raise ValueError("%s axis %d is outside dimension %d" % (where, axis, dimension))
    return RoleKey(family, axis)


def native_role_token(role: Any, *, dimension: Any = None) -> str:
    """Validate the exact lowering/identity token of a role descriptor."""
    if not isinstance(role, ComponentRole):
        raise TypeError("state role must implement ComponentRole")
    token = role.native_name
    if not isinstance(token, str) or not token:
        raise TypeError("ComponentRole.native_name must be a non-empty string")
    return parse_role(token, dimension=dimension,
                      where="ComponentRole.native_name").token


@dataclass(frozen=True, slots=True)
class StateSchema:
    """Resolved conservative-state role schema for one exact Cartesian dimension."""

    dimension: int
    roles: tuple[RoleKey, ...]

    @classmethod
    def resolve(cls, roles: Any, *, dimension: Any, where: str = "state schema") -> StateSchema:
        dim = _dimension(dimension, where="%s.dimension" % where)
        if isinstance(roles, (str, bytes)):
            raise TypeError("%s roles must be an ordered iterable, not one string" % where)
        try:
            values = tuple(roles)
        except TypeError:
            raise TypeError("%s roles must be an ordered iterable" % where) from None

        def resolve_one(role: Any) -> RoleKey:
            if isinstance(role, RoleKey):
                return parse_role(role.token, dimension=dim, where=where)
            if isinstance(role, ComponentRole):
                role = native_role_token(role, dimension=dim)
            return parse_role(role, dimension=dim, where=where)

        parsed = tuple(resolve_one(role) for role in values)
        seen: dict[str, int] = {}
        for index, role in enumerate(parsed):
            previous = seen.get(role.token)
            if previous is not None:
                raise ValueError(
                    "%s declares duplicate role token %r at components %d and %d"
                    % (where, role.token, previous, index))
            seen[role.token] = index
        return cls(dim, parsed)

    def require(self, *families: str) -> None:
        invalid = [family for family in families if family not in _PHYSICAL_FAMILIES]
        if invalid:
            raise ValueError("unknown physical role families: %s" % ", ".join(invalid))
        present = {role.family for role in self.roles if role.physical}
        missing = [family for family in families if family not in present]
        if missing:
            raise ValueError("state schema is missing required role families: %s" % ", ".join(missing))

    def axes(self, family: str) -> tuple[int, ...]:
        if family not in _AXIS_FAMILIES:
            raise ValueError("%r is not an axis-bearing role family" % family)
        return tuple(sorted(role.axis for role in self.roles if role.family == family))

    def index(self, token: Any) -> int:
        """Return the unique component index for an exact physical or user role token."""
        if isinstance(token, RoleKey):
            token = token.token
        elif isinstance(token, ComponentRole):
            token = native_role_token(token, dimension=self.dimension)
        parsed = parse_role(token, dimension=self.dimension, where="StateSchema.index")
        matches = [index for index, role in enumerate(self.roles) if role.token == parsed.token]
        if not matches:
            raise ValueError("state schema does not declare role token %r" % parsed.token)
        if len(matches) != 1:
            raise ValueError("state schema role token %r is not unique" % parsed.token)
        return matches[0]


@dataclass(frozen=True, slots=True)
class CouplingBlockContract:
    """Authenticated state schema plus model/provider parameters used by coupling presets."""

    schema: StateSchema
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for name, _ in self.parameters:
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("coupling block parameters require unique non-empty identities")
            seen.add(name)

    def parameter(self, name: str) -> Any:
        matches = [value for key, value in self.parameters if key == name]
        if len(matches) != 1:
            raise ValueError("coupling block contract does not provide parameter %r" % name)
        return matches[0]


@dataclass(frozen=True, slots=True)
class Custom(ComponentRole):
    """Exact user semantic for a non-physical state component.

    The label is scoped by the owning state/block at consumption time.  It must
    not collide with a physical token or the anonymous ``custom`` sentinel.
    """

    label: str

    def __post_init__(self) -> None:
        parsed = parse_role(self.label, where="Custom.label")
        if parsed.physical or parsed.label is None:
            raise ValueError(
                "Custom.label must be an exact non-physical label distinct from 'custom'"
            )

    @property
    def native_name(self) -> str:
        return self.label


@dataclass(frozen=True, slots=True)
class Density(ComponentRole):
    @property
    def native_name(self) -> str:
        return "density"


@dataclass(frozen=True, slots=True)
class Momentum(ComponentRole):
    axis: Any

    def __post_init__(self) -> None:
        index = getattr(self.axis, "index", None)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TypeError("Momentum axis must expose one non-negative axis index")

    @property
    def native_name(self) -> str:
        return "momentum:%d" % self.axis.index


@dataclass(frozen=True, slots=True)
class Energy(ComponentRole):
    @property
    def native_name(self) -> str:
        return "energy"


@dataclass(frozen=True, slots=True)
class Velocity(ComponentRole):
    axis: Any

    def __post_init__(self) -> None:
        index = getattr(self.axis, "index", None)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TypeError("Velocity axis must expose one non-negative axis index")

    @property
    def native_name(self) -> str:
        return "velocity:%d" % self.axis.index


@dataclass(frozen=True, slots=True)
class Axial(ComponentRole):
    """One component of an axial (pseudo-)vector under reflection."""

    axis: Any

    def __post_init__(self) -> None:
        index = getattr(self.axis, "index", None)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TypeError("Axial axis must expose one non-negative axis index")

    @property
    def native_name(self) -> str:
        return "axial:%d" % self.axis.index


@dataclass(frozen=True, slots=True)
class Pressure(ComponentRole):
    @property
    def native_name(self) -> str:
        return "pressure"


@dataclass(frozen=True, slots=True)
class Temperature(ComponentRole):
    @property
    def native_name(self) -> str:
        return "temperature"


@dataclass(frozen=True, slots=True)
class Scalar(ComponentRole):
    @property
    def native_name(self) -> str:
        return "scalar"


__all__ = [
    "Axial", "ComponentRole", "CouplingBlockContract", "Custom", "Density", "Energy", "Momentum",
    "Pressure", "Scalar", "Temperature", "Velocity", "RoleKey", "StateSchema",
    "native_role_token", "parse_role",
]
