"""Typed physical roles for conservative-state components."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_ROLE_TOKEN = re.compile(r"^(?:[a-z]+|(?:momentum|velocity|axial):[0-9]+)$")
_CANONICAL_ROLE_TOKENS = frozenset({
    "density", "energy", "pressure", "scalar", "temperature",
})


def native_role_token(role: Any) -> str:
    """Validate the exact lowering/identity token of a role descriptor."""
    if not isinstance(role, ComponentRole):
        raise TypeError("state role must implement ComponentRole")
    token = role.native_name
    if not isinstance(token, str) or not token:
        raise TypeError("ComponentRole.native_name must be a non-empty string")
    if _ROLE_TOKEN.fullmatch(token) is None:
        raise ValueError(
            "ComponentRole.native_name must be a structured native semantic token; got %r" % token)
    if token not in _CANONICAL_ROLE_TOKENS and not token.startswith(
            ("momentum:", "velocity:", "axial:")):
        raise ValueError(
            "ComponentRole.native_name %r is not implemented by the installed native role ABI"
            % token)
    return token


class ComponentRole:
    """Typed protocol translated to the native role vocabulary at the IR boundary."""

    __slots__ = ()

    @property
    def native_name(self) -> str:
        raise NotImplementedError


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
        if not isinstance(index, int) or index < 0:
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
        if not isinstance(index, int) or index < 0:
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
        if not isinstance(index, int) or index < 0:
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
    "Axial", "ComponentRole", "Density", "Energy", "Momentum", "Pressure", "Scalar",
    "Temperature", "Velocity", "native_role_token",
]
