"""Typed physical roles for conservative-state components."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_ROLE_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_ROLE_TOKENS = frozenset({"Custom"})


def native_role_token(role: Any) -> str:
    """Validate the exact lowering/identity token of a role descriptor."""
    if not isinstance(role, ComponentRole):
        raise TypeError("state role must implement ComponentRole")
    token = role.native_name
    if not isinstance(token, str) or not token:
        raise TypeError("ComponentRole.native_name must be a non-empty string")
    if _ROLE_TOKEN.fullmatch(token) is None:
        raise ValueError(
            "ComponentRole.native_name must be one canonical C++ role token; got %r" % token)
    if token in _RESERVED_ROLE_TOKENS:
        raise ValueError("ComponentRole.native_name %r is reserved by the native ABI" % token)
    return token


class ComponentRole:
    """Closed protocol translated to the native role vocabulary at the IR boundary."""

    __slots__ = ()

    @property
    def native_name(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Density(ComponentRole):
    @property
    def native_name(self) -> str:
        return "Density"


@dataclass(frozen=True, slots=True)
class Momentum(ComponentRole):
    axis: Any

    def __post_init__(self) -> None:
        name = getattr(self.axis, "name", None)
        if name not in ("x", "y", "z"):
            raise TypeError("Momentum axis must be a typed Cartesian x/y/z axis")

    @property
    def native_name(self) -> str:
        return "Momentum" + str(self.axis.name).upper()


__all__ = ["ComponentRole", "Density", "Momentum", "native_role_token"]
