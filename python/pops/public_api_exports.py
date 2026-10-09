"""Version 1 public compatibility aliases to canonical Python library classes.

This module routes public API names only. It accepts no physical expressions,
state or method descriptors, and never constructs or selects numerical recipes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

PUBLIC_LIBRARY_ALIAS_VERSION = 1


class PublicLibraryExportKind(str, Enum):
    CLASS = "python_library_class"


@dataclass(frozen=True, slots=True)
class PublicLibraryAlias:
    public_module: str
    public_name: str
    canonical_module: str
    canonical_name: str
    source_path: str
    contract_version: int = PUBLIC_LIBRARY_ALIAS_VERSION
    kind: PublicLibraryExportKind = PublicLibraryExportKind.CLASS

    def __post_init__(self) -> None:
        for module in (self.public_module, self.canonical_module):
            if (not isinstance(module, str) or not module.startswith("pops.")
                    or not all(part.isidentifier() for part in module.split("."))):
                raise ValueError("alias modules must be absolute pops module names")
        for name in (self.public_name, self.canonical_name):
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("alias names must be Python identifiers")
        if self.canonical_module.split(".")[1] not in {"moments", "physics", "lib"}:
            raise ValueError("canonical alias owner must be a Python library layer")
        if self.public_module == self.canonical_module:
            raise ValueError("compatibility alias must name a distinct public module")
        expected = "python/" + self.canonical_module.replace(".", "/") + ".py"
        if (not isinstance(self.source_path, str) or self.source_path != expected
                or str(PurePosixPath(self.source_path)) != self.source_path):
            raise ValueError("alias source_path must identify its canonical module")
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("unsupported public library alias contract version")
        if self.kind is not PublicLibraryExportKind.CLASS:
            raise ValueError("public library alias must route a class")


PUBLIC_LIBRARY_ALIASES: Mapping[tuple[str, str], PublicLibraryAlias] = MappingProxyType({
    ("pops.numerics", "FanLi15RawMomentPath"): PublicLibraryAlias(
        public_module="pops.numerics",
        public_name="FanLi15RawMomentPath",
        canonical_module="pops.moments.fan_li_path",
        canonical_name="FanLi15RawMomentPath",
        source_path="python/pops/moments/fan_li_path.py",
    ),
})


def resolve_public_library_alias(public_module: str, name: str) -> type:
    """Return the registered owner's existing class, without constructing it."""
    if type(public_module) is not str or type(name) is not str:
        raise AttributeError("public library alias lookup requires module/name strings")
    try:
        alias = PUBLIC_LIBRARY_ALIASES[(public_module, name)]
    except (KeyError, TypeError):
        raise AttributeError(f"module {public_module!r} has no attribute {name!r}") from None
    if ((alias.public_module, alias.public_name) != (public_module, name)
            or type(alias) is not PublicLibraryAlias):
        raise RuntimeError("public library alias registry key/owner mismatch")
    alias.__post_init__()
    module = import_module(alias.canonical_module)
    value = vars(module).get(alias.canonical_name)
    # Installed wheels and Source trees retain the same canonical pops/... source suffix.
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    suffix = alias.source_path.removeprefix("python/")
    if (not isinstance(origin, str) or not PurePosixPath(origin).as_posix().endswith("/" + suffix)
            or not isinstance(value, type) or value.__module__ != alias.canonical_module
            or value.__name__ != alias.canonical_name):
        raise RuntimeError("public library alias canonical class/source owner mismatch")
    return value
