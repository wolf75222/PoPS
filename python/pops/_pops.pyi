from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


__version__: str
__abi_version__: int
__cxx_std__: int
__cxx_compiler__: str
__has_kokkos__: bool
__aux_named_base__: int
__aux_max_extra__: int
__aux_base_comps__: int
__aux_max_comps__: int
__aux_canonical__: Mapping[str, int]


class SystemConfig:
    n: int
    L: float
    periodic: bool
    def __init__(self) -> None: ...


class AmrSystemConfig:
    n: int
    L: float
    periodic: bool
    max_level: int
    refine_ratio: int
    regrid_every: int
    def __init__(self) -> None: ...


class ModelSpec:
    state: str
    transport: str
    source: str
    elliptic: str
    scheme: str
    time: str
    name: str
    gamma: float
    def __init__(self) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


class System:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


class AmrSystem:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


_System = System
_AmrSystem = AmrSystem


def abi_key() -> str: ...
def set_threads(n: int) -> None: ...
def has_kokkos() -> bool: ...
def parallel_info() -> Mapping[str, Any]: ...
def my_rank() -> int: ...
def n_ranks() -> int: ...
def module_capabilities() -> Mapping[str, Any]: ...
def capability_report(target: str | None = None) -> Mapping[str, Any]: ...
def runtime_environment_report() -> Mapping[str, Any]: ...
def numerical_defaults_report() -> Mapping[str, Any]: ...
def fallback_diagnostics_report() -> Mapping[str, Any]: ...
def reset_fallback_diagnostics() -> None: ...
def inspect_amr(*args: Any, **kwargs: Any) -> Mapping[str, Any]: ...
def inspect_capabilities(*args: Any, **kwargs: Any) -> Sequence[Mapping[str, Any]]: ...
