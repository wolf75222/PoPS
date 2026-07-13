"""Native providers for hierarchy-scoped mathematical solves."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pops.identity import Identity, canonical_bytes, make_identity
from pops.solvers._numeric import exact_open_unit_real, optional_positive_int


_HIERARCHY_PROVIDER_SCHEMA_VERSION = 1


@runtime_checkable
class HierarchySolveProvider(Protocol):
    """Small structural extension contract for hierarchy-native solve providers."""

    provider_id: str
    capabilities: frozenset[str]
    __pops_ir_immutable__: bool

    def canonical_identity(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeTensorFAC:
    """Composite FAC provider for a per-level tensor elliptic assembly."""

    fine_sweeps: int | None = None
    coarse_rel_tol: Any = None
    coarse_cycles: int | None = None
    verbose: bool | None = None
    provider_id: str = field(init=False, default="composite_tensor_fac")
    capabilities: frozenset[str] = field(
        init=False, default_factory=lambda: frozenset({"amr_hierarchy", "tensor_elliptic"})
    )
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fine_sweeps",
            optional_positive_int(self.fine_sweeps, where="CompositeTensorFAC(fine_sweeps=)"),
        )
        object.__setattr__(
            self,
            "coarse_cycles",
            optional_positive_int(self.coarse_cycles, where="CompositeTensorFAC(coarse_cycles=)"),
        )
        if self.coarse_rel_tol is not None:
            object.__setattr__(
                self,
                "coarse_rel_tol",
                exact_open_unit_real(
                    self.coarse_rel_tol, where="CompositeTensorFAC(coarse_rel_tol=)"
                ),
            )
        if self.verbose is not None and type(self.verbose) is not bool:
            raise TypeError(
                "CompositeTensorFAC(verbose=) must be a Python bool or None (got %r)"
                % (self.verbose,)
            )

    def canonical_identity(self) -> dict[str, Any]:
        # Lazy by design: pops.solvers remains an import-graph sink and does not import pops.ir at
        # module scope merely because the provider catalog is imported.
        from pops.ir.literals import scalar_data

        return {
            "schema_version": _HIERARCHY_PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "capabilities": sorted(self.capabilities),
            "options": {
                "fine_sweeps": self.fine_sweeps,
                "coarse_rel_tol": (
                    None if self.coarse_rel_tol is None else scalar_data(self.coarse_rel_tol)
                ),
                "coarse_cycles": self.coarse_cycles,
                "verbose": self.verbose,
            },
        }

    def to_data(self) -> dict[str, Any]:
        return self.canonical_identity()

    @property
    def identity(self) -> Identity:
        return make_identity("hierarchy-solve-provider", self.canonical_identity())


def hierarchy_provider_data(provider: object | None) -> dict[str, Any] | None:
    """Authenticate one hierarchy provider and detach its canonical Program-IR identity."""
    if provider is None:
        return None
    if not isinstance(provider, HierarchySolveProvider):
        raise TypeError(
            "matrix_free_operator: provider must implement HierarchySolveProvider; got %r"
            % (provider,)
        )
    provider_id = provider.provider_id
    capabilities = provider.capabilities
    if not isinstance(provider_id, str) or not provider_id or provider_id.strip() != provider_id:
        raise TypeError("matrix_free_operator: provider_id must be canonical non-empty text")
    if not isinstance(capabilities, frozenset) or any(
        not isinstance(item, str) or not item or item.strip() != item for item in capabilities
    ):
        raise TypeError(
            "matrix_free_operator: provider capabilities must be a frozenset of canonical text"
        )
    if "amr_hierarchy" not in capabilities:
        raise ValueError(
            "matrix_free_operator: provider %r lacks the amr_hierarchy capability" % provider_id
        )
    if getattr(provider, "__pops_ir_immutable__", False) is not True:
        raise TypeError("matrix_free_operator: hierarchy provider must declare immutable IR state")
    first, second = provider.canonical_identity(), provider.canonical_identity()
    expected = {"schema_version", "provider_id", "capabilities", "options"}
    if (
        type(first) is not dict
        or type(second) is not dict
        or first != second
        or set(first) != expected
    ):
        raise TypeError(
            "matrix_free_operator: hierarchy provider canonical_identity() must return one "
            "deterministic v1 dict"
        )
    if first["schema_version"] != _HIERARCHY_PROVIDER_SCHEMA_VERSION:
        raise ValueError("matrix_free_operator: unsupported hierarchy provider identity schema")
    if first["provider_id"] != provider_id:
        raise ValueError(
            "matrix_free_operator: hierarchy provider identity disagrees with provider_id"
        )
    if first["capabilities"] != sorted(capabilities):
        raise ValueError(
            "matrix_free_operator: hierarchy provider identity disagrees with capabilities"
        )
    if type(first["options"]) is not dict:
        raise TypeError("matrix_free_operator: hierarchy provider options must be an exact dict")
    canonical_bytes(first)
    # Detach nested provider-owned containers before storing them in the mutable authoring IR.
    # ProgramGraph performs its own immutable snapshot later, but the authoring identity must not
    # retain aliases to a third-party descriptor either.
    return deepcopy(first)


def hierarchy_provider_id(provider: object | None) -> str | None:
    data = hierarchy_provider_data(provider)
    return None if data is None else data["provider_id"]


__all__ = ["HierarchySolveProvider", "CompositeTensorFAC"]
