"""Native providers for hierarchy-scoped mathematical solves."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HierarchySolveProvider:
    """Immutable extension interface implemented by hierarchy solve providers."""

    provider_id: str
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("HierarchySolveProvider.provider_id must be a non-empty string")
        if not isinstance(self.capabilities, frozenset):
            raise TypeError("HierarchySolveProvider.capabilities must be a frozenset")


class CompositeTensorFAC(HierarchySolveProvider):
    """Composite FAC provider for a per-level tensor elliptic assembly."""

    def __init__(self) -> None:
        super().__init__("composite_tensor_fac", frozenset({"amr_hierarchy", "tensor_elliptic"}))


def hierarchy_provider_id(provider: object | None) -> str | None:
    if provider is None:
        return None
    if not isinstance(provider, HierarchySolveProvider):
        raise TypeError(
            "matrix_free_operator: provider must implement HierarchySolveProvider; got %r"
            % (provider,))
    if "amr_hierarchy" not in provider.capabilities:
        raise ValueError("matrix_free_operator: provider %r lacks the amr_hierarchy capability"
                         % provider.provider_id)
    return provider.provider_id


__all__ = ["HierarchySolveProvider", "CompositeTensorFAC"]
