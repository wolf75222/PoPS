"""Pre-implemented AMR transfer and materialization policies.

These are authoring descriptors.  Their order, halo and conservation capabilities are intrinsic;
users select physics-level policies and never author compiler ``AccuracyRequirement`` objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


class _ImmutableTransferPolicy:
    """Marker shared by constructor-only, frozen transfer policy values."""

    __pops_ir_immutable__: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class ConservativeLinear(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "conservative_linear"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class VolumeAverage(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "volume_average"
    order: ClassVar[int] = 1
    ghost_depth: ClassVar[tuple[int, ...]] = (0,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class ConservativeCoarseFine(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "conservative_coarse_fine"
    order: ClassVar[int] = 1
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class LinearTimeInterpolation(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "linear_time_interpolation"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (0,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class DivergencePreservingFace(_ImmutableTransferPolicy):
    """Coupled normal-face vector prolongation; never an independent scalar interpolation."""

    native_route: ClassVar[str] = "face_divergence_preserving"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = True
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class BilinearNode(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "node_bilinear"
    order: ClassVar[int] = 2
    ghost_depth: ClassVar[tuple[int, ...]] = (1,)
    dimensions: ClassVar[tuple[int, ...]] = (2,)
    refinement_ratios: ClassVar[tuple[int, ...]] = (2,)
    conservative: ClassVar[bool] = False
    temporal: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class StateTransfer(_ImmutableTransferPolicy):
    prolongation: ConservativeLinear = field(default_factory=ConservativeLinear)
    restriction: VolumeAverage = field(default_factory=VolumeAverage)
    coarse_fine: ConservativeCoarseFine = field(default_factory=ConservativeCoarseFine)
    time_interpolation: LinearTimeInterpolation = field(
        default_factory=LinearTimeInterpolation
    )


@dataclass(frozen=True, slots=True)
class FaceTransfer(_ImmutableTransferPolicy):
    prolongation: DivergencePreservingFace = field(
        default_factory=DivergencePreservingFace
    )


@dataclass(frozen=True, slots=True)
class NodeTransfer(_ImmutableTransferPolicy):
    prolongation: BilinearNode = field(default_factory=BilinearNode)


@dataclass(frozen=True, slots=True)
class EllipticRecompute(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "elliptic_solve"


@dataclass(frozen=True, slots=True)
class PatchTopologyRebuild(_ImmutableTransferPolicy):
    native_route: ClassVar[str] = "patch_topology"


__all__ = [
    "ConservativeCoarseFine",
    "ConservativeLinear",
    "BilinearNode",
    "DivergencePreservingFace",
    "EllipticRecompute",
    "FaceTransfer",
    "LinearTimeInterpolation",
    "NodeTransfer",
    "PatchTopologyRebuild",
    "StateTransfer",
    "VolumeAverage",
]
