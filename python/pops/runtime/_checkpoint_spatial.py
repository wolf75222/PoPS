"""Rank-generic spatial authority for strict accepted-state checkpoints.

The authoring dimension is resolved before native installation.  Checkpoint/restart therefore
persists and authenticates that exact decision instead of inferring a rank from an array shape or
reconstructing a two-dimensional geometry from scalar compatibility fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import sys
from typing import Any


from pops._generated_release_contract import CHECKPOINT_SPATIAL_SCHEMA_VERSION


SPATIAL_CONTRACT_KEY = "pops_spatial_contract"
SPATIAL_CONTRACT_SCHEMA_VERSION = CHECKPOINT_SPATIAL_SCHEMA_VERSION

_LEGACY_SPATIAL_KEYS = frozenset({"nx", "ny", "n", "L", "Ly", "xlo", "ylo"})


def _exact_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("checkpoint spatial dimension must be an exact integer")
    if value not in (1, 2, 3):
        raise ValueError("checkpoint spatial dimension must be 1, 2, or 3")
    return value


def _shape(values: Any, *, dimension: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("checkpoint spatial shape must be an integer vector")
    result = tuple(values)
    if len(result) != dimension:
        raise ValueError("checkpoint spatial shape length must equal its dimension")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError("checkpoint spatial shape must contain exact integers")
    if any(value < 1 for value in result):
        raise ValueError("checkpoint spatial shape must be strictly positive on every axis")
    cell_count(result)
    return result


def _bounds(values: Any, *, dimension: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("checkpoint spatial %s must be a floating vector" % name)
    result = tuple(values)
    if len(result) != dimension:
        raise ValueError("checkpoint spatial %s length must equal its dimension" % name)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in result):
        raise TypeError("checkpoint spatial %s must contain real scalars" % name)
    normalized = tuple(float(value) for value in result)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("checkpoint spatial %s must contain finite values" % name)
    return normalized


def _periodicity(values: Any, *, dimension: int) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("checkpoint spatial periodicity must be a boolean vector")
    result = tuple(values)
    if len(result) != dimension:
        raise ValueError("checkpoint spatial periodicity length must equal its dimension")
    if any(type(value) is not bool for value in result):
        raise TypeError("checkpoint spatial periodicity must contain exact bool values")
    return result


def _refinement_ratios(values: Any, *, dimension: int) -> tuple[tuple[int, ...], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("checkpoint refinement ratios must be an ordered vector of vectors")
    result = []
    for transition, row in enumerate(values):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(
                "checkpoint refinement ratio %d must be an integer vector" % transition
            )
        ratio = tuple(row)
        if len(ratio) != dimension:
            raise ValueError(
                "checkpoint refinement ratio %d length must equal its dimension" % transition
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ratio):
            raise TypeError("checkpoint refinement ratios must contain exact integers")
        if any(value < 1 for value in ratio) or not any(value > 1 for value in ratio):
            raise ValueError(
                "checkpoint refinement ratios must be positive and refine at least one axis"
            )
        cell_count(ratio)
        result.append(ratio)
    return tuple(result)


def cell_count(shape: Sequence[int]) -> int:
    """Return one checked rank-generic product suitable for native allocation sizes."""
    if not shape:
        raise ValueError("checkpoint cell-count shape must be non-empty")
    count = 1
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError("checkpoint cell-count extents must be exact integers")
        if extent < 1:
            raise ValueError("checkpoint cell-count extents must be strictly positive")
        if count > sys.maxsize // extent:
            raise OverflowError("checkpoint cell count exceeds the native addressable range")
        count *= extent
    return count


@dataclass(frozen=True, slots=True)
class CheckpointSpatialContract:
    """One immutable 1D/2D/3D checkpoint specialization."""

    dimension: int
    shape: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    periodicity: tuple[bool, ...]
    refinement_ratios: tuple[tuple[int, ...], ...]
    native_layout_identity: str
    identity: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from pops.identity import Identity, make_identity

        dimension = _exact_dimension(self.dimension)
        shape = _shape(self.shape, dimension=dimension)
        lower = _bounds(self.lower, dimension=dimension, name="lower bounds")
        upper = _bounds(self.upper, dimension=dimension, name="upper bounds")
        if any(high <= low for low, high in zip(lower, upper, strict=True)):
            raise ValueError("checkpoint spatial upper bounds must exceed lower bounds")
        periodicity = _periodicity(self.periodicity, dimension=dimension)
        ratios = _refinement_ratios(self.refinement_ratios, dimension=dimension)
        if not isinstance(self.native_layout_identity, str):
            raise TypeError("checkpoint native layout identity must be text")
        native_identity = Identity.from_token(self.native_layout_identity)
        if native_identity.domain != "native-spatial-layout":
            raise ValueError("checkpoint native layout identity has the wrong domain")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(self, "refinement_ratios", ratios)
        object.__setattr__(self, "identity", make_identity("checkpoint-spatial-layout", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": SPATIAL_CONTRACT_SCHEMA_VERSION,
            "dimension": self.dimension,
            "shape": list(self.shape),
            "lower": [value.hex() for value in self.lower],
            "upper": [value.hex() for value in self.upper],
            "periodicity": list(self.periodicity),
            "refinement_ratios": [list(row) for row in self.refinement_ratios],
            "native_layout_identity": self.native_layout_identity,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.token}

    @classmethod
    def from_data(cls, data: Any) -> CheckpointSpatialContract:
        from pops.identity import Identity

        required = {
            "schema_version",
            "dimension",
            "shape",
            "lower",
            "upper",
            "periodicity",
            "refinement_ratios",
            "native_layout_identity",
            "identity",
        }
        if not isinstance(data, Mapping) or set(data) != required:
            raise TypeError("checkpoint spatial contract has an unsupported exact schema")
        if type(data["schema_version"]) is not int:
            raise TypeError("checkpoint spatial schema_version must be an exact integer")
        if data["schema_version"] != SPATIAL_CONTRACT_SCHEMA_VERSION:
            raise ValueError("checkpoint spatial contract schema version is unsupported")
        for name in ("lower", "upper"):
            values = data[name]
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise TypeError("checkpoint spatial %s must contain float.hex strings" % name)
        try:
            lower = tuple(float.fromhex(value) for value in data["lower"])
            upper = tuple(float.fromhex(value) for value in data["upper"])
        except ValueError:
            raise ValueError("checkpoint spatial bounds contain invalid float.hex data") from None
        result = cls(
            dimension=data["dimension"],
            shape=tuple(data["shape"]),
            lower=lower,
            upper=upper,
            periodicity=tuple(data["periodicity"]),
            refinement_ratios=tuple(tuple(row) for row in data["refinement_ratios"]),
            native_layout_identity=data["native_layout_identity"],
        )
        if Identity.from_token(data["identity"]) != result.identity or result.to_data() != dict(data):
            raise ValueError("checkpoint spatial contract does not authenticate its payload")
        return result

    @classmethod
    def from_native_layout(
        cls,
        native_layout: Any,
        *,
        transition_ratios: Sequence[Any] = (),
    ) -> CheckpointSpatialContract:
        from pops.mesh._layout_plan_contracts import NativeSpatialLayout

        if type(native_layout) is not NativeSpatialLayout:
            raise TypeError("checkpoint installation requires an exact NativeSpatialLayout")
        dimension = native_layout.dimension
        rows = []
        for transition, ratio in enumerate(transition_ratios):
            if isinstance(ratio, bool):
                raise TypeError("checkpoint transition ratio %d must be exact" % transition)
            if isinstance(ratio, int):
                rows.append(tuple(ratio for _axis in range(dimension)))
            else:
                rows.append(tuple(ratio))
        return cls(
            dimension=dimension,
            shape=native_layout.shape,
            lower=native_layout.lower,
            upper=native_layout.upper,
            periodicity=native_layout.periodicity,
            refinement_ratios=tuple(rows),
            native_layout_identity=native_layout.identity.token,
        )

    def shape_at_level(self, level: int) -> tuple[int, ...]:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("checkpoint AMR level must be an exact integer")
        if level < 0 or level > len(self.refinement_ratios):
            raise ValueError("checkpoint AMR level lies outside its refinement-ratio envelope")
        result = list(self.shape)
        for ratio in self.refinement_ratios[:level]:
            for axis, factor in enumerate(ratio):
                if result[axis] > sys.maxsize // factor:
                    raise OverflowError("checkpoint refined shape exceeds the native addressable range")
                result[axis] *= factor
        cell_count(result)
        return tuple(result)

    def cells_at_level(self, level: int) -> int:
        return cell_count(self.shape_at_level(level))


def _preflight_native_specialization(
    owner: Any, contract: CheckpointSpatialContract
) -> None:
    native = getattr(owner, "_s", None)
    if native is None:
        return
    prepare = getattr(native, "_prepare_checkpoint_spatial_contract", None)
    if not callable(prepare):
        raise TypeError(
            "native checkpoint engine lacks rank-generic spatial preparation"
        )
    supplied = prepare(contract.to_data())
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("native checkpoint spatial preparation must return level cell counts")
    counts = tuple(supplied)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("native checkpoint spatial cell counts must be exact integers")
    expected = tuple(
        contract.cells_at_level(level)
        for level in range(len(contract.refinement_ratios) + 1)
    )
    if counts != expected:
        raise RuntimeError(
            "native checkpoint spatial products differ from the authenticated Python contract"
        )


def install_checkpoint_spatial_contract(
    owner: Any,
    native_layout: Any,
    *,
    transition_ratios: Sequence[Any] = (),
) -> CheckpointSpatialContract:
    """Attach the already-resolved native specialization before Program installation."""
    contract = CheckpointSpatialContract.from_native_layout(
        native_layout, transition_ratios=transition_ratios
    )
    _preflight_native_specialization(owner, contract)
    existing = getattr(owner, "_checkpoint_spatial_contract", None)
    if existing is not None and existing != contract:
        raise RuntimeError("checkpoint spatial authority is immutable after installation")
    owner._checkpoint_spatial_contract = CheckpointSpatialContract.from_data(contract.to_data())
    return contract


def require_checkpoint_spatial_contract(owner: Any) -> CheckpointSpatialContract:
    contract = getattr(owner, "_checkpoint_spatial_contract", None)
    if type(contract) is not CheckpointSpatialContract:
        raise RuntimeError(
            "checkpoint requires the authenticated native spatial layout installed by pops.bind"
        )
    return CheckpointSpatialContract.from_data(contract.to_data())


def add_checkpoint_spatial_contract(
    payload: dict[str, Any], contract: CheckpointSpatialContract
) -> None:
    if type(contract) is not CheckpointSpatialContract:
        raise TypeError("checkpoint payload requires an exact spatial contract")
    if SPATIAL_CONTRACT_KEY in payload or _LEGACY_SPATIAL_KEYS.intersection(payload):
        raise ValueError("checkpoint payload contains a duplicate or legacy spatial authority")
    payload[SPATIAL_CONTRACT_KEY] = json.dumps(
        contract.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def inspect_checkpoint_spatial_contract(payload: Any) -> CheckpointSpatialContract:
    files = set(getattr(payload, "files", payload.keys() if isinstance(payload, Mapping) else ()))
    legacy = sorted(_LEGACY_SPATIAL_KEYS.intersection(files))
    if legacy:
        raise ValueError("checkpoint carries forbidden legacy spatial keys %r" % legacy)
    if SPATIAL_CONTRACT_KEY not in files:
        raise ValueError("checkpoint lacks its exact rank-generic spatial contract")
    from pops._manifest_protocol import strict_json_loads

    data = strict_json_loads(
        str(payload[SPATIAL_CONTRACT_KEY]), where="checkpoint spatial contract"
    )
    return CheckpointSpatialContract.from_data(data)


def authenticate_checkpoint_spatial_contract(
    owner: Any, payload: Any
) -> CheckpointSpatialContract:
    recorded = inspect_checkpoint_spatial_contract(payload)
    _preflight_native_specialization(owner, recorded)
    current = require_checkpoint_spatial_contract(owner)
    if recorded.dimension != current.dimension:
        raise ValueError(
            "restart checkpoint dimension %d does not match native dimension %d"
            % (recorded.dimension, current.dimension)
        )
    if recorded.identity != current.identity:
        raise ValueError("restart checkpoint spatial layout does not match the bound runtime")
    return recorded


__all__ = [
    "CheckpointSpatialContract",
    "SPATIAL_CONTRACT_KEY",
    "SPATIAL_CONTRACT_SCHEMA_VERSION",
    "add_checkpoint_spatial_contract",
    "authenticate_checkpoint_spatial_contract",
    "cell_count",
    "inspect_checkpoint_spatial_contract",
    "install_checkpoint_spatial_contract",
    "require_checkpoint_spatial_contract",
]
