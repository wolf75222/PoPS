"""Exact resolved data contract shared by scientific output consumers.

The consumer graph resolves authoring declarations before this boundary.  Writers therefore never
guess a block, layout, level or state: every selected array carries all four identities explicitly.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, cast

from pops._geometry_contracts import (
    CARTESIAN_1D_COORDINATES,
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_3D_COORDINATES,
    CARTESIAN_CELL_MEASURES_BY_DIMENSION,
    POLAR_ANNULUS_2D_COORDINATES,
    POLAR_ANNULUS_CELL_AREA,
)
from pops.identity import Identity, make_identity
from pops.model import Handle


_CENTERINGS = frozenset({"cell", "node", "face_x", "face_y", "face_z"})
EMBEDDED_BOUNDARY_ARRAY_NAMES = (
    "pops_active",
    "pops_phi",
    "pops_kappa",
)
_NATIVE_GEOMETRY_ARRAYS = object()
_CARTESIAN_COORDINATES = {
    1: CARTESIAN_1D_COORDINATES,
    2: CARTESIAN_2D_COORDINATES,
    3: CARTESIAN_3D_COORDINATES,
}
_CARTESIAN_CELL_MEASURES = {
    dimension: measure
    for dimension, measure in enumerate(CARTESIAN_CELL_MEASURES_BY_DIMENSION, start=1)
}
_CARTESIAN_AXIS_NAMES = ("x", "y", "z")


def _box_slices(lower: tuple[int, ...], upper: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))


def _centering_shape(
    cell_shape: tuple[int, ...], centering: str,
) -> tuple[int, ...]:
    if centering == "cell":
        return cell_shape
    if centering == "node":
        return tuple(extent + 1 for extent in cell_shape)
    face_axis = {"face_x": 0, "face_y": 1, "face_z": 2}.get(centering)
    if face_axis is None or face_axis >= len(cell_shape):
        raise ValueError(
            "field centering %r is not defined for spatial rank %d"
            % (centering, len(cell_shape))
        )
    # Dense scientific arrays use (..., z, y, x) order while geometry coordinates use
    # (x, y, z).  The corresponding face-normal axis is therefore counted from the end.
    array_axis = len(cell_shape) - 1 - face_axis
    result = list(cell_shape)
    result[array_axis] += 1
    return tuple(result)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def _identity(value: Any, where: str) -> Identity:
    if type(value) is not Identity:
        raise TypeError("%s must be an exact pops.identity.Identity" % where)
    return Identity.from_data(value.to_data())


def _array(value: Any, *, dtype: Any = None, borrow: bool = False) -> Any:
    import numpy as np

    array = np.asarray(value)
    exact_dtype = None if dtype is None else np.dtype(dtype)
    if borrow:
        if exact_dtype is not None and array.dtype != exact_dtype:
            raise TypeError("borrowed output array does not have its exact native dtype")
        if not array.flags.c_contiguous:
            raise TypeError("borrowed output array must be C-contiguous")
        result = array
    else:
        result = np.ascontiguousarray(np.asarray(value, dtype=exact_dtype)).copy()
    if result.dtype.hasobject:
        raise TypeError("output arrays cannot use object dtype")
    result.setflags(write=False)
    return result


def array_evidence(value: Any) -> dict[str, Any]:
    """Stable byte evidence for one already-normalized dense array."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    # Python refuses to cast a multidimensional memoryview when any extent is zero.
    # An empty dense array still has canonical dtype/shape evidence and contributes no
    # payload bytes, so keep the zero-copy path for non-empty scientific arrays only.
    if array.size:
        digest.update(memoryview(cast(Any, array)).cast("B"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": digest.hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class OutputClock:
    clock_id: str
    time_hex: str
    macro_step: int
    stage: str
    tick: int | None = None
    level: int = 0
    substep: int = 0
    stage_index: int = 0
    fraction_numerator: int = 1
    fraction_denominator: int = 1
    dt_hex: str = "0x0.0p+0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_id", _text(self.clock_id, "clock_id"))
        object.__setattr__(self, "stage", _text(self.stage, "clock stage"))
        if not isinstance(self.time_hex, str):
            raise TypeError("clock time must be a float.hex() string")
        try:
            value = float.fromhex(self.time_hex)
        except ValueError:
            raise ValueError("clock time is not a float.hex() string") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("clock time must be finite")
        if isinstance(self.macro_step, bool) or not isinstance(self.macro_step, int) \
                or self.macro_step < 0:
            raise ValueError("clock macro_step must be an integer >= 0")
        if self.tick is None:
            object.__setattr__(self, "tick", self.macro_step)
        for name in ("tick", "level", "substep", "stage_index"):
            item = getattr(self, name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError("clock %s must be an integer >= 0" % name)
        numerator, denominator = self.fraction_numerator, self.fraction_denominator
        if isinstance(numerator, bool) or not isinstance(numerator, int) or numerator < 0 \
                or isinstance(denominator, bool) or not isinstance(denominator, int) \
                or denominator <= 0 or numerator > denominator:
            raise ValueError("clock stage fraction must be canonical within [0,1]")
        import math
        if math.gcd(numerator, denominator) != 1:
            raise ValueError("clock stage fraction must be reduced")
        if not isinstance(self.dt_hex, str):
            raise TypeError("clock dt must be a float.hex() string")
        try:
            dt = float.fromhex(self.dt_hex)
        except ValueError:
            raise ValueError("clock dt is not a float.hex() string") from None
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("clock dt must be finite and non-negative")

    @classmethod
    def at(cls, clock_id: Any, time: Any, macro_step: Any, *, stage: Any,
           tick: Any = None, level: Any = 0, substep: Any = 0,
           stage_index: Any = 0, fraction: tuple[int, int] = (1, 1),
           dt: Any = 0.0) -> OutputClock:
        value = float(time)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("clock time must be finite")
        return cls(clock_id, value.hex(), macro_step, stage, tick, level, substep,
                   stage_index, fraction[0], fraction[1], float(dt).hex())

    def to_data(self) -> dict[str, Any]:
        return {
            "clock_id": self.clock_id, "time": self.time_hex,
            "macro_step": self.macro_step, "stage": self.stage, "tick": self.tick,
            "level": self.level, "substep": self.substep,
            "stage_index": self.stage_index,
            "fraction": [self.fraction_numerator, self.fraction_denominator],
            "dt": self.dt_hex,
        }


@dataclass(frozen=True, slots=True)
class OutputProvenance:
    plan_identity: Identity
    bind_identity: Identity
    run_identity: Identity
    source: str

    def __post_init__(self) -> None:
        for name in ("plan_identity", "bind_identity", "run_identity"):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(self, "source", _text(self.source, "provenance source"))

    def to_data(self) -> dict[str, Any]:
        return {
            "plan_identity": self.plan_identity.token,
            "bind_identity": self.bind_identity.token,
            "run_identity": self.run_identity.token,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class LevelGeometry:
    layout_identity: Identity
    layout_kind: str
    level: int
    origin: tuple[float, ...]
    spacing: tuple[float, ...]
    cell_shape: tuple[int, ...]
    boxes: tuple[tuple[int, ...], ...]
    coverage: Any = field(repr=False, compare=False)
    cell_volumes: Any = field(repr=False, compare=False)
    coordinate_system: str | None = None
    cell_measure: str | None = None
    axis_names: tuple[str, ...] = ()
    valid_cells: Any = field(init=False, repr=False, compare=False)
    _native_valid_cells: InitVar[Any] = None
    _native_arrays: InitVar[Any] = None

    def __post_init__(self, _native_valid_cells: Any, _native_arrays: Any) -> None:
        import numpy as np

        native = _native_arrays is _NATIVE_GEOMETRY_ARRAYS
        if _native_arrays is not None and not native:
            raise TypeError("LevelGeometry native array authority is private")

        object.__setattr__(self, "layout_identity", _identity(
            self.layout_identity, "layout_identity"))
        if self.layout_kind not in {"uniform", "amr"}:
            raise ValueError("layout_kind must be exactly 'uniform' or 'amr'")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("geometry level must be an integer >= 0")
        shape = tuple(self.cell_shape)
        if len(shape) not in (1, 2, 3) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in shape):
            raise ValueError(
                "cell_shape must have spatial rank 1, 2, or 3 and positive integer extents")
        object.__setattr__(self, "cell_shape", shape)
        dimension = len(shape)
        coordinate_system = self.coordinate_system
        if coordinate_system is None:
            coordinate_system = _CARTESIAN_COORDINATES[dimension]
        if not isinstance(coordinate_system, str) \
                or not coordinate_system.startswith("pops://") or "@" not in coordinate_system:
            raise ValueError("geometry coordinate_system must be a versioned pops:// URI")
        object.__setattr__(self, "coordinate_system", coordinate_system)
        cell_measure = self.cell_measure
        if cell_measure is None:
            if coordinate_system == _CARTESIAN_COORDINATES[dimension]:
                cell_measure = _CARTESIAN_CELL_MEASURES[dimension]
            elif coordinate_system == POLAR_ANNULUS_2D_COORDINATES and dimension == 2:
                cell_measure = POLAR_ANNULUS_CELL_AREA
            else:
                raise ValueError(
                    "geometry cell_measure must be explicit for coordinate system %s"
                    % coordinate_system
                )
        if not isinstance(cell_measure, str) \
                or not cell_measure.startswith("pops://") or "@" not in cell_measure:
            raise ValueError("geometry cell_measure must be a versioned pops:// URI")
        object.__setattr__(self, "cell_measure", cell_measure)
        axis_names = tuple(self.axis_names)
        if not axis_names:
            axis_names = _CARTESIAN_AXIS_NAMES[:dimension]
        if len(axis_names) != dimension or any(
                not isinstance(item, str) or not item for item in axis_names) \
                or len(set(axis_names)) != dimension:
            raise ValueError(
                "geometry axis_names must contain %d distinct non-empty names" % dimension)
        object.__setattr__(self, "axis_names", axis_names)
        for name in ("origin", "spacing"):
            values = tuple(float(item) for item in getattr(self, name))
            if len(values) != dimension or any(
                    item != item or item in (float("inf"), float("-inf"))
                    for item in values):
                raise ValueError(
                    "geometry %s must contain %d finite values" % (name, dimension))
            if name == "spacing" and any(item <= 0.0 for item in values):
                raise ValueError("geometry spacing must be positive")
            object.__setattr__(self, name, values)
        boxes = tuple(tuple(item) for item in self.boxes)
        if not boxes:
            raise ValueError("geometry boxes must explicitly cover the represented level")
        for index, box in enumerate(boxes):
            if len(box) != 2 * dimension or any(
                    isinstance(item, bool) or not isinstance(item, int) for item in box):
                raise TypeError(
                    "geometry boxes use %d integer lower bounds followed by %d upper bounds"
                    % (dimension, dimension))
            lower, upper = box[:dimension], box[dimension:]
            if any(
                    lo < 0 or hi <= lo or hi > extent
                    for lo, hi, extent in zip(lower, upper, shape, strict=True)):
                raise ValueError("geometry box %r is outside cell_shape %r" % (box, shape))
            # Native boxes come from the hierarchy and the mask/cardinality check below proves their
            # represented union without an O(patch_count**2) Python scan.
            if not native:
                for prior in boxes[:index]:
                    prior_lower, prior_upper = prior[:dimension], prior[dimension:]
                    if all(
                            lo < prior_hi and prior_lo < hi
                            for lo, hi, prior_lo, prior_hi in zip(
                                lower, upper, prior_lower, prior_upper, strict=True)):
                        raise ValueError("geometry boxes must not overlap")
        object.__setattr__(self, "boxes", boxes)
        if native:
            if _native_valid_cells is None:
                raise ValueError("native geometry requires its native valid-cell mask")
            valid = _array(_native_valid_cells, dtype=np.bool_, borrow=True)
            if valid.shape != shape:
                raise ValueError("native valid_cells must be a binary cell_shape mask")
            represented = sum(
                math.prod(
                    hi - lo for lo, hi in zip(
                        box[:dimension], box[dimension:], strict=True))
                for box in boxes
            )
            if int(np.count_nonzero(valid)) != represented:
                raise ValueError("native valid_cells count differs from geometry boxes")
        else:
            if _native_valid_cells is not None:
                raise TypeError("valid_cells is derived from boxes outside the native provider")
            valid = np.zeros(shape, dtype=np.bool_)
            for box in boxes:
                valid[_box_slices(box[:dimension], box[dimension:])] = True
            valid.setflags(write=False)
        object.__setattr__(self, "valid_cells", valid)
        coverage = _array(self.coverage, dtype=np.bool_, borrow=native)
        volumes = _array(self.cell_volumes, dtype=np.float64, borrow=native)
        if coverage.shape != shape or volumes.shape != shape:
            raise ValueError("coverage and cell_volumes must match cell_shape")
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("cell_volumes must be finite and strictly positive")
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "cell_volumes", volumes)

    @property
    def key(self) -> tuple[str, int]:
        return self.layout_identity.token, self.level

    @property
    def spatial_rank(self) -> int:
        """Return the immutable rank inferred from the authoritative cell shape."""

        return len(self.cell_shape)

    def to_data(self) -> dict[str, Any]:
        return {
            "layout_identity": self.layout_identity.token,
            "layout_kind": self.layout_kind,
            "coordinate_system": self.coordinate_system,
            "cell_measure": self.cell_measure,
            "axis_names": list(self.axis_names),
            "level": self.level,
            "origin": [item.hex() for item in self.origin],
            "spacing": [item.hex() for item in self.spacing],
            "cell_shape": list(self.cell_shape),
            "boxes": [list(item) for item in self.boxes],
            "valid_cells": array_evidence(self.valid_cells),
            "coverage": array_evidence(self.coverage),
            "cell_volumes": array_evidence(self.cell_volumes),
        }


@dataclass(frozen=True, slots=True)
class FieldKey:
    reference: Handle
    component_manifest_identity: Identity
    layout_identity: Identity
    level: int
    state_id: str

    def __post_init__(self) -> None:
        if type(self.reference) is not Handle and not isinstance(self.reference, Handle):
            raise TypeError("output field reference must be a Handle")
        if not self.reference.is_resolved:
            raise ValueError("output field reference must be owner-qualified and resolved")
        self.reference.canonical_identity()
        object.__setattr__(self, "component_manifest_identity", _identity(
            self.component_manifest_identity, "component_manifest_identity"))
        object.__setattr__(self, "layout_identity", _identity(
            self.layout_identity, "layout_identity"))
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("field level must be an integer >= 0")
        object.__setattr__(self, "state_id", _text(self.state_id, "field state_id"))

    @property
    def identity(self) -> Identity:
        return make_identity("output-field", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "reference": self.reference.canonical_identity(),
            "component_manifest_identity": self.component_manifest_identity.token,
            "layout_identity": self.layout_identity.token,
            "level": self.level,
            "state_id": self.state_id,
        }


def _field_family_identity(key: FieldKey) -> Identity:
    """Identity of one exact field across its explicitly selected AMR levels."""
    if type(key) is not FieldKey:
        raise TypeError("output field family requires an exact FieldKey")
    return make_identity("output-field-family", {
        "reference": key.reference.canonical_identity(),
        "component_manifest_identity": key.component_manifest_identity.token,
        "layout_identity": key.layout_identity.token,
        "state_id": key.state_id,
    })


def _composite_integral_authority_identity(
    family_identity: Identity, levels: tuple[int, ...],
) -> Identity:
    if type(family_identity) is not Identity:
        raise TypeError("composite integral authority requires an exact family Identity")
    return make_identity("native-composite-integral", {
        "family_identity": family_identity.token,
        "levels": list(levels),
    })


@dataclass(frozen=True, slots=True)
class _NativeCompositeIntegral:
    """Private evidence produced by the native accepted-state reduction path."""

    family_identity: Identity
    levels: tuple[int, ...]
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_identity", _identity(
            self.family_identity, "native composite integral family_identity"))
        levels = tuple(self.levels)
        if not levels or any(
                isinstance(level, bool) or type(level) is not int or level < 0
                for level in levels):
            raise TypeError(
                "native composite integral levels must be non-empty exact integers >= 0")
        if levels != tuple(sorted(set(levels))):
            raise ValueError(
                "native composite integral levels must be strictly increasing and unique")
        object.__setattr__(self, "levels", levels)
        value = float(self.value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("native composite integral must be finite")
        object.__setattr__(self, "value", value)

    @property
    def authority_identity(self) -> Identity:
        return _composite_integral_authority_identity(self.family_identity, self.levels)


@dataclass(frozen=True, slots=True)
class ArrayPiece:
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    values: Any = field(repr=False, compare=False)
    global_box_index: int
    owner_rank: int
    replicated: bool

    def __post_init__(self) -> None:
        lower, upper = tuple(self.lower), tuple(self.upper)
        if len(lower) not in (1, 2, 3) or len(upper) != len(lower) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in lower + upper):
            raise TypeError(
                "array piece bounds must be integer tuples of spatial rank 1, 2, or 3")
        if any(lo < 0 or hi <= lo for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("array piece bounds must be non-negative non-empty half-open ranges")
        values = _array(self.values)
        spatial_shape = tuple(hi - lo for lo, hi in zip(lower, upper, strict=True))
        if values.ndim not in (len(lower), len(lower) + 1) \
                or values.shape[-len(lower):] != spatial_shape:
            raise ValueError("array piece values do not match its spatial bounds")
        for name in ("global_box_index", "owner_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) is not int or value < 0:
                raise TypeError("array piece %s must be an integer >= 0" % name)
        if type(self.replicated) is not bool:
            raise TypeError("array piece replicated must be an exact bool")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "values", values)

    def to_data(self) -> dict[str, Any]:
        return {
            "lower": list(self.lower), "upper": list(self.upper),
            "global_box_index": self.global_box_index,
            "owner_rank": self.owner_rank,
            "replicated": self.replicated,
            "array": array_evidence(self.values),
        }


@dataclass(frozen=True, slots=True)
class EmbeddedBoundaryPayload:
    """Exact geometry sidecar for one selected ``(layout, level)``.

    Embedded-boundary arrays are intentionally not scientific ``FieldPayload`` values: they have
    reserved names, no user Handle, and describe the mesh on which physical fields live.  Keeping
    them separate prevents accidental concatenation with conservative variables while still
    carrying the exact per-patch MPI ownership needed by VTK, Catalyst, and durable observers.
    """

    layout_identity: Identity
    level: int
    global_shape: tuple[int, ...]
    arrays: Mapping[str, tuple[ArrayPiece, ...]]
    dtype: str = "<f8"

    def __post_init__(self) -> None:
        import numpy as np

        object.__setattr__(
            self,
            "layout_identity",
            _identity(self.layout_identity, "embedded-boundary layout_identity"),
        )
        if isinstance(self.level, bool) or type(self.level) is not int or self.level < 0:
            raise ValueError("embedded-boundary level must be an integer >= 0")
        shape = tuple(self.global_shape)
        if len(shape) not in (1, 2, 3) or any(
            isinstance(item, bool) or type(item) is not int or item < 1 for item in shape
        ):
            raise ValueError("embedded-boundary global_shape must have spatial rank 1, 2, or 3")
        object.__setattr__(self, "global_shape", shape)
        dtype = np.dtype(self.dtype).str
        if dtype != np.dtype(np.float64).str:
            raise TypeError("embedded-boundary sidecars require exact float64 arrays")
        object.__setattr__(self, "dtype", dtype)
        if not isinstance(self.arrays, Mapping) or set(self.arrays) != set(
            EMBEDDED_BOUNDARY_ARRAY_NAMES
        ):
            raise TypeError(
                "embedded-boundary sidecar arrays must be exactly %s"
                % (EMBEDDED_BOUNDARY_ARRAY_NAMES,)
            )
        normalized: dict[str, tuple[ArrayPiece, ...]] = {}
        ownership: tuple[tuple[Any, ...], ...] | None = None
        for name in EMBEDDED_BOUNDARY_ARRAY_NAMES:
            pieces = tuple(self.arrays[name])
            if any(type(piece) is not ArrayPiece for piece in pieces):
                raise TypeError(
                    "embedded-boundary %s pieces must be exact ArrayPiece values" % name
                )
            metadata = []
            for piece in pieces:
                if len(piece.lower) != len(shape) or any(
                    high > extent for high, extent in zip(piece.upper, shape, strict=True)
                ):
                    raise ValueError(
                        "embedded-boundary piece lies outside its exact ranked geometry"
                    )
                if (
                    piece.values.dtype.str != dtype
                    or piece.values.ndim != len(shape) + 1
                    or piece.values.shape[0] != 1
                ):
                    raise ValueError("embedded-boundary pieces must contain one float64 component")
                values = piece.values[0]
                if not np.all(np.isfinite(values)):
                    raise ValueError("embedded-boundary sidecar arrays must contain finite values")
                if name == "pops_active" and not np.all((values == 0.0) | (values == 1.0)):
                    raise ValueError("pops_active must be an exact binary cell mask")
                if name == "pops_kappa" and not np.all((values >= 0.0) & (values <= 1.0)):
                    raise ValueError("pops_kappa must lie in the closed interval [0, 1]")
                metadata.append(
                    (
                        piece.lower,
                        piece.upper,
                        piece.global_box_index,
                        piece.owner_rank,
                        piece.replicated,
                    )
                )
            for index, left in enumerate(pieces):
                for right in pieces[index + 1 :]:
                    if all(
                        left_lo < right_hi and right_lo < left_hi
                        for left_lo, left_hi, right_lo, right_hi in zip(
                            left.lower,
                            left.upper,
                            right.lower,
                            right.upper,
                            strict=True,
                        )
                    ):
                        raise ValueError("embedded-boundary array pieces overlap")
            box_indices = [piece.global_box_index for piece in pieces]
            if len(box_indices) != len(set(box_indices)):
                raise ValueError(
                    "embedded-boundary array contains duplicate global_box_index values"
                )
            current = tuple(metadata)
            if ownership is None:
                ownership = current
            elif current != ownership:
                raise ValueError(
                    "embedded-boundary arrays disagree on patch bounds or MPI ownership"
                )
            normalized[name] = pieces
        object.__setattr__(self, "arrays", MappingProxyType(normalized))

    @property
    def key(self) -> tuple[str, int]:
        return self.layout_identity.token, self.level

    @property
    def identity(self) -> Identity:
        return make_identity("output-embedded-boundary", self.to_data())

    def pieces(self, name: str) -> tuple[ArrayPiece, ...]:
        if name not in EMBEDDED_BOUNDARY_ARRAY_NAMES:
            raise KeyError("unknown embedded-boundary sidecar array %r" % name)
        return self.arrays[name]

    def to_data(self) -> dict[str, Any]:
        return {
            "layout_identity": self.layout_identity.token,
            "level": self.level,
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "arrays": {
                name: [piece.to_data() for piece in self.arrays[name]]
                for name in EMBEDDED_BOUNDARY_ARRAY_NAMES
            },
        }


@dataclass(frozen=True, slots=True)
class FieldPayload:
    key: FieldKey
    centering: str
    units: str
    component_names: tuple[str, ...]
    global_shape: tuple[int, ...]
    pieces: tuple[ArrayPiece, ...]
    dtype: str | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not FieldKey:
            raise TypeError("field payload key must be an exact FieldKey")
        if self.centering not in _CENTERINGS:
            raise ValueError("field centering must be one of %s" % sorted(_CENTERINGS))
        object.__setattr__(self, "units", _text(self.units, "field units"))
        names = tuple(self.component_names)
        if any(not isinstance(item, str) or not item for item in names) \
                or len(names) != len(set(names)):
            raise ValueError("field component_names must be unique non-empty strings")
        shape = tuple(self.global_shape)
        if len(shape) not in (1, 2, 3) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in shape):
            raise ValueError(
                "field global_shape must have spatial rank 1, 2, or 3 and positive extents")
        pieces = tuple(self.pieces)
        if any(type(piece) is not ArrayPiece for piece in pieces):
            raise TypeError("field payload pieces must be exact ArrayPiece values")
        if pieces:
            inferred_dtype = pieces[0].values.dtype.str
            if any(piece.values.dtype.str != inferred_dtype for piece in pieces):
                raise ValueError("field payload pieces must have one exact dtype")
            if self.dtype is not None and self.dtype != inferred_dtype:
                raise ValueError("declared field dtype differs from its array pieces")
        else:
            if self.dtype is None:
                raise ValueError("a rank with no field pieces must still declare the exact dtype")
            import numpy as np
            inferred_dtype = np.dtype(self.dtype).str
        expected_ndim = len(shape) + (1 if names else 0)
        for piece in pieces:
            if len(piece.lower) != len(shape):
                raise ValueError("array piece spatial rank differs from global_shape")
            if piece.values.ndim != expected_ndim:
                raise ValueError("component_names and array rank disagree")
            if names and piece.values.shape[0] != len(names):
                raise ValueError("component_names count does not match array components")
            if any(hi > extent for hi, extent in zip(piece.upper, shape, strict=True)):
                raise ValueError("array piece lies outside global_shape")
        for index, left in enumerate(pieces):
            for right in pieces[index + 1:]:
                if all(
                        left_lo < right_hi and right_lo < left_hi
                        for left_lo, left_hi, right_lo, right_hi in zip(
                            left.lower, left.upper, right.lower, right.upper, strict=True)):
                    raise ValueError("array pieces overlap")
        box_indices = [piece.global_box_index for piece in pieces]
        if len(box_indices) != len(set(box_indices)):
            raise ValueError("field payload contains duplicate global_box_index values")
        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "global_shape", shape)
        object.__setattr__(self, "pieces", pieces)
        object.__setattr__(self, "dtype", inferred_dtype)

    @property
    def array_dtype(self) -> str:
        if self.dtype is None:
            raise RuntimeError("validated field payload is missing its canonical dtype")
        return self.dtype

    def materialize(self) -> Any:
        """Build a complete dense array, refusing missing or overlapping cells."""
        import numpy as np

        if not self.pieces:
            raise ValueError("this rank owns no pieces; serial materialization is incomplete")
        prefix = (len(self.component_names),) if self.component_names else ()
        result = np.empty(prefix + self.global_shape, dtype=self.pieces[0].values.dtype)
        written = np.zeros(self.global_shape, dtype=np.uint8)
        for piece in self.pieces:
            spatial = _box_slices(piece.lower, piece.upper)
            if np.any(written[spatial]):
                raise ValueError("array pieces overlap")
            result[(...,) + spatial] = piece.values
            written[spatial] = 1
        if not np.all(written):
            raise ValueError("field payload does not completely cover global_shape")
        result.setflags(write=False)
        return result

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key.to_data(), "centering": self.centering, "units": self.units,
            "component_names": list(self.component_names), "global_shape": list(self.global_shape),
            "dtype": self.dtype, "pieces": [piece.to_data() for piece in self.pieces],
        }


@dataclass(frozen=True, slots=True)
class DiagnosticKey:
    reference: Handle
    component_manifest_identity: Identity
    layout_identity: Identity
    level: int
    state_id: str
    reduction: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Handle) or not self.reference.is_resolved:
            raise TypeError("diagnostic reference must be an owner-qualified resolved Handle")
        self.reference.canonical_identity()
        object.__setattr__(self, "component_manifest_identity", _identity(
            self.component_manifest_identity, "diagnostic component_manifest_identity"))
        object.__setattr__(self, "layout_identity", _identity(
            self.layout_identity, "diagnostic layout_identity"))
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("diagnostic level must be an integer >= 0")
        object.__setattr__(self, "state_id", _text(self.state_id, "diagnostic state_id"))
        object.__setattr__(self, "reduction", _text(self.reduction, "diagnostic reduction"))

    @property
    def identity(self) -> Identity:
        return make_identity("output-diagnostic", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "reference": self.reference.canonical_identity(),
            "component_manifest_identity": self.component_manifest_identity.token,
            "layout_identity": self.layout_identity.token,
            "level": self.level, "state_id": self.state_id, "reduction": self.reduction,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticPayload:
    key: DiagnosticKey
    value: float
    units: str
    terms: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.key) is not DiagnosticKey:
            raise TypeError("diagnostic payload key must be an exact DiagnosticKey")
        value = float(self.value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("diagnostic value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "units", _text(self.units, "diagnostic units"))
        if not isinstance(self.terms, Mapping):
            raise TypeError("diagnostic terms must be a mapping")
        terms = {}
        for name, item in self.terms.items():
            name = _text(name, "diagnostic term name")
            item = float(item)
            if item != item or item in (float("inf"), float("-inf")):
                raise ValueError("diagnostic term %r must be finite" % name)
            terms[name] = item
        object.__setattr__(self, "terms", MappingProxyType(dict(sorted(terms.items()))))

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key.to_data(), "value": self.value.hex(), "units": self.units,
            "terms": {name: value.hex() for name, value in self.terms.items()},
        }


@dataclass(frozen=True, slots=True)
class OutputRequest:
    consumer_id: str
    selection: tuple[FieldKey, ...]
    parallel_mode: Any
    rank: int = 0
    size: int = 1
    diagnostics: tuple[DiagnosticKey, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        selection = tuple(self.selection)
        if any(type(item) is not FieldKey for item in selection):
            raise TypeError("output request field selections must be exact FieldKey values")
        tokens = [item.identity.token for item in selection]
        if len(tokens) != len(set(tokens)):
            raise ValueError("output request selection contains duplicates")
        object.__setattr__(self, "selection", tuple(
            item for _, item in sorted(zip(tokens, selection, strict=True))))
        from ._consumer_contracts import ParallelMode

        if type(self.parallel_mode) is not ParallelMode:
            raise TypeError(
                "output request parallel_mode must be an exact pops.output.ParallelMode")
        for name in ("rank", "size"):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) is not int:
                raise TypeError("output request %s must be an exact int" % name)
        if self.size < 1 or self.rank < 0 or self.rank >= self.size:
            raise ValueError("output request rank/size are not a valid execution topology")
        if self.parallel_mode is ParallelMode.SERIAL:
            if (self.rank, self.size) != (0, 1):
                raise ValueError("SERIAL output requires the exact rank 0 / size 1 topology")
        diagnostics = tuple(self.diagnostics)
        if any(type(item) is not DiagnosticKey for item in diagnostics):
            raise TypeError("output request diagnostics must be exact DiagnosticKey values")
        diagnostic_tokens = [item.identity.token for item in diagnostics]
        if len(diagnostic_tokens) != len(set(diagnostic_tokens)):
            raise ValueError("output request diagnostic selection contains duplicates")
        if not selection and not diagnostics:
            raise ValueError("output request must select at least one exact field or diagnostic")
        object.__setattr__(self, "diagnostics", tuple(
            item for _, item in sorted(zip(diagnostic_tokens, diagnostics, strict=True))))

    @property
    def identity(self) -> Identity:
        return make_identity("output-selection", self.to_data())

    @property
    def publication_identity(self) -> Identity:
        """Identity shared by one artifact, or rank-qualified for PER_RANK artifacts."""
        return make_identity("output-publication-selection", self.publication_data())

    def publication_data(self) -> dict[str, Any]:
        """Canonical artifact selection with every participating rank made explicit."""
        from ._consumer_contracts import ParallelMode

        data = self.to_data()
        if self.parallel_mode in (ParallelMode.ROOT, ParallelMode.COLLECTIVE):
            data = dict(data)
            data.pop("rank")
            data["ranks"] = list(range(self.size))
        return data

    def to_data(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "selection": [item.to_data() for item in self.selection],
            "parallel_mode": self.parallel_mode.value,
            "rank": self.rank,
            "size": self.size,
            "diagnostics": [item.to_data() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class OutputSnapshot:
    clock: OutputClock
    provenance: OutputProvenance
    geometries: tuple[LevelGeometry, ...]
    fields: tuple[FieldPayload, ...]
    metadata: Any = field(default_factory=dict)
    diagnostics: tuple[DiagnosticPayload, ...] = ()
    embedded_boundaries: tuple[EmbeddedBoundaryPayload, ...] = ()
    _native_composite_integrals: tuple[_NativeCompositeIntegral, ...] = field(
        default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.clock) is not OutputClock or type(self.provenance) is not OutputProvenance:
            raise TypeError("output snapshot requires exact clock and provenance values")
        geometries, fields = tuple(self.geometries), tuple(self.fields)
        if not geometries or any(type(item) is not LevelGeometry for item in geometries):
            raise TypeError("output snapshot requires explicit LevelGeometry values")
        if any(type(item) is not FieldPayload for item in fields):
            raise TypeError("output snapshot fields must be exact FieldPayload values")
        geometry_map = {item.key: item for item in geometries}
        if len(geometry_map) != len(geometries):
            raise ValueError("output snapshot geometry keys must be unique")
        field_map = {item.key.identity.token: item for item in fields}
        if len(field_map) != len(fields):
            raise ValueError("output snapshot field keys must be unique")
        for item in fields:
            geometry = geometry_map.get((item.key.layout_identity.token, item.key.level))
            if geometry is None:
                raise ValueError("field has no exact layout/level geometry")
            expected = _centering_shape(geometry.cell_shape, item.centering)
            if item.global_shape != expected:
                raise ValueError("field shape does not match its centering and geometry")
        diagnostics = tuple(self.diagnostics)
        if any(type(item) is not DiagnosticPayload for item in diagnostics):
            raise TypeError("output snapshot diagnostics must be exact DiagnosticPayload values")
        diagnostic_map = {item.key.identity.token: item for item in diagnostics}
        if len(diagnostic_map) != len(diagnostics):
            raise ValueError("output snapshot diagnostic keys must be unique")
        if not fields and not diagnostics:
            raise ValueError("output snapshot must contain fields or diagnostics")
        if not isinstance(self.metadata, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, (str, int, bool))
                for key, value in self.metadata.items()):
            raise TypeError("output metadata must be a flat string/int/bool mapping")
        object.__setattr__(self, "geometries", tuple(sorted(geometries, key=lambda item: item.key)))
        object.__setattr__(self, "fields", tuple(
            field_map[token] for token in sorted(field_map)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(self.metadata.items()))))
        object.__setattr__(self, "diagnostics", tuple(
            diagnostic_map[token] for token in sorted(diagnostic_map)))
        embedded = tuple(self.embedded_boundaries)
        if any(type(item) is not EmbeddedBoundaryPayload for item in embedded):
            raise TypeError("output snapshot embedded boundaries must be exact sidecar payloads")
        embedded_map = {item.key: item for item in embedded}
        if len(embedded_map) != len(embedded):
            raise ValueError("output snapshot embedded-boundary layout/level keys must be unique")
        for item in embedded:
            geometry = geometry_map.get(item.key)
            if geometry is None:
                raise ValueError("embedded-boundary sidecar has no exact layout/level geometry")
            if item.global_shape != geometry.cell_shape:
                raise ValueError("embedded-boundary sidecar shape differs from its exact geometry")
            for name in EMBEDDED_BOUNDARY_ARRAY_NAMES:
                for piece in item.pieces(name):
                    if piece.global_box_index >= len(geometry.boxes) or (
                        piece.lower + piece.upper != geometry.boxes[piece.global_box_index]
                    ):
                        raise ValueError(
                            "embedded-boundary sidecar piece differs from its indexed geometry box"
                        )
        object.__setattr__(self, "embedded_boundaries", tuple(
            embedded_map[key] for key in sorted(embedded_map)))
        native_integrals = tuple(self._native_composite_integrals)
        if any(type(item) is not _NativeCompositeIntegral for item in native_integrals):
            raise TypeError(
                "native composite integral evidence must use the private native payload")
        native_map = {item.authority_identity.token: item for item in native_integrals}
        if len(native_map) != len(native_integrals):
            raise ValueError(
                "native composite integral family-and-level authorities must be unique")
        object.__setattr__(self, "_native_composite_integrals", tuple(
            native_map[token] for token in sorted(native_map)))

    def select(self, request: OutputRequest) -> tuple[FieldPayload, ...]:
        if type(request) is not OutputRequest:
            raise TypeError("snapshot selection requires an exact OutputRequest")
        available = {item.key.identity.token: item for item in self.fields}
        result = []
        for key in request.selection:
            try:
                result.append(available[key.identity.token])
            except KeyError:
                raise KeyError(
                    "requested owner/layout/level/state field %s is absent" % key.identity.token
                ) from None
        return tuple(result)

    def geometry(self, key: FieldKey) -> LevelGeometry:
        for geometry in self.geometries:
            if geometry.key == (key.layout_identity.token, key.level):
                return geometry
        raise KeyError("no geometry for selected field")

    def select_diagnostics(self, request: OutputRequest) -> tuple[DiagnosticPayload, ...]:
        available = {item.key.identity.token: item for item in self.diagnostics}
        result = []
        for key in request.diagnostics:
            try:
                result.append(available[key.identity.token])
            except KeyError:
                raise KeyError(
                    "requested owner/layout/state diagnostic %s is absent" % key.identity.token
                ) from None
        return tuple(result)

    def embedded_boundary(
        self, layout_identity: Identity, level: int,
    ) -> EmbeddedBoundaryPayload | None:
        token = _identity(layout_identity, "embedded-boundary lookup layout_identity").token
        for value in self.embedded_boundaries:
            if value.key == (token, level):
                return value
        return None

    def to_data(self, request: OutputRequest) -> dict[str, Any]:
        fields = self.select(request)
        geometries = {self.geometry(field.key).key: self.geometry(field.key) for field in fields}
        diagnostic_layouts = {item.key.layout_identity.token
                              for item in self.select_diagnostics(request)}
        geometries.update({item.key: item for item in self.geometries
                           if item.layout_identity.token in diagnostic_layouts})
        embedded = tuple(item for item in self.embedded_boundaries if item.key in geometries)
        return {
            "clock": self.clock.to_data(), "provenance": self.provenance.to_data(),
            "selection": request.publication_data(),
            "geometries": [item.to_data() for item in sorted(geometries.values(), key=lambda x: x.key)],
            "fields": [item.to_data() for item in fields],
            "embedded_boundaries": [item.to_data() for item in embedded],
            "diagnostics": [item.to_data() for item in self.select_diagnostics(request)],
            "metadata": dict(self.metadata),
        }


__all__ = [
    "ArrayPiece", "DiagnosticKey", "DiagnosticPayload", "EmbeddedBoundaryPayload",
    "EMBEDDED_BOUNDARY_ARRAY_NAMES", "FieldKey", "FieldPayload",
    "LevelGeometry", "OutputClock",
    "OutputProvenance", "OutputRequest", "OutputSnapshot", "array_evidence",
]
