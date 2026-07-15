"""Exact resolved data contract shared by scientific output consumers.

The consumer graph resolves authoring declarations before this boundary.  Writers therefore never
guess a block, layout, level or state: every selected array carries all four identities explicitly.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from pops.identity import Identity, make_identity
from pops.model import Handle


_CENTERINGS = frozenset({"cell", "node", "face_x", "face_y"})


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def _identity(value: Any, where: str) -> Identity:
    if type(value) is not Identity:
        raise TypeError("%s must be an exact pops.identity.Identity" % where)
    return Identity.from_data(value.to_data())


def _array(value: Any, *, dtype: Any = None) -> Any:
    import numpy as np

    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
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
    origin: tuple[float, float]
    spacing: tuple[float, float]
    cell_shape: tuple[int, int]
    boxes: tuple[tuple[int, int, int, int], ...]
    coverage: Any = field(repr=False, compare=False)
    cell_volumes: Any = field(repr=False, compare=False)
    coordinate_system: str = "pops://coordinates/cartesian-2d@1"
    cell_measure: str = "pops://cell-measures/cartesian-area@1"
    axis_names: tuple[str, str] = ("x", "y")
    valid_cells: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import numpy as np

        object.__setattr__(self, "layout_identity", _identity(
            self.layout_identity, "layout_identity"))
        if self.layout_kind not in {"uniform", "amr"}:
            raise ValueError("layout_kind must be exactly 'uniform' or 'amr'")
        for name in ("coordinate_system", "cell_measure"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("pops://") or "@" not in value:
                raise ValueError("geometry %s must be a versioned pops:// URI" % name)
        axis_names = tuple(self.axis_names)
        if len(axis_names) != 2 or any(
                not isinstance(item, str) or not item for item in axis_names) \
                or len(set(axis_names)) != 2:
            raise ValueError("geometry axis_names must contain two distinct non-empty names")
        object.__setattr__(self, "axis_names", axis_names)
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("geometry level must be an integer >= 0")
        for name in ("origin", "spacing"):
            values = tuple(float(item) for item in getattr(self, name))
            if len(values) != 2 or any(item != item or item in (float("inf"), float("-inf"))
                                       for item in values):
                raise ValueError("geometry %s must contain two finite values" % name)
            if name == "spacing" and any(item <= 0.0 for item in values):
                raise ValueError("geometry spacing must be positive")
            object.__setattr__(self, name, values)
        shape = tuple(self.cell_shape)
        if len(shape) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item < 1
                                  for item in shape):
            raise ValueError("cell_shape must be (ny, nx) with positive integers")
        object.__setattr__(self, "cell_shape", shape)
        boxes = tuple(tuple(item) for item in self.boxes)
        if not boxes:
            raise ValueError("geometry boxes must explicitly cover the represented level")
        ny, nx = shape
        valid = np.zeros(shape, dtype=np.bool_)
        for box in boxes:
            if len(box) != 4 or any(isinstance(item, bool) or not isinstance(item, int)
                                    for item in box):
                raise TypeError("geometry boxes use integer (jlo, ilo, jhi, ihi) bounds")
            jlo, ilo, jhi, ihi = box
            if jlo < 0 or ilo < 0 or jhi <= jlo or ihi <= ilo or jhi > ny or ihi > nx:
                raise ValueError("geometry box %r is outside cell_shape %r" % (box, shape))
            if np.any(valid[jlo:jhi, ilo:ihi]):
                raise ValueError("geometry boxes must not overlap")
            valid[jlo:jhi, ilo:ihi] = True
        object.__setattr__(self, "boxes", boxes)
        valid.setflags(write=False)
        object.__setattr__(self, "valid_cells", valid)
        coverage = _array(self.coverage, dtype=np.bool_)
        volumes = _array(self.cell_volumes, dtype=np.float64)
        if coverage.shape != shape or volumes.shape != shape:
            raise ValueError("coverage and cell_volumes must match cell_shape")
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("cell_volumes must be finite and strictly positive")
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "cell_volumes", volumes)

    @property
    def key(self) -> tuple[str, int]:
        return self.layout_identity.token, self.level

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


@dataclass(frozen=True, slots=True)
class ArrayPiece:
    lower: tuple[int, int]
    upper: tuple[int, int]
    values: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        lower, upper = tuple(self.lower), tuple(self.upper)
        if len(lower) != 2 or len(upper) != 2 or any(
                isinstance(item, bool) or not isinstance(item, int) for item in lower + upper):
            raise TypeError("array piece bounds must be integer (j, i) pairs")
        if lower[0] < 0 or lower[1] < 0 or upper[0] <= lower[0] or upper[1] <= lower[1]:
            raise ValueError("array piece bounds must be positive non-empty half-open ranges")
        values = _array(self.values)
        if values.ndim not in (2, 3) or values.shape[-2:] != (
                upper[0] - lower[0], upper[1] - lower[1]):
            raise ValueError("array piece values do not match its spatial bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "values", values)

    def to_data(self) -> dict[str, Any]:
        return {
            "lower": list(self.lower), "upper": list(self.upper),
            "array": array_evidence(self.values),
        }


@dataclass(frozen=True, slots=True)
class FieldPayload:
    key: FieldKey
    centering: str
    units: str
    component_names: tuple[str, ...]
    global_shape: tuple[int, int]
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
        if len(shape) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item < 1
                                  for item in shape):
            raise ValueError("field global_shape must be a positive (ny, nx)")
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
        expected_ndim = 3 if names else 2
        for piece in pieces:
            if piece.values.ndim != expected_ndim:
                raise ValueError("component_names and array rank disagree")
            if names and piece.values.shape[0] != len(names):
                raise ValueError("component_names count does not match array components")
            if piece.upper[0] > shape[0] or piece.upper[1] > shape[1]:
                raise ValueError("array piece lies outside global_shape")
        for index, left in enumerate(pieces):
            for right in pieces[index + 1:]:
                if not (left.upper[0] <= right.lower[0] or right.upper[0] <= left.lower[0]
                        or left.upper[1] <= right.lower[1] or right.upper[1] <= left.lower[1]):
                    raise ValueError("array pieces overlap")
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
            jlo, ilo = piece.lower
            jhi, ihi = piece.upper
            if np.any(written[jlo:jhi, ilo:ihi]):
                raise ValueError("array pieces overlap")
            result[..., jlo:jhi, ilo:ihi] = piece.values
            written[jlo:jhi, ilo:ihi] = 1
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
    parallel: bool
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
        if type(self.parallel) is not bool:
            raise TypeError("output request parallel flag must be a bool")
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

    def to_data(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "selection": [item.to_data() for item in self.selection],
            "parallel": self.parallel,
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
            ny, nx = geometry.cell_shape
            expected = {
                "cell": (ny, nx), "node": (ny + 1, nx + 1),
                "face_x": (ny, nx + 1), "face_y": (ny + 1, nx),
            }[item.centering]
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

    def to_data(self, request: OutputRequest) -> dict[str, Any]:
        fields = self.select(request)
        geometries = {self.geometry(field.key).key: self.geometry(field.key) for field in fields}
        diagnostic_layouts = {item.key.layout_identity.token
                              for item in self.select_diagnostics(request)}
        geometries.update({item.key: item for item in self.geometries
                           if item.layout_identity.token in diagnostic_layouts})
        return {
            "clock": self.clock.to_data(), "provenance": self.provenance.to_data(),
            "selection": request.to_data(),
            "geometries": [item.to_data() for item in sorted(geometries.values(), key=lambda x: x.key)],
            "fields": [item.to_data() for item in fields],
            "diagnostics": [item.to_data() for item in self.select_diagnostics(request)],
            "metadata": dict(self.metadata),
        }


__all__ = [
    "ArrayPiece", "DiagnosticKey", "DiagnosticPayload", "FieldKey", "FieldPayload",
    "LevelGeometry", "OutputClock",
    "OutputProvenance", "OutputRequest", "OutputSnapshot", "array_evidence",
]
