"""Host-side previews for bounded domains and generic implicit geometries.

Previewing is presentation only: NumPy samples the same canonical analytic expression that the
runtime lowers to its native evaluator, while Matplotlib is imported only when a figure is shown or
saved.  Geometry providers remain generic because every shape enters through
``Geometry.level_set(frame)``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pops.frames import Cartesian, Cartesian1D, Cartesian2D, Cartesian3D
from pops.identity import make_identity


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_AXIS_NAMES = ("x", "y", "z")
_RANKED_CARTESIAN = (Cartesian1D, Cartesian2D, Cartesian3D)


class AnalyticPreviewValue(Protocol):
    """Structural view of one canonical analytic scalar or predicate expression."""

    def to_data(self) -> Mapping[str, Any]:
        """Return the canonical data-only expression tree."""

        raise NotImplementedError


class GeometryPreviewProvider(Protocol):
    """Small presentation protocol shared by built-in and third-party geometries."""

    def level_set(self, frame: Any) -> Any:
        """Return an object exposing one canonical analytic ``expression``."""


class PreviewBoundaryNames(Protocol):
    """Structural labels used by the generic Cartesian preview renderer."""

    @property
    def x_min(self) -> str:
        raise NotImplementedError

    @property
    def x_max(self) -> str:
        raise NotImplementedError

    @property
    def y_min(self) -> str:
        raise NotImplementedError

    @property
    def y_max(self) -> str:
        raise NotImplementedError

    @property
    def z_min(self) -> str:
        raise NotImplementedError

    @property
    def z_max(self) -> str:
        raise NotImplementedError


class PreviewDomainProvider(Protocol):
    """Minimal bounded-domain protocol consumed by sampling and rendering."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def lower(self) -> tuple[float, ...]:
        raise NotImplementedError

    @property
    def upper(self) -> tuple[float, ...]:
        raise NotImplementedError

    @property
    def boundary_names(self) -> PreviewBoundaryNames:
        raise NotImplementedError

    @property
    def lengths(self) -> tuple[float, ...]:
        """Return positive Cartesian lengths."""

        raise NotImplementedError

    def frame(self, coordinates: Cartesian) -> Any:
        """Bind the domain to a typed Cartesian frame."""


@dataclass(frozen=True, slots=True)
class _DefaultBoundaryNames:
    x_min: str = "x_min"
    x_max: str = "x_max"
    y_min: str = "y_min"
    y_max: str = "y_max"
    z_min: str = "z_min"
    z_max: str = "z_max"


@dataclass(frozen=True, slots=True)
class _GeometryPreviewDomain:
    """Presentation-only bounded window owned by an implicit geometry preview."""

    name: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    frame_id: str | None
    boundary_names: _DefaultBoundaryNames = _DefaultBoundaryNames()

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(high - low for low, high in zip(self.lower, self.upper, strict=True))

    def frame(self, coordinates: Cartesian) -> _GeometryPreviewFrame:
        if not isinstance(coordinates, Cartesian):
            raise TypeError("geometry preview frame requires Cartesian")
        if coordinates.dimension != len(self.lower):
            raise ValueError("geometry preview coordinate and domain ranks differ")
        return _GeometryPreviewFrame(self, coordinates)


@dataclass(frozen=True, slots=True)
class _GeometryPreviewFrame:
    domain: _GeometryPreviewDomain
    coordinates: Cartesian

    @property
    def axes(self) -> Any:
        return self.coordinates.axes

    @property
    def x(self) -> Any:
        return self.coordinates.x

    @property
    def y(self) -> Any:
        return self.coordinates.y

    @property
    def z(self) -> Any:
        return self.coordinates.z

    @property
    def lower(self) -> tuple[float, ...]:
        return self.domain.lower

    @property
    def upper(self) -> tuple[float, ...]:
        return self.domain.upper

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_type": "geometry_preview_cartesian",
            "lower_binary64": [value.hex() for value in self.lower],
            "upper_binary64": [value.hex() for value in self.upper],
            "coordinates": self.coordinates.to_dict(),
        }

    @property
    def canonical_id(self) -> str:
        if self.domain.frame_id is not None:
            return self.domain.frame_id
        return make_identity("geometry-preview-frame", self.to_dict(), schema_version=1).token


def _preview_rank(lower: Any) -> int:
    if isinstance(lower, (str, bytes)) or not isinstance(lower, Sequence):
        raise TypeError("preview domain lower must be a coordinate sequence")
    rank = len(tuple(lower))
    if rank not in (1, 2, 3):
        raise ValueError("preview domain rank must be 1, 2, or 3")
    return rank


def _checked_resolution(value: Any, rank: int) -> tuple[int, ...]:
    if value is None:
        size = 24 if rank == 3 else 256
        values = (size,) * rank
    elif isinstance(value, int) and not isinstance(value, bool):
        values = (value,) * rank
    elif not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        if len(value) != rank:
            raise TypeError(
                "preview resolution must be an integer or a sequence of %d integers" % rank
            )
        values = tuple(value)
    else:
        raise TypeError("preview resolution must be an integer or a sequence of integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise TypeError("preview resolution entries must be integers, never bool")
    if any(item < 2 for item in values):
        raise ValueError("preview resolution entries must be >= 2")
    return values


def _checked_extent(value: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError("preview extent must contain lower and upper points")
    points: list[tuple[float, ...]] = []
    for point_index, point in enumerate(value):
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) \
                or len(point) not in (1, 2, 3):
            raise TypeError(
                "preview extent point %d must contain one, two, or three coordinates"
                % point_index
            )
        coordinates = []
        for coordinate in point:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise TypeError("preview extent coordinates must be real numbers, never bool")
            converted = float(coordinate)
            if not np.isfinite(converted):
                raise ValueError("preview extent coordinates must be finite")
            coordinates.append(converted)
        points.append(tuple(coordinates))
    lower, upper = points
    if len(lower) != len(upper):
        raise ValueError("preview extent lower and upper must have one common rank")
    if any(high <= low for low, high in zip(lower, upper, strict=True)):
        raise ValueError("preview extent upper coordinates must exceed lower coordinates")
    return lower, upper


def _validate_preview_domain(domain: Any) -> int:
    name = getattr(domain, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("DomainPreview.domain must expose a non-empty name")
    lower, upper = _checked_extent((getattr(domain, "lower", None),
                                    getattr(domain, "upper", None)))
    rank = len(lower)
    lengths = getattr(domain, "lengths", None)
    expected_lengths = tuple(high - low for low, high in zip(lower, upper, strict=True))
    if not isinstance(lengths, Sequence) or tuple(lengths) != expected_lengths:
        raise TypeError("DomainPreview.domain lengths must match its lower and upper bounds")
    labels = getattr(domain, "boundary_names", None)
    for name in _AXIS_NAMES[:rank]:
        for suffix in ("min", "max"):
            face = "%s_%s" % (name, suffix)
            label = getattr(labels, face, None)
            if not isinstance(label, str) or not label:
                raise TypeError(
                    "DomainPreview.domain must expose non-empty boundary labels for every axis"
                )
    if not callable(getattr(domain, "frame", None)):
        raise TypeError("DomainPreview.domain must implement frame(Cartesian)")
    return rank


def _bind_preview_frame(domain: Any, rank: int) -> Any:
    owned = getattr(domain, "coordinates", None)
    if isinstance(owned, Cartesian):
        try:
            return domain.frame(owned)
        except TypeError:
            return domain.frame()
    return domain.frame(_RANKED_CARTESIAN[rank - 1]())


def _readonly_float_array(value: Any, *, ndim: int, where: str) -> FloatArray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (where, ndim))
    if not np.isfinite(result).all():
        raise ValueError("%s must contain only finite values" % where)
    result.setflags(write=False)
    return result


def _readonly_bool_array(value: Any, *, shape: tuple[int, ...], where: str) -> BoolArray:
    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    if result.shape != shape:
        raise ValueError("%s has shape %r instead of %r" % (where, result.shape, shape))
    result.setflags(write=False)
    return result


def _sample_grids(
    x_values: FloatArray,
    y_values: FloatArray | None,
    z_values: FloatArray | None,
    rank: int,
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    if rank == 1:
        return x_values, None, None
    if rank == 2:
        xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
        return xx, yy, None
    zz, yy, xx = np.meshgrid(z_values, y_values, x_values, indexing="ij")
    return xx, yy, zz


@dataclass(frozen=True, slots=True, eq=False)
class DomainPreview:
    """Sampled presentation data for one domain, analytic field, and implicit geometry."""

    domain: PreviewDomainProvider
    geometry: GeometryPreviewProvider | None
    x: FloatArray
    y: FloatArray | None = None
    level_set_values: FloatArray | None = None
    active_mask: BoolArray | None = None
    field: AnalyticPreviewValue | None = None
    field_values: NDArray[Any] | None = None
    field_kind: str | None = None
    z: FloatArray | None = None

    def __post_init__(self) -> None:
        rank = _validate_preview_domain(self.domain)
        if self.geometry is not None and not callable(getattr(self.geometry, "level_set", None)):
            raise TypeError("DomainPreview.geometry must implement level_set(frame)")
        x_values = _readonly_float_array(self.x, ndim=1, where="DomainPreview.x")
        y_values = None
        z_values = None
        if rank == 1:
            if self.y is not None or self.z is not None:
                raise ValueError("DomainPreview y and z must be None for rank 1")
            expected_shape: tuple[int, ...] = (x_values.size,)
        elif rank == 2:
            if self.y is None:
                raise ValueError("DomainPreview.y is required for rank 2")
            if self.z is not None:
                raise ValueError("DomainPreview.z must be None for rank 2")
            y_values = _readonly_float_array(self.y, ndim=1, where="DomainPreview.y")
            expected_shape = (y_values.size, x_values.size)
        else:
            if self.y is None or self.z is None:
                raise ValueError("DomainPreview.y and DomainPreview.z are required for rank 3")
            y_values = _readonly_float_array(self.y, ndim=1, where="DomainPreview.y")
            z_values = _readonly_float_array(self.z, ndim=1, where="DomainPreview.z")
            expected_shape = (z_values.size, y_values.size, x_values.size)
        axes = {"x": x_values, "y": y_values, "z": z_values}
        if any(axes[name].size < 2 for name in _AXIS_NAMES[:rank]):
            raise ValueError("DomainPreview axes must each contain at least two samples")
        object.__setattr__(self, "x", x_values)
        object.__setattr__(self, "y", y_values)
        object.__setattr__(self, "z", z_values)

        if (self.level_set_values is None) != (self.active_mask is None):
            raise ValueError(
                "DomainPreview level-set values and active mask must be present together"
            )
        if self.geometry is None:
            if self.level_set_values is not None:
                raise ValueError("DomainPreview sampled level-set values require a geometry")
        else:
            if self.level_set_values is None:
                raise ValueError("DomainPreview geometry requires sampled level-set values")
            level_set_values = _readonly_float_array(
                self.level_set_values, ndim=rank, where="DomainPreview.level_set_values")
            if level_set_values.shape != expected_shape:
                raise ValueError(
                    "DomainPreview.level_set_values has shape %r instead of %r"
                    % (level_set_values.shape, expected_shape)
                )
            active_mask = _readonly_bool_array(
                self.active_mask, shape=expected_shape, where="DomainPreview.active_mask")
            if not np.array_equal(active_mask, level_set_values < 0.0):
                raise ValueError(
                    "DomainPreview.active_mask must equal level_set_values < 0"
                )
            object.__setattr__(self, "level_set_values", level_set_values)
            object.__setattr__(self, "active_mask", active_mask)

        if self.field is None:
            if self.field_values is not None or self.field_kind is not None:
                raise ValueError(
                    "DomainPreview sampled field data require an analytic field")
            return
        field_kind = _expression_kind(self.field, where="DomainPreview.field")
        if self.field_kind not in (None, field_kind):
            raise ValueError("DomainPreview.field_kind disagrees with the canonical expression")
        if self.field_values is None:
            raise ValueError("DomainPreview analytic field requires sampled field values")
        if field_kind == "predicate":
            field_values = _readonly_bool_array(
                self.field_values, shape=expected_shape, where="DomainPreview.field_values")
        else:
            field_values = _readonly_float_array(
                self.field_values, ndim=rank, where="DomainPreview.field_values")
            if field_values.shape != expected_shape:
                raise ValueError(
                    "DomainPreview.field_values has shape %r instead of %r"
                    % (field_values.shape, expected_shape)
                )
        object.__setattr__(self, "field_values", field_values)
        object.__setattr__(self, "field_kind", field_kind)

    @property
    def dimension(self) -> int:
        return _preview_rank(self.domain.lower)

    @property
    def resolution(self) -> tuple[int, ...]:
        """Return the sample count in Cartesian ``(x, y, z)`` order."""

        axes = {"x": self.x, "y": self.y, "z": self.z}
        return tuple(int(axes[name].size) for name in _AXIS_NAMES[:self.dimension])

    def show(self, *, path: str | PathLike[str] | None = None) -> Path | None:
        """Show interactively, or save when ``path`` is provided.

        The filename extension selects any format supported by the installed Matplotlib (for
        example ``.png``, ``.svg`` or ``.pdf``).  Saving always closes the figure and never opens an
        interactive window.
        """

        if path is not None:
            return self.export(path)
        return _show_matplotlib(self, path=None)

    def export(self, path: str | PathLike[str]) -> Path:
        """Write this preview without opening an interactive window."""

        result = _show_matplotlib(self, path=path)
        if result is None:
            raise RuntimeError("DomainPreview export did not produce an output path")
        return result


def preview_domain(
    domain: PreviewDomainProvider,
    *,
    geometry: GeometryPreviewProvider | None = None,
    field: AnalyticPreviewValue | None = None,
    resolution: int | Sequence[int] | None = None,
) -> DomainPreview:
    """Sample analytic data over ``domain`` through canonical expression contracts."""

    rank = _validate_preview_domain(domain)
    if geometry is not None and not callable(getattr(geometry, "level_set", None)):
        raise TypeError("domain preview geometry must implement level_set(frame)")
    if field is not None:
        _expression_kind(field, where="domain preview field")
    sizes = _checked_resolution(resolution, rank)
    axis_samples = {
        name: np.linspace(
            domain.lower[index], domain.upper[index], sizes[index], dtype=np.float64)
        for index, name in enumerate(_AXIS_NAMES[:rank])
    }
    x_values = axis_samples["x"]
    y_values = axis_samples.get("y")
    z_values = axis_samples.get("z")
    if geometry is None and field is None:
        return DomainPreview(domain, None, x_values, y_values, z=z_values)

    frame = _bind_preview_frame(domain, rank)
    sample_x, sample_y, sample_z = _sample_grids(x_values, y_values, z_values, rank)
    level_set_values = None
    active_mask = None
    if geometry is not None:
        level_set = geometry.level_set(frame)
        expression = getattr(level_set, "expression", None)
        level_set_values, level_set_kind = _sample_expression(
            expression, frame_id=frame.canonical_id, x=sample_x, y=sample_y, z=sample_z,
            where="geometry level set")
        if level_set_kind != "scalar":
            raise TypeError("geometry level set must be scalar, never a predicate")
        active_mask = level_set_values < 0.0

    field_values = None
    if field is not None:
        field_values, _ = _sample_expression(
            field, frame_id=frame.canonical_id, x=sample_x, y=sample_y, z=sample_z,
            where="analytic field")
    return DomainPreview(
        domain, geometry, x_values, y_values, level_set_values, active_mask,
        field, field_values, z=z_values,
    )


def preview_geometry(
    geometry: GeometryPreviewProvider,
    *,
    extent: Any = None,
    resolution: int | Sequence[int] | None = None,
) -> DomainPreview:
    """Preview any implicit-geometry provider in its own bounded presentation window."""

    if not callable(getattr(geometry, "level_set", None)):
        raise TypeError("preview_geometry requires a provider implementing level_set(frame)")
    if extent is None:
        extent_provider = getattr(geometry, "preview_extent", None)
        extent = extent_provider() if callable(extent_provider) else ((-1.0, -1.0), (1.0, 1.0))
    lower, upper = _checked_extent(extent)
    frame_provider = getattr(geometry, "preview_frame_id", None)
    frame_id = frame_provider() if callable(frame_provider) else None
    if frame_id is not None and (not isinstance(frame_id, str) or not frame_id):
        raise TypeError("geometry preview_frame_id() must return non-empty text or None")
    name = getattr(geometry, "name", type(geometry).__name__)
    if not isinstance(name, str) or not name:
        raise TypeError("geometry preview name must be non-empty text")
    domain = _GeometryPreviewDomain(name, lower, upper, frame_id)
    return preview_domain(domain, geometry=geometry, resolution=resolution)


def _sample_expression(
    expression: Any,
    *,
    frame_id: str,
    x: FloatArray,
    y: FloatArray | None = None,
    z: FloatArray | None = None,
    where: str,
) -> tuple[NDArray[Any], str]:
    expression_data = _expression_data(expression, where=where)
    expression_kind = expression_data["expression_type"]
    values, valid = _evaluate_node(
        expression_data["root"], frame_id=frame_id, x=x, y=y, z=z)
    dtype = np.bool_ if expression_kind == "predicate" else np.float64
    sampled = np.asarray(values, dtype=dtype)
    validity = np.asarray(valid, dtype=np.bool_)
    if sampled.shape != x.shape:
        sampled = np.broadcast_to(sampled, x.shape)
    if validity.shape != x.shape:
        validity = np.broadcast_to(validity, x.shape)
    if not validity.all():
        invalid_count = int(validity.size - np.count_nonzero(validity))
        raise ValueError(
            "%s is undefined at %d preview sample(s)" % (where, invalid_count))
    return sampled, expression_kind


def _expression_data(expression: Any, *, where: str) -> Mapping[str, Any]:
    to_data = getattr(expression, "to_data", None)
    if not callable(to_data):
        raise TypeError("%s must implement canonical to_data()" % where)
    data = to_data()
    if not isinstance(data, Mapping) or data.get("expression_type") not in {
            "scalar", "predicate"} or not isinstance(data.get("root"), Mapping):
        raise TypeError("%s must expose canonical analytic expression data" % where)

    # Built-in expressions validate their own immutable graph in ``to_data``. Structural
    # third-party providers are useful for inspection tooling, but their claim of exposing the
    # canonical schema must be authenticated before the evaluator traverses arbitrary mappings.
    # Reusing the owning decoder keeps preview and native lowering on one operation/arity contract
    # and turns malformed data into a precise authoring error rather than a late KeyError.
    from pops.analytic import PredicateExpr, ScalarExpr

    if isinstance(expression, (ScalarExpr, PredicateExpr)):
        return data
    expression_type = data["expression_type"]
    decoded = (ScalarExpr.from_data(data) if expression_type == "scalar"
               else PredicateExpr.from_data(data))
    return decoded.to_data()


def _expression_kind(expression: Any, *, where: str) -> str:
    return str(_expression_data(expression, where=where)["expression_type"])


def _constant_grid(value: float, shape: tuple[int, ...]) -> tuple[FloatArray, BoolArray]:
    values = np.full(shape, value, dtype=np.float64)
    return values, np.ones(shape, dtype=np.bool_)


def _evaluate_node(
    node: Mapping[str, Any],
    *,
    frame_id: str,
    x: FloatArray,
    y: FloatArray | None = None,
    z: FloatArray | None = None,
) -> tuple[NDArray[Any], BoolArray]:
    """Vectorized counterpart of the native analytic evaluator for presentation sampling."""

    op = node["op"]
    kind = node["kind"]
    shape = x.shape
    if kind == "scalar" and op == "constant":
        return _constant_grid(float.fromhex(node["value"]["binary64"]), shape)
    if kind == "scalar" and op == "coordinate":
        if node["frame_id"] != frame_id:
            raise ValueError("geometry preview expression belongs to another frame")
        direction = node["axis"]["direction"]
        available = {"x": x, "y": y, "z": z}
        if direction not in available or available[direction] is None:
            raise ValueError("geometry preview does not expose a %s coordinate" % direction)
        values = available[direction]
        return values, np.isfinite(values)
    if kind == "scalar" and op == "parameter":
        raise TypeError("geometry preview cannot sample unresolved analytic parameters")
    if kind == "scalar" and op == "input":
        raise TypeError("geometry preview cannot sample runtime analytic inputs")

    arguments = [
        _evaluate_node(argument, frame_id=frame_id, x=x, y=y, z=z)
        for argument in node["arguments"]
    ]
    values = [argument[0] for argument in arguments]
    validity = [argument[1] for argument in arguments]
    with np.errstate(all="ignore"):
        if kind == "scalar" and op in {
            "neg", "sqrt", "abs", "sin", "cos", "exp", "log",
        }:
            functions = {
                "neg": np.negative,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "log": np.log,
            }
            result = functions[op](values[0])
            return result, validity[0] & np.isfinite(result)
        if kind == "scalar" and op in {
            "add", "sub", "mul", "div", "pow", "atan2", "hypot", "minimum", "maximum",
        }:
            functions = {
                "add": np.add,
                "sub": np.subtract,
                "mul": np.multiply,
                "div": np.divide,
                "pow": np.power,
                "atan2": np.arctan2,
                "hypot": np.hypot,
                "minimum": np.fmin,
                "maximum": np.fmax,
            }
            result = functions[op](values[0], values[1])
            return result, validity[0] & validity[1] & np.isfinite(result)
        if kind == "scalar" and op == "where":
            condition = np.asarray(values[0], dtype=np.bool_)
            result = np.where(condition, values[1], values[2])
            valid = validity[0] & np.where(condition, validity[1], validity[2])
            return result, valid

        if kind == "predicate" and op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            functions = {
                "eq": np.equal,
                "ne": np.not_equal,
                "lt": np.less,
                "le": np.less_equal,
                "gt": np.greater,
                "ge": np.greater_equal,
            }
            return functions[op](values[0], values[1]), validity[0] & validity[1]
        if kind == "predicate" and op in {"and", "or"}:
            function = np.logical_and if op == "and" else np.logical_or
            return function(values[0], values[1]), validity[0] & validity[1]
        if kind == "predicate" and op == "not":
            return np.logical_not(values[0]), validity[0]
        if kind == "predicate" and op == "between":
            result = np.greater_equal(values[0], values[1]) \
                & np.less_equal(values[0], values[2])
            return result, validity[0] & validity[1] & validity[2]
    raise ValueError("unsupported analytic preview operation %r" % op)


def _checked_output_path(value: str | PathLike[str]) -> Path:
    try:
        result = Path(value)
    except TypeError:
        raise TypeError("DomainPreview output path must be text or path-like") from None
    if not result.name or not result.suffix:
        raise ValueError("DomainPreview output path must include a filename extension")
    return result


def _finish_matplotlib_figure(plt: Any, figure: Any, output_path: Path | None) -> Path | None:
    figure.tight_layout()
    if output_path is None:
        try:
            plt.show()
        finally:
            plt.close(figure)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path


def _show_matplotlib_1d(preview: DomainPreview, plt: Any, output_path: Path | None) -> Path | None:
    domain = preview.domain
    figure, axes = plt.subplots(figsize=(7.0, 3.2))
    axes.set_facecolor("#f7f8fa")
    if preview.field_values is not None:
        if preview.field_kind == "predicate":
            axes.fill_between(
                preview.x,
                0.0,
                preview.field_values.astype(np.float64),
                color="#4c9bd3",
                alpha=0.85,
                step="mid",
            )
        else:
            axes.plot(preview.x, preview.field_values, color="#16618f", linewidth=1.8)
    elif preview.active_mask is not None:
        axes.fill_between(
            preview.x,
            0.0,
            preview.active_mask.astype(np.float64),
            color="#b9dcf5",
            alpha=0.9,
            step="mid",
        )
    else:
        axes.plot(
            preview.x,
            np.zeros(preview.x.shape, dtype=np.float64),
            color="#20252b",
            linewidth=2.4,
            solid_capstyle="butt",
        )
        axes.set_yticks([])
    labels = domain.boundary_names
    y_min, y_max = axes.get_ylim()
    y_mid = 0.5 * (y_min + y_max)
    x_offset = 0.025 * domain.lengths[0]
    axes.text(domain.lower[0] - x_offset, y_mid, labels.x_min,
              ha="right", va="center", rotation=90, clip_on=False)
    axes.text(domain.upper[0] + x_offset, y_mid, labels.x_max,
              ha="left", va="center", rotation=90, clip_on=False)
    axes.set_xlim(domain.lower[0], domain.upper[0])
    axes.set_xlabel("x")
    axes.set_title(domain.name)
    axes.grid(color="#d7dce1", linewidth=0.5, alpha=0.6)
    return _finish_matplotlib_figure(plt, figure, output_path)


def _show_matplotlib_2d(
    preview: DomainPreview,
    plt: Any,
    ListedColormap: Any,
    RectanglePatch: Any,
    output_path: Path | None,
) -> Path | None:
    domain = preview.domain
    width, height = domain.lengths
    figure_width = 7.0
    figure_height = max(4.0, min(8.0, figure_width * height / width))
    figure, axes = plt.subplots(figsize=(figure_width, figure_height))
    axes.set_facecolor("#f7f8fa")
    extent = (domain.lower[0], domain.upper[0], domain.lower[1], domain.upper[1])
    if preview.field_values is not None:
        if preview.field_kind == "predicate":
            axes.imshow(
                preview.field_values.astype(np.float64),
                extent=extent,
                origin="lower",
                interpolation="nearest",
                cmap=ListedColormap(["#f7f8fa", "#4c9bd3"]),
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
                aspect="auto",
            )
        else:
            image = axes.imshow(
                preview.field_values,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
                aspect="auto",
            )
            figure.colorbar(image, ax=axes, label="value", shrink=0.85)
    if preview.active_mask is not None and preview.level_set_values is not None:
        if preview.field_values is None:
            axes.imshow(
                preview.active_mask.astype(np.float64),
                extent=extent,
                origin="lower",
                interpolation="nearest",
                cmap=ListedColormap(["#f7f8fa", "#b9dcf5"]),
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
                aspect="auto",
            )
        minimum = float(np.min(preview.level_set_values))
        maximum = float(np.max(preview.level_set_values))
        if minimum < 0.0 < maximum:
            axes.contour(
                preview.x,
                preview.y,
                preview.level_set_values,
                levels=(0.0,),
                colors=("#16618f",),
                linewidths=(1.6,),
            )

    axes.add_patch(RectanglePatch(
        domain.lower, width, height, fill=False, edgecolor="#20252b", linewidth=1.8))
    labels = domain.boundary_names
    x_mid = 0.5 * (domain.lower[0] + domain.upper[0])
    y_mid = 0.5 * (domain.lower[1] + domain.upper[1])
    x_offset = 0.025 * width
    y_offset = 0.025 * height
    axes.text(domain.lower[0] - x_offset, y_mid, labels.x_min,
              ha="right", va="center", rotation=90, clip_on=False)
    axes.text(domain.upper[0] + x_offset, y_mid, labels.x_max,
              ha="left", va="center", rotation=90, clip_on=False)
    axes.text(x_mid, domain.lower[1] - y_offset, labels.y_min,
              ha="center", va="top", clip_on=False)
    axes.text(x_mid, domain.upper[1] + y_offset, labels.y_max,
              ha="center", va="bottom", clip_on=False)
    axes.set_xlim(domain.lower[0], domain.upper[0])
    axes.set_ylim(domain.lower[1], domain.upper[1])
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(domain.name)
    axes.grid(color="#d7dce1", linewidth=0.5, alpha=0.6)
    return _finish_matplotlib_figure(plt, figure, output_path)


def _box_edges(
    lower: tuple[float, ...], upper: tuple[float, ...]
) -> tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...]:
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    return (
        ((x0, x1), (y0, y0), (z0, z0)),
        ((x0, x1), (y1, y1), (z0, z0)),
        ((x0, x1), (y0, y0), (z1, z1)),
        ((x0, x1), (y1, y1), (z1, z1)),
        ((x0, x0), (y0, y1), (z0, z0)),
        ((x1, x1), (y0, y1), (z0, z0)),
        ((x0, x0), (y0, y1), (z1, z1)),
        ((x1, x1), (y0, y1), (z1, z1)),
        ((x0, x0), (y0, y0), (z0, z1)),
        ((x1, x1), (y0, y0), (z0, z1)),
        ((x0, x0), (y1, y1), (z0, z1)),
        ((x1, x1), (y1, y1), (z0, z1)),
    )


def _show_matplotlib_3d(preview: DomainPreview, plt: Any, output_path: Path | None) -> Path | None:
    import_module("mpl_toolkits.mplot3d")
    domain = preview.domain
    figure = plt.figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor("#f7f8fa")
    for xs, ys, zs in _box_edges(domain.lower, domain.upper):
        axes.plot(xs, ys, zs, color="#20252b", linewidth=1.6)
    values = preview.field_values
    if values is None and preview.active_mask is not None:
        values = preview.active_mask.astype(np.float64)
    if values is not None and preview.y is not None and preview.z is not None:
        if preview.field_kind == "predicate":
            values = values.astype(np.float64)
        iz = int(preview.z.size) // 2
        iy = int(preview.y.size) // 2
        ix = int(preview.x.size) // 2
        xx, yy = np.meshgrid(preview.x, preview.y, indexing="xy")
        axes.contourf(
            xx, yy, values[iz], zdir="z", offset=float(preview.z[iz]),
            cmap="viridis", alpha=0.7)
        xx, zz = np.meshgrid(preview.x, preview.z, indexing="xy")
        axes.contourf(
            xx, values[:, iy, :], zz, zdir="y", offset=float(preview.y[iy]),
            cmap="viridis", alpha=0.7)
        yy, zz = np.meshgrid(preview.y, preview.z, indexing="xy")
        axes.contourf(
            values[:, :, ix], yy, zz, zdir="x", offset=float(preview.x[ix]),
            cmap="viridis", alpha=0.7)
    axes.set_xlim(domain.lower[0], domain.upper[0])
    axes.set_ylim(domain.lower[1], domain.upper[1])
    axes.set_zlim(domain.lower[2], domain.upper[2])
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title(domain.name)
    return _finish_matplotlib_figure(plt, figure, output_path)


def _show_matplotlib(
    preview: DomainPreview,
    *,
    path: str | PathLike[str] | None,
) -> Path | None:
    output_path = None if path is None else _checked_output_path(path)
    try:
        plt = import_module("matplotlib.pyplot")
        ListedColormap = import_module("matplotlib.colors").ListedColormap
        RectanglePatch = import_module("matplotlib.patches").Rectangle
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "DomainPreview.show requires Matplotlib; "
            "install it with 'python -m pip install matplotlib'"
        ) from None

    rank = preview.dimension
    if rank == 1:
        return _show_matplotlib_1d(preview, plt, output_path)
    if rank == 2:
        return _show_matplotlib_2d(preview, plt, ListedColormap, RectanglePatch, output_path)
    return _show_matplotlib_3d(preview, plt, output_path)


__all__ = [
    "AnalyticPreviewValue", "DomainPreview", "GeometryPreviewProvider", "PreviewDomainProvider",
    "preview_domain", "preview_geometry",
]
