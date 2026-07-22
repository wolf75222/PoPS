"""Host-side previews for rectangular domains and generic implicit geometries.

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

from pops.frames import Cartesian2D
from .rectangle import Rectangle


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class AnalyticPreviewValue(Protocol):
    """Structural view of one canonical analytic scalar or predicate expression."""

    def to_data(self) -> Mapping[str, Any]:
        """Return the canonical data-only expression tree."""

        raise NotImplementedError


class GeometryPreviewProvider(Protocol):
    """Small presentation protocol shared by built-in and third-party geometries."""

    def level_set(self, frame: Any) -> Any:
        """Return an object exposing one canonical analytic ``expression``."""


def _checked_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value, value)
    elif not isinstance(value, (str, bytes)) and isinstance(value, Sequence) \
            and len(value) == 2:
        values = tuple(value)
    else:
        raise TypeError("preview resolution must be an integer or a pair of integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise TypeError("preview resolution entries must be integers, never bool")
    if any(item < 2 for item in values):
        raise ValueError("preview resolution entries must be >= 2")
    return (values[0], values[1])


def _readonly_float_array(value: Any, *, ndim: int, where: str) -> FloatArray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (where, ndim))
    if not np.isfinite(result).all():
        raise ValueError("%s must contain only finite values" % where)
    result.setflags(write=False)
    return result


def _readonly_bool_array(value: Any, *, shape: tuple[int, int], where: str) -> BoolArray:
    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    if result.shape != shape:
        raise ValueError("%s has shape %r instead of %r" % (where, result.shape, shape))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class DomainPreview:
    """Sampled presentation data for one domain, analytic field, and implicit geometry."""

    domain: Rectangle
    geometry: GeometryPreviewProvider | None
    x: FloatArray
    y: FloatArray
    level_set_values: FloatArray | None = None
    active_mask: BoolArray | None = None
    field: AnalyticPreviewValue | None = None
    field_values: NDArray[Any] | None = None
    field_kind: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Rectangle):
            raise TypeError("DomainPreview.domain must be a Rectangle")
        if self.geometry is not None and not callable(getattr(self.geometry, "level_set", None)):
            raise TypeError("DomainPreview.geometry must implement level_set(frame)")
        x_values = _readonly_float_array(self.x, ndim=1, where="DomainPreview.x")
        y_values = _readonly_float_array(self.y, ndim=1, where="DomainPreview.y")
        if x_values.size < 2 or y_values.size < 2:
            raise ValueError("DomainPreview axes must each contain at least two samples")
        object.__setattr__(self, "x", x_values)
        object.__setattr__(self, "y", y_values)

        expected_shape = (y_values.size, x_values.size)
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
                self.level_set_values, ndim=2, where="DomainPreview.level_set_values")
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
                self.field_values, ndim=2, where="DomainPreview.field_values")
            if field_values.shape != expected_shape:
                raise ValueError(
                    "DomainPreview.field_values has shape %r instead of %r"
                    % (field_values.shape, expected_shape)
                )
        object.__setattr__(self, "field_values", field_values)
        object.__setattr__(self, "field_kind", field_kind)

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the sample count in Cartesian ``(x, y)`` order."""

        return (int(self.x.size), int(self.y.size))

    def show(self, *, path: str | PathLike[str] | None = None) -> Path | None:
        """Show interactively, or save when ``path`` is provided.

        The filename extension selects any format supported by the installed Matplotlib (for
        example ``.png``, ``.svg`` or ``.pdf``).  Saving always closes the figure and never opens an
        interactive window.
        """

        return _show_matplotlib(self, path=path)


def preview_rectangle(
    domain: Rectangle,
    *,
    geometry: GeometryPreviewProvider | None = None,
    field: AnalyticPreviewValue | None = None,
    resolution: int | tuple[int, int] = (256, 256),
) -> DomainPreview:
    """Sample analytic data over ``domain`` through canonical expression contracts."""

    if not isinstance(domain, Rectangle):
        raise TypeError("preview_rectangle requires a Rectangle")
    if geometry is not None and not callable(getattr(geometry, "level_set", None)):
        raise TypeError("Rectangle.preview geometry must implement level_set(frame)")
    if field is not None:
        _expression_kind(field, where="Rectangle.preview field")
    nx, ny = _checked_resolution(resolution)
    x_values = np.linspace(domain.lower[0], domain.upper[0], nx, dtype=np.float64)
    y_values = np.linspace(domain.lower[1], domain.upper[1], ny, dtype=np.float64)
    if geometry is None and field is None:
        return DomainPreview(domain, None, x_values, y_values)

    frame = domain.frame(Cartesian2D())
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    level_set_values = None
    active_mask = None
    if geometry is not None:
        level_set = geometry.level_set(frame)
        expression = getattr(level_set, "expression", None)
        level_set_values, level_set_kind = _sample_expression(
            expression, frame_id=frame.canonical_id, x=xx, y=yy,
            where="geometry level set")
        if level_set_kind != "scalar":
            raise TypeError("geometry level set must be scalar, never a predicate")
        active_mask = level_set_values < 0.0

    field_values = None
    if field is not None:
        field_values, _ = _sample_expression(
            field, frame_id=frame.canonical_id, x=xx, y=yy, where="analytic field")
    return DomainPreview(
        domain, geometry, x_values, y_values, level_set_values, active_mask,
        field, field_values,
    )


def _sample_expression(
    expression: Any,
    *,
    frame_id: str,
    x: FloatArray,
    y: FloatArray,
    where: str,
) -> tuple[NDArray[Any], str]:
    expression_data = _expression_data(expression, where=where)
    expression_kind = expression_data["expression_type"]
    values, valid = _evaluate_node(
        expression_data["root"], frame_id=frame_id, x=x, y=y)
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
    return data


def _expression_kind(expression: Any, *, where: str) -> str:
    return str(_expression_data(expression, where=where)["expression_type"])


def _constant_grid(value: float, shape: tuple[int, int]) -> tuple[FloatArray, BoolArray]:
    values = np.full(shape, value, dtype=np.float64)
    return values, np.ones(shape, dtype=np.bool_)


def _evaluate_node(
    node: Mapping[str, Any],
    *,
    frame_id: str,
    x: FloatArray,
    y: FloatArray,
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
        if direction not in {"x", "y"}:
            raise ValueError("geometry preview supports only Cartesian x/y coordinates")
        values = x if direction == "x" else y
        return values, np.isfinite(values)
    if kind == "scalar" and op == "parameter":
        raise TypeError("geometry preview cannot sample unresolved analytic parameters")
    if kind == "scalar" and op == "input":
        raise TypeError("geometry preview cannot sample runtime analytic inputs")

    arguments = [
        _evaluate_node(argument, frame_id=frame_id, x=x, y=y)
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
        raise TypeError("DomainPreview.show path must be text or path-like") from None
    if not result.name or not result.suffix:
        raise ValueError("DomainPreview.show path must include a filename extension")
    return result


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


__all__ = [
    "AnalyticPreviewValue", "DomainPreview", "GeometryPreviewProvider", "preview_rectangle",
]
