"""Pre-implemented, data-only initial profiles.

Profiles contain no Python callback.  They expose a small open protocol consumed by
``pops.initial.InitialCondition``: ``validate_for``, ``initial_source_options`` and ``to_data``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a finite numeric scalar" % where)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % where)
    return result


def _declaration(state: Any) -> Any:
    return getattr(state, "declaration_ref", None) or state


def _component_count(state: Any) -> int:
    declaration = _declaration(state)
    components = getattr(declaration, "components", None)
    if components is None:
        components = getattr(getattr(declaration, "space", None), "components", None)
    if not isinstance(components, tuple) or not components:
        raise TypeError("initial profile requires a state with declared components")
    return len(components)


def _reference_key(value: Any, *, where: str) -> str:
    """Canonical comparison key for one captured Handle projection."""

    if type(value) is not dict:
        raise TypeError("%s must be an exact Handle data mapping" % where)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must contain strict JSON Handle data" % where) from exc


def _replace_captured_parameter_references(
    value: Any,
    *,
    replacements: Mapping[str, dict[str, Any]],
    used: set[str],
    where: str,
) -> Any:
    """Replace only exact analytic parameter leaves in one detached JSON tree."""

    if type(value) is dict:
        if value.get("kind") == "scalar" and value.get("op") == "parameter":
            if set(value) != {"kind", "op", "reference"}:
                raise TypeError("%s has a malformed analytic parameter leaf" % where)
            key = _reference_key(value["reference"], where=where + ".reference")
            replacement = replacements.get(key)
            if replacement is None:
                raise ValueError(
                    "%s references a parameter absent from the captured authorities" % where
                )
            used.add(key)
            return {"kind": "scalar", "op": "parameter", "reference": replacement}
        return {
            key: _replace_captured_parameter_references(
                item,
                replacements=replacements,
                used=used,
                where="%s.%s" % (where, key),
            )
            for key, item in value.items()
        }
    if type(value) is list:
        return [
            _replace_captured_parameter_references(
                item,
                replacements=replacements,
                used=used,
                where="%s[%d]" % (where, index),
            )
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("%s contains non-JSON captured analytic data" % where)


@dataclass(frozen=True, slots=True)
class BindArray:
    """A complete state array supplied explicitly to :func:`pops.bind`.

    ``BindArray`` authors ownership and bootstrap semantics without embedding a potentially large
    runtime array in the immutable Case snapshot.  The concrete value is keyed by the same typed
    state Handle in ``pops.bind(initial_values=...)``.  Level zero consumes the full conservative
    vector and finer levels are populated through the resolved AMR transfer provider.
    """

    native_route = "bound_level_zero"
    reprojectable = False
    __pops_ir_immutable__ = True

    def validate_for(self, state: Any) -> bool:
        _component_count(state)
        return True

    def initial_source_options(self) -> dict[str, Any]:
        return {"native_route": self.native_route}

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "bind_array",
        }

    canonical_identity = to_data
    inspect = to_data


@dataclass(frozen=True, slots=True, init=False)
class Constant:
    """One constant per state component; supported by the native AMR bootstrap route."""

    components: tuple[float, ...]
    native_route = "constant_field"
    reprojectable = True
    __pops_ir_immutable__ = True

    def __init__(self, components: Any) -> None:
        if isinstance(components, (str, bytes)):
            raise TypeError("Constant components must be an ordered numeric sequence")
        try:
            values = tuple(components)
        except TypeError as exc:
            raise TypeError("Constant components must be an ordered numeric sequence") from exc
        if not values:
            raise ValueError("Constant requires at least one component")
        object.__setattr__(self, "components", tuple(
            _finite(value, where="Constant.components[]") for value in values))

    def validate_for(self, state: Any) -> bool:
        if len(self.components) != _component_count(state):
            raise ValueError(
                "Constant component count does not match the target state")
        return True

    def initial_source_options(self) -> dict[str, Any]:
        return {"native_route": self.native_route, "components": list(self.components)}

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "constant",
            "components": list(self.components),
        }

    canonical_identity = to_data
    inspect = to_data


@dataclass(frozen=True, slots=True, init=False)
class Gaussian:
    """A scalar Gaussian profile expressed in one immutable physical frame."""

    frame: Any
    center: tuple[tuple[Any, float], ...]
    background: float
    amplitude: float
    inverse_width: float
    native_route = "gaussian_field"
    reprojectable = True
    __pops_ir_immutable__ = True

    def __init__(
        self,
        *,
        frame: Any,
        center: Any,
        background: Any = 0.0,
        amplitude: Any = 1.0,
        inverse_width: Any = 1.0,
    ) -> None:
        axes = getattr(frame, "axes", None)
        if not isinstance(axes, tuple) or not axes:
            raise TypeError("Gaussian frame must expose immutable typed axes")
        if not isinstance(center, Mapping) or set(center) != set(axes):
            raise ValueError("Gaussian center must map every typed frame axis exactly once")
        width = _finite(inverse_width, where="Gaussian.inverse_width")
        if width <= 0.0:
            raise ValueError("Gaussian.inverse_width must be > 0")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "center", tuple(
            (axis, _finite(center[axis], where="Gaussian.center[%s]" % axis.name))
            for axis in axes))
        object.__setattr__(self, "background", _finite(background, where="Gaussian.background"))
        object.__setattr__(self, "amplitude", _finite(amplitude, where="Gaussian.amplitude"))
        object.__setattr__(self, "inverse_width", width)

    def validate_for(self, state: Any) -> bool:
        if _component_count(state) != 1:
            raise ValueError("Gaussian is a scalar profile and requires a one-component state")
        space = getattr(_declaration(state), "space", None)
        frame_id = getattr(self.frame, "canonical_id", None)
        if space is None or getattr(space, "frame", None) != frame_id:
            raise ValueError("Gaussian frame differs from the target state frame")
        return True

    def initial_source_options(self) -> dict[str, Any]:
        return {
            "native_route": self.native_route,
            "frame_id": self.frame.canonical_id,
            "center": {axis.name: value for axis, value in self.center},
            "background": self.background,
            "amplitude": self.amplitude,
            "inverse_width": self.inverse_width,
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "gaussian",
            "frame_id": self.frame.canonical_id,
            "center": {axis.name: value for axis, value in self.center},
            "background": self.background,
            "amplitude": self.amplitude,
            "inverse_width": self.inverse_width,
        }

    canonical_identity = to_data
    inspect = to_data


@dataclass(frozen=True, slots=True, init=False)
class Analytic:
    """An ordered conservative state assembled from generic analytic expressions.

    The expressions are immutable data bound to one physical frame.  They are lowered to the
    native analytic evaluator and sampled by the selected projection policy; no Python callback is
    retained by the Case or invoked while the simulation runs.
    """

    frame: Any
    components: tuple[Any, ...]
    native_route = "analytic_expression"
    reprojectable = True
    __pops_ir_immutable__ = True

    def __init__(self, *, frame: Any, components: Any) -> None:
        frame_id = getattr(frame, "canonical_id", None)
        axes = getattr(frame, "axes", None)
        if not isinstance(frame_id, str) or not frame_id \
                or not isinstance(axes, tuple) or not axes:
            raise TypeError("Analytic frame must expose a canonical id and immutable typed axes")
        if isinstance(components, (str, bytes)):
            raise TypeError("Analytic components must be an ordered ScalarExpr sequence")
        try:
            values = tuple(components)
        except TypeError as exc:
            raise TypeError(
                "Analytic components must be an ordered ScalarExpr sequence") from exc
        if not values:
            raise ValueError("Analytic requires at least one component expression")
        from pops.analytic import ScalarExpr

        for index, expression in enumerate(values):
            if type(expression) is not ScalarExpr:
                raise TypeError(
                    "Analytic.components[%d] must be a ScalarExpr, got %s"
                    % (index, type(expression).__name__))
            expression.validate()
            if expression.frame_id not in (None, frame_id):
                raise ValueError(
                    "Analytic.components[%d] belongs to another physical frame" % index)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "components", values)

    def validate_for(self, state: Any) -> bool:
        if len(self.components) != _component_count(state):
            raise ValueError("Analytic component count does not match the target state")
        space = getattr(_declaration(state), "space", None)
        if space is None or getattr(space, "frame", None) != self.frame.canonical_id:
            raise ValueError("Analytic frame differs from the target state frame")
        return True

    def resolve_references(self, resolver: Any) -> Analytic:
        """Return the same profile with every parameter Handle registry-authenticated."""

        if not callable(resolver):
            raise TypeError("Analytic resolver must be callable")
        return Analytic(
            frame=self.frame,
            components=tuple(
                expression.resolve_references(resolver) for expression in self.components
            ),
        )

    def captured_reference_handles(self) -> tuple[Any, ...]:
        """Capture exact immutable parameter authorities in expression order."""

        ordered = []
        seen = set()
        for expression in self.components:
            for handle in expression.parameter_handles():
                if handle not in seen:
                    seen.add(handle)
                    ordered.append(handle)
        return tuple(ordered)

    @classmethod
    def resolve_captured_references(
        cls,
        *,
        value_identity: Any,
        source_options: Any,
        references: Any,
        resolver: Any,
    ) -> dict[str, Any]:
        """Resolve a detached analytic snapshot without consulting a provider instance.

        ``InitialCondition`` invokes this class protocol with the JSON it captured during
        authoring plus the exact immutable Handles captured at the same boundary.  No frame,
        expression or provider attribute is read again.
        """

        if not callable(resolver):
            raise TypeError("captured Analytic resolver must be callable")
        expected_value_keys = {
            "schema_version", "profile", "frame_id", "components",
        }
        expected_source_keys = {"native_route", "frame_id", "components"}
        if type(value_identity) is not dict or set(value_identity) != expected_value_keys \
                or value_identity.get("schema_version") != 1 \
                or value_identity.get("profile") != "analytic":
            raise TypeError("captured Analytic value identity has an unsupported schema")
        if type(source_options) is not dict or set(source_options) != expected_source_keys \
                or source_options.get("native_route") != cls.native_route:
            raise TypeError("captured Analytic source options have an unsupported schema")
        if value_identity["frame_id"] != source_options["frame_id"] \
                or value_identity["components"] != source_options["components"]:
            raise ValueError("captured Analytic identity and source options disagree")
        if type(references) is not tuple:
            raise TypeError("captured Analytic references must be an immutable tuple")

        from pops.model import ParamHandle

        replacements: dict[str, dict[str, Any]] = {}
        for index, handle in enumerate(references):
            if type(handle) is not ParamHandle:
                raise TypeError(
                    "captured Analytic references[%d] must be an exact ParamHandle" % index
                )
            authored = handle.canonical_identity() if handle.is_resolved else handle.inspect()
            key = _reference_key(authored, where="captured Analytic references[%d]" % index)
            if key in replacements:
                raise ValueError("captured Analytic references contain a duplicate authority")
            resolved = resolver(handle)
            if type(resolved) is not ParamHandle or not resolved.is_resolved:
                raise TypeError(
                    "captured Analytic resolver must return an exact canonical ParamHandle"
                )
            if resolved.param_kind != handle.param_kind:
                raise ValueError("captured Analytic resolver changed parameter kind")
            replacements[key] = resolved.canonical_identity()

        used: set[str] = set()
        components = _replace_captured_parameter_references(
            value_identity["components"],
            replacements=replacements,
            used=used,
            where="captured Analytic components",
        )
        if used != set(replacements):
            raise ValueError(
                "captured Analytic parameter authorities do not exactly match its expression data"
            )
        if not isinstance(components, list) or not components:
            raise TypeError("captured Analytic components must be a non-empty list")
        from pops.analytic import ScalarExpr

        canonical_components = [
            ScalarExpr.from_data(component).to_data() for component in components
        ]
        resolved_value = {
            "schema_version": 1,
            "profile": "analytic",
            "frame_id": value_identity["frame_id"],
            "components": canonical_components,
        }
        resolved_source = {
            "native_route": cls.native_route,
            "frame_id": source_options["frame_id"],
            "components": canonical_components,
        }
        return {
            "value_identity": resolved_value,
            "source_options": resolved_source,
        }

    def initial_source_options(self) -> dict[str, Any]:
        return {
            "native_route": self.native_route,
            "frame_id": self.frame.canonical_id,
            "components": [expression.to_data() for expression in self.components],
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "analytic",
            "frame_id": self.frame.canonical_id,
            "components": [expression.to_data() for expression in self.components],
        }

    canonical_identity = to_data
    inspect = to_data


@dataclass(frozen=True, slots=True, init=False)
class FieldMappedAnalytic:
    """Analytic seed followed by a native field solve and a discrete analytic state map."""

    frame: Any
    seed_components: tuple[Any, ...]
    components: tuple[Any, ...]
    inputs: tuple[tuple[int, str, int], ...]
    native_route = "field_mapped_analytic_expression"
    reprojectable = False
    __pops_ir_immutable__ = True

    def __init__(self, *, frame: Any, seed_components: Any,
                 components: Any, inputs: Any) -> None:
        frame_id = getattr(frame, "canonical_id", None)
        axes = getattr(frame, "axes", None)
        if not isinstance(frame_id, str) or not frame_id \
                or not isinstance(axes, tuple) or not axes:
            raise TypeError(
                "FieldMappedAnalytic frame must expose a canonical id and typed axes")
        from pops.analytic import ScalarExpr

        def expr_tuple(values: Any, *, where: str) -> tuple[Any, ...]:
            if isinstance(values, (str, bytes)):
                raise TypeError("%s must be an ordered ScalarExpr sequence" % where)
            try:
                rows = tuple(values)
            except TypeError as exc:
                raise TypeError("%s must be an ordered ScalarExpr sequence" % where) from exc
            if not rows:
                raise ValueError("%s requires at least one expression" % where)
            for index, expression in enumerate(rows):
                if type(expression) is not ScalarExpr:
                    raise TypeError(
                        "%s[%d] must be a ScalarExpr, got %s"
                        % (where, index, type(expression).__name__))
                expression.validate()
                if expression.frame_id not in (None, frame_id):
                    raise ValueError("%s[%d] belongs to another physical frame" % (where, index))
            return rows

        if not isinstance(inputs, Mapping):
            raise TypeError("FieldMappedAnalytic inputs must map integer ids to sources")
        normalized_inputs = []
        for value_id, source in inputs.items():
            if isinstance(value_id, bool) or not isinstance(value_id, int) or value_id < 0:
                raise TypeError("FieldMappedAnalytic input ids must be non-negative integers")
            if not isinstance(source, (tuple, list)) or len(source) != 2:
                raise TypeError(
                    "FieldMappedAnalytic inputs values must be ('state'|'aux', component)")
            kind, component = source
            if kind not in ("state", "aux"):
                raise ValueError("FieldMappedAnalytic input source must be 'state' or 'aux'")
            if isinstance(component, bool) or not isinstance(component, int) or component < 0:
                raise TypeError("FieldMappedAnalytic input component must be a non-negative integer")
            normalized_inputs.append((value_id, kind, component))
        if not normalized_inputs or [row[0] for row in sorted(normalized_inputs)] \
                != list(range(len(normalized_inputs))):
            raise ValueError("FieldMappedAnalytic input ids must be contiguous from zero")
        seed = expr_tuple(seed_components, where="FieldMappedAnalytic.seed_components")
        mapped = expr_tuple(components, where="FieldMappedAnalytic.components")
        used = sorted({
            value_id
            for expression in mapped
            for value_id, _component in expression.input_references()
        })
        if used != list(range(len(normalized_inputs))):
            raise ValueError(
                "FieldMappedAnalytic components must use exactly every declared input id")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "seed_components", seed)
        object.__setattr__(self, "components", mapped)
        object.__setattr__(self, "inputs", tuple(sorted(normalized_inputs)))

    def validate_for(self, state: Any) -> bool:
        count = _component_count(state)
        if len(self.seed_components) != count or len(self.components) != count:
            raise ValueError(
                "FieldMappedAnalytic seed and mapped component counts must match the target state")
        space = getattr(_declaration(state), "space", None)
        if space is None or getattr(space, "frame", None) != self.frame.canonical_id:
            raise ValueError("FieldMappedAnalytic frame differs from the target state frame")
        return True

    def initial_source_options(self) -> dict[str, Any]:
        return {
            "native_route": self.native_route,
            "frame_id": self.frame.canonical_id,
            "seed_components": [expression.to_data() for expression in self.seed_components],
            "components": [expression.to_data() for expression in self.components],
            "inputs": [
                {"value_id": value_id, "source": source, "component": component}
                for value_id, source, component in self.inputs
            ],
        }

    def to_data(self) -> dict[str, Any]:
        data = self.initial_source_options()
        data["schema_version"] = 1
        data["profile"] = "field_mapped_analytic"
        return data

    canonical_identity = to_data
    inspect = to_data


__all__ = ["Analytic", "BindArray", "Constant", "FieldMappedAnalytic", "Gaussian"]
