"""Immutable Python presentation contract for scheduled console diagnostics."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from inspect import isfunction
from string import Formatter
from types import MappingProxyType
from typing import Any


_DEFAULT_TEMPLATE = (
    "PoPS monitor | t={time:.12g} | step={step} | dt={dt:.12g} | {diagnostics}"
)


def _immutable_mapping(value: Mapping[str, Any], *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    rows = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("%s keys must be non-empty strings" % where)
        rows[key] = item
    return MappingProxyType(dict(sorted(rows.items())))


@dataclass(frozen=True, slots=True)
class ConsoleSample:
    """One immutable rank-zero snapshot containing only reduced scalar diagnostics."""

    time: float
    step: int
    dt: float
    values: Mapping[str, float | None]
    unavailable: Mapping[str, str]

    def __post_init__(self) -> None:
        time, dt = float(self.time), float(self.dt)
        if time != time or dt != dt or time in (float("inf"), float("-inf")) \
                or dt in (float("inf"), float("-inf")):
            raise ValueError("ConsoleSample time and dt must be finite")
        if isinstance(self.step, bool) or type(self.step) is not int or self.step < 0:
            raise TypeError("ConsoleSample step must be an integer >= 0")
        normalized = {}
        for name, value in self.values.items():
            if value is None:
                normalized[name] = None
                continue
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                raise ValueError("ConsoleSample value %r must be finite" % name)
            normalized[name] = number
        unavailable = {}
        for name, reason in self.unavailable.items():
            if not isinstance(reason, str) or not reason:
                raise TypeError("ConsoleSample unavailable reasons must be non-empty strings")
            unavailable[name] = reason
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(
            self, "values", _immutable_mapping(normalized, where="ConsoleSample.values"))
        object.__setattr__(
            self,
            "unavailable",
            _immutable_mapping(unavailable, where="ConsoleSample.unavailable"),
        )

    def __getitem__(self, name: str) -> float | None:
        """Return one scalar by its short or block-qualified diagnostic name."""
        return self.values[name]

    @property
    def diagnostics(self) -> str:
        qualified = [name for name in self.values if "." in name]
        block_names = {name.split(".", 1)[0] for name in qualified}
        parts = []
        for name in qualified:
            block, reduction = name.split(".", 1)
            label = "dU_L2" if reduction == "step_change_l2" else reduction
            if len(block_names) > 1:
                label = "%s[%s]" % (label, block)
            value = self.values[name]
            if value is None:
                parts.append("%s=n/a (%s)" % (label, self.unavailable[name]))
            else:
                parts.append("%s=%.6e" % (label, value))
        return " | ".join(parts)


class _Unavailable:
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __format__(self, _format_spec: str) -> str:
        return "n/a (%s)" % self.reason


class _Namespace:
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _template(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("ConsoleMonitor.template must be non-empty canonical text")
    try:
        fields = tuple(Formatter().parse(value))
    except ValueError as exc:
        raise ValueError("ConsoleMonitor.template has invalid format syntax") from exc
    for _literal, field_name, _spec, conversion in fields:
        if field_name is None:
            continue
        if conversion is not None or "[" in field_name or "]" in field_name:
            raise ValueError(
                "ConsoleMonitor.template supports named fields without conversion or indexing")
        if field_name not in {"time", "step", "dt", "diagnostics"} \
                and "." not in field_name:
            raise ValueError("ConsoleMonitor.template field %r is unsupported" % field_name)
    return value


def _handler_reference(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isfunction(value) or value.__name__ == "<lambda>":
        raise TypeError("ConsoleMonitor.handler must be a named Python function")
    if "<locals>" in value.__qualname__ or value.__closure__:
        raise TypeError("ConsoleMonitor.handler must not be local or close over Python state")
    return {"module": value.__module__, "qualname": value.__qualname__}


@dataclass(frozen=True, slots=True)
class ConsolePresentation:
    """Identity-bearing Python renderer carried by a diagnostic consumer."""

    __pops_ir_immutable__ = True

    template: str | None
    handler: Callable[[ConsoleSample], None] | None
    _handler_data: tuple[str, str] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.handler is not None and self.template is not None:
            raise ValueError("ConsoleMonitor accepts either template= or handler=, not both")
        selected_template = None if self.handler is not None else _template(
            _DEFAULT_TEMPLATE if self.template is None else self.template)
        handler_data = _handler_reference(self.handler)
        object.__setattr__(self, "template", selected_template)
        object.__setattr__(
            self,
            "_handler_data",
            None if handler_data is None else (
                handler_data["module"], handler_data["qualname"]),
        )

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.output.console-presentation.v1",
            "parallel_mode": "root",
            # A serial run has one natural root. Native diagnostic reductions already
            # support that singleton communicator, so no MPI context is required.
            "supports_singleton_collective": True,
            "template": self.template,
            "handler": (
                None if self._handler_data is None
                else {"module": self._handler_data[0], "qualname": self._handler_data[1]}
            ),
        }

    def emit(self, sample: ConsoleSample) -> None:
        if type(sample) is not ConsoleSample:
            raise TypeError("ConsolePresentation.emit expects an exact ConsoleSample")
        if self.handler is not None:
            self.handler(sample)
            return
        context: dict[str, Any] = {
            "time": sample.time,
            "step": sample.step,
            "dt": sample.dt,
            "diagnostics": sample.diagnostics,
        }
        blocks: dict[str, dict[str, Any]] = {}
        for name, value in sample.values.items():
            rendered: Any = (
                _Unavailable(sample.unavailable[name]) if value is None else value)
            context[name] = rendered
            if "." in name:
                block, reduction = name.split(".", 1)
                blocks.setdefault(block, {})[reduction] = rendered
        context.update({
            block: _Namespace(values) for block, values in blocks.items()
        })
        print(self.template.format_map(context), flush=True)


__all__ = ["ConsolePresentation", "ConsoleSample"]
