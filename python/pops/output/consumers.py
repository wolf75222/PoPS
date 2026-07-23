"""Final direct scientific-output and checkpoint consumer descriptors."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from pops.descriptors import Descriptor
from pops.model import Handle
from pops.time import Schedule

from .levels import AllLevels, LevelSelection


_WRITABLE_KINDS = frozenset({"state", "field", "aux"})


def _target(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    if "\\" in value:
        raise ValueError("%s must use '/' as its logical path separator" % where)
    pieces = value.split("/")
    if any(piece in {"", ".", ".."} for piece in pieces):
        raise ValueError("%s must be a canonical relative output target" % where)
    if PurePosixPath(value).suffix:
        raise ValueError(
            "%s is a logical target and must not contain a file suffix; "
            "the selected provider owns its extension" % where)
    return value


def _schedule(value: Any, *, where: str) -> Schedule:
    if type(value) is not Schedule:
        raise TypeError("%s must be an exact pops.time.Schedule" % where)
    return value


def _format_data(value: Any) -> dict[str, Any]:
    from .provider import consumer_format_data

    return consumer_format_data(value, where="ScientificOutput.format")


def _failure_action(value: Any) -> Any:
    from ._consumer_contracts import FailRun, _FAILURE_ACTIONS

    selected = FailRun() if value is None else value
    if type(selected) not in _FAILURE_ACTIONS:
        raise TypeError(
            "ScientificOutput.failure_action must be pops.output.FailRun(), Retry(...), "
            "or SkipSampleReported()"
        )
    return selected


def _diagnostic(value: Any, *, index: int) -> None:
    where = "ScientificOutput diagnostics[%d]" % index
    for method in ("declaration_references", "resolve_references", "consumer_data", "freeze"):
        if not callable(getattr(value, method, None)):
            raise TypeError("%s must implement %s()" % (where, method))


class ScientificOutput(Descriptor):
    """One direct writer consumer: exact format, schedule, quantities and target.

    Parallel mode is owned by the selected typed format. There is no second boolean switch or
    runtime inference that can disagree with ``HDF5(mode=ParallelMode.COLLECTIVE)``.
    """

    category = "scientific_output"

    def __init__(
        self,
        *,
        format: Any,
        schedule: Any,
        fields: Any = (),
        diagnostics: Any = (),
        levels: Any = None,
        target: Any,
        failure_action: Any = None,
    ) -> None:
        format_data = _format_data(format)
        field_rows = tuple(fields)
        if any(not isinstance(reference, Handle) for reference in field_rows):
            raise TypeError("ScientificOutput fields must contain declaration Handles")
        if any(reference.kind not in _WRITABLE_KINDS for reference in field_rows):
            raise TypeError("ScientificOutput fields accept only state, field, or aux Handles")
        if len(set(field_rows)) != len(field_rows):
            raise ValueError("ScientificOutput fields must be unique")
        diagnostic_rows = tuple(diagnostics)
        for index, diagnostic in enumerate(diagnostic_rows):
            _diagnostic(diagnostic, index=index)
            cadence = getattr(diagnostic, "cadence", None)
            if cadence is not None and cadence != schedule:
                raise ValueError(
                    "a diagnostic embedded in ScientificOutput must use the same schedule")
        if not field_rows and not diagnostic_rows:
            raise ValueError("ScientificOutput requires at least one field or diagnostic")
        selected_levels = AllLevels() if levels is None else levels
        if not isinstance(selected_levels, LevelSelection):
            raise TypeError("ScientificOutput levels must be a typed LevelSelection")
        self.format = format
        self._format_data = format_data
        self.schedule = _schedule(schedule, where="ScientificOutput.schedule")
        self.fields = field_rows
        self.diagnostics = diagnostic_rows
        self.levels = selected_levels
        self.target = _target(target, where="ScientificOutput.target")
        self.failure_action = _failure_action(failure_action)

    def declaration_references(self) -> tuple[Handle, ...]:
        result = list(self.fields)
        for index, diagnostic in enumerate(self.diagnostics):
            references = diagnostic.declaration_references()
            if not isinstance(references, tuple) or any(
                    not isinstance(reference, Handle) for reference in references):
                raise TypeError(
                    "ScientificOutput diagnostics[%d].declaration_references() must return "
                    "a tuple of Handles" % index)
            for reference in references:
                if reference not in result:
                    result.append(reference)
        return tuple(result)

    def consumer_authoring(self) -> tuple[Any, ...]:
        from ._consumer_authoring import ConsumerAuthoringNode
        from ._consumer_contracts import ConsumerKind, ParallelMode

        mode = {
            "serial": ParallelMode.SERIAL,
            "root": ParallelMode.ROOT,
            "collective": ParallelMode.COLLECTIVE,
            "per_rank": ParallelMode.PER_RANK,
        }[self._format_data["parallel_mode"]]
        return (ConsumerAuthoringNode(
            label="scientific-output-%s" % self.target.replace("/", "-"),
            kind=ConsumerKind.SCIENTIFIC_OUTPUT,
            references=self.fields,
            schedule=self.schedule,
            target_uri=self.target,
            output_format=self.format,
            parallel_mode=mode,
            levels=self.levels,
            operation=None,
            diagnostics=self.diagnostics,
            failure_action=self.failure_action,
        ),)

    def options(self) -> dict[str, Any]:
        return {
            "format": self._format_data,
            "schedule": self.schedule.to_data(),
            "fields": [reference.inspect() for reference in self.fields],
            "n_diagnostics": len(self.diagnostics),
            "levels": self.levels.to_data(),
            "target": self.target,
            "failure_action": self.failure_action.to_data(),
        }


class ConsoleMonitor(Descriptor):
    """Print scheduled native diagnostics for accepted states on MPI rank zero."""

    category = "console_monitor"

    def __init__(
        self,
        *,
        schedule: Any,
        diagnostics: Any,
        levels: Any = None,
        enabled: Any = True,
        failure_action: Any = None,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("ConsoleMonitor.enabled must be an exact bool")
        rows = tuple(diagnostics)
        if not rows:
            raise ValueError("ConsoleMonitor requires at least one diagnostic")
        for index, diagnostic in enumerate(rows):
            _diagnostic(diagnostic, index=index)
            cadence = getattr(diagnostic, "cadence", None)
            if cadence is not None and cadence != schedule:
                raise ValueError(
                    "a diagnostic embedded in ConsoleMonitor must use the same schedule")
        selected_levels = AllLevels() if levels is None else levels
        if not isinstance(selected_levels, LevelSelection):
            raise TypeError("ConsoleMonitor levels must be a typed LevelSelection")
        self.schedule = _schedule(schedule, where="ConsoleMonitor.schedule")
        self.diagnostics = rows
        self.levels = selected_levels
        self.enabled = enabled
        from ._consumer_contracts import SkipSampleReported
        self.failure_action = (
            SkipSampleReported() if failure_action is None else _failure_action(failure_action)
        )

    def declaration_references(self) -> tuple[Handle, ...]:
        result = []
        for index, diagnostic in enumerate(self.diagnostics):
            references = diagnostic.declaration_references()
            if not isinstance(references, tuple) or any(
                    not isinstance(reference, Handle) for reference in references):
                raise TypeError(
                    "ConsoleMonitor diagnostics[%d].declaration_references() must return "
                    "a tuple of Handles" % index)
            for reference in references:
                if reference not in result:
                    result.append(reference)
        return tuple(result)

    def consumer_authoring(self) -> tuple[Any, ...]:
        if not self.enabled:
            return ()
        from ._consumer_authoring import ConsumerAuthoringNode
        from ._consumer_contracts import ConsumerKind, ParallelMode

        return (ConsumerAuthoringNode(
            label="console-monitor",
            kind=ConsumerKind.DIAGNOSTIC,
            references=(),
            schedule=self.schedule,
            target_uri="console/diagnostics",
            output_format=None,
            parallel_mode=ParallelMode.ROOT,
            levels=self.levels,
            operation=None,
            diagnostics=self.diagnostics,
            failure_action=self.failure_action,
        ),)

    def options(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_data(),
            "n_diagnostics": len(self.diagnostics),
            "levels": self.levels.to_data(),
            "enabled": self.enabled,
            "failure_action": self.failure_action.to_data(),
        }


class Checkpoint(Descriptor):
    """A restartable checkpoint consumer; bit identity is the only optional stronger guarantee."""

    category = "checkpoint"

    def __init__(self, *, schedule: Any, target: Any, bit_identical: Any = False) -> None:
        if type(bit_identical) is not bool:
            raise TypeError("Checkpoint.bit_identical must be a bool")
        self.schedule = _schedule(schedule, where="Checkpoint.schedule")
        self.target = _target(target, where="Checkpoint.target")
        self.bit_identical = bit_identical

    def declaration_references(self) -> tuple[Handle, ...]:
        return ()

    def consumer_authoring(self) -> tuple[Any, ...]:
        from ._consumer_authoring import ConsumerAuthoringNode
        from ._consumer_contracts import ConsumerKind, ParallelMode
        from ._restart_provider import RestartV3

        return (ConsumerAuthoringNode(
            label="checkpoint-%s" % self.target.replace("/", "-"),
            kind=ConsumerKind.CHECKPOINT,
            references=(),
            schedule=self.schedule,
            target_uri=self.target,
            output_format=None,
            parallel_mode=ParallelMode.COLLECTIVE,
            levels=AllLevels(),
            operation=RestartV3(bit_identical=self.bit_identical),
        ),)

    def options(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_data(),
            "target": self.target,
            "restartable": True,
            "bit_identical": self.bit_identical,
        }


__all__ = ["Checkpoint", "ConsoleMonitor", "ScientificOutput"]
