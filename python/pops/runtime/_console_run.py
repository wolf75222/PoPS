"""Rank-zero console presentation for the public :func:`pops.run` transition.

The renderer owns no numerical decision and reads no field array.  Every technical value printed
below comes from the authenticated install plan, the run manifest, or the native runtime report.
Keeping this module private leaves the final lifecycle unchanged while making the default script
experience useful without tutorial-specific logging helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from pops._version import __version__
from pops.runtime_environment import runtime_environment_report


_LOGO = r"""
 ____       ____  ____
|  _ \ ___ |  _ \/ ___|
| |_) / _ \| |_) \___ \
|  __/ (_) |  __/ ___) |
|_|   \___/|_|   |____/
""".strip("\n")


def _float_literal(value: Any) -> Any:
    if isinstance(value, dict) and value.get("kind") == "binary64" \
            and isinstance(value.get("value"), str):
        try:
            return float.fromhex(value["value"])
        except ValueError:
            return value
    return value


def _strategy_text(strategy: Any) -> str:
    projected = strategy.to_data()
    if not isinstance(projected, dict):
        raise TypeError("step strategy to_data() must return a dict")
    data = dict(projected)
    kind = str(data.pop("kind", type(strategy).__name__))
    options = ", ".join(
        "%s=%s" % (name, _float_literal(value))
        for name, value in data.items()
    )
    return "%s (%s)" % (kind, options) if options else kind


def _layout_text(row: Any) -> str:
    geometry = row.geometry
    cells = " x ".join(str(value) for value in geometry.cells)
    bounds = " x ".join(
        "[%g, %g]" % pair
        for pair in zip(geometry.lower, geometry.upper, strict=True)
    )
    if row.adaptive:
        ratios = ":".join(str(value) for value in row.transition_ratios) or "none"
        detail = "%d levels, ratios %s" % (len(row.levels), ratios)
    else:
        detail = "1 level"
    return "%s; %s cells; %s; %s" % (
        row.descriptor_name, cells, detail, bounds,
    )


def _consumer_text(graph: Any) -> str:
    if not graph.nodes:
        return "none"
    rows = []
    for node in graph.nodes:
        target = getattr(node, "target_uri", "")
        format_data = getattr(node, "output_format_data", None)
        format_name = format_data.get("format_name") if format_data is not None else None
        label = node.kind.value
        if format_name:
            label += " %s" % format_name
        schedule = getattr(node, "schedule", None)
        schedule_projector = getattr(schedule, "to_data", None)
        schedule_data = schedule_projector() if callable(schedule_projector) else {}
        trigger = schedule_data.get("trigger", {}) if isinstance(schedule_data, dict) else {}
        if isinstance(trigger, dict) and isinstance(trigger.get("type"), str):
            cadence = trigger["type"]
            if cadence == "every" and isinstance(trigger.get("n"), int):
                cadence += "(%d)" % trigger["n"]
            label += " @ %s" % cadence
        if target:
            label += " -> %s" % target
        rows.append(label)
    return "; ".join(rows)


def _numerics_text(block: Any) -> str:
    spatial = block.spatial
    projector = getattr(spatial, "to_data", None)
    data = projector() if callable(projector) else {}
    if not isinstance(data, dict):
        data = {}
    parts = [str(data.get("method", type(spatial).__name__))]
    for key in ("variables", "reconstruction", "riemann"):
        value = data.get(key)
        if isinstance(value, dict) and value.get("name"):
            parts.append("%s=%s" % (key, value["name"]))
    for key in ("formal_order", "ghost_depth"):
        if key in data:
            parts.append("%s=%s" % (key, data[key]))
    boundaries = getattr(block, "boundaries", ())
    if boundaries:
        parts.append("boundaries=%d" % len(boundaries))
    return ", ".join(parts)


@dataclass(slots=True)
class ConsoleRunSession:
    """One best-effort presentation session owned by a single public run call."""

    enabled: bool
    started_at: float

    def completed(self, report: Any) -> None:
        if not self.enabled:
            return
        elapsed = perf_counter() - self.started_at
        print("-" * 64)
        print("PoPS run completed")
        print("  accepted / rejected : %d / %d" % (
            report.accepted_steps, report.rejected_steps))
        print("  final time / step   : %.12g / %d" % (
            report.final_time, report.final_macro_step))
        print("  elapsed             : %.6f s" % elapsed)
        print("  run identity        : %s" % report.run_identity.hexdigest[:16])

    def failed(self, error: BaseException, *, accepted_steps: int, final_time: float) -> None:
        if not self.enabled:
            return
        elapsed = perf_counter() - self.started_at
        print("-" * 64)
        print("PoPS run failed after %d accepted step(s) at t=%.12g (%.6f s)" % (
            accepted_steps, final_time, elapsed), file=sys.stderr)
        print("  %s: %s" % (type(error).__name__, error), file=sys.stderr)


def _rank_size(instance: Any) -> tuple[str, int, int]:
    communicator = instance._execution_context.communicator
    identity = str(communicator.identity)
    if identity == "serial":
        if communicator.handle is not None:
            raise ValueError("serial execution context hides a communicator handle")
        return identity, 0, 1
    handle = communicator.handle
    rank = getattr(handle, "rank", None)
    size = getattr(handle, "size", None)
    if type(rank) is not int or type(size) is not int or rank < 0 or size < 1 or rank >= size:
        raise ValueError("execution communicator does not expose a valid native rank/size")
    return identity, rank, size


def _warning(phase: str, error: Exception) -> None:
    try:
        print(
            "PoPS console %s disabled: %s: %s" % (
                phase, type(error).__name__, error),
            file=sys.stderr,
        )
    except Exception:
        pass


def begin_console_run(instance: Any, manifest: Any, strategy: Any) -> ConsoleRunSession:
    """Print the actual resolved/native configuration once on rank zero."""
    communicator, rank, ranks = _rank_size(instance)
    if rank != 0:
        return ConsoleRunSession(False, perf_counter())
    environment = runtime_environment_report()

    install = instance._install_plan
    plan = install.artifact.plan
    snapshot = plan.snapshot.to_dict()
    backend = str(environment.get("kokkos_backend", "unknown"))
    concurrency = int(environment.get("kokkos_concurrency", 0) or 0)
    compute = "native C++"
    if bool(environment.get("has_kokkos")):
        compute += " / Kokkos %s" % backend
    lane_text = str(concurrency) if concurrency > 0 else "not reported"
    print(_LOGO)
    print("PoPS %s | resolved simulation launch" % __version__)
    print("=" * 64)
    print("  case                : %s" % snapshot.get("name", "(unnamed)"))
    print("  target / backend    : %s / %s" % (plan.target, plan.backend))
    print("  compute             : %s" % compute)
    print("  execution lanes     : %s" % lane_text)
    print("  precision           : %s (%s bytes)" % (
        environment.get("precision", "unknown"),
        environment.get("real_bytes", "?"),
    ))
    print("  communicator        : %s (%d rank%s)" % (
        communicator, ranks, "s" if ranks != 1 else ""))
    print("  blocks              : %s" % (", ".join(instance.block_names()) or "none"))
    for block in plan.blocks:
        print("  numerics %-10s : %s" % (block.name, _numerics_text(block)))
    for index, layout in enumerate(instance._layout_plan.layouts):
        print("  layout %d            : %s" % (index, _layout_text(layout)))
    print("  step strategy       : %s" % _strategy_text(strategy))
    print("  interval            : %.12g -> %.12g (max %d accepted steps)" % (
        manifest.start_time, manifest.controls["t_end"], manifest.controls["max_steps"]))
    print("  consumers           : %s" % _consumer_text(instance.consumer_graph))
    persistent_output = any(
        node.kind.value in {"scientific_output", "checkpoint"}
        for node in instance.consumer_graph.nodes
    )
    if persistent_output:
        output_root = instance._output_root
        print("  output root         : %s (%s)" % (
            Path.cwd() if output_root is None else Path(output_root),
            manifest.controls["output_mode"],
        ))
    else:
        print("  output root         : none")
    print("  artifact identity   : %s" % install.artifact.artifact_identity.hexdigest[:16])
    print("  run identity        : %s" % manifest.run_identity.hexdigest[:16])
    print("=" * 64, flush=True)
    return ConsoleRunSession(True, perf_counter())


def safe_begin_console_run(instance: Any, manifest: Any, strategy: Any) -> ConsoleRunSession:
    """Start presentation without letting a terminal failure alter numerical execution."""

    try:
        return begin_console_run(instance, manifest, strategy)
    except Exception as error:
        _warning("startup", error)
        return ConsoleRunSession(False, perf_counter())


def safe_console_completed(session: ConsoleRunSession, report: Any) -> None:
    """Render a success report without converting success into failure."""

    try:
        session.completed(report)
    except Exception as error:
        _warning("completion", error)


def safe_console_failed(
    session: ConsoleRunSession,
    error: BaseException,
    *,
    accepted_steps: int,
    final_time: float,
) -> None:
    """Render a failed run without masking its original exception."""

    try:
        session.failed(error, accepted_steps=accepted_steps, final_time=final_time)
    except Exception as presentation_error:
        _warning("failure", presentation_error)


__all__: list[str] = []
