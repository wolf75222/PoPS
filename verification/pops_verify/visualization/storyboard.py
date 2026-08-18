"""Static storyboards from accepted-state frames (plan §40.1, §40.4.3)."""
from __future__ import annotations

from typing import Any

from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.style import configure_matplotlib


def render_storyboard(
    payload: dict[str, Any],
    output_path,
    *,
    caption: str | None = None,
    provenance_sha: str | None = None,
) -> None:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise VisualsError("storyboard has no accepted-state frames")
    if any("interpolated" in str(frame.get("source", "")).lower() for frame in frames):
        raise VisualsError("storyboard frames must be accepted states, not interpolated")
    plt = configure_matplotlib()
    count = len(frames)
    figure, axes_list = plt.subplots(1, count, figsize=(3.2 * count, 3.6), constrained_layout=True)
    if count == 1:
        axes_list = [axes_list]
    units = payload.get("units") or {}
    for axes, frame in zip(axes_list, frames, strict=True):
        series = frame.get("series") or []
        for item in series:
            axes.plot(item["x"], item["y"], label=item.get("name"))
        event = frame.get("event") or "state"
        time = frame.get("time")
        axes.set_title(f"{event} t={time}")
        axes.set_xlabel(units.get("x") or "x")
        axes.set_ylabel(units.get("y") or "y")
        axes.legend(fontsize=7)
    note = " | ".join(part for part in (caption, provenance_sha) if part)
    if note:
        figure.text(0.01, 0.01, note, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, metadata={"Creator": "pops.verification.visuals.v1", "Date": None})
    plt.close(figure)
