"""Deterministic animation frames and optional FFmpeg masters (plan §40.4.3)."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from verification.pops_verify.visualization.data import VisualsError
from verification.pops_verify.visualization.style import configure_matplotlib


def write_animation_frames(
    payload: dict[str, Any],
    frames_dir: Path,
    *,
    caption: str | None = None,
    provenance_sha: str | None = None,
) -> list[Path]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise VisualsError("animation has no accepted-state frames")
    limits = payload.get("color_limits")
    if not isinstance(limits, list) or len(limits) != 2:
        raise VisualsError("animation requires global color_limits")
    vmin, vmax = float(limits[0]), float(limits[1])
    if vmin >= vmax:
        raise VisualsError("animation color_limits must be ordered")
    frames_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib()
    written: list[Path] = []
    units = payload.get("units") or {}
    note = " | ".join(part for part in (caption, provenance_sha) if part)
    for index, frame in enumerate(frames):
        figure, axes = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
        mesh = axes.pcolormesh(
            frame["x"],
            frame["y"],
            frame["field"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )
        colorbar = figure.colorbar(mesh, ax=axes)
        colorbar.set_label(units.get("field") or "field")
        axes.set_xlabel(units.get("x") or "x")
        axes.set_ylabel(units.get("y") or "y")
        axes.set_title(f"t={frame.get('time')} step={frame.get('step')}")
        if note:
            figure.text(0.01, 0.01, note, fontsize=8)
        path = frames_dir / f"frame_{index:06d}.png"
        figure.savefig(
            path,
            metadata={"Creator": "pops.verification.visuals.v1", "Date": None},
        )
        plt.close(figure)
        written.append(path)
    return written


def _usable_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    probe = subprocess.run([ffmpeg, "-version"], check=False, capture_output=True, text=True)
    if probe.returncode != 0:
        return None
    return ffmpeg


def assemble_ffmpeg(
    frames_dir: Path,
    mp4_path: Path,
    gif_path: Path,
    *,
    periodic: bool,
    fps: int = 12,
) -> dict[str, str]:
    ffmpeg = _usable_ffmpeg()
    if ffmpeg is None:
        raise VisualsError("ffmpeg is required to assemble MP4/GIF masters")
    pattern = str(frames_dir / "frame_%06d.png")
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    mp4_cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-pix_fmt",
        "yuv420p",
        "-s",
        "1920x1080",
        str(mp4_path),
    ]
    completed = subprocess.run(mp4_cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise VisualsError(f"ffmpeg mp4 failed: {completed.stderr[-400:]}")
    gif_cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp4_path),
        "-vf",
        "fps=8,scale=640:-1:flags=lanczos",
        "-loop",
        "0" if periodic else "-1",
        str(gif_path),
    ]
    completed = subprocess.run(gif_cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise VisualsError(f"ffmpeg gif failed: {completed.stderr[-400:]}")
    return {"mp4": str(mp4_path), "gif": str(gif_path)}
