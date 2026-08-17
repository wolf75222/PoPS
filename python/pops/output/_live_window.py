"""Fail-closed live ParaView window capability for the post-commit worker.

Apple's Cocoa backend cannot create a ``RenderView`` from a background thread.
PoPS therefore never pretends that a macOS GUI spawned from the solver worker.
File and Catalyst data paths remain valid; an off-screen EGL/OSMesa or
``VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN`` backend may render frames without a window.
"""
from __future__ import annotations

import ctypes.util
import os
import sys
from pathlib import Path
from typing import Any


_LIVE_WINDOW_MARKERS = (
    'CreateView("RenderView")',
    "CreateView('RenderView')",
    'GetActiveViewOrCreate("RenderView")',
    "GetActiveViewOrCreate('RenderView')",
    "CreateRenderView(",
)

_OFFSCREEN_LIBRARIES = ("OSMesa", "osmesa", "EGL", "egl")
_TRUTHY = frozenset({"1", "true", "TRUE", "yes", "YES", "on", "ON"})


def pipeline_requests_render_view(path: Path) -> bool:
    """Return whether a Catalyst pipeline script asks for a local RenderView."""

    text = path.read_text(encoding="utf-8")
    return any(marker in text for marker in _LIVE_WINDOW_MARKERS)


def offscreen_render_backend() -> str | None:
    """Return a worker-safe off-screen render backend, if any.

    A ``pvpython`` executable is not sufficient: it is a separate process and does not
    make in-process Cocoa ``RenderView`` creation safe on the post-commit worker.
    """

    env = os.environ.get("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "").strip()
    if env in _TRUTHY:
        return "vtk-offscreen-env"
    for name in _OFFSCREEN_LIBRARIES:
        if ctypes.util.find_library(name):
            return name
    return None


def live_paraview_window_capability(*, requested: bool) -> dict[str, Any]:
    """Describe or refuse a live ParaView window on the post-commit worker.

    When ``requested`` is false the Catalyst data path is allowed on every
    platform.  When a pipeline asks for ``RenderView``, macOS Cocoa without an
    off-screen backend fails closed instead of crashing inside vtkCocoaRenderWindow.
    """
    backend = offscreen_render_backend()
    if not requested:
        return {
            "window": False,
            "offscreen": backend,
            "platform": sys.platform,
        }
    if backend is None and sys.platform == "darwin":
        raise RuntimeError(
            "macOS Cocoa cannot create a ParaView RenderView from the PoPS post-commit "
            "worker; publish the file or Catalyst data path instead, or use an off-screen "
            "EGL/OSMesa backend (VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1)"
        )
    if backend is None:
        raise RuntimeError(
            "live ParaView RenderView on the post-commit worker requires an off-screen "
            "EGL/OSMesa backend; PoPS will not spawn an interactive GUI from the solver thread"
        )
    return {
        "window": False,
        "offscreen": backend,
        "platform": sys.platform,
    }


def reject_unsafe_live_render_window(pipeline: Path) -> dict[str, Any]:
    """Fail closed when a Catalyst pipeline would create an unsafe Cocoa RenderView."""

    requested = pipeline_requests_render_view(pipeline)
    capability = live_paraview_window_capability(requested=requested)
    if requested and capability["offscreen"] is not None:
        os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    return capability


__all__ = [
    "live_paraview_window_capability",
    "offscreen_render_backend",
    "pipeline_requests_render_view",
    "reject_unsafe_live_render_window",
]
