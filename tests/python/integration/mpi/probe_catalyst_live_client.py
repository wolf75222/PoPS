#!/usr/bin/env python3
"""Real client for the two-rank PoPS Catalyst Live end-to-end probe.

Launch this file with ParaView's ``pvpython --no-mpi`` before starting
``probe_catalyst_live_mpi.py``.  It keeps every connection callback and rendering operation on the
main thread, extracts the distributed ``mesh`` channel, checks its data, configures the requested
view, and emits atomic evidence files for an external test orchestrator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--handshake", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.port < 1 or arguments.port > 65535:
        parser.error("--port must be between 1 and 65535")
    arguments.handshake = arguments.handshake.expanduser().resolve()
    if not arguments.handshake.is_dir():
        parser.error("--handshake must name an existing directory")
    return arguments


def _signal(root: Path, name: str, **evidence: Any) -> None:
    target = root / name
    temporary = root / (".%s.%d.tmp" % (name, os.getpid()))
    temporary.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def main() -> None:
    from paraview.live import (
        ConnectToCatalyst,
        ExtractCatalystData,
        ProcessServerNotifications,
    )
    from paraview.simple import (
        ColorBy,
        CreateView,
        GetColorTransferFunction,
        Render,
        ResetCamera,
        SaveScreenshot,
        Show,
    )

    arguments = _arguments()
    view = CreateView("RenderView")
    view.ViewSize = [640, 480]

    class State:
        source: Any = None
        display: Any = None
        received = False
        closed = False
        failure: str | None = None

    state = State()

    def guarded(callback: Any) -> Any:
        def invoke(link: Any, event: Any) -> None:
            try:
                callback(link, event)
            except BaseException:  # noqa: BLE001 - callback failures cross the VTK boundary
                state.failure = traceback.format_exc()
                _signal(arguments.handshake, "client-failed.json", error=state.failure)

        return invoke

    @guarded
    def connected(link: Any, event: Any) -> None:
        del event
        manager = link.GetInsituProxyManager()
        if manager.GetProxy("sources", "mesh") is None:
            raise RuntimeError("in-situ state has no exact sources/mesh proxy")
        state.source = ExtractCatalystData(link, "mesh")
        if state.source is None or not link.HasExtract("sources", "mesh", 0):
            raise RuntimeError("Catalyst Live client did not register the mesh extract")
        _signal(
            arguments.handshake,
            "client-extract-requested.json",
            port=arguments.port,
            source="mesh",
        )

    @guarded
    def updated(link: Any, event: Any) -> None:
        del event
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Catalyst Live client callback left the main thread")
        step = int(link.GetTimeStep())
        if step < 5:
            return
        if step != 5 or state.source is None:
            raise RuntimeError("Catalyst Live client received unexpected step %d" % step)
        state.source.UpdatePipeline()
        field = state.source.GetCellDataInformation()["U"]
        cells = int(state.source.GetDataInformation().GetNumberOfCells())
        if field is None or int(field.GetNumberOfComponents()) != 1:
            raise RuntimeError("Catalyst Live client did not receive scalar cell field U")
        lower, upper = (float(value) for value in field.GetRange(0))
        if cells != 8 or abs(lower - 1.0) > 1.0e-12 or abs(upper - 2.0) > 1.0e-12:
            raise RuntimeError(
                "Catalyst Live distributed data differs: cells=%d range=(%r, %r)"
                % (cells, lower, upper))
        if state.display is None:
            state.display = Show(state.source, view, "UnstructuredGridRepresentation")
            state.display.SetRepresentationType("Surface With Edges")
            ColorBy(state.display, ("CELLS", "U"))
            color_map = GetColorTransferFunction("U")
            if color_map.ApplyPreset("Viridis", True) is False:
                raise RuntimeError("ParaView does not provide the Viridis color preset")
            state.display.RescaleTransferFunctionToDataRange(True)
            state.display.SetScalarBarVisibility(view, True)
            ResetCamera(view)
        Render(view)
        screenshot = arguments.handshake / "live-client.png"
        SaveScreenshot(str(screenshot), view, ImageResolution=[640, 480])
        if not screenshot.is_file() or screenshot.stat().st_size == 0:
            raise RuntimeError("Catalyst Live client produced no screenshot")
        state.received = True
        _signal(
            arguments.handshake,
            "client-frame.json",
            cells=cells,
            color_map="Viridis",
            field="U",
            range=[lower, upper],
            representation="Surface With Edges",
            step=step,
        )

    @guarded
    def closed(link: Any, event: Any) -> None:
        del link, event
        state.closed = True
        _signal(
            arguments.handshake,
            "client-closed.json",
            received=state.received,
        )

    try:
        link = ConnectToCatalyst(arguments.host, arguments.port)
        link.AddObserver("ConnectionCreatedEvent", connected)
        link.AddObserver("UpdateEvent", updated)
        link.AddObserver("ConnectionClosedEvent", closed)
        _signal(
            arguments.handshake,
            "client-ready.json",
            host=arguments.host,
            pid=os.getpid(),
            port=arguments.port,
        )
        deadline = time.monotonic() + 60.0
        while not state.closed:
            if state.failure is not None:
                raise RuntimeError(state.failure)
            if (arguments.handshake / "abort").exists():
                raise RuntimeError("Catalyst Live orchestrator aborted")
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Catalyst Live completion")
            ProcessServerNotifications()
            time.sleep(0.01)
        if not state.received:
            raise RuntimeError("Catalyst Live connection closed before the certified frame")
    except BaseException:
        if not (arguments.handshake / "client-failed.json").exists():
            _signal(
                arguments.handshake,
                "client-failed.json",
                error=traceback.format_exc(),
            )
        raise

    print("PASS Catalyst Live client: socket=connected frame=5 render=verified", flush=True)


if __name__ == "__main__":
    main()
