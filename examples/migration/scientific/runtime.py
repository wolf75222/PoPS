"""Runtime and measurements shared by the examples; no physics or numerical method."""

from __future__ import annotations

from typing import Any
import numpy as np
import pops


def _execution_resources(artifact: Any) -> dict[str, Any]:
    communicator = artifact.platform_manifest.communicator.require("migration example communicator")
    if communicator == "serial":
        return {}
    if communicator == "MPI_COMM_WORLD":
        return {"execution_context": pops.ExecutionContext.mpi_world(artifact)}
    raise RuntimeError("unsupported communicator %r" % communicator)


def _compile(case: Any, layout: Any, *, components: tuple[Any, ...] = ()) -> Any:
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout, components=components)
    return pops.compile(resolved)


def _bind(artifact: Any, initial_state: dict[str, np.ndarray]) -> Any:
    return pops.bind(
        artifact, initial_state=initial_state, resources=_execution_resources(artifact)
    )


def _run_case(
    case: Any,
    layout: Any,
    initial_state: dict[str, np.ndarray],
    *,
    dt: float,
    steps: int,
    components: tuple[Any, ...] = (),
) -> tuple[Any, Any, Any]:
    artifact = _compile(case, layout, components=components)
    runtime = _bind(artifact, initial_state)
    report = pops.run(runtime, t_end=steps * dt, max_steps=steps, console=False)
    return (runtime, report, artifact)


def _cell_centers(cells: int) -> tuple[np.ndarray, np.ndarray]:
    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    return np.meshgrid(coordinate, coordinate, indexing="xy")


def _state_summary(runtime: Any, block: str) -> dict[str, Any]:
    values = np.asarray(runtime.state_global(block), dtype=np.float64)
    return {
        "block": block,
        "shape": list(values.shape),
        "component_means": np.mean(values, axis=tuple(range(1, values.ndim))).tolist(),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
