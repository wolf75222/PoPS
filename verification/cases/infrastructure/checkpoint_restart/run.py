"""IF-04 in-memory JSON checkpoint plus public RuntimeInstance restart.

Serialize a manufactured state dict (centers, q, t) to JSON and reload.
Optional ``run_native`` installs a ``Checkpoint`` consumer on TR-01, then
uses ``RuntimeInstance.checkpoint`` / ``restart``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
_TR01_RUN = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
)

RESTORE_MISSING = (
    "restore_checkpoint_payload requires owner and executor; "
    "no public path restore returns scientific {centers, q, t}"
)


class NativeUnavailable(RuntimeError):
    """Raised when the optional native checkpoint/restore path cannot complete."""


def _as_float64_list(values):
    return [float(item) for item in np.asarray(values, dtype=np.float64).ravel()]


def dump_state(state) -> str:
    """Serialize {centers, q, t} to a JSON object string."""
    payload = {
        "centers": _as_float64_list(state["centers"]),
        "q": _as_float64_list(state["q"]),
        "t": float(state["t"]),
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def load_state(text: str) -> dict:
    """Reload a JSON checkpoint into {centers, q, t} float64 arrays."""
    payload = json.loads(text)
    return {
        "centers": np.asarray(payload["centers"], dtype=np.float64),
        "q": np.asarray(payload["q"], dtype=np.float64),
        "t": float(payload["t"]),
    }


def round_trip(state=None) -> dict:
    """Dump then load a manufactured (or supplied) state dict."""
    original = _exact.manufactured_state() if state is None else state
    return load_state(dump_state(original))


def _local_work_dir() -> Path:
    """Prefer a local disk. GPFS scratch rejects renameat2 (EINVAL)."""
    import tempfile

    candidates = []
    override = os.environ.get("POPS_IF04_TMP")
    if override:
        candidates.append(Path(override))
    candidates.append(Path("/tmp"))
    scratch = os.environ.get("TMPDIR")
    if scratch:
        candidates.append(Path(scratch))
    candidates.append(Path(tempfile.gettempdir()))
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="if04-", dir=str(root)))
        except OSError:
            continue
        return work
    raise NativeUnavailable("no writable local work directory for IF-04")


def refuse_native_restore() -> str:
    """Return the documented missing public restore piece."""
    return RESTORE_MISSING


def json_round_trip_is_not_restart_proof() -> bool:
    """A manufactured JSON dump is not a RuntimeInstance restart."""
    return True


def restart_semantic_fields() -> tuple[str, ...]:
    """Fields a real continuous-vs-restart compare must expose."""
    return ("continuous", "restarted", "linf", "checkpoint_time", "final_time")


def install_checkpoint_consumer(case, target: str = "checkpoints/restart"):
    """Install a public Checkpoint consumer. Does not compile or run."""
    from pops.output import ConsumerGraph
    from pops.output.consumers import Checkpoint
    from pops.time import Clock, every

    program = getattr(case, "_time", None)
    clock = getattr(program, "clock", None)
    if clock is None:
        clock = Clock("if04_checkpoint", owner=case.owner_path)
    # Do not use on_end: the consumer writer hits EINVAL renaming native.npz
    # on GPFS. Install Checkpoint only for RestartAuthority, then snapshot
    # via RuntimeInstance.checkpoint().
    case.consumers(
        ConsumerGraph.from_consumers(
            (
                Checkpoint(
                    schedule=every(10**9, clock=clock),
                    target=str(target),
                    bit_identical=True,
                ),
            )
        )
    )
    return case


def run_native(n_cells: int = _exact.N_CELLS, t=_exact.T, path=None, request=None):
    """TR-01 continuous vs checkpoint/restart via public RuntimeInstance APIs.

    Installs ``Checkpoint``, compiles, runs to ``t/2``, ``checkpoint(path)``,
    ``restart(path)``, then continues to ``t``. Returns continuous / restarted
    fields and their L∞. Raises NativeUnavailable without a compiler or if
    the public restart contract cannot complete.
    """
    import pops

    _v15.refuse_invalid_mode(request)
    if request is not None and request.min_resolution is not None:
        n_cells = int(request.min_resolution)

    from verification.pops_verify.case_authoring import (
        bind_public,
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.reference_errors import reference_errors

    from verification.pops_verify.tr01_runtime import author

    tr01 = load_sibling_module(_TR01_RUN)
    missing = tr01._native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored_c = author(int(n_cells))
    authored_r = author(int(n_cells))
    work = _local_work_dir()
    install_checkpoint_consumer(authored_r.case, target="checkpoints/restart")
    layout = uniform_periodic_layout(authored_c.frame, (authored_c.n_cells,))
    try:
        plan_c = resolve_case(authored_c.case, layout=layout)
        plan_r = resolve_case(authored_r.case, layout=layout)
        artifact_c = pops.compile(plan_c)
        artifact_r = pops.compile(plan_r)
    except Exception as exc:
        raise NativeUnavailable(f"IF-04 Checkpoint Case failed: {exc}") from exc
    exact = load_sibling_module(_TR01_RUN.with_name("exact.py"))
    centers, volumes = exact.uniform_cell_centers(authored_c.n_cells)
    initial = np.ascontiguousarray(
        exact.exact_sine(centers, 0.0)[np.newaxis, :],
        dtype=np.float64,
    )
    t_end = float(t)
    half = 0.5 * t_end
    snapshot = Path(path) if path is not None else work / "accepted"
    mpi_mode = request.mpi_mode if request is not None else "off"
    try:
        continuous = bind_public(
            artifact_c, mpi_mode=mpi_mode, initial_values={authored_c.instance: initial}
        )
        pops.run(continuous, t_end=t_end, max_steps=tr01.MAX_STEPS, output_dir=work)
        field_c = np.ravel(
            np.asarray(continuous.state_global("tracer"), dtype=np.float64)
        )

        interrupted = bind_public(
            artifact_r, mpi_mode=mpi_mode, initial_values={authored_r.instance: initial}
        )
        pops.run(interrupted, t_end=half, max_steps=tr01.MAX_STEPS, output_dir=work)
        ckpt = interrupted.checkpoint(snapshot)
        restarted = bind_public(
            artifact_r, mpi_mode=mpi_mode, initial_values={authored_r.instance: initial}
        )
        restarted.restart(ckpt)
        pops.run(restarted, t_end=t_end, max_steps=tr01.MAX_STEPS, output_dir=work)
        field_r = np.ravel(
            np.asarray(restarted.state_global("tracer"), dtype=np.float64)
        )
    except NativeUnavailable:
        raise
    except Exception as exc:
        raise NativeUnavailable(f"IF-04 public restart failed: {exc}") from exc
    errors = reference_errors(field_r, field_c, volumes)
    payload = {
        "continuous": field_c,
        "restarted": field_r,
        "linf": float(errors.linf),
        "l2": float(errors.l2),
        "checkpoint_time": half,
        "final_time": t_end,
        "comparison_artifacts": {
            "kind": "checkpoint_restart",
            "checkpoint": str(ckpt),
            "linf": float(errors.linf),
        },
    }
    if request is None:
        return payload
    fields = _v15.campaign_run_fields(
        request, n_cells=n_cells, t_end=t_end, comparison=payload["comparison_artifacts"]
    )
    fields.update(payload)
    return fields
