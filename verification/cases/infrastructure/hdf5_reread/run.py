"""IF-10 in-memory npz dump/load. No compile, bind, pops.run, or h5py.

Serialize a manufactured HDF5-shaped state
{centers, q, components=["q"], owner="rank0"} through numpy .npz and reload.
Optional native path inspects the TR-01 Case, then raises NativeUnavailable:
no public state Handle for ScientificOutput, and no public reread of that
shaped dict.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_TR01_RUN = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
)

REREAD_MISSING = (
    "TR-01 Case.blocks() exposes no public state Handle for ScientificOutput.fields; "
    "read_hdf5 is not a {centers, q, components, owner} reread"
)


class NativeUnavailable(RuntimeError):
    """Raised when the optional native HDF5 reread path cannot complete."""


def dump_state(state) -> bytes:
    """Serialize {centers, q, components, owner} to an in-memory .npz blob."""
    payload = {
        "centers": np.asarray(state["centers"], dtype=np.float64),
        "q": np.asarray(state["q"], dtype=np.float64),
        "components": np.asarray(list(state["components"]), dtype="U"),
        "owner": np.asarray([str(state["owner"])], dtype="U"),
    }
    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    return buffer.getvalue()


def load_state(blob: bytes) -> dict:
    """Reload an npz blob into {centers, q, components, owner}."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as payload:
        return {
            "centers": np.asarray(payload["centers"], dtype=np.float64),
            "q": np.asarray(payload["q"], dtype=np.float64),
            "components": [str(name) for name in np.asarray(payload["components"]).ravel()],
            "owner": str(np.asarray(payload["owner"]).reshape(-1)[0]),
        }


def round_trip(state=None) -> dict:
    """Dump then load a manufactured (or supplied) HDF5-shaped state dict."""
    original = _exact.manufactured_state() if state is None else state
    return load_state(dump_state(original))


def refuse_native_reread() -> str:
    """Return the documented missing public HDF5 reread piece."""
    return REREAD_MISSING


def public_state_handles(case) -> tuple:
    """Return public state Handles. ``Case.blocks()`` exposes only BlockHandles."""
    blocks = case.blocks()
    if not isinstance(blocks, dict) or any(
        getattr(handle, "kind", None) != "block" for handle in blocks.values()
    ):
        raise TypeError("Case.blocks() must return BlockHandles")
    return ()


def run_native(n_cells: int = _exact.N_CELLS, t_end: float = 0.1):
    """Write TR-01 through public NPZ ScientificOutput and reread with read_npz.

    HDF5 is attempted when the writer completes; otherwise NPZ is the
    public reread. Staging uses ``/tmp`` (GPFS rejects renameat2).
    """
    import tempfile

    from pops.output import ConsumerGraph, NPZ, ScientificOutput
    from pops.output import read_npz
    from pops.time import on_end

    from verification.pops_verify.tr01_runtime import advance, prepare

    work = Path(tempfile.mkdtemp(prefix="if10-", dir="/tmp" if Path("/tmp").is_dir() else None))

    def _attach(authored) -> None:
        clock = authored.case._time.clock
        authored.case.consumers(
            ConsumerGraph.from_consumers(
                (
                    ScientificOutput(
                        format=NPZ(),
                        schedule=on_end(clock=clock),
                        fields=(authored.instance,),
                        target="state/tracer",
                    ),
                )
            )
        )

    try:
        prepared = prepare(int(n_cells), attach=_attach)
        field = advance(prepared, float(t_end), output_dir=work)
        blobs = sorted(work.rglob("*.npz"))
        if not blobs:
            raise NativeUnavailable("IF-10 NPZ writer produced no file")
        reopened = read_npz(blobs[-1])
    except NativeUnavailable:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "NativeUnavailable":
            raise NativeUnavailable(str(exc)) from exc
        raise NativeUnavailable(f"IF-10 NPZ reread failed: {exc}") from exc
    return {
        "field": field,
        "path": str(blobs[-1]),
        "reopened": reopened,
    }
