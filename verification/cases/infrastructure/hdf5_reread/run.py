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
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
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


def npz_round_trip_is_not_hdf5() -> bool:
    """In-memory npz dump/load is not an HDF5 reread."""
    return True


def authenticated_hdf5_collective(path=None) -> bool:
    """Native parallel-HDF5 capability and an actual HDF5 path. Never request.mpi_mode."""
    return _v15.authenticated_hdf5_collective(path)


def campaign_hdf5_fields(request, path) -> dict:
    """Campaign provenance with a collective flag from the authenticated native path."""
    _v15.refuse_invalid_mode(request)
    return _v15.campaign_run_fields(
        request,
        n_cells=int(getattr(request, "min_resolution", None) or _exact.N_CELLS),
        t_end=0.1,
        comparison={"kind": "hdf5_reread", "path": str(path), "hdf5": True},
        hdf5_collective_enabled=authenticated_hdf5_collective(path),
    )


def public_state_handles(case) -> tuple:
    """Return public state Handles. ``Case.blocks()`` exposes only BlockHandles."""
    blocks = case.blocks()
    if not isinstance(blocks, dict) or any(
        getattr(handle, "kind", None) != "block" for handle in blocks.values()
    ):
        raise TypeError("Case.blocks() must return BlockHandles")
    return ()


def run_native(n_cells: int = _exact.N_CELLS, t_end: float = 0.1, request=None):
    """Write TR-01 through public HDF5 ScientificOutput and reread with read_hdf5.

    NPZ is not HDF5. Collective HDF5 requires MPI. Staging uses ``/tmp``.
    """
    import tempfile

    from pops.output import ConsumerGraph, HDF5, ParallelMode, ScientificOutput, read_hdf5
    from pops.time import on_end

    from verification.pops_verify.tr01_runtime import advance, prepare

    _v15.bind_campaign(request, NativeUnavailable)
    if request is not None and request.min_resolution is not None:
        n_cells = int(request.min_resolution)
    hdf5_mode = (
        ParallelMode.COLLECTIVE
        if _v15.native_has_parallel_hdf5()
        else ParallelMode.SERIAL
    )
    work = Path(tempfile.mkdtemp(prefix="if10-", dir="/tmp" if Path("/tmp").is_dir() else None))

    def _attach(authored) -> None:
        clock = authored.case._time.clock
        authored.case.consumers(
            ConsumerGraph.from_consumers(
                (
                    ScientificOutput(
                        format=HDF5(mode=hdf5_mode),
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
        npz_blobs = sorted(work.rglob("*.npz"))
        if npz_blobs:
            raise NativeUnavailable("IF-10 NPZ is not HDF5")
        blobs = sorted(work.rglob("*.h5")) + sorted(work.rglob("*.hdf5"))
        if not blobs:
            raise NativeUnavailable("IF-10 produced no HDF5 file")
        _v15.refuse_npz_as_hdf5(blobs[-1])
        reopened = read_hdf5(blobs[-1])
    except NativeUnavailable:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "NativeUnavailable":
            raise NativeUnavailable(str(exc)) from exc
        raise NativeUnavailable(f"IF-10 HDF5 reread failed: {exc}") from exc
    payload = {
        "field": field,
        "path": str(blobs[-1]),
        "reopened": reopened,
        "comparison_artifacts": {
            "kind": "hdf5_reread",
            "path": str(blobs[-1]),
            "hdf5": True,
        },
    }
    if request is None:
        return payload
    fields = _v15.campaign_run_fields(
        request,
        n_cells=n_cells,
        t_end=t_end,
        comparison=payload["comparison_artifacts"],
        hdf5_collective_enabled=authenticated_hdf5_collective(blobs[-1]),
    )
    fields.update(payload)
    return fields
