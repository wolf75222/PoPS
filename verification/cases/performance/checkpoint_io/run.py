"""PF-10 npz checkpoint write/read.

Write/read a 1-d field. Record bytes and a fake write-time observation.
A no-output path must not write the artifact. Optional ``run_native``
times IF-04 or TR-01 when those public paths exist.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
_CASES = Path(__file__).resolve().parents[2]
_IF04_RUN = _CASES / "infrastructure" / "checkpoint_restart" / "run.py"
_TR01_RUN = _CASES / "transport" / "advection_sine" / "run.py"

CHECKPOINT_PATH_REFUSAL = "public IF-04 / TR-01 native checkpoint path is not available"


class NativeUnavailable(RuntimeError):
    """Raised when IF-04 / TR-01 cannot be timed."""


def write_checkpoint(path, field=None, *, output=True) -> dict:
    """Write the 1-d field to npz when output is enabled; otherwise skip I/O."""
    original = (
        _exact.manufactured_field()
        if field is None
        else np.asarray(field, dtype=np.float64)
    )
    dest = Path(path)
    if not output:
        return {
            "path": dest,
            "wrote": False,
            "bytes": 0,
            "write_time_s": 0.0,
            "field": original,
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **{_exact.FIELD_KEY: original})
    return {
        "path": dest,
        "wrote": True,
        "bytes": dest.stat().st_size,
        "write_time_s": float(_exact.FAKE_WRITE_TIME_S),
        "field": original,
    }


def read_checkpoint(path) -> np.ndarray:
    """Reload the 1-d field from an npz checkpoint."""
    with np.load(path) as loaded:
        return np.asarray(loaded[_exact.FIELD_KEY], dtype=np.float64)


def round_trip(path, field=None) -> dict:
    """Write then read a manufactured (or supplied) 1-d field."""
    written = write_checkpoint(path, field, output=True)
    restored = read_checkpoint(written["path"])
    return {**written, "restored": restored}


def run_no_output(path, field=None) -> dict:
    """No-output path: record a skipped write and do not create the artifact."""
    return write_checkpoint(path, field, output=False)


def public_checkpoint_native():
    """Return ``(source, module)`` for IF-04 or TR-01 ``run_native``, or ``None``."""
    for source, path in (("if04", _IF04_RUN), ("tr01", _TR01_RUN)):
        if not path.is_file():
            continue
        sibling = load_sibling_module(path)
        if callable(getattr(sibling, "run_native", None)):
            return source, sibling
    return None


def refuse_public_checkpoint() -> str:
    """Return why a native checkpoint timer cannot wrap IF-04 / TR-01."""
    return CHECKPOINT_PATH_REFUSAL



def official_authority() -> dict:
    """PF-10 is absent from benchmarks/manifest.toml."""
    return _v15.official_authority("PF-10")


def run_native(*args, request=None, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import OfficialBenchmarkUnavailable

    try:
        return _v15.run_mapped_or_refuse("PF-10", request)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc

