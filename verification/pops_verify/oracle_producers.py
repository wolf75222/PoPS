"""Case oracles from trusted resolved config and coordinates at analysis time.

Callers cannot pass an oracle array or ``error_fn``. A missing producer is a
hard failure.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.cell_averages import analytic_cell_averages

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES = REPO_ROOT / "verification" / "cases"


class OracleProducerError(ValueError):
    """Raised when a case has no independent oracle producer."""


def _exact(family: str, slug: str):
    return load_sibling_module(CASES / family / slug / "exact.py")


def _averages_1d(fn, n_cells: int, length: float = 1.0):
    width = float(length) / int(n_cells)
    lo = np.arange(int(n_cells), dtype=np.float64) * width
    hi = lo + width
    return analytic_cell_averages(fn, lo, hi)


def _averages_2d(fn, n_cells: int, length: float = 1.0, indexing: str = "ij"):
    width = float(length) / int(n_cells)
    axis_lo = np.arange(int(n_cells), dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing=indexing)
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing=indexing)
    lo = np.stack((x_lo, y_lo), axis=-1)
    hi = np.stack((x_hi, y_hi), axis=-1)
    return analytic_cell_averages(fn, lo, hi)


def _n_cells(result: Any, resolved: Mapping[str, Any]) -> int:
    field = np.asarray(result)
    job = resolved.get("job") if isinstance(resolved.get("job"), Mapping) else {}
    if job.get("min_resolution") is not None:
        return int(job["min_resolution"])
    if field.ndim == 0:
        raise OracleProducerError("result has no spatial shape")
    return int(field.shape[-1])


def _time(provenance: Mapping[str, Any]) -> float:
    return float(provenance.get("final_time", 0.0))


def _oracle_tr02(resolved, result, provenance):
    exact = _exact("transport", "gaussian_pulse")
    n_cells = _n_cells(result, resolved)
    return _averages_1d(lambda x: exact.exact_gaussian(x, _time(provenance)), n_cells)


def _oracle_tr06(resolved, result, provenance):
    exact = _exact("transport", "axis_permutation")
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)
    return _averages_2d(lambda x, y: exact.exact_product(x, y, t), n_cells)


def _oracle_tr07(resolved, result, provenance):
    exact = _exact("transport", "discontinuous_slot")
    n_cells = _n_cells(result, resolved)
    return _averages_1d(lambda x: exact.exact_slot(x, _time(provenance)), n_cells)


def _oracle_eu01(resolved, result, provenance):
    exact = _exact("euler", "linear_waves")
    field = np.asarray(result, dtype=np.float64)
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)

    def density(x):
        return exact.exact_mode(np.asarray(x, dtype=np.float64).reshape(-1), t, mode="entropy")[
            0
        ].reshape(np.shape(x))

    rho = _averages_1d(density, n_cells)
    if field.ndim == 1:
        return rho
    if field.shape == (3, n_cells):
        def vel(x):
            return exact.exact_mode(np.asarray(x, dtype=np.float64).reshape(-1), t, mode="entropy")[
                1
            ].reshape(np.shape(x))

        def pres(x):
            return exact.exact_mode(np.asarray(x, dtype=np.float64).reshape(-1), t, mode="entropy")[
                2
            ].reshape(np.shape(x))

        u = _averages_1d(vel, n_cells)
        p = _averages_1d(pres, n_cells)
        gamma = 1.4
        energy = p / (gamma - 1.0) + 0.5 * rho * u * u
        return np.stack((rho, rho * u, energy), axis=0)
    raise OracleProducerError("EU-01 oracle cannot match result shape")


def _oracle_eu02(resolved, result, provenance):
    exact = _exact("euler", "isentropic_vortex")
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)
    length = float(getattr(exact, "PERIOD", 1.0))
    rho = _averages_2d(
        lambda x, y: exact.exact_vortex(x, y, t, u_inf=1.0, v_inf=0.0)["rho"],
        n_cells,
        length,
        indexing="xy",
    )
    field = np.asarray(result)
    if field.shape == rho.shape:
        return rho
    raise OracleProducerError("EU-02 oracle expects a density array")


def _oracle_eu04(resolved, result, provenance):
    exact = _exact("euler", "standing_acoustic")
    field = np.asarray(result, dtype=np.float64)
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)

    def density(x):
        return np.asarray(exact.primitives_1d(np.asarray(x).reshape(-1), t)[0]).reshape(
            np.shape(x)
        )

    rho = _averages_1d(density, n_cells)
    if field.ndim == 1 or field.shape[-1] == n_cells and field.ndim == 1:
        return rho
    if field.shape[-1] == n_cells:
        return rho
    raise OracleProducerError("EU-04 oracle cannot match result shape")


def _oracle_eu05(resolved, result, provenance):
    exact = _exact("euler", "gresho")
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)
    rho = _averages_2d(lambda x, y: exact.exact_gresho(x, y, t)["rho"], n_cells)
    if np.asarray(result).shape == rho.shape:
        return rho
    raise OracleProducerError("EU-05 oracle expects a density array")


def _oracle_eu06(resolved, result, provenance):
    exact = _exact("euler", "uniform_flow")
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)
    rho = _averages_2d(lambda x, y: exact.exact_primitives(x, y, t)["rho"], n_cells)
    if np.asarray(result).shape == rho.shape:
        return rho
    raise OracleProducerError("EU-06 oracle expects a density array")


def _oracle_po01(resolved, result, provenance):
    exact = _exact("poisson", "periodic_trig")
    n_cells = _n_cells(result, resolved)
    return _averages_1d(exact.phi_exact, n_cells)


def _oracle_po03(resolved, result, provenance):
    exact = _exact("poisson", "neumann_nullspace")
    n_cells = _n_cells(result, resolved)
    phi = _averages_1d(exact.phi_exact, n_cells)
    return phi - float(np.mean(phi))


def _oracle_tm01(resolved, result, provenance):
    exact = _exact("time", "pure_temporal")
    n_cells = int(getattr(exact, "N_CELLS", 64))
    return _averages_1d(lambda x: exact.exact_sine(x, _time(provenance)), n_cells)


def _oracle_tr01(resolved, result, provenance):
    exact = _exact("transport", "advection_sine")
    field = np.asarray(result)
    n_cells = _n_cells(result, resolved)
    dim = int(field.ndim)
    if dim not in (1, 2, 3):
        raise OracleProducerError("TR-01 result rank must be 1, 2, or 3")
    job = resolved.get("job") if isinstance(resolved.get("job"), Mapping) else {}
    declared = job.get("pops_native_dim")
    if declared is not None and int(declared) != dim:
        raise OracleProducerError("TR-01 result rank does not match pops_native_dim")
    velocity = tuple(float(value) for value in exact.A[:dim])
    wave = tuple(float(value) for value in exact.K[:dim])
    lo, hi = exact.cell_bounds_nd(n_cells, dim)

    def _u(*args):
        *coords, time = args
        return exact.exact_sine_nd(coords, time, a=velocity, k=wave)

    return analytic_cell_averages(_u, lo, hi, _time(provenance))


PRODUCERS: dict[str, Callable[..., Any]] = {
    "TR-01": _oracle_tr01,
    "TR-02": _oracle_tr02,
    "TR-06": _oracle_tr06,
    "TR-07": _oracle_tr07,
    "EU-01": _oracle_eu01,
    "EU-02": _oracle_eu02,
    "EU-04": _oracle_eu04,
    "EU-05": _oracle_eu05,
    "EU-06": _oracle_eu06,
    "PO-01": _oracle_po01,
    "PO-03": _oracle_po03,
    "TM-01": _oracle_tm01,
}

CASE_SOURCES = {
    "TR-01": ("transport", "advection_sine"),
    "TR-02": ("transport", "gaussian_pulse"),
    "TR-06": ("transport", "axis_permutation"),
    "TR-07": ("transport", "discontinuous_slot"),
    "EU-01": ("euler", "linear_waves"),
    "EU-02": ("euler", "isentropic_vortex"),
    "EU-04": ("euler", "standing_acoustic"),
    "EU-05": ("euler", "gresho"),
    "EU-06": ("euler", "uniform_flow"),
    "PO-01": ("poisson", "periodic_trig"),
    "PO-03": ("poisson", "neumann_nullspace"),
    "TM-01": ("time", "pure_temporal"),
}


def _oracle_tr06_pair(resolved, result, provenance):
    exact = _exact("transport", "axis_permutation")
    n_cells = _n_cells(result, resolved)
    t = _time(provenance)
    return _averages_2d(lambda x, y: exact.exact_product(y, x, t), n_cells)


def producer_source_files(case_id: str) -> list[Path]:
    files = [
        Path(__file__).resolve(),
        (REPO_ROOT / "verification" / "pops_verify" / "cell_averages.py").resolve(),
        (REPO_ROOT / "verification" / "pops_verify" / "case_authoring.py").resolve(),
    ]
    spec = CASE_SOURCES.get(str(case_id))
    if spec is not None:
        exact = (CASES / spec[0] / spec[1] / "exact.py").resolve()
        files.append(exact)
        text = exact.read_text(encoding="utf-8")
        if "advection_sine" in text:
            files.append((CASES / "transport" / "advection_sine" / "exact.py").resolve())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _git_blob(rel: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", f"HEAD:{rel}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    blob = completed.stdout.strip()
    if completed.returncode != 0 or not blob:
        raise OracleProducerError(f"no committed git blob for {rel}")
    return blob


def _git_head_sha256(rel: str) -> str:
    import hashlib
    import subprocess

    completed = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OracleProducerError(f"cannot read committed blob for {rel}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_path_dirty(rel: str) -> bool:
    import subprocess

    completed = subprocess.run(
        ["git", "status", "--porcelain", "--", rel],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OracleProducerError(f"cannot read git status for {rel}")
    return bool(completed.stdout.strip())


def dirty_producer_paths(case_id: str) -> list[str]:
    from verification.pops_verify.capabilities import sha256_file

    dirty: list[str] = []
    for path in producer_source_files(case_id):
        rel = str(path.relative_to(REPO_ROOT))
        if _git_path_dirty(rel) or sha256_file(path) != _git_head_sha256(rel):
            dirty.append(rel)
    return dirty


def hash_producer_files(case_id: str) -> dict[str, dict[str, str]]:
    from verification.pops_verify.capabilities import sha256_file

    manifest: dict[str, dict[str, str]] = {}
    for path in producer_source_files(case_id):
        rel = str(path.relative_to(REPO_ROOT))
        manifest[rel] = {
            "sha256": sha256_file(path),
            "git_blob": _git_blob(rel),
            "head_sha256": _git_head_sha256(rel),
        }
    return manifest


def verify_committed_producers(case_id: str) -> dict[str, dict[str, str]]:
    """Fail if any oracle-affecting source is dirty or differs from HEAD."""
    dirty = dirty_producer_paths(case_id)
    if dirty:
        raise OracleProducerError(f"dirty producer files: {dirty}")
    manifest = hash_producer_files(case_id)
    for rel, row in manifest.items():
        if row["sha256"] != row["head_sha256"]:
            raise OracleProducerError(f"producer differs from HEAD: {rel}")
        if row["git_blob"] != _git_blob(rel):
            raise OracleProducerError(f"producer git blob mismatch: {rel}")
    return manifest


def produce_paired_oracle(
    case_id: str,
    resolved: Mapping[str, Any],
    result: Any,
    provenance: Mapping[str, Any],
) -> np.ndarray:
    if str(case_id) != "TR-06":
        raise OracleProducerError(f"no paired oracle producer for {case_id}")
    oracle = np.ascontiguousarray(
        np.asarray(_oracle_tr06_pair(resolved, result, provenance), dtype=np.float64)
    )
    field = np.asarray(result, dtype=np.float64)
    if oracle.shape != field.shape:
        raise OracleProducerError("paired oracle shape does not match pair result")
    return oracle


def produce_oracle(
    case_id: str,
    resolved: Mapping[str, Any],
    result: Any,
    provenance: Mapping[str, Any],
) -> np.ndarray:
    """Return the case oracle. Raises if no independent producer exists."""
    producer = PRODUCERS.get(str(case_id))
    if producer is None:
        raise OracleProducerError(f"absent oracle producer for {case_id}")
    oracle = np.ascontiguousarray(np.asarray(producer(resolved, result, provenance), dtype=np.float64))
    field = np.asarray(result, dtype=np.float64)
    if oracle.shape != field.shape:
        raise OracleProducerError(
            f"oracle shape {oracle.shape} does not match result {field.shape}"
        )
    return oracle
