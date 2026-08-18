"""CP-02 analysis from native fields or an EvidenceBundle; fail-closed otherwise."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.evidence_bundle import EvidenceBundle, EvidenceError
from verification.pops_verify.metrics import write_metrics
from verification.pops_verify.native_evidence import (
    fail_closed_report,
    native_diagnostics,
    native_report_sections,
    order_rows,
)
from verification.pops_verify.phase import frequency_error, numerical_frequency, phase_error

_RUN = load_sibling_module(_CASE_DIR / "run.py")
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import write_verification_report

_exact = load_sibling_module(_CASE_DIR / "exact.py")
CASE_ID = "CP-02"
NATIVE_DIMS = [1]
SUITE = "pr"
COMPONENT = "euler_poisson"
ORDER_THRESHOLD = 1.8
ROUNDING_FLOOR = 1.0e-14
BLOCKER = (
    "no native Kokkos output; exact-vs-exact and synthetic identities "
    "are not scientific evidence"
)


def final_campaign_job(series_dir) -> Path:
    """Last job named in series.json, including temporal ``dt*`` names."""
    root = Path(series_dir)
    series_file = root / "series.json"
    if not series_file.is_file():
        raise EvidenceError("series.json is required to resolve the final campaign job")
    jobs = list(json.loads(series_file.read_text(encoding="utf-8")).get("jobs") or [])
    if not jobs:
        raise EvidenceError("series.json jobs is empty")
    job = root / str(jobs[-1])
    if not job.is_dir():
        raise EvidenceError(f"final series job {jobs[-1]!r} is missing")
    return job


def analyze_native(native):
    """Compute phase/frequency/field/energy/order from native arrays only."""
    return native_diagnostics(native)


def _spectral_dx(field, length: float = 1.0) -> np.ndarray:
    samples = np.asarray(field, dtype=np.float64)
    spacing = float(length) / float(samples.size)
    wave = 2.0 * np.pi * np.fft.fftfreq(samples.size, d=spacing)
    return np.fft.ifft(1j * wave * np.fft.fft(samples)).real


def _gauss_sign_ok(electric, density) -> bool:
    field = np.asarray(electric, dtype=np.float64)
    n = np.asarray(density, dtype=np.float64)
    if field.size == 0 or n.size == 0 or field.size != n.size:
        return False
    d_e = _spectral_dx(field)
    rhs = _exact.E_CHARGE * (_exact.N_I - n) / _exact.EPS0
    if float(np.linalg.norm(d_e)) == 0.0 or float(np.linalg.norm(rhs)) == 0.0:
        return False
    return float(np.dot(d_e, rhs)) > 0.0


def _floats(values) -> list[float]:
    return [float(item) for item in np.asarray(values, dtype=np.float64).ravel()]


def _series(name: str, x, y, unit: str) -> dict[str, Any]:
    return {"name": name, "x": _floats(x), "y": _floats(y), "unit": unit}


def _profile_block(kind: str, x, numerical, exact, error, ylabel: str) -> dict[str, Any]:
    return {
        "figure_id": kind,
        "kind": kind,
        "data_kind": "campaign",
        "units": {"x": "x / L", "y": ylabel},
        "variables": [ylabel],
        "series": [
            _series("exact", x, exact, ylabel),
            _series("numerical", x, numerical, ylabel),
            _series("error", x, error, ylabel),
        ],
        "times": [1.0],
        "step_numbers": [0],
        "title": kind.replace("_", " "),
    }


def _array_or(value, default) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def write_campaign_visual_data(output_dir, campaign: dict) -> dict[str, Any]:
    """Write Phase 8 visual_data from a native campaign mapping. Does not mint pass."""
    if not isinstance(campaign, dict) or campaign.get("source") == "exact-vs-exact":
        raise ValueError("campaign visual_data requires native arrays")
    root = Path(output_dir)
    visual_dir = root / "analysis" / "visual_data"
    visual_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(campaign.get("x"), dtype=np.float64)
    density = campaign.get("density") or {}
    velocity = campaign.get("velocity") or {}
    potential = campaign.get("potential") or {}
    electric = campaign.get("electric") or {}
    times = _array_or(campaign.get("times"), [0.0])
    written: dict[str, Any] = {}

    reference = _profile_block(
        "reference_profile",
        x,
        density.get("numerical", x),
        density.get("exact", x),
        density.get("error", np.zeros_like(x)),
        "n_e",
    )
    # Overlay u, φ, E on the same 1-d profile figure.
    for name, block, unit in (
        ("velocity", velocity, "u_e"),
        ("potential", potential, "phi"),
        ("electric", electric, "E"),
    ):
        if block:
            reference["series"].extend(
                [
                    _series(f"exact_{name}", x, block.get("exact", np.zeros_like(x)), unit),
                    _series(f"numerical_{name}", x, block.get("numerical", np.zeros_like(x)), unit),
                ]
            )
    written["reference_profile"] = reference

    error = np.asarray(density.get("error", np.zeros_like(x)), dtype=np.float64)
    if np.max(error) <= 0.0 or np.min(error) >= 0.0:
        # Prefer a signed field when density error is one-sided.
        signed = np.asarray(electric.get("error", error), dtype=np.float64)
        if np.max(signed) <= 0.0 or np.min(signed) >= 0.0:
            signed = np.asarray(density.get("numerical", x), dtype=np.float64) - float(
                _exact.N0
            )
    else:
        signed = error
    written["signed_error_profile"] = {
        "figure_id": "signed_error_profile",
        "kind": "signed_error_profile",
        "data_kind": "campaign",
        "units": {"x": "x / L", "y": "signed error"},
        "variables": ["error"],
        "series": [_series("error", x, signed, "1")],
        "times": _floats(times[:1] if times.size else [0.0]),
        "step_numbers": [0],
        "title": "signed error",
    }

    family = str(campaign.get("family") or "global")
    resolutions = [int(item) for item in (campaign.get("resolutions") or [])]
    if family == "temporal":
        axis = [float(value) for value in (campaign.get("spacings") or [])]
        x_unit = "dt"
    else:
        axis = [float(value) for value in resolutions]
        x_unit = "1/h"
    written["spatial_convergence"] = {
        "figure_id": "spatial_convergence",
        "kind": "spatial_convergence",
        "data_kind": "campaign",
        "units": {"x": x_unit, "y": "error"},
        "variables": ["density"],
        "series": [
            _series("L1", axis, _array_or(campaign.get("l1"), []), "1"),
            _series("L2", axis, _array_or(campaign.get("l2"), []), "1"),
            _series("Linf", axis, _array_or(campaign.get("linf"), []), "1"),
        ],
        "reference_slopes": [
            {
                "name": "§9.3 last-two L∞ ≥ 1.8",
                "order": ORDER_THRESHOLD,
                "anchor": [axis[0] if axis else 16, 1.0e-3],
            }
        ],
        "times": _floats(times[:1] if times.size else [1.0]),
        "step_numbers": [0],
        "title": "CP-02 %s convergence"
        % (campaign.get("family") or "global"),
    }

    probe_e = _array_or(campaign.get("probe_e"), [0.0])
    written["phase_amplitude"] = {
        "figure_id": "phase_amplitude",
        "kind": "phase_amplitude",
        "data_kind": "campaign",
        "units": {"x": "t / T", "y": "E probe"},
        "variables": ["E"],
        "series": [_series("probe_e", times, probe_e, "E")],
        "times": _floats(times),
        "step_numbers": list(range(len(times))),
        "title": "probe electric field",
        "omega_fft": campaign.get("omega_fft"),
        "omega_phase": campaign.get("omega_phase"),
        "omega_zero": campaign.get("omega_zero"),
        "omega_num": campaign.get("omega_num"),
        "e_omega": campaign.get("e_omega"),
        "method_disagreement": campaign.get("method_disagreement"),
        "harmonic_h2": campaign.get("harmonic_h2"),
        "fft_official": False,
    }

    omegas = [
        float(campaign[key])
        for key in ("omega_fft", "omega_phase", "omega_zero")
        if campaign.get(key) is not None
    ]
    freqs = omegas or [1.0]
    written["frequency_spectrum"] = {
        "figure_id": "frequency_spectrum",
        "kind": "frequency_spectrum",
        "data_kind": "campaign",
        "units": {"x": "method", "y": "omega"},
        "variables": ["omega"],
        "series": [_series("omega", list(range(1, len(freqs) + 1)), freqs, "1/t")],
        "times": [float(times[-1]) if times.size else 1.0],
        "step_numbers": [0],
        "title": "frequency estimators",
    }

    written["invariants_vs_time"] = {
        "figure_id": "invariants_vs_time",
        "kind": "invariants_vs_time",
        "data_kind": "campaign",
        "units": {"x": "t / T", "y": "energy"},
        "variables": ["energy"],
        "series": [
            _series("ke", times, _array_or(campaign.get("ke"), [0.0] * max(times.size, 1)), "1"),
            _series("ese", times, _array_or(campaign.get("ese"), [0.0] * max(times.size, 1)), "1"),
        ],
        "times": _floats(times),
        "step_numbers": list(range(max(times.size, 1))),
        "title": "kinetic and electrostatic energy",
    }

    space_time = np.asarray(campaign.get("space_time_n"), dtype=np.float64)
    if space_time.ndim != 2:
        space_time = np.stack([np.asarray(density.get("numerical", x), dtype=np.float64)])
    written["hero_figure"] = {
        "figure_id": "hero_figure",
        "kind": "hero_figure",
        "data_kind": "campaign",
        "units": {"x": "x / L", "y": "t / T", "field": "n_e"},
        "variables": ["n_e"],
        "x": _floats(x),
        "y": _floats(times[: space_time.shape[0]]),
        "field": [[float(value) for value in row] for row in space_time],
        "times": _floats(times[: space_time.shape[0]]),
        "step_numbers": list(range(space_time.shape[0])),
        "title": "space-time density",
    }

    written["report_figure"] = {
        "figure_id": "report_figure",
        "kind": "report_figure",
        "data_kind": "campaign",
        "units": {"x": "x / L", "y": "n_e"},
        "variables": ["n_e"],
        "series": reference["series"][:3],
        "panels": [
            {
                "type": "profile",
                "name": "density",
                "x": _floats(x),
                "y": _floats(density.get("numerical", x)),
                "xlabel": "x / L",
                "ylabel": "n_e",
                "title": "density",
            }
        ],
        "times": _floats(times[:1] if times.size else [1.0]),
        "step_numbers": [0],
        "title": "CP-02 report figure",
    }

    frames = []
    for index, row in enumerate(space_time):
        stamp = float(times[index]) if index < times.size else float(index)
        frames.append(
            {
                "event": "accepted",
                "time": stamp,
                "step": index,
                "source": "native",
                "x": _floats(x),
                "y": [0.0, 1.0],
                "field": [_floats(row), _floats(row)],
                "series": [
                    _series("numerical", x, row, "n_e"),
                    _series("exact", x, density.get("exact", row), "n_e"),
                ],
            }
        )
    written["storyboard"] = {
        "figure_id": "storyboard",
        "kind": "storyboard",
        "data_kind": "campaign",
        "units": {"x": "x / L", "y": "n_e"},
        "frames": frames[:3] or frames,
        "times": [float(frame["time"]) for frame in (frames[:3] or frames)],
        "step_numbers": [int(frame["step"]) for frame in (frames[:3] or frames)],
    }
    peak = float(np.max(np.abs(space_time))) if space_time.size else 1.0
    written["animation"] = {
        "figure_id": "animation",
        "kind": "animation",
        "data_kind": "campaign",
        "units": {"x": "x / L", "y": "strip", "field": "n_e"},
        "frames": frames,
        "color_limits": [float(_exact.N0 - peak), float(_exact.N0 + peak)]
        if peak
        else [0.999, 1.001],
        "periodic": True,
        "times": [float(frame["time"]) for frame in frames],
        "step_numbers": [int(frame["step"]) for frame in frames],
    }

    for name, document in written.items():
        (visual_dir / f"{name}.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    return written


def _cell_oracle(n_cells: int, time: float) -> dict[str, np.ndarray]:
    width = 1.0 / float(n_cells)
    lo = np.arange(int(n_cells), dtype=np.float64) * width
    hi = lo + width
    volumes = np.full(int(n_cells), width, dtype=np.float64)
    return {
        "n": analytic_cell_averages(lambda x: _exact.n_e(x, time), lo, hi),
        "u": analytic_cell_averages(lambda x: _exact.u_e(x, time), lo, hi),
        "nu": analytic_cell_averages(
            lambda x: _exact.n_e(x, time) * _exact.u_e(x, time), lo, hi
        ),
        "e": analytic_cell_averages(lambda x: _exact.e_field(x, time), lo, hi),
        "phi": analytic_cell_averages(lambda x: _exact.phi(x, time), lo, hi),
        "volumes": volumes,
    }


def _claim_from_errors(l1, l2, linf, spacings) -> dict[str, Any]:
    orders = [float(value) for value in observed_order(linf, spacings)]
    usable = [index for index, error in enumerate(linf) if float(error) > ROUNDING_FLOOR]
    usable_set = set(usable)
    interval_ids = [
        index
        for index in range(len(orders))
        if index in usable_set and (index + 1) in usable_set
    ]
    if len(interval_ids) < 2:
        return {
            "order_pass": False,
            "orders": orders,
            "gated_orders": [],
            "reason": (
                "insufficient finest L∞ intervals above rounding floor "
                f"{ROUNDING_FLOOR} to apply spatial_order_min {ORDER_THRESHOLD}"
            ),
        }
    gated = [orders[index] for index in interval_ids[-2:]]
    order_pass = all(value >= ORDER_THRESHOLD for value in gated)
    return {
        "order_pass": order_pass,
        "orders": orders,
        "gated_orders": gated,
        "reason": None
        if order_pass
        else (
            f"observed L∞ gated orders {gated} below spatial_order_min "
            f"{ORDER_THRESHOLD}; all intervals {orders}"
        ),
    }


def _frequency_block(times, probe, oracle_probe, omega_ref: float) -> dict[str, Any]:
    stamps = np.asarray(times, dtype=np.float64)
    samples = np.asarray(probe, dtype=np.float64)
    methods = {}
    for name in ("fft", "phase_fit", "zero_crossing"):
        series_t, series_p = stamps, samples
        if name == "zero_crossing":
            start = 0
            while start < samples.size and samples[start] == 0.0:
                start += 1
            if 0 < start < samples.size - 2:
                series_t = stamps[start:]
                series_p = samples[start:]
        try:
            methods[name] = float(numerical_frequency(series_t, series_p, method=name))
        except ValueError:
            methods[name] = None
    omega_num = methods.get("phase_fit")
    if omega_num is None:
        omega_num = float("nan")
    known = [value for value in methods.values() if value is not None]
    disagreement = float(max(known) - min(known)) if len(known) >= 2 else None
    e_omega = (
        float(frequency_error(omega_num, omega_ref)) if np.isfinite(omega_num) else None
    )
    wrapped_phase = None
    try:
        wrapped_phase = float(phase_error(samples, oracle_probe))
    except ValueError:
        wrapped_phase = None
    return {
        "omega_fft": methods.get("fft"),
        "omega_phase": methods.get("phase_fit"),
        "omega_zero": methods.get("zero_crossing"),
        "omega_num": float(omega_num) if np.isfinite(omega_num) else float("nan"),
        "e_omega": e_omega,
        "frequency_error": e_omega,
        "method_disagreement": disagreement,
        "phase_error": wrapped_phase,
        "harmonic_h2": None,
        "fft_official": False,
        "omega_fft_note": (
            "one-period 65-sample FFT is a disagreement diagnostic, not omega_num"
        ),
    }


def _campaign_from_bundle(bundle: EvidenceBundle) -> dict[str, Any]:
    records = list(bundle.records)
    if len(records) < 4:
        raise EvidenceError("CP-02 order claim requires four native resolutions")
    resolutions = []
    l1 = []
    l2 = []
    linf = []
    fields = {}
    oracles = {}
    volumes = {}
    finest = records[-1]
    for record in records:
        provenance = record["provenance"]
        n_cells = int((provenance.get("resolution") or [record["result"].shape[-1]])[0])
        time = float(provenance.get("final_time") or 0.0)
        result = np.asarray(record["result"], dtype=np.float64)
        oracle = _cell_oracle(n_cells, time)
        errors = reference_errors(result[0], oracle["n"], oracle["volumes"])
        resolutions.append(n_cells)
        l1.append(float(errors.l1))
        l2.append(float(errors.l2))
        linf.append(float(errors.linf))
        fields[n_cells] = result
        oracles[n_cells] = oracle
        volumes[n_cells] = oracle["volumes"]
        if float(errors.linf) <= 0.0:
            raise EvidenceError("refusing exact-vs-exact native errors as an order pass")
    resolved = finest.get("resolved_case") or {}
    family = str(resolved.get("family") or "global")
    if family == "temporal":
        spacings = []
        for record in records:
            job = (record.get("resolved_case") or {}).get("job") or {}
            step = job.get("dt")
            spacings.append(float(step) if step is not None else 0.0)
        if any(value <= 0.0 for value in spacings):
            raise EvidenceError("CP-02 temporal campaign requires positive dt in resolved job")
    else:
        spacings = [1.0 / float(n) for n in resolutions]
    claim = _claim_from_errors(l1, l2, linf, spacings)
    claim["family"] = family
    claim["kind"] = family if family in {"temporal", "spatial"} else "global"
    coupling = {}
    coupling_candidates = []
    try:
        coupling_candidates.append(final_campaign_job(bundle.path) / "coupling.json")
    except EvidenceError:
        pass
    for coupling_path in coupling_candidates:
        if coupling_path.is_file():
            coupling = json.loads(coupling_path.read_text(encoding="utf-8"))
            break
    times = np.asarray(coupling.get("times") or [float(finest["provenance"].get("final_time") or 0.0)])
    probe = np.asarray(coupling.get("probe_e") or [0.0], dtype=np.float64)
    centers, _ = _exact.uniform_cell_centers(int(resolutions[-1]))
    index = min(int(0.25 * resolutions[-1]), resolutions[-1] - 1)
    oracle_probe = _exact.e_field(centers[index], times)
    frequency = _frequency_block(times, probe, oracle_probe, _exact.plasma_frequency())
    electric = np.asarray(coupling.get("electric") or [], dtype=np.float64)
    if electric.size:
        spectrum = np.fft.rfft(electric)
        k_index = 1
        h2 = (
            float(abs(spectrum[2]) / abs(spectrum[k_index]))
            if spectrum.size > 2 and abs(spectrum[k_index]) > 0.0
            else None
        )
        frequency["harmonic_h2"] = h2
        phi = np.asarray(coupling.get("phi"), dtype=np.float64)
        phi_oracle = oracles[resolutions[-1]]["phi"]
        e_oracle = oracles[resolutions[-1]]["e"]
        phi_err = reference_errors(phi, phi_oracle, volumes[resolutions[-1]])
        e_err = reference_errors(electric, e_oracle, volumes[resolutions[-1]])
    else:
        phi_err = None
        e_err = None
    finest_n = resolutions[-1]
    density = fields[finest_n][0]
    momentum = fields[finest_n][1]
    velocity = momentum / np.maximum(density, 1.0e-30)
    campaign = {
        "source": "native",
        "case_id": CASE_ID,
        "resolutions": resolutions,
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "orders": claim["orders"],
        "x": (np.arange(finest_n, dtype=np.float64) + 0.5) / float(finest_n),
        "density": {
            "numerical": density,
            "exact": oracles[finest_n]["n"],
            "error": density - oracles[finest_n]["n"],
        },
        "velocity": {
            "numerical": velocity,
            "exact": oracles[finest_n]["u"],
            "error": velocity - oracles[finest_n]["u"],
        },
        "potential": {
            "numerical": np.asarray(coupling.get("phi") or np.zeros(finest_n)),
            "exact": oracles[finest_n]["phi"],
            "error": np.asarray(coupling.get("phi") or np.zeros(finest_n))
            - oracles[finest_n]["phi"],
        },
        "electric": {
            "numerical": np.asarray(coupling.get("electric") or np.zeros(finest_n)),
            "exact": oracles[finest_n]["e"],
            "error": np.asarray(coupling.get("electric") or np.zeros(finest_n))
            - oracles[finest_n]["e"],
        },
        "times": times,
        "probe_e": probe,
        "ke": coupling.get("ke") or [0.0],
        "ese": coupling.get("ese") or [0.0],
        "space_time_n": np.asarray(coupling.get("snapshots_n") or [density.tolist()]),
        **frequency,
        "claim": claim,
        "phi_err": phi_err,
        "e_err": e_err,
        "bundle": bundle,
        "mass": coupling.get("mass") or [],
        "momentum": coupling.get("momentum") or [],
        "charge": coupling.get("charge") or [],
        "sign_ok": _gauss_sign_ok(electric, density) if electric.size else False,
        "family": family,
        "spacings": spacings,
    }
    return campaign


def _write_campaign_metrics(output_dir: Path, campaign: dict, claim: dict) -> None:
    finest_l1 = float(campaign["l1"][-1])
    finest_l2 = float(campaign["l2"][-1])
    finest_linf = float(campaign["linf"][-1])
    observed = float(claim["orders"][-1]) if claim["orders"] else None
    mass = campaign.get("mass") or []
    momentum = campaign.get("momentum") or []
    ke = campaign.get("ke") or []
    ese = campaign.get("ese") or []
    charge = campaign.get("charge") or []
    energy_block = _RUN.energy_baseline(ke, ese) if ke and ese else {
        "initial": None,
        "final": None,
        "max_relative_drift": None,
    }
    energy = energy_block.get("values") or []
    phi_err = campaign.get("phi_err")
    e_err = campaign.get("e_err")
    velocity = campaign.get("velocity") or {}
    n_cells = int(len(campaign["density"]["numerical"]))
    volumes = np.full(n_cells, 1.0 / float(n_cells), dtype=np.float64)
    vel_err = None
    if velocity:
        vel_err = reference_errors(
            np.asarray(velocity["numerical"], dtype=np.float64),
            np.asarray(velocity["exact"], dtype=np.float64),
            volumes,
        )
    electric_num = np.asarray(campaign["electric"]["numerical"], dtype=np.float64)
    density_num = np.asarray(campaign["density"]["numerical"], dtype=np.float64)
    gauss_defect = None
    if electric_num.size == density_num.size and electric_num.size:
        defect = _spectral_dx(electric_num) - _exact.E_CHARGE * (
            _exact.N_I - density_num
        ) / _exact.EPS0
        gauss_defect = float(np.sqrt(np.mean(defect * defect)))
    reasons = {
        "extrema.p_min": "cold-fluid model has no pressure variable",
        "symmetry.error": "one-dimensional mode",
        "amr.interface_error": "uniform-grid configuration",
        "timings_seconds.ghost_fill": "ghost fill not timed in CP-02",
        "timings_seconds.reflux": "no reflux timing in CP-02",
        "errors.*.observed_order": "observed_order is the last global interval; §9.3 gates the last two L∞ pairs",
    }
    document = {
        "schema": "pops.verification.metrics.v1",
        "case_id": CASE_ID,
        "errors": {
            "density": {
                "l1": finest_l1,
                "l2": finest_l2,
                "linf": finest_linf,
                "observed_order": observed,
            },
            "velocity_x": {
                "l1": None if vel_err is None else float(vel_err.l1),
                "l2": None if vel_err is None else float(vel_err.l2),
                "linf": None if vel_err is None else float(vel_err.linf),
                "observed_order": None,
            },
            "potential": {
                "l1": None if phi_err is None else float(phi_err.l1),
                "l2": None if phi_err is None else float(phi_err.l2),
                "linf": None if phi_err is None else float(phi_err.linf),
                "observed_order": None,
            },
            "electric_field_x": {
                "l1": None if e_err is None else float(e_err.l1),
                "l2": None if e_err is None else float(e_err.l2),
                "linf": None if e_err is None else float(e_err.linf),
                "observed_order": None,
            },
        },
        "conservation": {
            "mass_total": {
                "initial": float(mass[0]) if mass else None,
                "final": float(mass[-1]) if mass else None,
                "max_relative_drift": (
                    float(max(abs(value - mass[0]) for value in mass) / abs(mass[0]))
                    if mass and mass[0] != 0.0
                    else None
                ),
            },
            "momentum_total": {
                "initial": [float(momentum[0])] if momentum else None,
                "final": [float(momentum[-1])] if momentum else None,
                "max_relative_drift": (
                    float(max(abs(value - momentum[0]) for value in momentum) / abs(momentum[0]))
                    if momentum and momentum[0] != 0.0
                    else (
                        float(max(abs(value) for value in momentum))
                        if momentum
                        else None
                    )
                ),
            },
            "energy_total": {
                "initial": energy_block.get("initial"),
                "final": energy_block.get("final"),
                "max_relative_drift": energy_block.get("max_relative_drift"),
            },
            "charge_total": {
                "initial": float(charge[0]) if charge else None,
                "final": float(charge[-1]) if charge else None,
                "max_absolute_drift": (
                    float(max(abs(value - charge[0]) for value in charge)) if charge else None
                ),
            },
            "electrostatic_energy": {
                "initial": float(ese[0]) if ese else None,
                "final": float(ese[-1]) if ese else None,
                "max_relative_drift": None,
            },
        },
        "poisson": {
            "residual_l2": None,
            "gauss_defect_l2": gauss_defect,
        },
        "extrema": {
            "rho_min": float(np.min(campaign["density"]["numerical"])),
            "p_min": None,
        },
        "symmetry": {"error": None},
        "amr": {
            "interface_error": None,
            "bulk_error": finest_linf,
            "leaf_cells": int(campaign["resolutions"][-1]),
            "patch_count": 1,
            "regrid_count": 0,
        },
        "not_applicable_reason": reasons,
        "timings_seconds": {"ghost_fill": None, "poisson": 0, "reflux": None},
        "csv_files": ["analysis/visual_data/spatial_convergence.json"],
        "time_series": {
            "omega_num": campaign.get("omega_num"),
            "omega_fft": campaign.get("omega_fft"),
            "omega_phase": campaign.get("omega_phase"),
            "omega_zero": campaign.get("omega_zero"),
            "e_omega": campaign.get("e_omega"),
            "method_disagreement": campaign.get("method_disagreement"),
            "harmonic_h2": campaign.get("harmonic_h2"),
            "fft_official": False,
            "ese_oscillation": energy_block.get("ese_oscillation"),
            "total_energy_drift": energy_block.get("max_relative_drift"),
        },
    }
    if document["errors"]["velocity_x"]["observed_order"] is None:
        reasons["errors.velocity_x.observed_order"] = "velocity order is not the gated series"
    if document["poisson"]["residual_l2"] is None:
        reasons["poisson.residual_l2"] = "algebraic residual is not exported by the public FFT runtime"
    if document["poisson"]["gauss_defect_l2"] is None:
        reasons["poisson.gauss_defect_l2"] = "no native electric field in coupling.json"
    if document["errors"]["potential"]["observed_order"] is None:
        reasons["errors.potential.observed_order"] = "potential order is not the gated series"
    if document["errors"]["electric_field_x"]["observed_order"] is None:
        reasons["errors.electric_field_x.observed_order"] = "electric-field order is not the gated series"
    if document["conservation"]["energy_total"]["initial"] is None:
        reasons["conservation.energy_total.initial"] = "no non-zero energy sample after skipping fake E=0"
    if document["conservation"]["energy_total"]["max_relative_drift"] is None:
        reasons["conservation.energy_total.max_relative_drift"] = (
            "energy baseline skipped fake E=0; remaining series has no usable drift"
            if document["conservation"]["energy_total"]["initial"] is None
            else "energy baseline is zero"
        )
    reasons["conservation.electrostatic_energy.max_relative_drift"] = (
        "ESE oscillation is standing-wave KE↔ESE exchange, not total-energy drift; "
        f"ese_oscillation={energy_block.get('ese_oscillation')}"
    )
    write_metrics(output_dir / "metrics.json", document)


def write_cp02_report(
    output_dir,
    *,
    native=None,
    request=None,
    series_dir=None,
) -> dict:
    """Write artifacts. Required cases fail; never not-supported unless gated."""
    output = Path(output_dir)
    if series_dir is not None:
        try:
            bundle = EvidenceBundle(series_dir)
        except EvidenceError as exc:
            return write_verification_report(
                fail_closed_report(
                    case_id=CASE_ID,
                    component=COMPONENT,
                    native_dims=list(NATIVE_DIMS),
                    reason=str(exc),
                    suite=SUITE,
                    request=request,
                ),
                output,
            )
        campaign = _campaign_from_bundle(bundle)
        claim = campaign["claim"]
        write_campaign_visual_data(output, campaign)
        _write_campaign_metrics(output, campaign, claim)
        orders = [
            {
                "case_id": CASE_ID,
                "kind": claim.get("kind") or "global",
                "variable": "density",
                "observed_order": float(value),
                "threshold": ORDER_THRESHOLD,
            }
            for value in claim["orders"]
        ]
        failures = []
        if not campaign.get("sign_ok", False):
            failures.append(
                {
                    "case_id": CASE_ID,
                    "reason": "Gauss sign/coupling failed: spectral dE/dx is not positively aligned with e(n_i-n_e)/ε0",
                    "metrics_ref": "metrics.json",
                    "provenance_ref": "provenance.json",
                }
            )
        if not claim["order_pass"]:
            failures.append(
                {
                    "case_id": CASE_ID,
                    "reason": claim["reason"] or "order gate failed",
                    "metrics_ref": "metrics.json",
                    "provenance_ref": "provenance.json",
                }
            )
        passed = bool(claim["order_pass"] and campaign.get("sign_ok", False))
        ke = campaign.get("ke") or []
        ese = campaign.get("ese") or []
        energy_block = _RUN.energy_baseline(ke, ese) if ke and ese else {
            "initial": None,
            "max_relative_drift": None,
        }
        first = bundle.records[0]
        summary = {
            "schema": "pops.verification.report.v1",
            "repository": "wolf75222/PoPS",
            "repository_sha": first["provenance"]["repository_sha"],
            "suite": SUITE,
            "max_nodes": 2,
            "native_dimensions": list(NATIVE_DIMS),
            "execution_spaces": [first["provenance"]["kokkos_execution_space"]],
            "coverage": {
                "components": [COMPONENT],
                "cases_planned": 1,
                "cases_run": 1,
                "cases_passed": 1 if passed else 0,
                "cases_failed": 0 if passed else 1,
                "cases_not_supported": 0,
                "not_tested": [],
            },
            "failures": failures,
            "orders": orders,
            "amr": {
                "order_retained": None,
                "invariants_ok": None,
                "interface_error": None,
                "bulk_error": float(campaign["linf"][-1]),
            },
            "poisson": {
                "potential_error": None
                if campaign.get("phi_err") is None
                else float(campaign["phi_err"].linf),
                "field_error": None
                if campaign.get("e_err") is None
                else float(campaign["e_err"].linf),
                "residual_l2": None,
            },
            "coupling": {
                "phase_error": campaign.get("phase_error"),
                "sign_ok": bool(campaign.get("sign_ok", False)),
                "energy_drift": energy_block.get("max_relative_drift"),
            },
            "parallel_invariance": {
                "ranks_ok": None,
                "threads_ok": None,
                "gpu_ok": None,
            },
            "performance": {"one_node": None, "two_node": None},
            "not_applicable_reason": {
                "amr.*": "uniform 1-d CP-02 does not claim AMR",
                "parallel_invariance.*": "OpenMP/MPI smokes are separate series",
                "performance.*": "performance not measured",
                **(
                    {
                        "coupling.energy_drift": "no non-zero energy sample after skipping fake E=0"
                    }
                    if energy_block.get("max_relative_drift") is None
                    else {}
                ),
            },
            "artifacts": {
                "report_md": "REPORT.md",
                "summary_json": "summary.json",
                "coverage_csv": "coverage.csv",
                "failures_csv": "failures.csv",
            },
        }
        written = write_verification_report(summary, output)
        report_md = output / "REPORT.md"
        if report_md.is_file():
            report_md.write_text(
                report_md.read_text(encoding="utf-8")
                + "\n## Frequency\n\n"
                + (
                    f"- official omega_num (phase-fit, t0 E corrected) = {campaign.get('omega_num')}\n"
                    f"- E_omega = {campaign.get('e_omega')}\n"
                    f"- FFT (not official) = {campaign.get('omega_fft')}; "
                    f"zero-crossing = {campaign.get('omega_zero')}\n"
                    f"- method_disagreement = {campaign.get('method_disagreement')}\n"
                    f"- H2 = {campaign.get('harmonic_h2')}\n"
                    f"- family = {campaign.get('family')}\n"
                    f"- total energy drift = {energy_block.get('max_relative_drift')}\n"
                    f"- ESE oscillation = {energy_block.get('ese_oscillation')} "
                    f"(KE↔ESE exchange, not conservation drift)\n"
                ),
                encoding="utf-8",
            )
        provenance = dict(first["provenance"])
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        from verification.pops_verify.visualization.live import write_run_status

        write_run_status(
            output,
            case_id=CASE_ID,
            run_id=output.name,
            verdict="pass" if passed else "fail",
            scientific_pass=passed,
        )
        finest = final_campaign_job(bundle.path)
        for name in ("resolved_case.json", "native_artifact.json"):
            source = finest / name
            if source.is_file():
                (output / name).write_bytes(source.read_bytes())
        program_digest = finest / "program.sha256"
        (output / "program.json").write_text(
            json.dumps(
                {
                    "time_program": provenance.get("time_program"),
                    "cfl": provenance.get("cfl"),
                    "program_sha256": (
                        program_digest.read_text(encoding="utf-8").strip()
                        if program_digest.is_file()
                        else None
                    ),
                    "source_job": finest.name,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return written

    orders = []
    poisson = None
    coupling = None
    extra_reasons = None
    reason = BLOCKER
    if native is not None:
        diagnostics = analyze_native(native)
        orders = order_rows(CASE_ID, diagnostics)
        sections = native_report_sections(diagnostics)
        poisson = sections["poisson"]
        coupling = sections["coupling"]
        extra_reasons = sections["extra_reasons"]
        reason = (
            "native diagnostics computed from supplied arrays; Kokkos campaign "
            "is not authenticated in this isolated stream"
        )
    return write_verification_report(
        fail_closed_report(
            case_id=CASE_ID,
            component=COMPONENT,
            native_dims=list(NATIVE_DIMS),
            reason=reason,
            suite=SUITE,
            request=request,
            orders=orders,
            poisson=poisson,
            coupling=coupling,
            extra_reasons=extra_reasons,
        ),
        output,
    )


def render_campaign_figures(run_dir, *, suite: str = "nightly") -> dict[str, str]:
    """Render Phase 8 figures from a completed campaign directory."""
    from verification.pops_verify.visualization.render import render_run

    return render_run(run_dir, suite=suite, formats=("svg", "png", "pdf", "gif"))
