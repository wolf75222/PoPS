"""IF-05 in-memory dump cadences plus a capability-gated native wrapper.

Each cadence evaluates the same exact TR-01 state at dump times.
Dumps copy the field and must not mutate it.

A public ScientificOutput / ConsumerGraph exists, but TR-01 ``build_case``
does not expose a program clock or state Handle, so a dump-every-step
consumer cannot be attached without re-authoring the Case. ``run_native``
refuses that missing attach; it does not compile or call pops.run.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.native_evidence import campaign_run_fields
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")

MUTATION_AMPLITUDE = 0.25
OUTPUT_CADENCE_REFUSAL = "public output-cadence consumer attach not active"


class NativeUnavailable(RuntimeError):
    """Raised when two TR-01 dump cadences cannot be attached publicly."""


def dump_state(field) -> np.ndarray:
    """Return an independent copy of the field. Does not mutate the source."""
    return np.asarray(field, dtype=np.float64).copy()


def mutate_dump(field, amplitude: float = MUTATION_AMPLITUDE, index: int = 0):
    """Return a leftover dump with `amplitude` added at one cell."""
    leftover = dump_state(field)
    leftover[int(index)] = leftover[int(index)] + float(amplitude)
    return leftover


def advance_cadence(
    cadence,
    n_cells: int = _exact.DEFAULT_N_CELLS,
    *,
    n_steps: int = _exact.N_STEPS,
    t=_exact.T,
):
    """Advance analytically to t, dumping every `cadence` steps."""
    steps = int(n_steps)
    dt = float(t) / float(steps)
    centers = _exact.cell_centers(n_cells)
    volumes = _exact.cell_volumes(n_cells)
    field = _exact.exact_field(centers, 0.0)
    dumps = []
    for step in range(1, steps + 1):
        time = float(step) * dt
        field = _exact.exact_field(centers, time)
        if _exact.should_dump(step, cadence):
            dumps.append({"t": time, "q": dump_state(field)})
    return {
        "t": float(t),
        "field": field,
        "dumps": dumps,
        "centers": centers,
        "volumes": volumes,
    }


def exact_fields_for_cadences(
    n_cells: int = _exact.DEFAULT_N_CELLS, t=_exact.T
):
    """Return final exact fields keyed by dump cadence 1 / 2 / 10."""
    return {
        cadence: advance_cadence(cadence, n_cells, t=t)["field"]
        for cadence in _exact.CADENCES
    }


def max_cadence_difference(
    n_cells: int = _exact.DEFAULT_N_CELLS, t=_exact.T
) -> float:
    """Return the max pairwise L∞ between the three cadence finals."""
    fields = exact_fields_for_cadences(n_cells, t)
    volumes = _exact.cell_volumes(n_cells)
    linf = 0.0
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        linf = max(linf, errors.linf)
    return float(linf)


def leftover_dump_linf(
    n_cells: int = _exact.DEFAULT_N_CELLS,
    *,
    amplitude: float = MUTATION_AMPLITUDE,
    t=_exact.T,
) -> float:
    """Return L∞ of an in-place mutated dump versus the unmutated field."""
    centers = _exact.cell_centers(n_cells)
    volumes = _exact.cell_volumes(n_cells)
    field = _exact.exact_field(centers, t)
    dumped = dump_state(field)
    dumped[0] += float(amplitude)
    return reference_errors(dumped, field, volumes).linf


def refuse_output_cadence_consumer() -> str:
    """Return why a public dump-every-step consumer cannot be attached."""
    return OUTPUT_CADENCE_REFUSAL


def run_native(n_cells: int = _exact.DEFAULT_N_CELLS, t_end=_exact.T, request=None):
    """Compare TR-01 with no dumps vs NPZ dump every step.

    Uses ``_author.instance`` (public StateHandle) and ``ScientificOutput``.
    Output is staged on ``/tmp`` because GPFS rejects renameat2.
    """
    import os
    import tempfile

    from pops.output import ConsumerGraph, NPZ, ScientificOutput
    from pops.time import every

    from verification.pops_verify.tr01_runtime import advance, prepare

    _v15.bind_campaign(request, NativeUnavailable)
    if request is not None and request.min_resolution is not None:
        n_cells = int(request.min_resolution)
    work = Path(tempfile.mkdtemp(prefix="if05-", dir="/tmp" if Path("/tmp").is_dir() else None))
    try:
        plain = prepare(int(n_cells))
        field_off = advance(plain, float(t_end), output_dir=work)

        def _attach(authored) -> None:
            clock = authored.case._time.clock
            authored.case.consumers(
                ConsumerGraph.from_consumers(
                    (
                        ScientificOutput(
                            format=NPZ(),
                            schedule=every(1, clock=clock),
                            fields=(authored.instance,),
                            target="dumps/q",
                        ),
                    )
                )
            )

        dumped = prepare(int(n_cells), attach=_attach)
        field_on = advance(dumped, float(t_end), output_dir=work)
    except Exception as exc:
        if exc.__class__.__name__ == "NativeUnavailable":
            raise NativeUnavailable(str(exc)) from exc
        raise NativeUnavailable(f"IF-05 cadence compare failed: {exc}") from exc
    volumes = _exact.cell_volumes(n_cells)
    errors = reference_errors(field_on, field_off, volumes)
    payload = {
        "off": field_off,
        "on": field_on,
        "linf": float(errors.linf),
        "l2": float(errors.l2),
        "dumps": list(work.rglob("*.npz")),
        "comparison_artifacts": {
            "kind": "output_cadence",
            "dumps": [str(path) for path in work.rglob("*.npz")],
            "linf": float(errors.linf),
        },
    }
    if request is None:
        return payload
    fields = campaign_run_fields(
        request=request,
        n_cells=n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=0.45,
        comparison=payload["comparison_artifacts"],
    )
    fields.update(payload)
    return fields
