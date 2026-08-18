"""Shared TR-01 compile / bind / run for IF and PF campaigns.

Uses the public Case from ``transport/advection_sine`` plus optional
consumers. ``pops.run`` stays here so case modules can call this helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import (
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)

_REPO = Path(__file__).resolve().parents[2]
_TR01 = _REPO / "verification" / "cases" / "transport" / "advection_sine" / "run.py"


class Prepared:
    __slots__ = ("module", "authored", "artifact", "initial")

    def __init__(self, module: Any, authored: Any, artifact: Any, initial: np.ndarray) -> None:
        self.module = module
        self.authored = authored
        self.artifact = artifact
        self.initial = initial


def tr01_module():
    return load_sibling_module(_TR01)


def prepare(n_cells: int, *, consumers=None, attach=None) -> Prepared:
    """Validate/compile TR-01. Optional ``attach(authored)`` or ConsumerGraph."""
    import pops

    module = tr01_module()
    missing = module._native_unavailable_reason()
    if missing:
        raise module.NativeUnavailable(missing)
    if hasattr(module, "_require_native_dim3"):
        module._require_native_dim3()
    authored = module._author(int(n_cells))
    if attach is not None:
        attach(authored)
    if consumers is not None:
        authored.case.consumers(consumers)
    count = authored.n_cells
    layout = uniform_periodic_layout(authored.frame, (count, count, count))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    return Prepared(module, authored, artifact, np.zeros(0, dtype=np.float64))


def advance(prepared: Prepared, t_end: float, *, output_dir=None) -> np.ndarray:
    """Bind and run one prepared artifact. Returns the gathered 1-d field."""
    import pops

    from verification.pops_verify.case_authoring import bind_public

    if prepared.initial.size:
        simulation = bind_public(
            prepared.artifact,
            initial_values={prepared.authored.instance: prepared.initial},
        )
    else:
        simulation = bind_public(prepared.artifact)
    kwargs = {"t_end": float(t_end), "max_steps": prepared.module.MAX_STEPS}
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    pops.run(simulation, **kwargs)
    return np.ravel(np.asarray(simulation.state_global("tracer"), dtype=np.float64))
