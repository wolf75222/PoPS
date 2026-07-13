"""One real scalar-advection assembly crosses every typed lifecycle phase."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pops


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_m1_scalar_advection", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scalar_advection_completes_typed_phase_pipeline():
    example = _load_example()
    target = example.build_final_case()

    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout)
    artifact = pops.compile(resolved)
    simulation = pops.bind(
        artifact,
        params=example.build_bind_params(target.authoring),
    )

    assert validated is target.authoring.case and validated.frozen
    assert artifact.plan is resolved
    artifact.verify()
    assert simulation.lifecycle_state() == "bound"
    assert simulation.bind_identity().domain == "bind"
