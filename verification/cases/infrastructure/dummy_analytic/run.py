"""In-memory numerical sample of the manufactured 1-d cosine.

Does not compile, bind, or launch a solver.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PERTURBATION = 1.0e-12


def _exact_module():
    path = Path(__file__).with_name("exact.py")
    spec = importlib.util.spec_from_file_location("dummy_analytic_exact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numerical_sample(n_cells: int = 32, perturbation: float = PERTURBATION):
    """Same grid as the oracle, with an optional tiny pointwise perturbation."""
    u_exact, volumes = _exact_module().exact_sample(n_cells=n_cells)
    field = np.asarray(u_exact, dtype=np.float64) + float(perturbation)
    return field, volumes
