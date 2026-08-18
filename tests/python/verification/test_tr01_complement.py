"""TR-01 complement catalog, oracle, and diagnostics (no native run)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine"


def _load(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tr01_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_covers_obligatory_velocities_and_layouts():
    complement = _load("complement")
    one = complement.catalog(dim=1)
    two = complement.catalog(dim=2)
    three = complement.catalog(dim=3)
    assert {(row.velocity[0],) for row in one} >= {(1.0,), (-1.0,)}
    velocities_2d = {tuple(row.velocity) for row in two}
    assert (1.0, 0.0) in velocities_2d
    assert (0.0, 1.0) in velocities_2d
    assert (1.0, 1.0) in velocities_2d
    assert (1.0, 0.37) in velocities_2d
    velocities_3d = {tuple(row.velocity) for row in three}
    assert (1.0, 0.0, 0.0) in velocities_3d
    assert (0.0, 1.0, 0.0) in velocities_3d
    assert (0.0, 0.0, 1.0) in velocities_3d
    assert (1.0, 1.0, 1.0) in velocities_3d
    assert (1.0, 0.37, 0.61) in velocities_3d
    assert {row.layout for row in one} >= {
        "U-C",
        "U-F",
        "A-S0",
        "A-S2",
        "A-DP",
        "A-DT",
    }
    assert {row.periods for row in one} >= {1, 2, 4}
    assert {row.block_size for row in one} >= {8, 16, 32, 64}
    assert any(row.n_cells == 256 for row in one)
    assert any(row.family == "temporal" for row in one)
    assert any(row.family == "perm" for row in two)
    assert any(row.family == "perm" for row in three)


def test_exact_nd_is_a_periodic_translation():
    exact = _load("exact")
    coords, volumes = exact.uniform_cell_mesh_nd(16, 2)
    q0 = exact.exact_sine_nd(coords, 0.0, a=(1.0, 1.0), k=(1.0, 2.0))
    q1 = exact.exact_sine_nd(coords, 1.0, a=(1.0, 1.0), k=(1.0, 2.0))
    np.testing.assert_allclose(q0, q1, atol=1.0e-12)
    errors = reference_errors(q0, q1, volumes)
    assert errors.linf < 1.0e-15
    assert exact.directional_cfl((1.0, 1.0, 1.0)) == 0.15
    assert exact.directional_cfl((1.0,)) == 0.45


def test_field_diagnostics_on_exact_field():
    exact = _load("exact")
    complement = _load("complement")
    coords, volumes = exact.uniform_cell_mesh_nd(32, 1)
    variant = complement.Variant(
        dim=1, velocity=(1.0,), wave=(1.0,), n_cells=32, family="spatial"
    )
    field = exact.exact_sine_nd(coords, 0.0, a=(1.0,), k=(1.0,))
    diagnostics = complement.field_diagnostics(field, field, volumes, variant=variant)
    assert diagnostics["l1"] == 0.0
    assert diagnostics["linf"] == 0.0
    assert abs(diagnostics["mass_error"]) < 1.0e-12
    assert abs(diagnostics["amplitude_loss"]) < 0.02


def test_manufactured_orders_and_summarize():
    complement = _load("complement")
    results = []
    for n_cells, linf in ((16, 4.0e-2), (32, 1.0e-2), (64, 2.5e-3)):
        results.append(
            {
                "id": f"n{n_cells}",
                "status": "ok",
                "family": "spatial",
                "dim": 1,
                "velocity": [1.0],
                "layout": "U-C",
                "periods": 1,
                "block_size": None,
                "mesh_cells": n_cells,
                "linf": linf,
                "conservation_ok": True,
            }
        )
    spacings = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0]
    orders = observed_order([4.0e-2, 1.0e-2, 2.5e-3], spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))
    summary = complement.summarize(results)
    assert summary["acceptance_order_met"]
    assert summary["n_ok"] == 3


def test_canonical_run_py_stays_three_d_only():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "Cartesian1D" not in text
    assert "Cartesian2D" not in text
    assert "Cartesian3D" in text


def test_resolve_plan_1d_uniform_is_dimension_one():
    complement = _load("complement")
    variant = complement.Variant(
        dim=1, velocity=(1.0,), wave=(1.0,), n_cells=8, family="spatial"
    )
    plan = complement.resolve_plan(variant)
    assert getattr(plan, "resolved_dimension", None) == 1
