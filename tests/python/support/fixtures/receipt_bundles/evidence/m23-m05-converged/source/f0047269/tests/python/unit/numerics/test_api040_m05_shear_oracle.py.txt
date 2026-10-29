"""Pure M05 shear mathematics and public Diffusion source admission."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
sys.path.insert(0, str(EXAMPLES))
from api040_m05_shear_oracle import (  # noqa: E402
    energy, forward_euler_energy_defect, forward_euler_states,
    fourier_amplitudes, initial_means, integrated_initial_means,
    periodic_faces, periodic_rate, semidiscrete_work,
)


@pytest.mark.parametrize("cells", (32, 64, 128))
def test_true_cell_means_match_antiderivative_and_differ_from_center_samples(cells):
    averages = initial_means(cells)
    np.testing.assert_allclose(averages, integrated_initial_means(cells), rtol=0, atol=2e-14)
    centers = (np.arange(cells) + .5) / cells
    assert np.max(np.abs(averages - np.sin(2 * math.pi * centers))) > 1e-5


def test_semidiscrete_shear_work_is_exact_negative_face_jump_form():
    state = np.random.default_rng(785).normal(size=37)
    work, jumps = semidiscrete_work(state, .03)
    assert abs(work - jumps) < 5e-13
    assert work < 0 and jumps < 0
    assert abs(np.sum(periodic_rate(state, .03))) < 2e-12
    constant = np.full(37, 2.)
    assert semidiscrete_work(constant, .03) == (0., 0.)


@pytest.mark.parametrize("cells", (32, 64, 128))
def test_forward_euler_energy_has_explicit_temporal_correction(cells):
    before, expected = forward_euler_states(cells, .03, .1, cells)
    step = .1 / cells
    after = before + step * periodic_rate(before, .03)
    np.testing.assert_allclose(after, expected, rtol=0, atol=2e-15)
    increment, decomposition = forward_euler_energy_defect(before, .03, step)
    assert abs(increment - decomposition) < 5e-15
    work, _ = semidiscrete_work(before, .03)
    assert increment > step * work  # FE adds a nonnegative, method-only term.
    assert energy(after) < energy(before)


@pytest.mark.parametrize("cells", (32, 64, 128))
def test_continuous_spatial_and_time_amplitudes_are_distinct(cells):
    continuous, semidiscrete, forward_euler, symbol = fourier_amplitudes(
        cells, .03, .1, cells)
    assert 0 < forward_euler < semidiscrete < 1
    assert 0 < continuous < 1 and symbol < (2 * math.pi)**2
    assert abs(forward_euler - continuous) < 2e-4
    assert abs(forward_euler - semidiscrete) > 1e-5


def test_public_case_selects_true_diffusion_provider_and_accepted_faces():
    import pops
    from pops.codegen import Production
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from api040_m05_periodic_shear import build_case

    case, layout, subject = build_case(32)
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout, backend=Production())
    model = case._block_registry.spec("shear")["model"]
    plan = next(iter(resolved.resolved_operations.values()))
    code = emit_cpp_program(
        resolved.time, model=lower_and_validate(model, resolved_operations=plan)[0])
    assert resolved.resolved_dimension == 1
    assert resolved.initial_condition_plan.bindings[0].subject == validated.resolve(subject)
    assert "PreparedDiffusion<pops::kNativeDimension>" in code
    assert code.count(".stage_accepted_exchanges(") == 1
    assert "PreparedCoupledGradient" not in code


def test_oriented_ledger_reconstruction_uses_both_owned_cell_incidences():
    from api040_m05_periodic_shear import _ledger_metrics

    cells, step = 32, .1 / 32
    before = initial_means(cells)
    after = before + step * periodic_rate(before, .03)
    faces = periodic_faces(before, .03)
    rows = []
    for cell in range(cells):
        for side in (0, 1):
            orientation = -1 if side == 0 else 1
            flux = faces[(cell - 1) % cells if side == 0 else cell]
            rows.append({
                "quadrature_identity": f"cell:{cell}/axis:0/side:{side}",
                "occurrence_identity": "same-exact-flux",
                "evaluation_context": "accepted-stage",
                "orientation": orientation, "multiplicity": 1,
                "face_measure": 1., "temporal_weight": step,
                "numerical_flux": flux,
                "integrated_amount": orientation * flux * step,
            })
    metrics = _ledger_metrics(rows, before, after, step)
    assert metrics["accepted_exchange_count"] == 2 * cells
    assert metrics["ledger_flux_max_error"] == 0
    assert metrics["ledger_cell_change_error"] < 2e-17
    rows[0] = {**rows[0], "numerical_flux": rows[0]["numerical_flux"] + .01}
    assert _ledger_metrics(rows, before, after, step)["ledger_flux_max_error"] > .009


def test_negative_viscosity_is_refused_before_native_publication():
    import pops
    from pops import math as pops_math
    from pops.frames import Cartesian1D
    from pops.numerics import Diffusion

    model = pops.Model("negative_viscosity", frame=Cartesian1D())
    state = model.state("u", components=("value",))
    flux = model.diffusive_flux(
        "bad", state=state, value=-.03 * pops_math.grad(state[0]))
    with pytest.raises(ValueError, match="nonnegative"):
        Diffusion(flux=flux)
