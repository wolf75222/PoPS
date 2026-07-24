"""ADC-694 canonical HyQMOM15 model, closure and final authoring contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from pops.lib.models.moments import HyQMOM15
from pops.moments import (
    HyQMOM15Closure,
    LocalClosure,
    RealizabilityProjection,
    closure,
    moment_flux_expressions,
    moment_names,
)
from pops.domain import RectangleFrame
from pops.frames import Cartesian2D
from pops.physics import Model
from pops.time import ProjectAndRecheck, RejectAttempt


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"


def test_moment_flux_generator_is_public_for_explicit_python_models() -> None:
    assert callable(moment_flux_expressions)


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_final_hyqmom15", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hyqmom15_is_a_real_model_with_exact_generic_handles() -> None:
    model = HyQMOM15.vlasov_lorentz(exact_speeds=False)
    assert type(model) is Model
    state = model.states["U"]
    flux = model.fluxes["transport"]
    explicit_rate = model.operators["transport"]
    implicit_source = model.operators["magnetic_rotation"]
    electric_source = model.sources["electric"]
    assert tuple(state.components) == tuple(moment_names(4))
    assert len(state.components) == 15
    assert model.rate_contract(explicit_rate) == {
        "state": state, "flux": flux, "sources": (electric_source,)}
    assert implicit_source.kind == "local_linear_operator"
    assert set(model.module.operator_registry().names()) >= {
        explicit_rate.registered_operator_name,
        electric_source.reg_name,
        implicit_source.registered_operator_name,
    }


def test_local_closure_is_model_agnostic_and_order_checked() -> None:
    @closure(4)
    def zero_fifth_order(_standardized):
        return {"S%d%d" % (p, 5 - p): 0.0 for p in range(6)}

    assert isinstance(zero_fifth_order, LocalClosure)
    model = HyQMOM15.vlasov_lorentz(
        closure=zero_fifth_order, exact_speeds=False)
    assert type(model) is Model
    assert tuple(model.states["U"].components) == tuple(moment_names(4))

    @closure(2)
    def wrong_order(_standardized):
        return {"S30": 0.0, "S21": 0.0, "S12": 0.0, "S03": 0.0}

    with pytest.raises(ValueError, match="declares order 2"):
        HyQMOM15.vlasov_lorentz(
            closure=wrong_order, exact_speeds=False)


def test_hyqmom15_closure_matches_closure_s5_matlab_oracle() -> None:
    """Pin the six non-Gaussian polynomial relations used by closureS5.m."""

    standardized = {
        "S03": -0.2,
        "S04": 2.8,
        "S11": 0.15,
        "S12": -0.35,
        "S13": 0.42,
        "S20": 1.0,
        "S21": 0.25,
        "S22": 1.2,
        "S30": 0.3,
        "S31": -0.1,
        "S40": 3.1,
        "S02": 1.0,
    }

    closed = HyQMOM15Closure()(standardized)

    assert closed == pytest.approx({
        "S50": 2.1345,
        "S41": 1.1932375,
        "S32": -0.64775,
        "S23": 0.425,
        "S14": -1.9052,
        "S05": -1.288,
    })
    assert any(value != 0.0 for value in closed.values())


def test_board_model_exposes_generic_native_hooks_without_a_private_preset_seam() -> None:
    frame = Cartesian2D()
    model = Model("generic_native_hooks", frame=frame)
    state = model.state("U", components=("q",))
    cached = model.module
    model.projection((state[0],))
    projected = model.module
    assert projected is not cached
    assert projected.operator_registry().get("projection").kind == "projection"
    values = np.array([[[2.0, 3.0]]])
    assert np.array_equal(model.projection_value(values), values)

    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={frame.x: (state[0],), frame.y: (state[0],)},
    )
    cached = model.module
    model.wave_speeds_from_jacobian()
    assert model.module is not cached
    cached = model.module
    model.roe_from_jacobian()
    assert model.module is not cached
    with pytest.raises(ValueError, match="eig 'numeric' \\| 'fd'"):
        Model("invalid_hook", frame=frame).wave_speeds_from_jacobian(eig="invalid")

    preset_source = (ROOT / "python/pops/lib/models/moments/hyqmom15.py").read_text(
        encoding="utf-8"
    )
    assert "._dsl" not in preset_source


def test_final_authoring_derives_field_storage_and_complete_generic_program() -> None:
    target = _load_example().build_authoring()

    assert type(target.model) is Model
    assert isinstance(target.model.frame, RectangleFrame)
    assert target.components == tuple(moment_names(4))
    assert target.model.field_spaces()[target.field.local_id].components == (
        "phi", "grad_x", "grad_y")
    assert target.field_provider == target.model.operators["fields"]
    assert target.program.transaction_plan() is not None
    guards = target.program.transaction_plan().guards
    assert [guard.name for guard in guards] == [
        "hyqmom15_realizability_density",
        "hyqmom15_realizability_moment_matrix",
    ]
    assert all(type(guard.action) is ProjectAndRecheck for guard in guards)
    assert all(type(guard.action.on_failure) is RejectAttempt for guard in guards)
    assert len(target.case._consumers.inspect()["nodes"]) == 3
    local_map = target.implicit_operator.signature.output
    assert len(local_map.domain.components) == 15
    assert local_map.domain == local_map.range
    projection = target.model.module.operator_registry().get("projection")
    assert projection.kind == "projection"


def test_hyqmom15_projection_checks_all_moments_and_refuses_to_manufacture_density() -> None:
    example = _load_example()
    target = example.build_authoring()
    projection = target.realizability
    assert type(projection) is RealizabilityProjection
    state = example.build_initial_state(cells=4)["plasma"]
    assert projection.is_hyqmom15_realizable(state)
    assert np.array_equal(projection.project_hyqmom15_array(state), state)

    component = {name: index for index, name in enumerate(target.components)}
    invalid_fourth_moment = state.copy()
    invalid_fourth_moment[component["M40"]] = -1.0
    assert not projection.is_hyqmom15_realizable(invalid_fourth_moment)
    repaired = projection.project_hyqmom15_array(invalid_fourth_moment)
    assert projection.is_hyqmom15_realizable(repaired)
    assert not np.array_equal(repaired, invalid_fourth_moment)
    assert np.array_equal(projection.project_hyqmom15_array(repaired), repaired)
    emitted_projection = target.model.projection_value(invalid_fourth_moment)
    assert projection.is_hyqmom15_realizable(emitted_projection)
    assert np.allclose(emitted_projection, repaired, rtol=1.0e-13, atol=1.0e-13)

    invalid_covariance = state.copy()
    invalid_covariance[component["M20"]] = 0.0
    assert not projection.is_hyqmom15_realizable(invalid_covariance)
    repaired_covariance = projection.project_hyqmom15_array(invalid_covariance)
    assert projection.is_hyqmom15_realizable(repaired_covariance)
    emitted_covariance = target.model.projection_value(invalid_covariance)
    assert projection.is_hyqmom15_realizable(emitted_covariance)
    assert np.allclose(emitted_covariance, repaired_covariance, rtol=1.0e-13, atol=1.0e-13)

    negative_density = -state
    assert not projection.is_hyqmom15_realizable(negative_density)
    assert np.array_equal(
        projection.project_hyqmom15_array(negative_density), negative_density)


def test_final_example_uses_only_the_root_lifecycle_and_public_layout_home() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "from pops.layouts import Uniform" in source
    assert "pops.mesh.layouts" not in source
    assert "BindInputs" not in source
    assert "simulation.run(" not in source
    assert "from pops.runtime import System" not in source
    assert "from pops.runtime import AmrSystem" not in source
    for call in (
        "pops.validate(", "pops.resolve(", "pops.compile(", "pops.bind(", "pops.run(",
    ):
        assert call in source


def test_magnetic_wave_selects_the_matlab_complex_spectrum_order() -> None:
    source = (ROOT / "docs/tuto/hyqmom/05_openmp_magnetic_wave_hll.py").read_text(
        encoding="utf-8")
    assert "np.lexsort((np.angle(eigenvalues), np.abs(eigenvalues)))" in source
    assert "np.argsort(eigenvalues.real)" not in source
