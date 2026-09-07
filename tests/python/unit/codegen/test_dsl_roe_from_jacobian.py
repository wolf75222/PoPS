"""Compiled generic Roe regression through the final public lifecycle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops._dense_spectral import (
    DenseSpectralCapacityError,
    is_exact_block_triangular,
)
from pops._ir.expr import Const
from pops._ir.lowering import diff
from pops.codegen import Production
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.module_emit_riemann import has_characteristic_no_inflow_provider
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.riemann.providers import (
    ROE_FLUX_JACOBIAN,
    authoring_provider_evidence,
)
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context


ROOT = Path(__file__).resolve().parents[4]
N = 24
DT = 5.0e-4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def test_exact_block_triangular_certificate_is_structural_and_complete() -> None:
    lower = [
        [Const(1.0), Const(0.0), Const(0.0)],
        [Const(2.0), Const(3.0), Const(4.0)],
        [Const(5.0), Const(6.0), Const(7.0)],
    ]
    assert is_exact_block_triangular(lower, [[0], [1, 2]])
    assert not is_exact_block_triangular(lower, [[0], [1]])

    coupled = [
        [Const(1.0), Const(2.0)],
        [Const(3.0), Const(4.0)],
    ]
    assert not is_exact_block_triangular(coupled, [[0], [1]])


def _nonhyperbolic_roe_model() -> Model:
    frame = Rectangle(
        "nonhyperbolic-roe-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("nonhyperbolic_dense_roe", frame=frame)
    state = model.state("U", components=("q1", "q2"))
    q1, q2 = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (-q2, q1), y_axis: (-q2, q1)},
    )
    # Roe supplies the interface dissipation, while the time-step authority still needs the
    # exact signed spectrum of the same flux Jacobian.  Register both explicitly: neither
    # provider is allowed to stand in for the other or fall back to a scalar radius.
    model.wave_speeds_from_jacobian()
    model.roe_from_jacobian(entropy_fix=riemann.Harten(1.0e-6))
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def _diagonal_roe_model(name: str, components: int) -> Model:
    frame = Rectangle(
        "%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state(
        "U", components=tuple("q%d" % index for index in range(components)))
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: tuple(state), y_axis: tuple(state)},
    )
    return model


def _emit_cpp_brick(model: Model, *, name: str) -> str:
    """Resolve the public Model through its canonical Module/provider-pack route."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    return emit_model._m.emit_cpp_brick(name=name)


def _lower_model(model: Model):
    """Resolve the public Model without requiring an independently emitted brick."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    return emit_model


def test_qualified_flux_jacobian_retains_nonzero_entries_through_freeze_and_lowering() -> None:
    model = _nonhyperbolic_roe_model()
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])

    def assert_jacobians(carrier) -> None:
        for axis in ("x", "y"):
            for matrix in (
                carrier.flux_jacobian(axis),
                carrier._roe_jacobian[axis],
                carrier._ws_jacobian["rows"][axis],
            ):
                np.testing.assert_array_equal(
                    [[entry.eval({}) for entry in row] for row in matrix], expected)

    assert_jacobians(model._dsl._m)
    assert_jacobians(_lower_model(model)._m)
    model.freeze()
    assert_jacobians(model._dsl._m)
    assert_jacobians(_lower_model(model)._m)


def test_qualified_derivative_distinguishes_homonymous_owners_and_legacy_variables() -> None:
    from pops.physics._model import HyperbolicModel

    first = _diagonal_roe_model("qualified_derivative_first", 1)
    second = _diagonal_roe_model("qualified_derivative_second", 1)
    (own_q,) = first.states["U"]
    (foreign_q,) = second.states["U"]
    expression = own_q * foreign_q
    environment = {own_q.qualified_id: 2.0, foreign_q.qualified_id: 7.0}
    assert diff(expression, own_q).eval(environment) == 7.0
    assert diff(expression, foreign_q).eval(environment) == 2.0
    assert diff(own_q, foreign_q).eval({}) == 0
    assert diff(own_q, "q0").eval({}) == 0

    legacy = HyperbolicModel("legacy_derivative")
    (q,) = legacy.conservative_vars("q")
    legacy.set_flux(x=[q * q], y=[3 * q])
    assert legacy.flux_jacobian("x")[0][0].eval({"q": 4.0}) == 8.0
    assert legacy.flux_jacobian("y")[0][0].eval({}) == 3.0


def test_dense_roe_complex_spectrum_fails_without_rusanov_fallback(
    isolated_native_cache, native_cxx, kokkos_root
) -> None:
    del isolated_native_cache, kokkos_root
    model = _nonhyperbolic_roe_model()
    state = model.states["U"]
    flux = model.fluxes["transport"]
    rate = model.operators["transport"]
    case = pops.Case("nonhyperbolic_dense_roe_case")
    block = case.block("toy", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Roe(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(N, N),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    initial = np.ones((2, N, N), dtype=np.float64)
    initial[1] = 0.25
    simulation = pops.bind(
        artifact,
        initial_state={"toy": initial},
        resources={"execution_context": artifact_execution_context(artifact)},
    )

    with pytest.raises(
        RuntimeError,
        match=r"solve status=invalid_evaluation.*numerical flux evaluation reject",
    ):
        pops.run(simulation, t_end=DT, max_steps=1)
    np.testing.assert_array_equal(
        np.asarray(simulation.get_state("toy"), dtype=np.float64).reshape(initial.shape),
        initial,
    )


def test_roe_dense_spectral_capacity_fails_during_authoring() -> None:
    boundary = _diagonal_roe_model("dense_roe_boundary", 16)
    boundary.roe_from_jacobian(entropy_fix=riemann.Harten(1.0e-6))
    assert authoring_provider_evidence(_lower_model(boundary)).roe_provider == ROE_FLUX_JACOBIAN

    too_large = _diagonal_roe_model("dense_roe_too_large", 17)
    with pytest.raises(DenseSpectralCapacityError) as caught:
        too_large.roe_from_jacobian(entropy_fix=riemann.Harten(1.0e-6))
    assert caught.value.components == 17
    assert caught.value.max_components == 16
    assert "HLL" in str(caught.value)
    assert "model.wave_speeds" in str(caught.value)
    assert "native Roe spectral provider" in str(caught.value)
    assert authoring_provider_evidence(_lower_model(too_large)).roe_provider != ROE_FLUX_JACOBIAN


def test_flux_jacobian_roe_emits_generic_characteristic_no_inflow_provider() -> None:
    model = _diagonal_roe_model("dense_characteristic_boundary", 2)
    model.wave_speeds_from_jacobian()
    model.roe_from_jacobian()
    source = _emit_cpp_brick(model, name="DenseCharacteristicBoundary")
    assert "bool characteristic_no_inflow(" in source
    assert "pops::characteristic_incoming_apply" in source
    assert "outward_sign" in source
    assert "Euler" not in source


@pytest.mark.parametrize("through_primitive", [False, True])
def test_auxiliary_dependent_jacobian_does_not_advertise_characteristic_provider(
    through_primitive: bool,
) -> None:
    frame = Rectangle(
        "aux-characteristic-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("aux_characteristic_boundary", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    coefficient = model.aux("coefficient")
    physical_flux = coefficient * q
    if through_primitive:
        physical_flux = model.primitive("scaled_q", physical_flux)
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (physical_flux,), y_axis: (physical_flux,)},
    )
    model.wave_speeds_from_jacobian()
    model.roe_from_jacobian()

    assert not has_characteristic_no_inflow_provider(model._dsl._m)
    for axis in ("x", "y"):
        jacobian = model._dsl._m._roe_jacobian[axis]
        assert jacobian[0][0].eval({coefficient.name: 2.5}) == 2.5
    source = _emit_cpp_brick(model, name="AuxCharacteristicBoundary")
    assert "bool characteristic_no_inflow(" not in source


def test_state_only_jacobian_remains_characteristic_with_additive_auxiliary_flux() -> None:
    frame = Rectangle(
        "additive-aux-characteristic-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = Model("additive_aux_characteristic_boundary", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    coefficient = model.aux("coefficient")
    model.flux(
        "transport", frame=frame, state=state,
        components={axis: (q + coefficient,) for axis in frame.axes},
    )
    model.wave_speeds_from_jacobian()
    model.roe_from_jacobian()
    assert has_characteristic_no_inflow_provider(model._dsl._m)
    source = _emit_cpp_brick(model, name="AdditiveAuxCharacteristicBoundary")
    assert "bool characteristic_no_inflow(" in source
