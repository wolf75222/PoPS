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
from pops.codegen import Production
from pops.codegen.component_provider_packs import resolve_component_provider_packs
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
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
    frame = Rectangle("nonhyperbolic-roe-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
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
    frame = Rectangle("%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=tuple("q%d" % index for index in range(components)))
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: tuple(state), y_axis: tuple(state)},
    )
    return model


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
    assert boundary._dsl._m._roe_jacobian is not None

    too_large = _diagonal_roe_model("dense_roe_too_large", 17)
    with pytest.raises(DenseSpectralCapacityError) as caught:
        too_large.roe_from_jacobian(entropy_fix=riemann.Harten(1.0e-6))
    assert caught.value.components == 17
    assert caught.value.max_components == 16
    assert "HLL" in str(caught.value)
    assert "model.wave_speeds" in str(caught.value)
    assert "native Roe spectral provider" in str(caught.value)
    assert too_large._dsl._m._roe_jacobian is None


def test_flux_jacobian_roe_emits_generic_characteristic_no_inflow_provider() -> None:
    model = _diagonal_roe_model("dense_characteristic_boundary", 2)
    model.wave_speeds_from_jacobian()
    model.roe_from_jacobian()
    model._dsl.__pops_bind_component_provider_packs__(
        resolve_component_provider_packs(model.module)
    )
    source = model._dsl._m.emit_cpp_brick(name="DenseCharacteristicBoundary")
    assert "bool characteristic_no_inflow(" in source
    assert "characteristic_no_inflow_contract_version = 1" in source
    assert "characteristic_no_inflow_dimension = dimension" in source
    assert "characteristic_no_inflow_components = n_vars" in source
    assert "characteristic_no_inflow_conservative = true" in source
    assert "pops::characteristic_incoming_apply" in source
    assert "outward_sign" in source
    assert "Euler" not in source


def test_auxiliary_dependent_jacobian_does_not_advertise_characteristic_provider() -> None:
    frame = Rectangle("aux-characteristic-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = Model("aux_characteristic_boundary", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    coefficient = model.aux("coefficient")
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (coefficient * q,), y_axis: (coefficient * q,)},
    )
    model.wave_speeds_from_jacobian()
    model.roe_from_jacobian()
    model._dsl.__pops_bind_component_provider_packs__(
        resolve_component_provider_packs(model.module)
    )

    source = model._dsl._m.emit_cpp_brick(name="AuxCharacteristicBoundary")
    assert "bool characteristic_no_inflow(" not in source
