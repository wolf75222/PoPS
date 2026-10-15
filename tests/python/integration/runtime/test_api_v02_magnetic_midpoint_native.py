"""Complete M09 accepted fields received from the installed PoPS C++ artifact."""
import math
import numpy as np
import pytest

from examples.api_v02.models_magnetic_midpoint import build_m09_midpoint, COMPONENTS
from tests.python.support import magnetic_midpoint_oracle as reference
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("solver,source_only,omega", (
    ("schur", False, 4.), ("monolithic", False, 4.), ("schur", True, 1.e5)))
def test_native_complete_magnetic_midpoint(solver, source_only, omega,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, lengths, dt, steps = (3, 3), (2*math.pi, 3*math.pi), .003, 2
    _, world = native_profile(dimension=2, scope="complete magnetic midpoint "+solver)
    builder = lambda: build_m09_midpoint(cells=cells, lengths=lengths, dt=dt,
        steps=steps, solver=solver, source_only=source_only, omega=omega)
    spec = collective_call(world, builder)
    runtime, artifact, _ = execute_case(lambda: spec, world, "magnetic midpoint "+solver)
    image = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        packet = _decode(spec, image)["root"]
        rows = packet["state"]["values"]
        actual = np.array([[row[key] for row in rows] for key in COMPONENTS])
        actual = actual.reshape(6, cells[1], cells[0])
        old = reference.initial(cells, lengths=lengths, omega=omega)
        expected, residuals = reference.evolve(old, dt, steps, lengths=lengths,
                                              source_only=source_only)
        error = float(np.max(abs(actual-expected)))
        assert error < 5.e-10, (solver, source_only, error)
        assert max(residuals) < 1.e-12
        assert packet["source_residual"] < 256.e-24
        energy_error = abs(reference.energy(actual, lengths)-reference.energy(old, lengths))
        assert energy_error < 1.e-9
        gauss = float(np.max(abs(reference.gauss(actual, lengths))))
        assert packet["step"] == steps and packet["time"] == steps*dt
        assert packet["source_iterations"] > 0
        assert np.max(abs(actual[1:3]-old[1:3])) > 1.e-6
        if not source_only:
            assert np.max(abs(actual[0]-old[0])) > 1.e-5
            assert gauss > 1.e-6
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_complete_image_bytes", len(image))
    record_property("original_monolithic_reference_error", error)
    record_property("total_energy_error", energy_error)
    record_property("gauss_splitting_defect", gauss)
