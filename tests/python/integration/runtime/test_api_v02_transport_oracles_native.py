"""PoPS native states must match complete discrete transport, including cross terms."""
import numpy as np
import pytest

from examples.api_v02.models_M01_M14 import build_m04, build_m08
from examples.api_v02.continuum_transport_oracles import anisotropic_ssprk2, guiding_center_ssprk2
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("identifier", ("M04", "M08"))
def test_native_transport_rejects_missing_cross_diffusion_or_reversed_field_velocity(
        identifier, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope=identifier+" independent transport")
    builder = build_m04 if identifier == "M04" else build_m08
    spec = collective_call(world, builder)
    runtime, artifact, metrics = execute_case(lambda: spec, world, identifier+" transport equations")
    values = collective_call(world, lambda: runtime.state_global("fluid"))
    with collective_check(world):
        if values is not None:
            initial = next(iter(spec.initial_values.values()))
            actual = np.asarray(values).reshape(initial.shape)[0]
            if identifier == "M04":
                wrong = anisotropic_ssprk2(initial[0], velocity=(.4, -.2),
                    tensor=((.02, 0.), (0., .015)), lengths=(1., 1.), dt=spec.dt, steps=2)
            else:
                wrong, _ = guiding_center_ssprk2(initial[0], lengths=(2*np.pi, 2*np.pi),
                    dt=spec.dt, steps=2, velocity_sign=-1)
            assert np.max(abs(actual-wrong)) > 1e-6
            assert np.max(abs(actual-initial[0])) > 1e-6
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_metrics", str(metrics))
