"""Native full-f 3x2v, geometric drifts and original Ampere compatibility."""

import numpy as np
import pops
import pytest

from examples.api_v02.model_gyrokinetic_nd import build_gyrokinetic_nd
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.gyrokinetic_nd_oracle import gyrokinetic_reference
from tests.python.support.public_native_profile import native_profile, profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("nonlinear", (False, True))
def test_native_full_f_3x2v(isolated_native_cache, record_property, nonlinear):
    del isolated_native_cache
    _native, world = native_profile(dimension=2, scope="API v0.2 M21 full-f 3x2v")
    ranks, rank = (1, 0) if world is None else (world.size, world.rank)
    options = dict(nonlinear_polarization=nonlinear, collision_frequency=0.02)
    spec = collective_call(world, lambda: build_gyrokinetic_nd(**options, ranks=ranks))
    runtime, artifact, metrics = execute_case(lambda: spec, world, "M21 full-f 3x2v")
    image = collective_call(world, lambda: _image(runtime))
    decoded = collective_call(world, lambda: _decode(spec, image, rank=rank))
    shards = profile_gather_value(world, decoded["root"])
    _, expected, records = gyrokinetic_reference(**options)
    with collective_check(world):
        for species in range(2):
            actual = np.concatenate(
                [np.asarray(shard["fields"]["g" + str(species)]["values"]) for shard in shards]
            ).reshape(expected[species].shape)
            np.testing.assert_allclose(actual, expected[species], atol=3e-11, rtol=3e-10)
        assert all(shard["phase"]["size"] == 27 * 12 for shard in shards)
        assert runtime.macro_step() == 2
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_package", pops.__file__)
    record_property("nonlinear_polarization_with_H2", nonlinear)
    record_property(
        "oracle_original_Ampere_derivative_residual",
        max(row["derivative_residual"] for row in records),
    )
