"""Actual distributed C++ moving-volume updates in two and three dimensions."""
import numpy as np
import pops
import pytest

from examples.api_v02.model_ale_nd import build_ale_nd
from tests.python.support.ale_nd_oracle import ale_reference
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile, profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("shape,amplitudes,velocities", [
    ((8, 6), (.025, .018), (.7, -.3)),
    ((5, 4, 3), (.02, .015, -.01), (.3, -.2, .1)),
])
def test_native_moving_volumes_match_independent_quadrature(
    isolated_native_cache, record_property, shape, amplitudes, velocities
):
    del isolated_native_cache
    _, world = native_profile(dimension=2, scope="API v0.2 M28 moving tensor volumes")
    options = dict(shape=shape, amplitudes=amplitudes, velocities=velocities)
    spec = collective_call(world, lambda: build_ale_nd(**options, ranks=1 if world is None else world.size))
    runtime, artifact, _ = execute_case(lambda: spec, world, "M28 moving tensor volumes")
    image = collective_call(world, lambda: _image(runtime))
    decoded = collective_call(world, lambda: _decode(spec, image, rank=0 if world is None else world.rank))
    shards = profile_gather_value(world, decoded["root"])
    expected, records = ale_reference(**options)
    with collective_check(world):
        for name in ("density", "uniform"):
            actual = np.concatenate([np.asarray(s[name]["values"]) for s in shards]).reshape(shape)
            np.testing.assert_allclose(actual, expected[name], atol=3e-13, rtol=2e-13)
        cells = [cell for s in shards for cell in s["geometry"]["values"]]
        actual_volume = np.asarray([cell["volume"] for cell in cells]).reshape(shape)
        np.testing.assert_allclose(actual_volume, expected["geometry"]["volume"], atol=1e-16, rtol=1e-14)
        for axis in range(len(shape)):
            for side in ("lower", "upper"):
                actual = np.asarray([cell[side+str(axis)] for cell in cells]).reshape(shape)
                np.testing.assert_allclose(actual, expected["geometry"][side][axis], atol=1e-15, rtol=1e-14)
        assert all(row["gcl"] < 2e-16 and row["mass"] < 1e-13 for row in records)
        assert all(shard["step"] == 4 for shard in shards)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_package", pops.__file__)
    record_property("physical_dimension", str(len(shape)))
