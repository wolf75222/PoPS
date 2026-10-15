"""Execute the complete selected 3x3v equations in the installed native PoPS."""

import numpy as np
import pops
import pytest

from examples.api_v02.models_kinetic_nd import build_kinetic_nd
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.kinetic_nd_oracle import kinetic_reference
from tests.python.support.public_native_profile import native_profile, profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize(
    "model,masses,charges", [("M19", (1.0,), (-1.0,)), ("M20", (1.0, 1.5, 2.0), (-1.0, 1.0, 1.0))]
)
def test_native_distributed_3x3v_equations(
    isolated_native_cache, record_property, model, masses, charges
):
    del isolated_native_cache
    _native, world = native_profile(dimension=2, scope="API v0.2 " + model + " full tensor 3x3v")
    ranks = 1 if world is None else world.size
    shape, velocity = (3, 3, 3), (4, 4, 4)
    options = dict(
        model=model,
        configuration_shape=shape,
        velocity_shape=velocity,
        masses=masses,
        charges=charges,
        dt=1e-5,
        steps=2,
        frequency=0.2,
    )
    spec = collective_call(world, lambda: build_kinetic_nd(**options, ranks=ranks))
    runtime, artifact, metrics = execute_case(lambda: spec, world, model + " distributed 3x3v")
    image = collective_call(world, lambda: _image(runtime))
    rank = 0 if world is None else world.rank
    decoded = collective_call(world, lambda: _decode(spec, image, rank=rank))
    shards = profile_gather_value(world, decoded["root"])
    _, expected, _ = kinetic_reference(**options)
    with collective_check(world):
        for species in range(len(masses)):
            actual = np.concatenate(
                [np.asarray(shard["fields"]["f" + str(species)]["values"]) for shard in shards]
            ).reshape(shape + velocity)
            np.testing.assert_allclose(actual, expected[species], atol=2e-11, rtol=2e-10)
        assert all(shard["phase"]["size"] == 27 * 64 for shard in shards)
        assert all(shard["configuration"]["size"] == 27 for shard in shards)
        assert runtime.macro_step() == 2
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_package", pops.__file__)
    record_property("physical_dimension", "3x3v independent of Dim2 runtime context")
    record_property("phase_partition", str(shards[0]["phase"]["offsets"]))
