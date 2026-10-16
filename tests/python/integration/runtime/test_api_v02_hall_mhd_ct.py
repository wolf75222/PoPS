"""Real native C++ execution of every CT Hall-MHD field, including MPI shards."""

import numpy as np
import pops
import pytest

from examples.api_v02.model_hall_mhd_ct import NAMES, build_hall_mhd_ct
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.hall_mhd_ct_oracle import hall_mhd_reference
from tests.python.support.public_native_profile import native_profile, profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def test_native_hall_ohm_ct_two_dimensional(isolated_native_cache, record_property):
    del isolated_native_cache
    _native, world = native_profile(dimension=2, scope="API v0.2 M23 Hall CT")
    ranks, rank = (1, 0) if world is None else (world.size, world.rank)
    spec = collective_call(world, lambda: build_hall_mhd_ct(ranks=ranks))
    runtime, artifact, metrics = execute_case(lambda: spec, world, "M23 Hall CT 2D")
    image = collective_call(world, lambda: _image(runtime))
    decoded = collective_call(world, lambda: _decode(spec, image, rank=rank))
    shards = profile_gather_value(world, decoded["root"])
    _, expected, records = hall_mhd_reference()
    with collective_check(world):
        actual = np.stack(
            [
                np.concatenate([np.asarray(shard["fields"][name]["values"]) for shard in shards])
                for name in NAMES
            ]
        ).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
        divergence = 6 * (np.roll(actual[5], -1, 0) - actual[5]) + 8 * (
            np.roll(actual[6], -1, 1) - actual[6]
        )
        assert abs(divergence).max() < 2e-13
        assert runtime.macro_step() == 3
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("native_package", pops.__file__)
    record_property(
        "oracle_Hall_work",
        max(abs(stage["hall_work"]) for row in records for stage in row["stages"]),
    )
    record_property("oracle_Ohmic_heat", records[0]["stages"][0]["ohmic_heat"])
