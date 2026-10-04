"""CP12 replicated ownership is empty; partitioned ownership is exact."""

import numpy as np
import pytest

from tests.review.sol61_m19_amr_consumed_offline import _carrier_owner_map


def child(mode, owners, dtype="int64"):
    return {"distribution_mode_0": np.asarray(mode), "dmap_0": np.asarray(owners, dtype=dtype)}


@pytest.mark.parametrize("patches, ranks", ((1, 1), (3, 2), (5, 4)))
def test_replicated_level_has_no_owner_map(patches, ranks):
    assert _carrier_owner_map(child("replicated", []), 0, patches, ranks) == (-1,) * patches


@pytest.mark.parametrize("owners, ranks", (((0,), 1), ((1, 0, 1), 2), ((3, 0, 2, 1), 4)))
def test_partitioned_level_preserves_patch_order(owners, ranks):
    assert _carrier_owner_map(child("partitioned", owners), 0, len(owners), ranks) == owners


@pytest.mark.parametrize(
    "payload, patches, ranks",
    (
        (child("replicated", [0]), 1, 1),
        (child("partitioned", []), 1, 1),
        (child("partitioned", [0]), 2, 1),
        (child("partitioned", [0, 0]), 1, 1),
        (child("partitioned", [-1]), 1, 1),
        (child("partitioned", [2]), 1, 2),
        (child("partitioned", [0], "float64"), 1, 1),
        (child("partitioned", [True], "bool"), 1, 1),
        (child("partitioned", [[0]]), 1, 1),
        (child(["replicated"], []), 1, 1),
        (child("unrecorded", []), 1, 1),
        (child("replicated", []), 0, 1),
        (child("replicated", []), 1, True),
    ),
)
def test_distribution_counterfeits_remain_refused(payload, patches, ranks):
    with pytest.raises(ValueError, match="carrier distribution authority"):
        _carrier_owner_map(payload, 0, patches, ranks)
