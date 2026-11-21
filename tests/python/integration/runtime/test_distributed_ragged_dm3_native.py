"""DM3 free placement receipt on native Serial, MPI2 and MPI3."""

import json
import struct

import pops
import pytest

from examples.api_v02.distributed_ragged_dm3 import (
    COLUMNS, ROW_COUNT, ROWS, SAMPLES, SIGNS, build_distributed_ragged_dm3,
)
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile, profile_gather_value, rank_and_size

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

# Cuts for rows, occurrences, and columns, respectively.
WRITTEN_OFFSETS = {
    1: ((0, 4), (0, 9), (0, 6)),
    2: ((0, 1, 4), (0, 1, 9), (0, 2, 6)),
    3: ((0, 1, 1, 4), (0, 1, 1, 9), (0, 2, 2, 6)),
}


def _transport(operation, *args, **kwargs):
    return operation(*args, byte_capacity=65_536, **kwargs)


def _receive(ranks, *, balanced):
    placement = "balanced" if balanced else "written"

    def place_domains(d, sizes):
        if balanced:
            return tuple(d.balanced_domain(size, capacity=size) for size in sizes)
        cuts = WRITTEN_OFFSETS[ranks]
        assert tuple(offsets[-1] for offsets in cuts) == sizes
        return tuple(d.domain(offsets, capacity=size)
                     for offsets, size in zip(cuts, sizes, strict=True))

    spec, artifact, world = _compile(
        f"DM3 ragged {placement}", builder=build_distributed_ragged_dm3,
        place_domains=place_domains, transport=_transport)
    rank, actual_ranks = rank_and_size(world)
    assert actual_ranks == ranks
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=1, console=False))
    image = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        root = _decode(spec, image, rank=rank)["root"]
        assert root["step"] == 1 and root["time"] == spec.t_end
        local_gids = root["owned_gids"]["values"]
        local_occurrences = root["occurrences"]["values"]
        local_rows = root["rows"]["values"]

    # MPI supplies the rank of each local checkpoint shard. This inventory
    # neither evaluates Distributed.rank() nor infers ownership from cuts.
    inventory = collective_call(world, lambda: profile_gather_value(world, local_gids))
    occurrences = collective_call(world, lambda: profile_gather_value(world, local_occurrences))
    rows = collective_call(world, lambda: profile_gather_value(world, local_rows))
    with collective_check(world):
        assert len(inventory) == ranks
        assert sorted(gid for shard in inventory for gid in shard) == list(range(len(SAMPLES)))
        owners = {gid: owner for owner, shard in enumerate(inventory) for gid in shard}
        if not balanced:
            for shards, cuts in zip((rows, occurrences, inventory), WRITTEN_OFFSETS[ranks],
                                    strict=True):
                for shard, start, end in zip(shards, cuts[:-1], cuts[1:], strict=True):
                    gids = [item["gid"] if isinstance(item, dict) else item for item in shard]
                    assert gids == list(range(start, end))
            if ranks == 3:
                assert not inventory[1] and not occurrences[1] and not rows[1]

        by_entry = sorted((item for shard in occurrences for item in shard),
                          key=lambda item: item["gid"])
        by_row = sorted((item for shard in rows for item in shard), key=lambda item: item["gid"])
        assert [item["gid"] for item in by_entry] == list(range(len(COLUMNS)))
        assert [item["gid"] for item in by_row] == list(range(ROW_COUNT))

        occurrence_bytes = []
        for item, column in zip(by_entry, COLUMNS, strict=True):
            assert item["column"] == column
            assert item["owner"] == owners[column], item
            assert struct.pack(">d", item["value"]) == struct.pack(">d", SAMPLES[column])
            occurrence_bytes.append(struct.pack(">qqd", item["gid"], item["column"], item["value"]))

        expected = [0.] * ROW_COUNT
        for row, column, sign in zip(ROWS, COLUMNS, SIGNS, strict=True):
            expected[row] += sign * SAMPLES[column]
        # Swapping occurrences 1 and 2 would change row 0 from +0 to 1.
        assert expected[0] == 0. and (SAMPLES[0] + SAMPLES[2]) + SAMPLES[1] == 1.
        row_bytes = []
        for item, value in zip(by_row, expected, strict=True):
            assert struct.pack(">d", item["value"]) == struct.pack(">d", value), item
            row_bytes.append(struct.pack(">qd", item["gid"], item["value"]))
    return inventory, owners, b"".join(occurrence_bytes), b"".join(row_bytes)


def test_ragged_typed_import_and_ordered_scatter_survive_free_placement(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    _, world = native_profile(dimension=2, scope="DM3 ragged")
    _, ranks = rank_and_size(world)
    assert ranks in (1, 2, 3)
    written_inventory, written_owners, written_entries, written_rows = _receive(ranks, balanced=False)
    balanced_inventory, balanced_owners, balanced_entries, balanced_rows = _receive(ranks, balanced=True)
    with collective_check(world):
        assert written_entries == balanced_entries
        assert written_rows == balanced_rows
        moved = [gid for gid in written_owners if written_owners[gid] != balanced_owners[gid]]
        if ranks > 1:
            assert moved
    record_property("dm3_column_inventory", json.dumps(
        {"written": written_inventory, "balanced": balanced_inventory, "moved_gids": sorted(moved)}))
