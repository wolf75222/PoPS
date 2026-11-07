"""All eight evolved-original fixture cases, one declared partition per MPI2 world."""
import pytest

from tests.python.support.public_uniform_profile_mpi import (
    EVOLVED_ORIGINAL_PARTITIONS,
    run_evolved_original_world,
)


pytestmark = pytest.mark.mpi


@pytest.mark.parametrize("partition", tuple(EVOLVED_ORIGINAL_PARTITIONS))
def test_public_evolved_original_stage_on_authenticated_mpi2_world(tmp_path, partition):
    run_evolved_original_world(ranks=2, directory=tmp_path / (partition + "-mpi2"),
                               partition=partition)
