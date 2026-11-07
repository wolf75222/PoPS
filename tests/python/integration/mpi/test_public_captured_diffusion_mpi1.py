"""All four captured/candidate native diffusion cases, one declared partition per MPI1 world."""
import pytest

from tests.python.support.public_uniform_profile_mpi import (
    CAPTURED_DIFFUSION_PARTITIONS,
    run_captured_diffusion_world,
)


pytestmark = pytest.mark.mpi


@pytest.mark.parametrize("partition", tuple(CAPTURED_DIFFUSION_PARTITIONS))
def test_public_captured_and_candidate_diffusion_on_authenticated_mpi1_world(tmp_path, partition):
    run_captured_diffusion_world(ranks=1, directory=tmp_path / (partition + "-mpi1"),
                                 partition=partition)
