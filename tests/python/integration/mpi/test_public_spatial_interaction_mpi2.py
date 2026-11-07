"""All eight spatial-interaction native cases, one declared partition per MPI2 world."""
import pytest

from tests.python.support.public_field_profile_mpi import (
    SPATIAL_INTERACTION_PARTITIONS,
    run_public_field_profile_world,
)


pytestmark = pytest.mark.mpi


@pytest.mark.parametrize("partition", tuple(SPATIAL_INTERACTION_PARTITIONS))
def test_public_spatial_interaction_on_authenticated_mpi2_world(tmp_path, partition):
    run_public_field_profile_world(group="spatial_interaction", ranks=2,
                                   directory=tmp_path / (partition + "-mpi2"), partition=partition)
