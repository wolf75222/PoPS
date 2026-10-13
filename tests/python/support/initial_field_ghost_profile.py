"""Current initial Field/Ghost profile; historical ABI8 receipts remain historical."""
from tests.python.support.public_native_profile import native_profile as _native_profile, rank_and_size


def native_profile():
    return _native_profile(scope="initial Field/Ghost",
                           expected_rank_env="POPS_TEST_INITIAL_GHOST_EXPECT_RANKS")


__all__ = ["native_profile", "rank_and_size"]
