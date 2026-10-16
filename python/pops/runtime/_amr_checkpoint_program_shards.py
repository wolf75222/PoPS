"""Archive rank-local accepted Program images without promoting rank zero to authority."""

from collections.abc import Mapping, Sequence
from typing import Any

_STATE = "program_accepted_state"
_COUNT = _STATE + "_rank_count"
_PREFIX = _STATE + "_rank_"
_AUTHORITY = _STATE + "_source_authority"


def _bytes(payload: Mapping[str, Any], key: str) -> bytes:
    import numpy as np

    if key not in payload:
        raise ValueError("AMR accepted Program archive lacks " + key)
    value = np.asarray(payload[key])
    if value.dtype != np.dtype("uint8") or value.ndim != 1:
        raise ValueError("AMR accepted Program archive requires a uint8 vector: " + key)
    return value.tobytes()


def capture_program_shards(payload: dict[str, Any], states: Sequence[bytes]) -> None:
    """Called after native all-rank common-metadata and local-image authentication."""
    import numpy as np

    if not states or any(type(state) is not bytes for state in states):
        raise TypeError("AMR accepted Program shards must be a nonempty sequence of bytes")
    if any(bool(state) != bool(states[0]) for state in states):
        raise ValueError("AMR accepted Program presence differs across ranks")
    payload[_COUNT] = np.asarray(len(states), dtype=np.int64)
    for rank, state in enumerate(states):
        payload[_PREFIX + str(rank)] = np.frombuffer(state, dtype=np.uint8).copy()
    # Kept for the historical rank-independent archive contract, never selected
    # implicitly by a forest reader.
    payload[_STATE] = np.frombuffer(states[0], dtype=np.uint8).copy()


def select_program_shard(payload: Mapping[str, Any], *, source_ranks: int,
                         target_ranks: int, target_rank: int,
                         allow_native_migration: bool = False) -> dict[str, Any]:
    """Validate the entire sealed table before selecting an exact native rank image."""
    import numpy as np

    if type(allow_native_migration) is not bool:
        raise TypeError("AMR native migration capability must be an exact bool")
    for value in (source_ranks, target_ranks):
        if type(value) is not int or value < 1:
            raise ValueError("AMR accepted Program rank count must be a positive integer")
    if type(target_rank) is not int or not 0 <= target_rank < target_ranks:
        raise ValueError("AMR accepted Program target rank is outside its communicator")
    canonical = _bytes(payload, _STATE)
    keys = {key for key in payload if key.startswith(_PREFIX) and key != _COUNT}
    authority = _bytes(payload, _AUTHORITY) if _AUTHORITY in payload else b""
    forest = canonical.startswith(b"POPSANDF")
    if _COUNT not in payload:
        if keys or forest or authority.startswith(b"POPSRSA1"):
            raise ValueError("AMR accepted Program forest requires its complete rank table")
        return dict(payload)
    count = np.asarray(payload[_COUNT])
    if count.dtype != np.dtype("int64") or count.ndim != 0 or int(count) != source_ranks:
        raise ValueError("AMR accepted Program rank table differs from the recorded communicator")
    if len(keys) != source_ranks:
        raise ValueError("AMR accepted Program rank table is incomplete or has surplus shards")
    expected = {_PREFIX + str(rank) for rank in range(source_ranks)}
    if keys != expected:
        raise ValueError("AMR accepted Program rank table has invalid rank identities")
    states = tuple(_bytes(payload, _PREFIX + str(rank)) for rank in range(source_ranks))
    if states[0] != canonical:
        raise ValueError("AMR accepted Program canonical image differs from archived rank zero")
    if any(bool(state) != bool(canonical) for state in states):
        raise ValueError("AMR accepted Program presence differs across archived ranks")
    if any(state.startswith(b"POPSANDF") != forest for state in states):
        raise ValueError("AMR accepted Program forest framing differs across archived ranks")
    if forest != authority.startswith(b"POPSRSA1"):
        raise ValueError("AMR accepted Program forest lacks its ranked native source authority")
    if source_ranks != target_ranks and (forest or any(state != canonical for state in states)):
        if not forest or not allow_native_migration:
            raise ValueError("AMR rank-local accepted Program restart requires native ownership rematerialization")
    # When cardinality changes this representative is used only for common topology preflight.
    # Execution must consume the complete table through native RSA1 rematerialization below.
    selected = dict(payload)
    selected[_STATE] = np.frombuffer(
        states[target_rank] if source_ranks == target_ranks else canonical, dtype=np.uint8
    ).copy()
    return selected
