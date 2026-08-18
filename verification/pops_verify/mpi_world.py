"""Live native MPI communicator identity. Never trust launcher env."""

from __future__ import annotations

from typing import Any


class NativeWorldError(RuntimeError):
    """Raised when the authenticated native communicator is missing or too small."""


def _selected_module() -> Any:
    from pops._native_selector import selected_native_module

    return selected_native_module(required=False)


def native_has_mpi() -> bool:
    try:
        module = _selected_module()
    except Exception:
        return False
    if module is None:
        return False
    return bool(getattr(module, "__has_mpi__", False))


def native_world_size(*, required: bool = False) -> int | None:
    try:
        module = _selected_module()
        if module is None:
            raise NativeWorldError("native communicator unavailable")
        n_ranks = getattr(module, "n_ranks", None)
        if not callable(n_ranks):
            raise NativeWorldError("native n_ranks unavailable")
        return int(n_ranks())
    except NativeWorldError:
        if required:
            raise
        return None
    except Exception as exc:
        if required:
            raise NativeWorldError(f"native communicator unavailable: {exc}") from exc
        return None


def native_world_rank(*, required: bool = False) -> int | None:
    try:
        module = _selected_module()
        if module is None:
            raise NativeWorldError("native communicator unavailable")
        my_rank = getattr(module, "my_rank", None)
        if callable(my_rank):
            return int(my_rank())
        world = getattr(module, "mpi_world", None)
        if callable(world):
            comm = world()
            rank = getattr(comm, "rank", None)
            if rank is not None:
                return int(rank)
        raise NativeWorldError("native communicator rank unavailable")
    except NativeWorldError:
        if required:
            raise
        return None
    except Exception as exc:
        if required:
            raise NativeWorldError(f"native communicator unavailable: {exc}") from exc
        return None


def is_native_writer_rank() -> bool:
    """Serial rank 0 may write. An MPI singleton world must not write a ledger."""
    if native_has_mpi():
        size = native_world_size(required=False)
        if size is None or int(size) < 2:
            return False
    rank = native_world_rank(required=False)
    return (0 if rank is None else int(rank)) == 0


def require_native_world_size(expected: int) -> int:
    size = native_world_size(required=True)
    if size != int(expected):
        raise NativeWorldError(
            f"native communicator world size {size} != requested {expected}"
        )
    return int(size)
