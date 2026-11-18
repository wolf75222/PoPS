"""Cold CC13 agreement for a Program bound on the native world."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SchemaMismatch(ValueError):
    """The ranks could not authenticate the same Program."""


def bind_world(resources: Any) -> Any:
    """Find the actual world authority before verifying local artifact contents."""
    from pops._native_collectives import require_world
    from pops._platform_contracts import ExecutionContext

    if not isinstance(resources, Mapping):
        return None
    context = resources.get("execution_context")
    if type(context) is not ExecutionContext or context.communicator.identity != "MPI_COMM_WORLD":
        return None
    return require_world(context.communicator.handle)


def verify_world_bind(artifact: Any, world: Any) -> None:
    """Converge local artifact verification and agree its CC2 identity before install."""
    try:
        artifact.verify()
        program = bytes.fromhex(artifact.program_id.hexdigest)
    except Exception:
        # Other participants still enter the same rendezvous when local evidence fails.
        program = None
    status = world._agree_program_bind(program)
    if status == 1:
        raise SchemaMismatch("MPI bind could not authenticate every rank's Program identity")
    if status == 2:
        raise SchemaMismatch("MPI bind program_id differs across the native world")
