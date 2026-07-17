"""Collectively publish a content-addressed DSL artifact in an MPI test.

The production artifact cache is shared by ranks of one MPI job.  A test must
therefore not let every rank race through the compiler lock: rank 0 publishes
the artifact, then every peer authenticates and loads that exact cache entry.
Every failure is reported through a collective before a later MPI runtime call
can be reached, so no rank can wait forever behind a rank-local exception.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from pops._native_collectives import allgather_value, broadcast_value, require_world


class _PlanIdentity(Protocol):
    """Minimal resolved-plan identity contract required by this helper."""

    @property
    def hexdigest(self) -> str: ...


class ResolvedPlan(Protocol):
    """Minimal public resolved-plan contract needed for cache publication."""

    @property
    def plan_identity(self) -> _PlanIdentity: ...


ArtifactT = TypeVar("ArtifactT")
PlanT = TypeVar("PlanT", bound=ResolvedPlan)


def _phase(comm: object, label: str) -> None:
    """Keep potentially blocking compiler/cache phases visible in CI logs."""
    native = require_world(comm)
    print("[rank %d] %s" % (native.rank, label), flush=True)


def compile_resolved_plan_once(
    comm: object,
    resolved: PlanT,
    *,
    route: str,
    compile_artifact: Callable[[PlanT], ArtifactT],
) -> ArtifactT:
    """Compile once on rank 0 and collectively authenticate cache loading.

    This deliberately has no barriers.  The ordered ``bcast`` and ``allgather``
    communicate rank-local failures before any rank reaches runtime construction,
    avoiding the deadlock pattern where a peer waits in a later MPI collective.
    """
    native = require_world(comm)
    identities = allgather_value(native, resolved.plan_identity.hexdigest)
    if len(set(identities)) != 1:
        raise RuntimeError("resolved AMR plan identity differs across MPI ranks")

    rank = int(native.rank)
    artifact: ArtifactT | None = None
    publication: tuple[bool, str] | None = None
    if rank == 0:
        _phase(comm, route + ": compile and publish start")
        try:
            artifact = compile_artifact(resolved)
            if artifact is None:
                raise RuntimeError("compiler returned no artifact")
        except Exception as exc:  # noqa: BLE001 -- broadcast rank-0 cause to every peer
            publication = (False, "%s: %s" % (type(exc).__name__, exc))
        else:
            publication = (True, "")
            _phase(comm, route + ": compile and publish done")

    publication = broadcast_value(native, publication, root=0)
    if not publication[0]:
        raise RuntimeError("rank 0 artifact publication failed: " + publication[1])

    load_error = ""
    if rank != 0:
        _phase(comm, route + ": authenticated cache load start")
        try:
            artifact = compile_artifact(resolved)
            if artifact is None:
                raise RuntimeError("compiler returned no artifact")
        except Exception as exc:  # noqa: BLE001 -- collect every peer error before continuing
            load_error = "%s: %s" % (type(exc).__name__, exc)
        else:
            _phase(comm, route + ": authenticated cache load done")

    load_errors = allgather_value(native, load_error)
    if any(load_errors):
        details = "; ".join(
            "rank %d: %s" % (peer_rank, error)
            for peer_rank, error in enumerate(load_errors)
            if error
        )
        raise RuntimeError("authenticated artifact cache load failed: " + details)
    if artifact is None:
        raise RuntimeError("rank did not obtain the compiled AMR artifact")
    return artifact
