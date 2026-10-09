"""Collectively publish a content-addressed DSL artifact in an MPI test.

Rank-local pytest fixtures may name different native caches. Rank zero publishes
its per-test cache resource and complete artifact identity, then every peer
authenticates and loads that exact publication before runtime construction.
Every failure is reported through a collective before a later MPI runtime call
can be reached, so no rank can wait forever behind a rank-local exception.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
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


class CompiledArtifact(Protocol):
    @property
    def artifact_identity(self) -> _PlanIdentity: ...

    def verify(self) -> None: ...


ArtifactT = TypeVar("ArtifactT", bound=CompiledArtifact)
PlanT = TypeVar("PlanT", bound=ResolvedPlan)


def _phase(comm: object, label: str) -> None:
    """Keep potentially blocking compiler/cache phases visible in CI logs."""
    native = require_world(comm)
    print("[rank %d] %s" % (native.rank, label), flush=True)


@contextmanager
def _published_cache(path: str):
    """Borrow this test's publisher cache only while a peer loads its artifact."""
    previous = os.environ.get("POPS_CACHE_DIR")
    os.environ["POPS_CACHE_DIR"] = path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("POPS_CACHE_DIR", None)
        else:
            os.environ["POPS_CACHE_DIR"] = previous


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
    The per-test publisher cache remains alive through every peer's verified load;
    peer fixture environments are restored before returning. No global isolation
    setting or native package authentication is disabled.
    """
    native = require_world(comm)
    identities = allgather_value(native, resolved.plan_identity.hexdigest)
    if len(set(identities)) != 1:
        raise RuntimeError("resolved AMR plan identity differs across MPI ranks")

    rank = int(native.rank)
    artifact: ArtifactT | None = None
    publication: tuple[bool, str, str] | None = None
    if rank == 0:
        _phase(comm, route + ": compile and publish start")
        try:
            from pops.codegen.cache import pops_cache_dir

            # tmp_path fixtures are rank-local even on a shared filesystem. Publish
            # rank zero's actual cache resource, not an assumption that paths agree.
            published_cache = str(Path(pops_cache_dir()).resolve())
            artifact = compile_artifact(resolved)
            if artifact is None:
                raise RuntimeError("compiler returned no artifact")
            artifact.verify()
            published_identity = artifact.artifact_identity.hexdigest
        except Exception as exc:  # noqa: BLE001 -- broadcast rank-0 cause to every peer
            publication = (False, "%s: %s" % (type(exc).__name__, exc), "")
        else:
            publication = (True, published_cache, published_identity)
            _phase(comm, route + ": compile and publish done")

    publication = broadcast_value(native, publication, root=0)
    if not publication[0]:
        raise RuntimeError("rank 0 artifact publication failed: " + publication[1])

    load_error = ""
    if rank != 0:
        _phase(comm, route + ": authenticated cache load start")
        try:
            if not Path(publication[1]).is_dir():
                raise RuntimeError("published MPI test cache is not visible on this rank")
            with _published_cache(publication[1]):
                artifact = compile_artifact(resolved)
            if artifact is None:
                raise RuntimeError("compiler returned no artifact")
        except Exception as exc:  # noqa: BLE001 -- collect every peer error before continuing
            load_error = "%s: %s" % (type(exc).__name__, exc)
        else:
            _phase(comm, route + ": authenticated cache load done")

    if not load_error:
        try:
            if artifact is None:
                raise RuntimeError("rank did not obtain the compiled AMR artifact")
            # Recompute evidence for every model and Program binary, sidecar-owned
            # component, and ABI/platform contract before any rank may bind.
            artifact.verify()
            if artifact.artifact_identity.hexdigest != publication[2]:
                raise RuntimeError("compiled artifact differs from rank zero's exact publication")
        except Exception as exc:  # noqa: BLE001 -- one collective failure schedule
            load_error = "%s: %s" % (type(exc).__name__, exc)

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
