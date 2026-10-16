"""Collectively publish a content-addressed DSL artifact in an MPI test.

Rank-local pytest fixtures may name different native caches. Rank zero publishes
its per-test cache resource and complete artifact identity, then every peer
authenticates and loads that exact publication before runtime construction.
Every failure is reported through a collective before a later MPI runtime call
can be reached, so no rank can wait forever behind a rank-local exception.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
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


def compile_resolved_plan_cached(
    comm: object,
    resolved: PlanT,
    *,
    route: str,
    compile_artifact: Callable[[PlanT], ArtifactT],
) -> ArtifactT:
    """Prepare public cached compilations concurrently, then authenticate together.

    Only the public compiler's ordinary content-addressed cache route opts in.
    Every other route keeps the historical rank-zero-first protocol. Model and
    Program cache locks serialize each successful cache entry's publication;
    Python preparation still happens independently on every rank. External
    source components retain their own separate compilation protocol, so this
    helper does not promise one compiler invocation for those components.
    """
    native = require_world(comm)

    def converge(error: str, phase: str) -> None:
        errors = allgather_value(native, error)
        if any(errors):
            details = "; ".join(
                "rank %d: %s" % (peer_rank, item)
                for peer_rank, item in enumerate(errors) if item)
            raise RuntimeError(phase + " failed: " + details)

    eligible = False
    eligibility_error = ""
    try:
        import pops

        # A callback may contain collectives or choose an uncached output. Do not
        # change its schedule merely because it returns the same Protocol type.
        if compile_artifact is pops.compile and not os.environ.get("POPS_CODEGEN_DIR"):
            options = getattr(resolved, "compile_options", None)
            eligible = (isinstance(options, Mapping)
                        and options.get("so_path") is None
                        and not options.get("force", False)
                        and options.get("model_source_policy", "allow_missing") != "recompile")
    except Exception as exc:  # noqa: BLE001 -- agree on the route before any branch
        eligibility_error = "%s: %s" % (type(exc).__name__, exc)
    decisions = allgather_value(native, (eligibility_error, eligible))
    if any(row[0] for row in decisions):
        details = "; ".join("rank %d: %s" % (rank, row[0])
                            for rank, row in enumerate(decisions) if row[0])
        raise RuntimeError("cached compilation eligibility failed: " + details)
    if not all(row[1] for row in decisions):
        return compile_resolved_plan_once(
            comm, resolved, route=route, compile_artifact=compile_artifact)

    identity = ""
    identity_error = ""
    try:
        identity = resolved.plan_identity.hexdigest
        if type(identity) is not str or not identity:
            raise RuntimeError("resolved plan has no exact identity")
    except Exception as exc:  # noqa: BLE001 -- even a failed identity getter must converge
        identity_error = "%s: %s" % (type(exc).__name__, exc)
    identities = allgather_value(native, (identity_error, identity))
    if any(row[0] for row in identities):
        details = "; ".join("rank %d: %s" % (rank, row[0])
                            for rank, row in enumerate(identities) if row[0])
        raise RuntimeError("resolved plan authentication failed: " + details)
    if len({row[1] for row in identities}) != 1:
        raise RuntimeError("resolved AMR plan identity differs across MPI ranks")

    rank = int(native.rank)
    publication: tuple[bool, str, str, str] | None = None
    if rank == 0:
        try:
            import secrets
            import tempfile
            from pops.codegen.cache import pops_cache_dir

            cache = str(Path(pops_cache_dir()).resolve())
            nonce = secrets.token_hex(32)
            # Directory equality alone does not prove that ranks see the same
            # filesystem. Keep a unique, closed marker for this test's lifetime.
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="ascii", prefix=".pops-mpi-cache-",
                    suffix=".publication", dir=cache, delete=False) as marker:
                marker.write(nonce)
                marker_name = Path(marker.name).name
            publication = (True, cache, marker_name, nonce)
        except Exception as exc:  # noqa: BLE001 -- publish setup failure before compilation
            publication = (False, "%s: %s" % (type(exc).__name__, exc), "", "")
    publication = broadcast_value(native, publication, root=0)
    resource_error = ""
    try:
        if not publication[0]:
            raise RuntimeError("rank 0 cache publication failed: " + publication[1])
        cache_path = Path(publication[1])
        if not cache_path.is_dir():
            raise RuntimeError("published MPI test cache is not visible on this rank")
        if (cache_path / publication[2]).read_text(encoding="ascii") != publication[3]:
            raise RuntimeError("published MPI test cache marker differs on this rank")
    except Exception as exc:  # noqa: BLE001 -- no rank compiles behind a failed peer
        resource_error = "%s: %s" % (type(exc).__name__, exc)
    converge(resource_error, "shared MPI cache publication")

    artifact: ArtifactT | None = None
    prepared_identity = ""
    compile_error = ""
    try:
        with _published_cache(publication[1]):
            _phase(comm, route + ": concurrent cached preparation start")
            artifact = compile_artifact(resolved)
            if artifact is None:
                raise RuntimeError("compiler returned no artifact")
            artifact.verify()
            # Read this potentially failing property before the error collective,
            # not on rank zero between that collective and the next broadcast.
            prepared_identity = artifact.artifact_identity.hexdigest
            if type(prepared_identity) is not str or not prepared_identity:
                raise RuntimeError("compiled artifact has no exact identity")
            _phase(comm, route + ": concurrent cached preparation done")
    except Exception as exc:  # noqa: BLE001 -- restore the environment, then converge
        compile_error = "%s: %s" % (type(exc).__name__, exc)
    converge(compile_error, "concurrent cached artifact preparation")

    published_identity = broadcast_value(
        native, prepared_identity if rank == 0 else None, root=0)
    verification_error = ""
    try:
        if artifact is None:
            raise RuntimeError("rank did not obtain the compiled artifact")
        # Recompute the complete evidence after every rank has finished preparing;
        # a late publication or tampering must not escape the pre-bind frontier.
        artifact.verify()
        if artifact.artifact_identity.hexdigest != published_identity:
            raise RuntimeError("compiled artifact differs from rank zero's exact publication")
    except Exception as exc:  # noqa: BLE001 -- every rank leaves on the same failure schedule
        verification_error = "%s: %s" % (type(exc).__name__, exc)
    converge(verification_error, "authenticated concurrent artifact publication")
    if artifact is None:
        raise RuntimeError("rank did not obtain the compiled artifact")
    return artifact
