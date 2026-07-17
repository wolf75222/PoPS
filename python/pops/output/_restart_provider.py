"""Output-owned restart operation provider used by checkpoint ConsumerGraph nodes."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._checkpoint_collective import (
    canonical_checkpoint_path,
    checkpoint_topology,
    consensus,
    root_value,
)


@dataclass(frozen=True, slots=True)
class ReopenedRestart:
    target: Path
    payload: bytes
    cursors: Any


class _RestartSnapshot:
    """One collectively captured file whose publication is still compensatable."""

    __slots__ = (
        "_runtime", "_topology", "_staging", "_staging_inode",
        "_published_target", "_published_inode", "_discarded",
    )

    @staticmethod
    def _inode(path: Path) -> tuple[int, int]:
        status = path.stat(follow_symlinks=False)
        return int(status.st_dev), int(status.st_ino)

    @classmethod
    def _unlink_owned(
        cls, path: Path, inode: tuple[int, int], *, phase: str,
    ) -> None:
        try:
            current = cls._inode(path)
        except FileNotFoundError:
            return
        if current != inode:
            raise RuntimeError(
                "checkpoint %s refuses to delete replaced path %s" % (phase, path))
        path.unlink()

    def __init__(self, runtime: Any, directory: Any) -> None:
        self._runtime = runtime
        self._topology = checkpoint_topology(runtime)
        local_directory = Path(os.path.abspath(os.path.normpath(os.fspath(directory))))

        def choose_staging() -> dict[str, str]:
            local_directory.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=".pops-restart-snapshot.", suffix=".npz", dir=local_directory)
            os.close(fd)
            os.unlink(name)
            return {"directory": str(local_directory), "staging": name}

        selected = root_value(self._topology, "staging selection", choose_staging)
        selection_error = None
        try:
            if not isinstance(selected, dict) or set(selected) != {"directory", "staging"}:
                raise RuntimeError("rank zero returned an invalid checkpoint staging selection")
            if str(local_directory) != selected["directory"]:
                raise ValueError(
                    "checkpoint staging directory differs across ranks: local %s, rank-0 %s"
                    % (local_directory, selected["directory"])
                )
            staging = canonical_checkpoint_path(selected["staging"])
            if staging.parent != local_directory:
                raise ValueError("checkpoint staging path escaped its authenticated directory")
        except BaseException as error:
            selection_error = error
            staging = Path(selected.get("staging", ".invalid-checkpoint.npz")) \
                if isinstance(selected, dict) else Path(".invalid-checkpoint.npz")
        consensus(self._topology, "staging agreement", error=selection_error)
        self._staging = staging
        self._staging_inode: tuple[int, int] | None = None
        self._published_target: Path | None = None
        self._published_inode: tuple[int, int] | None = None
        self._discarded = False

        # Every rank enters the exact native capture with the same staging path.  The RuntimeInstance
        # performs a consensus after native collection and after rank-zero envelope sealing.
        try:
            produced = Path(runtime._checkpoint_payload(self._staging))
        except BaseException:
            # Capture providers are required to publish their private staging path only after a
            # complete sealed payload exists.  On failure there is therefore no owned final inode
            # to remove here.  Blindly unlinking the lexical name would risk deleting a concurrent
            # replacement for which this transaction has no ownership proof.
            self._discarded = True
            raise
        exact_error = None
        if produced != self._staging:
            exact_error = RuntimeError(
                "restart provider did not capture the exact shared staged snapshot")
        consensus(
            self._topology,
            "staged snapshot identity",
            error=exact_error,
            value=str(produced),
        )
        staged_inode = root_value(
            self._topology,
            "staged snapshot inode",
            lambda: list(self._inode(self._staging)),
        )
        if not isinstance(staged_inode, list) or len(staged_inode) != 2:
            raise RuntimeError("rank zero returned an invalid staged checkpoint inode")
        self._staging_inode = (int(staged_inode[0]), int(staged_inode[1]))

    @property
    def path(self) -> Path:
        return self._staging

    def publish(self, target: Any) -> Path:
        if self._discarded:
            raise RuntimeError("discarded restart snapshot cannot be published")
        local_target = canonical_checkpoint_path(target)
        selected_target = Path(root_value(
            self._topology, "target selection", lambda: str(local_target)))
        target_error = None
        if local_target != selected_target:
            target_error = ValueError(
                "checkpoint target differs across ranks: local %s, rank-0 %s"
                % (local_target, selected_target)
            )
        if self._published_target is not None and self._published_target != selected_target:
            target_error = ValueError("restart snapshot was already published to another target")
        consensus(self._topology, "target agreement", error=target_error)
        if self._published_target is not None:
            return self._published_target

        def publish_root() -> dict[str, Any]:
            selected_target.parent.mkdir(parents=True, exist_ok=True)
            linked = False
            if self._staging_inode is None:
                raise RuntimeError("restart snapshot has no authenticated staging inode")
            try:
                # Staging and target are deliberately in the same directory.  A hard link is an
                # atomic no-clobber publication: unlike exists()+replace(), a competing creator can
                # never be overwritten between the collision check and the namespace mutation.
                os.link(self._staging, selected_target)
                linked = True
                if self._inode(selected_target) != self._staging_inode:
                    raise RuntimeError("checkpoint hard link does not retain the staging inode")
                self._runtime._inspect_checkpoint_file(selected_target)
                self._unlink_owned(
                    self._staging, self._staging_inode, phase="successful staging cleanup")
            except FileExistsError as error:
                raise FileExistsError(
                    "checkpoint target collision: %s" % selected_target) from error
            except BaseException as error:
                cleanup_error = None
                if linked:
                    try:
                        # This transaction created this exact link.  Staging remains as the durable
                        # owner until authentication succeeds, so cleanup cannot delete a peer's file.
                        self._unlink_owned(
                            selected_target,
                            self._staging_inode,
                            phase="failed publication cleanup",
                        )
                    except BaseException as caught:
                        cleanup_error = caught
                add_note = getattr(error, "add_note", None)
                if cleanup_error is not None and callable(add_note):
                    add_note("failed checkpoint publication cleanup: %s" % cleanup_error)
                raise
            return {
                "target": str(selected_target),
                "device": self._staging_inode[0],
                "inode": self._staging_inode[1],
            }

        publication = root_value(self._topology, "publication", publish_root)
        if not isinstance(publication, dict) or set(publication) != {
            "target", "device", "inode",
        }:
            raise RuntimeError("checkpoint publication returned invalid ownership evidence")
        published = Path(publication["target"])
        if published != selected_target:
            raise RuntimeError("checkpoint publication returned a different target")
        self._published_target = published
        self._published_inode = (int(publication["device"]), int(publication["inode"]))
        return published

    def discard(self) -> None:
        if self._discarded or self._published_target is not None:
            return

        def discard_root() -> None:
            if self._staging_inode is None:
                raise RuntimeError("restart snapshot has no authenticated staging inode")
            self._unlink_owned(
                self._staging, self._staging_inode, phase="snapshot discard")

        root_value(self._topology, "discard", discard_root)
        self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return

        def rollback_root() -> None:
            if self._staging_inode is not None:
                self._unlink_owned(
                    self._staging, self._staging_inode, phase="rollback staging cleanup")
            if self._published_target is not None:
                if self._published_inode is None:
                    raise RuntimeError("published checkpoint has no ownership evidence")
                self._unlink_owned(
                    self._published_target,
                    self._published_inode,
                    phase="rollback publication cleanup",
                )

        root_value(self._topology, "rollback", rollback_root)
        self._published_target = None
        self._published_inode = None
        self._discarded = True


@dataclass(frozen=True, slots=True)
class RestartV3:
    """Immutable adapter over the strict Uniform/AMR accepted-state v3 codecs."""

    __pops_ir_immutable__ = True
    bit_identical: bool = False

    def __post_init__(self) -> None:
        if type(self.bit_identical) is not bool:
            raise TypeError("RestartV3.bit_identical must be an exact bool")

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.restart.accepted-state-v3",
            "extension": ".npz",
            "bit_identical": self.bit_identical,
            # A serial runtime is the one-member case of this collective operation.  Providers
            # without this explicit capability (notably parallel HDF5) remain fail-closed when no
            # distributed communicator is present.
            "supports_singleton_collective": True,
        }

    def snapshot(self, runtime: Any, directory: Any) -> Any:
        return self.validate_snapshot(_RestartSnapshot(runtime, directory))

    @staticmethod
    def validate_snapshot(snapshot: Any) -> Any:
        from ._consumer_contracts import validate_checkpoint_snapshot

        return validate_checkpoint_snapshot(snapshot)

    def write(self, snapshot: Any, target: Any) -> Path:
        if type(snapshot) is not _RestartSnapshot:
            raise TypeError("RestartV3.write requires its exact prepared restart snapshot")
        return snapshot.publish(target)

    def reopen(self, runtime: Any, path: Any) -> ReopenedRestart:
        from ._checkpoint_collective import root_bytes

        topology = checkpoint_topology(runtime)
        local_target = canonical_checkpoint_path(path)
        target = Path(root_value(topology, "restart target selection", lambda: str(local_target)))
        target_error = None if target == local_target else ValueError(
            "restart target differs across ranks: local %s, rank-0 %s"
            % (local_target, target)
        )
        consensus(topology, "restart target agreement", error=target_error)
        root_payload = b""

        def read_and_authenticate_root() -> dict[str, Any]:
            nonlocal root_payload
            root_payload = target.read_bytes()
            cursors = runtime._inspect_checkpoint_payload(root_payload)
            return cursors.to_data()

        cursor_data = root_value(
            topology, "restart read and authentication", read_and_authenticate_root)
        payload = root_bytes(
            topology, "restart payload broadcast", lambda: root_payload)
        cursors = None
        cursor_error = None
        try:
            if not isinstance(payload, bytes) or not payload:
                raise RuntimeError("rank zero returned an invalid restart payload")
            cursors = runtime._checkpoint_cursors_from_data(cursor_data)
        except BaseException as error:
            cursor_error = error
        consensus(topology, "restart payload decoding", error=cursor_error)
        if cursors is None:
            raise RuntimeError("restart cursor consensus returned no cursor set")
        return ReopenedRestart(Path(target), payload, cursors)

    def restore(self, runtime: Any, reopened: Any) -> Any:
        if type(reopened) is not ReopenedRestart:
            raise TypeError("RestartV3.restore requires an exact ReopenedRestart")
        return runtime._restore_checkpoint(reopened.payload, reopened.cursors)


@dataclass(frozen=True, slots=True)
class RestartAuthority:
    """Resolved, plan-owned authority for manual and scheduled restart checkpoints."""

    operation: Any = field(repr=False)
    source: str = "builtin-v3"
    operation_data: Any = field(init=False, repr=False)
    identity: Any = field(init=False)

    def __post_init__(self) -> None:
        from pops.identity import make_identity
        from pops._frozen_data import thaw_data
        from ._consumer_contracts import _provider_data

        if self.source not in {"builtin-v3", "consumer-graph"}:
            raise ValueError("restart authority has an unsupported source")
        data = _provider_data(
            self.operation,
            where="RestartAuthority.operation",
            methods=("snapshot", "validate_snapshot", "write", "reopen", "restore"),
        )
        object.__setattr__(self, "operation_data", data)
        object.__setattr__(self, "identity", make_identity("restart-authority", {
            "source": self.source,
            "operation": thaw_data(data),
        }))

    @classmethod
    def from_consumer_graph(cls, graph: Any) -> RestartAuthority:
        from pops.identity import make_identity
        from pops._frozen_data import thaw_data
        from ._consumer_contracts import ConsumerGraph, ConsumerKind

        if graph is None:
            return cls(RestartV3())
        if type(graph) is not ConsumerGraph or not graph.is_resolved:
            raise TypeError("restart authority requires a resolved ConsumerGraph or None")
        rows = tuple(row for row in graph.nodes if row.kind is ConsumerKind.CHECKPOINT)
        if not rows:
            return cls(RestartV3())
        identities = {
            make_identity("restart-provider", thaw_data(row.operation_data)).token
            for row in rows
        }
        if len(identities) != 1:
            raise ValueError("ConsumerGraph declares incompatible restart authorities")
        return cls(rows[0].operation, source="consumer-graph")

    def to_data(self) -> dict[str, Any]:
        from pops._frozen_data import thaw_data

        return {
            "schema_version": 1,
            "source": self.source,
            "operation": thaw_data(self.operation_data),
            "identity": self.identity.to_data(),
        }


__all__ = ["ReopenedRestart", "RestartAuthority", "RestartV3"]
