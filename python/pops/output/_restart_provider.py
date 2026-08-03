"""Output-owned restart operation provider used by checkpoint ConsumerGraph nodes."""

from __future__ import annotations

import os
import stat
import sys
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


def _checkpoint_path_inode(path: Path) -> tuple[int, int]:
    """Return the exact non-following filesystem identity of one checkpoint path."""
    status = path.stat(follow_symlinks=False)
    return int(status.st_dev), int(status.st_ino)


def _unlink_checkpoint_path_if_owned(
    path: Path,
    inode: tuple[int, int],
    *,
    phase: str,
) -> None:
    """Atomically detach and remove only the exact checkpoint inode owned by PoPS."""
    from ._writers.common import _StagedOutputFile

    _StagedOutputFile._quarantine_owned_path(
        path,
        inode,
        replaced_message="checkpoint %s refuses to delete replaced path %s" % (phase, path),
    )


class _CheckpointTransactionReceipt:
    """Authenticated private directory spanning native capture and Python reseal.

    The path-only native checkpoint ABI cannot attest which inode it created.  RuntimeInstance
    therefore accepts a native candidate only inside this retained, mode-0700 directory and only
    after authenticating the candidate payload through an anchored descriptor.  Calls without this
    receipt are refused rather than pretending that a post-hoc ``stat`` proves creator ownership.

    The native provider still accepts only a path: it cannot promise no-clobber creation between
    the absence proof and its own open/replace, nor identify a same-principal substitution with
    another *valid, authenticated* PoPS payload before this descriptor is acquired.  Those stronger
    claims require a future fd/receipt native ABI and are deliberately not advertised here.  Invalid
    or later substitutions are detected and are never cleaned as PoPS-owned entries.
    """

    __slots__ = ("directory", "owner", "_descriptor", "_parent_descriptor")

    def __init__(
        self,
        directory: Any,
        owner: tuple[int, int],
        descriptor: int | None,
        parent_descriptor: int | None,
    ) -> None:
        if (
            type(owner) is not tuple
            or len(owner) != 2
            or any(type(value) is not int or value < 0 for value in owner)
        ):
            raise ValueError("checkpoint transaction receipt requires an exact directory inode")
        if (descriptor is None) != (parent_descriptor is None):
            raise ValueError("checkpoint transaction descriptors must be retained together")
        self.directory = Path(directory)
        self.owner = owner
        self._descriptor = descriptor
        self._parent_descriptor = parent_descriptor
        self.authenticate_directory()

    @staticmethod
    def _directory_flags() -> int:
        return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    @classmethod
    def created(cls, parent: Path) -> _CheckpointTransactionReceipt:
        parent.mkdir(parents=True, exist_ok=True)
        parent_descriptor = os.open(parent, cls._directory_flags())
        directory: Path | None = None
        descriptor: int | None = None
        try:
            directory = Path(tempfile.mkdtemp(prefix=".pops-restart-transaction.", dir=str(parent)))
            descriptor = os.open(
                directory.name,
                cls._directory_flags(),
                dir_fd=parent_descriptor,
            )
            created = os.fstat(descriptor)
            return cls(
                directory,
                (int(created.st_dev), int(created.st_ino)),
                descriptor,
                parent_descriptor,
            )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            # Authentication did not complete, so the lexical directory name is not owned and
            # must not be removed even when it still appears empty.
            os.close(parent_descriptor)
            raise

    @classmethod
    def observed(cls, directory: Any, owner: tuple[int, int]) -> _CheckpointTransactionReceipt:
        return cls(directory, owner, None, None)

    @property
    def has_root_descriptor(self) -> bool:
        return self._descriptor is not None

    def to_data(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "device": self.owner[0],
            "inode": self.owner[1],
        }

    def authenticate_directory(self) -> None:
        named = self.directory.lstat()
        named_owner = (int(named.st_dev), int(named.st_ino))
        if (
            not stat.S_ISDIR(named.st_mode)
            or stat.S_IMODE(named.st_mode) & 0o077
            or named_owner != self.owner
        ):
            raise RuntimeError("checkpoint private transaction directory authority changed")
        if self._descriptor is not None:
            retained = os.fstat(self._descriptor)
            if (
                not stat.S_ISDIR(retained.st_mode)
                or (int(retained.st_dev), int(retained.st_ino)) != self.owner
            ):
                raise RuntimeError("checkpoint private transaction descriptor authority changed")

    def require_entry_path(self, path: Path) -> None:
        self.authenticate_directory()
        if path.parent != self.directory or path.name in {"", ".", ".."}:
            raise RuntimeError("checkpoint staging path escaped its private transaction directory")

    def open_candidate(self, path: Path) -> tuple[int, tuple[int, int]]:
        """Open an unowned native candidate without granting cleanup authority."""
        self.require_entry_path(path)
        if self._descriptor is None:
            raise RuntimeError("rank zero lacks the checkpoint transaction descriptor")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=self._descriptor)
        try:
            candidate = os.fstat(descriptor)
            if not stat.S_ISREG(candidate.st_mode):
                raise RuntimeError("native checkpoint candidate is not a regular file")
            return descriptor, (int(candidate.st_dev), int(candidate.st_ino))
        except BaseException:
            os.close(descriptor)
            raise

    def require_absent_entry(self, path: Path) -> None:
        """Prove that PoPS has not yet acquired or inherited this directory entry."""
        self.require_entry_path(path)
        if self._descriptor is None:
            raise RuntimeError("rank zero lacks the checkpoint transaction descriptor")
        try:
            os.stat(path.name, dir_fd=self._descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(
            "checkpoint private staging entry existed before native creation: %s" % path
        )

    def authenticate_entry(self, path: Path, owner: tuple[int, int]) -> None:
        """Acquire/confirm one name only while it still denotes the authenticated inode."""
        self.require_entry_path(path)
        if self._descriptor is None:
            raise RuntimeError("rank zero lacks the checkpoint transaction descriptor")
        try:
            current = os.stat(path.name, dir_fd=self._descriptor, follow_symlinks=False)
        except FileNotFoundError as error:
            raise RuntimeError(
                "checkpoint transaction entry disappeared before ownership acquisition"
            ) from error
        if (
            not stat.S_ISREG(current.st_mode)
            or (
                int(current.st_dev),
                int(current.st_ino),
            )
            != owner
        ):
            raise RuntimeError(
                "checkpoint transaction entry was replaced before ownership acquisition"
            )

    def rename_no_replace(self, source: Path, destination: Path) -> None:
        self.require_entry_path(source)
        self.require_entry_path(destination)
        if self._descriptor is None:
            raise RuntimeError("rank zero lacks the checkpoint transaction descriptor")
        from ._writers.common import _rename_no_replace

        _rename_no_replace(
            source.name,
            destination.name,
            src_dir_fd=self._descriptor,
            dst_dir_fd=self._descriptor,
        )

    def cleanup_empty(self) -> None:
        """Atomically detach then remove this exact empty private directory."""
        descriptor = self._descriptor
        parent_descriptor = self._parent_descriptor
        if descriptor is None:
            return
        cleanup_name = ".pops-restart-cleanup-%s" % os.urandom(16).hex()
        moved = False
        try:
            self.authenticate_directory()
            if os.listdir(descriptor):
                raise RuntimeError(
                    "checkpoint private transaction directory is not empty; retained at %s"
                    % self.directory
                )
            if parent_descriptor is None:
                raise RuntimeError("checkpoint transaction parent descriptor is unavailable")
            from ._writers.common import _rename_no_replace

            _rename_no_replace(
                self.directory.name,
                cleanup_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            moved = True
            detached = os.stat(cleanup_name, dir_fd=parent_descriptor, follow_symlinks=False)
            detached_owner = (int(detached.st_dev), int(detached.st_ino))
            if detached_owner != self.owner:
                recovery = self.directory.parent / cleanup_name
                try:
                    _rename_no_replace(
                        cleanup_name,
                        self.directory.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
                except BaseException as restore_error:
                    raise RuntimeError(
                        "checkpoint transaction directory was replaced; replacement retained at "
                        "%s; restoration failed: %s" % (recovery, restore_error)
                    ) from restore_error
                moved = False
                raise RuntimeError(
                    "checkpoint transaction directory was replaced and restored without deletion"
                )
            os.rmdir(cleanup_name, dir_fd=parent_descriptor)
            moved = False
        except BaseException as error:
            if moved:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "authenticated checkpoint transaction retained at %s"
                        % (self.directory.parent / cleanup_name)
                    )
            raise
        finally:
            self._descriptor = None
            self._parent_descriptor = None
            primary = sys.exc_info()[1]
            close_failures = []
            try:
                os.close(descriptor)
            except BaseException as close_error:
                close_failures.append(close_error)
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except BaseException as close_error:
                    close_failures.append(close_error)
            if close_failures:
                message = "checkpoint transaction descriptor cleanup also failed: " + "; ".join(
                    "%s: %s" % (type(error).__name__, error) for error in close_failures
                )
                if primary is not None:
                    add_note = getattr(primary, "add_note", None)
                    if callable(add_note):
                        add_note(message)
                else:
                    raise RuntimeError(message)


def _recorded_hierarchy() -> Any:
    from .restart import RestoreRecordedHierarchy

    return RestoreRecordedHierarchy()


@dataclass(frozen=True, slots=True)
class ReopenedRestart:
    target: Path
    payload: bytes
    cursors: Any


class _RestartSnapshot:
    """One collectively captured file whose publication is still compensatable."""

    __slots__ = (
        "_runtime",
        "_topology",
        "_staging",
        "_staging_inode",
        "_published_target",
        "_published_inode",
        "_discarded",
        "_transaction",
    )

    @staticmethod
    def _inode(path: Path) -> tuple[int, int]:
        return _checkpoint_path_inode(path)

    @staticmethod
    def _unlink_owned(
        path: Path,
        inode: tuple[int, int],
        *,
        phase: str,
    ) -> None:
        _unlink_checkpoint_path_if_owned(path, inode, phase=phase)

    def __init__(self, runtime: Any, directory: Any) -> None:
        self._runtime = runtime
        self._topology = checkpoint_topology(runtime)
        local_directory = Path(os.path.abspath(os.path.normpath(os.fspath(directory))))
        created_transaction: _CheckpointTransactionReceipt | None = None

        def choose_staging() -> dict[str, Any]:
            nonlocal created_transaction
            created_transaction = _CheckpointTransactionReceipt.created(local_directory)
            selected = created_transaction.to_data()
            selected["parent"] = str(local_directory)
            selected["staging"] = str(created_transaction.directory / "native.npz")
            return selected

        selected = root_value(self._topology, "staging selection", choose_staging)
        selection_error = None
        try:
            if not isinstance(selected, dict) or set(selected) != {
                "parent",
                "directory",
                "device",
                "inode",
                "staging",
            }:
                raise RuntimeError("rank zero returned an invalid checkpoint staging selection")
            if str(local_directory) != selected["parent"]:
                raise ValueError(
                    "checkpoint staging directory differs across ranks: local %s, rank-0 %s"
                    % (local_directory, selected["parent"])
                )
            if any(
                isinstance(selected[key], bool) or type(selected[key]) is not int
                for key in ("device", "inode")
            ):
                raise RuntimeError("rank zero returned invalid transaction directory evidence")
            transaction_owner = (int(selected["device"]), int(selected["inode"]))
            if self._topology.rank == 0:
                if created_transaction is None:
                    raise RuntimeError("rank zero lost its checkpoint transaction receipt")
                transaction = created_transaction
                if transaction.to_data() != {
                    key: selected[key] for key in ("directory", "device", "inode")
                }:
                    raise RuntimeError("rank zero transaction receipt differs from its broadcast")
            else:
                transaction = _CheckpointTransactionReceipt.observed(
                    selected["directory"], transaction_owner
                )
            staging = canonical_checkpoint_path(selected["staging"])
            transaction.require_entry_path(staging)
        except BaseException as error:
            selection_error = error
            transaction = created_transaction
            staging = (
                Path(selected.get("staging", ".invalid-checkpoint.npz"))
                if isinstance(selected, dict)
                else Path(".invalid-checkpoint.npz")
            )
        try:
            consensus(self._topology, "staging agreement", error=selection_error)
        except BaseException as error:
            if created_transaction is not None:
                try:
                    created_transaction.cleanup_empty()
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note("checkpoint transaction cleanup also failed: %s" % cleanup_error)
            raise
        if transaction is None:
            raise RuntimeError("checkpoint staging selection returned no transaction receipt")
        self._transaction = transaction
        self._staging = staging
        self._staging_inode: tuple[int, int] | None = None
        self._published_target: Path | None = None
        self._published_inode: tuple[int, int] | None = None
        self._discarded = False

        # Every rank enters the exact native capture with the same staging path.  The RuntimeInstance
        # performs a consensus after native collection and after rank-zero envelope sealing.
        try:
            produced = Path(
                runtime._checkpoint_payload(
                    self._staging,
                    transaction_receipt=self._transaction,
                )
            )
        except BaseException as error:
            try:
                root_value(
                    self._topology,
                    "failed capture transaction cleanup",
                    self._transaction.cleanup_empty,
                )
            except BaseException as cleanup_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note("checkpoint transaction cleanup also failed: %s" % cleanup_error)
            self._discarded = True
            raise
        exact_error = None
        if produced != self._staging:
            exact_error = RuntimeError(
                "restart provider did not capture the exact shared staged snapshot"
            )
        consensus(
            self._topology,
            "staged snapshot identity",
            error=exact_error,
            value=str(produced),
        )
        staged_inode = root_value(
            self._topology,
            "staged snapshot inode",
            lambda: list(self._transaction_entry_inode(self._staging)),
        )
        if not isinstance(staged_inode, list) or len(staged_inode) != 2:
            raise RuntimeError("rank zero returned an invalid staged checkpoint inode")
        self._staging_inode = (int(staged_inode[0]), int(staged_inode[1]))

    def _transaction_entry_inode(self, path: Path) -> tuple[int, int]:
        descriptor, owner = self._transaction.open_candidate(path)
        os.close(descriptor)
        self._transaction.authenticate_entry(path, owner)
        return owner

    @property
    def path(self) -> Path:
        return self._staging

    def publish(self, target: Any) -> Path:
        if self._discarded:
            raise RuntimeError("discarded restart snapshot cannot be published")
        local_target = canonical_checkpoint_path(target)
        selected_target = Path(
            root_value(self._topology, "target selection", lambda: str(local_target))
        )
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
                # Staging lives in a private child of the target directory, hence on the same
                # filesystem.  A hard link is an atomic no-clobber publication: unlike
                # exists()+replace(), it cannot overwrite a competing creator.
                os.link(self._staging, selected_target)
                linked = True
                if self._inode(selected_target) != self._staging_inode:
                    raise RuntimeError("checkpoint hard link does not retain the staging inode")
                self._runtime._inspect_checkpoint_file(selected_target)
                self._unlink_owned(
                    self._staging, self._staging_inode, phase="successful staging cleanup"
                )
                self._transaction.cleanup_empty()
            except FileExistsError as error:
                raise FileExistsError(
                    "checkpoint target collision: %s" % selected_target
                ) from error
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
            "target",
            "device",
            "inode",
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
            self._unlink_owned(self._staging, self._staging_inode, phase="snapshot discard")
            self._transaction.cleanup_empty()

        root_value(self._topology, "discard", discard_root)
        self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return

        def rollback_root() -> None:
            if self._staging_inode is not None:
                self._unlink_owned(
                    self._staging, self._staging_inode, phase="rollback staging cleanup"
                )
            if self._published_target is not None:
                if self._published_inode is None:
                    raise RuntimeError("published checkpoint has no ownership evidence")
                self._unlink_owned(
                    self._published_target,
                    self._published_inode,
                    phase="rollback publication cleanup",
                )
            self._transaction.cleanup_empty()

        root_value(self._topology, "rollback", rollback_root)
        self._published_target = None
        self._published_inode = None
        self._discarded = True


@dataclass(frozen=True, slots=True)
class RestartV3:
    """Compatibility-named adapter over strict Uniform v5 / AMR v7 accepted-state payloads."""

    __pops_ir_immutable__ = True
    bit_identical: bool = False
    hierarchy: Any = field(default_factory=_recorded_hierarchy)

    def __post_init__(self) -> None:
        if type(self.bit_identical) is not bool:
            raise TypeError("RestartV3.bit_identical must be an exact bool")
        from .restart import RegridOnRestart, require_restart_hierarchy

        policy = require_restart_hierarchy(self.hierarchy, where="RestartV3.hierarchy")
        object.__setattr__(self, "hierarchy", policy)
        if self.bit_identical and type(policy) is RegridOnRestart:
            raise ValueError("RestartV3 cannot combine bit_identical=True with RegridOnRestart()")

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.restart.accepted-state-v5",
            "extension": ".npz",
            "bit_identical": self.bit_identical,
            "guarantee": (
                "bit_identical_accepted_state" if self.bit_identical else self.hierarchy.guarantee
            ),
            "hierarchy": self.hierarchy.to_data(),
            "hierarchy_identity": self.hierarchy.identity.token,
            # A serial runtime is the one-member case of this collective operation.  Providers
            # without this explicit capability (notably parallel HDF5) remain fail-closed when no
            # distributed communicator is present.
            "supports_singleton_collective": True,
            "supports_regrid_on_restart": True,
        }

    def validate_configuration(self) -> None:
        """Validate the policy pair before capture/reopen touches collective state."""
        if self.bit_identical and self.hierarchy.mode == "regrid_on_restart":
            raise ValueError("RestartV3 cannot combine bit_identical=True with RegridOnRestart()")

    def snapshot(self, runtime: Any, directory: Any) -> Any:
        self.validate_configuration()
        return self.validate_snapshot(_RestartSnapshot(runtime, directory))

    @staticmethod
    def validate_snapshot(snapshot: Any) -> Any:
        from ._consumer_contracts import validate_checkpoint_snapshot

        return validate_checkpoint_snapshot(snapshot)

    def write(self, snapshot: Any, target: Any) -> Path:
        self.validate_configuration()
        if type(snapshot) is not _RestartSnapshot:
            raise TypeError("RestartV3.write requires its exact prepared restart snapshot")
        return snapshot.publish(target)

    def reopen(self, runtime: Any, path: Any) -> ReopenedRestart:
        self.validate_configuration()
        from ._checkpoint_collective import root_bytes

        topology = checkpoint_topology(runtime)
        local_target = canonical_checkpoint_path(path)
        target = Path(root_value(topology, "restart target selection", lambda: str(local_target)))
        target_error = (
            None
            if target == local_target
            else ValueError(
                "restart target differs across ranks: local %s, rank-0 %s" % (local_target, target)
            )
        )
        consensus(topology, "restart target agreement", error=target_error)
        root_payload = b""

        def read_and_authenticate_root() -> dict[str, Any]:
            nonlocal root_payload
            root_payload = target.read_bytes()
            cursors = runtime._inspect_checkpoint_payload(root_payload)
            return cursors.to_data()

        cursor_data = root_value(
            topology, "restart read and authentication", read_and_authenticate_root
        )
        payload = root_bytes(topology, "restart payload broadcast", lambda: root_payload)
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
        self.validate_configuration()
        if type(reopened) is not ReopenedRestart:
            raise TypeError("RestartV3.restore requires an exact ReopenedRestart")
        if self.hierarchy.mode == "regrid_on_restart":
            return runtime._restore_checkpoint(
                reopened.payload,
                reopened.cursors,
                bit_identical=self.bit_identical,
                hierarchy_mode=self.hierarchy.mode,
                hierarchy_identity=self.hierarchy.identity.token,
            )
        return runtime._restore_checkpoint(
            reopened.payload,
            reopened.cursors,
            bit_identical=self.bit_identical,
        )


@dataclass(frozen=True, slots=True)
class RestartAuthority:
    """Resolved, plan-owned authority for manual and scheduled restart checkpoints."""

    operation: Any = field(repr=False)
    source: str = "builtin-v5"
    operation_data: Any = field(init=False, repr=False)
    identity: Any = field(init=False)

    def __post_init__(self) -> None:
        from pops.identity import make_identity
        from pops._frozen_data import thaw_data
        from ._consumer_contracts import _provider_data

        if self.source not in {"builtin-v5", "consumer-graph"}:
            raise ValueError("restart authority has an unsupported source")
        data = _provider_data(
            self.operation,
            where="RestartAuthority.operation",
            methods=("snapshot", "validate_snapshot", "write", "reopen", "restore"),
        )
        object.__setattr__(self, "operation_data", data)
        object.__setattr__(
            self,
            "identity",
            make_identity(
                "restart-authority",
                {
                    "source": self.source,
                    "operation": thaw_data(data),
                },
            ),
        )

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
            make_identity("restart-provider", thaw_data(row.operation_data)).token for row in rows
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
