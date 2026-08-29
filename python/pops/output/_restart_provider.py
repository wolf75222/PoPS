"""Output-owned restart operation provider used by checkpoint ConsumerGraph nodes."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._checkpoint_collective import (
    canonical_checkpoint_path,
    checkpoint_topology,
    consensus,
    root_value,
)


def _append_exception_note(error: BaseException, note: str) -> None:
    """Attach diagnostic context when the configured Python runtime supports it."""
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def _owner(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _validate_owner(value: Any, *, where: str) -> tuple[int, int]:
    if (
        type(value) not in {list, tuple}
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise TypeError("%s must be exact opaque inode evidence" % where)
    return int(value[0]), int(value[1])


def _raise_cleanup_failures(message: str, failures: list[BaseException]) -> None:
    if failures:
        raise RuntimeError(
            message
            + ": "
            + "; ".join("%s: %s" % (type(error).__name__, error) for error in failures)
        )


class _CheckpointTransportFailure(RuntimeError):
    """A broken control transport after which no second collective is legal."""


class _CheckpointEntryAuthority:
    """One exact directory entry plus the retained descriptor of its inode on rank zero."""

    __slots__ = ("name", "owner", "_descriptor")

    def __init__(self, name: str, owner: tuple[int, int], descriptor: int | None) -> None:
        if not isinstance(name, str) or not name or "/" in name or "\x00" in name:
            raise ValueError("checkpoint entry authority requires one local name")
        self.name = name
        self.owner = _validate_owner(owner, where="checkpoint entry owner")
        self._descriptor = descriptor
        if descriptor is not None:
            retained = os.fstat(descriptor)
            if not stat.S_ISREG(retained.st_mode) or _owner(retained) != self.owner:
                raise RuntimeError("checkpoint entry descriptor differs from its inode authority")

    @property
    def is_open(self) -> bool:
        return self._descriptor is not None

    def fileno(self) -> int:
        if self._descriptor is None:
            raise RuntimeError("this checkpoint peer has no rank-zero entry descriptor")
        return self._descriptor

    def duplicate(self) -> int:
        return os.dup(self.fileno())

    def transfer(self, name: str) -> _CheckpointEntryAuthority:
        descriptor = self.fileno()
        transferred = _CheckpointEntryAuthority(name, self.owner, descriptor)
        self._descriptor = None
        return transferred

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        os.close(descriptor)


class _CheckpointTransactionReceipt:
    """Private mkdirat/openat namespace retained from capture through publication."""

    __slots__ = (
        "parent",
        "directory_name",
        "owner",
        "_directory_fd",
        "_parent_fd",
        "_native_entry",
    )

    _NATIVE_NAME = "native.npz"

    def __init__(
        self,
        parent: Any,
        directory_name: str,
        owner: tuple[int, int],
        directory_fd: int | None,
        parent_fd: int | None,
        native_entry: _CheckpointEntryAuthority,
    ) -> None:
        if (
            not isinstance(directory_name, str)
            or not directory_name.startswith(".pops-restart-transaction.")
            or "/" in directory_name
        ):
            raise ValueError("checkpoint transaction requires one private directory name")
        if (directory_fd is None) != (parent_fd is None):
            raise ValueError("checkpoint transaction descriptors must be retained together")
        self.parent = Path(parent)
        self.directory_name = directory_name
        self.owner = _validate_owner(owner, where="checkpoint transaction owner")
        self._directory_fd = directory_fd
        self._parent_fd = parent_fd
        self._native_entry = native_entry
        if directory_fd is not None:
            self.authenticate_directory_at()

    @staticmethod
    def _directory_flags() -> int:
        return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    @property
    def directory(self) -> Path:
        return self.parent / self.directory_name

    @property
    def staging_path(self) -> Path:
        return self.directory / self._NATIVE_NAME

    @property
    def has_root_descriptor(self) -> bool:
        return self._directory_fd is not None

    @classmethod
    def created(cls, parent: Path) -> _CheckpointTransactionReceipt:
        parent.mkdir(parents=True, exist_ok=True)
        parent_fd = os.open(parent, cls._directory_flags())
        directory_name = ""
        directory_fd: int | None = None
        transaction: _CheckpointTransactionReceipt | None = None
        try:
            for _attempt in range(32):
                candidate = ".pops-restart-transaction.%s" % os.urandom(16).hex()
                try:
                    os.mkdir(candidate, 0o700, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                directory_name = candidate
                break
            if not directory_name:
                raise RuntimeError("checkpoint could not allocate a private transaction directory")
            directory_fd = os.open(directory_name, cls._directory_flags(), dir_fd=parent_fd)
            transaction = cls(
                parent,
                directory_name,
                _owner(os.fstat(directory_fd)),
                directory_fd,
                parent_fd,
                _CheckpointEntryAuthority(cls._NATIVE_NAME, (0, 0), None),
            )
            transaction._native_entry = transaction.created_at(cls._NATIVE_NAME)
            return transaction
        except BaseException as error:
            failures = []
            if transaction is not None:
                try:
                    transaction.cleanup_owned()
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            else:
                if directory_fd is not None:
                    try:
                        os.close(directory_fd)
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                try:
                    os.close(parent_fd)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            if failures:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "checkpoint transaction construction cleanup also failed: "
                        + "; ".join(str(item) for item in failures)
                    )
            raise

    @classmethod
    def observed(cls, data: Any) -> _CheckpointTransactionReceipt:
        if not isinstance(data, dict) or set(data) != {
            "parent",
            "directory_name",
            "directory_owner",
            "staging_name",
            "staging_owner",
        }:
            raise RuntimeError("rank zero returned invalid checkpoint transaction evidence")
        if data["staging_name"] != cls._NATIVE_NAME:
            raise RuntimeError("rank zero returned a different native staging name")
        # Device/inode values are opaque transport scalars on peers; they are never compared with
        # a rank-local mount.
        native = _CheckpointEntryAuthority(
            data["staging_name"],
            _validate_owner(data["staging_owner"], where="native staging evidence"),
            None,
        )
        return cls(
            data["parent"],
            data["directory_name"],
            _validate_owner(data["directory_owner"], where="transaction evidence"),
            None,
            None,
            native,
        )

    def to_data(self) -> dict[str, Any]:
        if self.has_root_descriptor:
            self.authenticate_directory_at()
            if self._native_entry.is_open:
                self.authenticate_entry_at(self._native_entry)
        return {
            "parent": str(self.parent),
            "directory_name": self.directory_name,
            "directory_owner": list(self.owner),
            "staging_name": self._native_entry.name,
            "staging_owner": list(self._native_entry.owner),
        }

    def directory_fileno(self) -> int:
        if self._directory_fd is None:
            raise RuntimeError("rank zero lacks the checkpoint transaction directory descriptor")
        return self._directory_fd

    def authenticate_directory_at(self) -> None:
        directory_fd = self.directory_fileno()
        if self._parent_fd is None:
            raise RuntimeError("checkpoint transaction parent descriptor is unavailable")
        retained = os.fstat(directory_fd)
        named = os.stat(self.directory_name, dir_fd=self._parent_fd, follow_symlinks=False)
        parent = os.fstat(self._parent_fd)
        if (
            not stat.S_ISDIR(retained.st_mode)
            or stat.S_IMODE(retained.st_mode) & 0o077
            or _owner(retained) != self.owner
            or _owner(named) != self.owner
            or int(retained.st_dev) != int(parent.st_dev)
        ):
            raise RuntimeError("checkpoint private transaction directory authority changed")

    def created_at(self, name: str) -> _CheckpointEntryAuthority:
        self.authenticate_directory_at()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=self.directory_fileno())
        try:
            return _CheckpointEntryAuthority(name, _owner(os.fstat(descriptor)), descriptor)
        except BaseException:
            os.close(descriptor)
            raise

    def create_unique_at(self, *, suffix: str) -> _CheckpointEntryAuthority:
        for _attempt in range(32):
            name = ".native.npz.%s%s" % (os.urandom(12).hex(), suffix)
            try:
                return self.created_at(name)
            except FileExistsError:
                continue
        raise RuntimeError("checkpoint could not allocate a unique transaction staging entry")

    def open_candidate_at(self, name: str) -> _CheckpointEntryAuthority:
        self.authenticate_directory_at()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=self.directory_fileno())
        try:
            return _CheckpointEntryAuthority(name, _owner(os.fstat(descriptor)), descriptor)
        except BaseException:
            os.close(descriptor)
            raise

    def authenticate_entry_at(self, entry: _CheckpointEntryAuthority) -> None:
        self.authenticate_directory_at()
        retained = os.fstat(entry.fileno())
        named = os.stat(entry.name, dir_fd=self.directory_fileno(), follow_symlinks=False)
        if (
            not stat.S_ISREG(retained.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or _owner(retained) != entry.owner
            or _owner(named) != entry.owner
        ):
            raise RuntimeError(
                "checkpoint transaction entry was replaced before ownership acquisition"
            )

    def rename_no_replace_at(
        self, source: _CheckpointEntryAuthority, destination_name: str
    ) -> _CheckpointEntryAuthority:
        self.authenticate_entry_at(source)
        from ._writers.common import _rename_no_replace

        _rename_no_replace(
            source.name,
            destination_name,
            src_dir_fd=self.directory_fileno(),
            dst_dir_fd=self.directory_fileno(),
        )
        # Keep the same object/fd in the caller's cleanup ledger until post-rename
        # authentication succeeds.  If that check fails, cleanup still knows the new entry name.
        source.name = destination_name
        self.authenticate_entry_at(source)
        return source

    def quarantine_entry_at(
        self,
        entry: _CheckpointEntryAuthority,
        *,
        phase: str,
        close_entry: bool = True,
    ) -> None:
        from ._writers.common import _StagedOutputFile

        try:
            _StagedOutputFile._quarantine_owned_path(
                self.directory / entry.name,
                entry.owner,
                replaced_message=(
                    "checkpoint %s refuses to delete replaced transaction entry %s"
                    % (phase, entry.name)
                ),
                directory_fd=self.directory_fileno(),
            )
        finally:
            if close_entry:
                entry.close()

    def take_native_entry(self) -> _CheckpointEntryAuthority:
        entry = self._native_entry
        self._native_entry = _CheckpointEntryAuthority(self._NATIVE_NAME, entry.owner, None)
        return entry

    def cleanup_empty(self) -> None:
        directory_fd = self._directory_fd
        parent_fd = self._parent_fd
        if directory_fd is None:
            return
        cleanup_name = ".pops-restart-cleanup-%s" % os.urandom(16).hex()
        moved = False
        primary = None
        try:
            self.authenticate_directory_at()
            if os.listdir(directory_fd):
                raise RuntimeError(
                    "checkpoint private transaction directory is not empty; retained as %s"
                    % self.directory_name
                )
            if parent_fd is None:
                raise RuntimeError("checkpoint transaction parent descriptor is unavailable")
            from ._writers.common import _rename_no_replace

            _rename_no_replace(
                self.directory_name,
                cleanup_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            moved = True
            detached = os.stat(cleanup_name, dir_fd=parent_fd, follow_symlinks=False)
            if _owner(detached) != self.owner:
                try:
                    _rename_no_replace(
                        cleanup_name,
                        self.directory_name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                except BaseException as restore_error:
                    raise RuntimeError(
                        "checkpoint transaction directory was substituted; replacement "
                        "retained as %s because restoration failed" % cleanup_name
                    ) from restore_error
                else:
                    moved = False
                raise RuntimeError("checkpoint transaction directory was substituted and restored")
            os.rmdir(cleanup_name, dir_fd=parent_fd)
            moved = False
        except BaseException as error:
            primary = error
            if moved:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note("checkpoint transaction retained as %s" % cleanup_name)
            raise
        finally:
            failures = self.close_descriptors()
            if failures:
                message = "checkpoint transaction descriptor cleanup also failed: " + "; ".join(
                    str(error) for error in failures
                )
                if primary is not None:
                    add_note = getattr(primary, "add_note", None)
                    if callable(add_note):
                        add_note(message)
                else:
                    raise RuntimeError(message)

    def close_descriptors(self) -> list[BaseException]:
        failures = []
        for attribute in ("_directory_fd", "_parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            setattr(self, attribute, None)
            try:
                os.close(descriptor)
            except BaseException as error:
                failures.append(error)
        return failures

    def close(self) -> None:
        failures = []
        native = self._native_entry
        self._native_entry = _CheckpointEntryAuthority(self._NATIVE_NAME, native.owner, None)
        if native.is_open:
            try:
                native.close()
            except BaseException as error:
                failures.append(error)
        failures.extend(self.close_descriptors())
        _raise_cleanup_failures(
            "checkpoint transaction descriptor cleanup failed",
            failures,
        )

    def cleanup_owned(self) -> None:
        failures = []
        native = self._native_entry
        self._native_entry = _CheckpointEntryAuthority(self._NATIVE_NAME, native.owner, None)
        if native.is_open:
            try:
                self.quarantine_entry_at(native, phase="transaction construction cleanup")
            except BaseException as error:
                failures.append(error)
        try:
            self.cleanup_empty()
        except BaseException as error:
            failures.append(error)
        _raise_cleanup_failures("checkpoint transaction cleanup failed", failures)


class _CheckpointPayloadProof:
    """Exact resealed inode handoff; only rank zero retains its open descriptor."""

    __slots__ = ("transaction", "entry")

    def __init__(
        self,
        transaction: _CheckpointTransactionReceipt,
        entry: _CheckpointEntryAuthority,
    ) -> None:
        self.transaction = transaction
        self.entry = entry
        if transaction.has_root_descriptor:
            transaction.authenticate_entry_at(entry)

    @property
    def path(self) -> Path:
        return self.transaction.directory / self.entry.name

    @property
    def owner(self) -> tuple[int, int]:
        return self.entry.owner

    def to_data(self) -> dict[str, Any]:
        if self.transaction.has_root_descriptor:
            # The collective handoff is evidence for this still-open inode, not for whichever
            # object a later lexical lookup might find under the same name.
            self.transaction.authenticate_entry_at(self.entry)
        return {
            "path": str(self.path),
            "entry_name": self.entry.name,
            "entry_owner": list(self.entry.owner),
            "directory_name": self.transaction.directory_name,
            "directory_owner": list(self.transaction.owner),
        }

    @classmethod
    def observed(
        cls, transaction: _CheckpointTransactionReceipt, data: Any
    ) -> _CheckpointPayloadProof:
        if not isinstance(data, dict) or set(data) != {
            "path",
            "entry_name",
            "entry_owner",
            "directory_name",
            "directory_owner",
        }:
            raise RuntimeError("rank zero returned invalid checkpoint payload proof")
        if (
            data["path"] != str(transaction.directory / data["entry_name"])
            or data["directory_name"] != transaction.directory_name
            or _validate_owner(data["directory_owner"], where="payload proof transaction owner")
            != transaction.owner
        ):
            raise RuntimeError("checkpoint payload proof differs from its transaction receipt")
        return cls(
            transaction,
            _CheckpointEntryAuthority(
                data["entry_name"],
                _validate_owner(data["entry_owner"], where="payload proof entry owner"),
                None,
            ),
        )

    def close(self) -> None:
        self.entry.close()


def _recorded_hierarchy() -> Any:
    from .restart import RestoreRecordedHierarchy

    return RestoreRecordedHierarchy()


@dataclass(frozen=True, slots=True)
class ReopenedRestart:
    target: Path
    payload: bytes
    cursors: Any


class _RestartSnapshot:
    """One exact resealed-fd handoff whose publication remains compensatable."""

    __slots__ = (
        "_runtime",
        "_topology",
        "_proof",
        "_staging_owned",
        "_published_target",
        "_published_entry",
        "_published_parent_fd",
        "_discarded",
    )

    def __init__(self, runtime: Any, directory: Any) -> None:
        from ._checkpoint_collective import root_attempt

        self._runtime = runtime
        self._topology = checkpoint_topology(runtime)
        self._proof: _CheckpointPayloadProof | None = None
        self._staging_owned = False
        self._published_target: Path | None = None
        self._published_entry: _CheckpointEntryAuthority | None = None
        self._published_parent_fd: int | None = None
        self._discarded = False
        local_directory = Path(os.path.abspath(os.path.normpath(os.fspath(directory))))
        created_transaction: _CheckpointTransactionReceipt | None = None

        def choose_transaction() -> dict[str, Any]:
            nonlocal created_transaction
            created_transaction = _CheckpointTransactionReceipt.created(local_directory)
            return created_transaction.to_data()

        attempt = root_attempt(self._topology, "staging selection", choose_transaction)
        if attempt.transport_error is not None:
            error = _CheckpointTransportFailure(
                "checkpoint transport failed during staging selection: %s" % attempt.transport_error
            )
            if attempt.producer_error is not None:
                _append_exception_note(
                    error, "rank-zero producer also failed: %s" % attempt.producer_error
                )
            if self._topology.rank == 0 and created_transaction is not None:
                try:
                    created_transaction.cleanup_owned()
                except BaseException as cleanup_error:
                    _append_exception_note(
                        error,
                        "rank-zero checkpoint cleanup also failed: %s" % cleanup_error,
                    )
            raise error from attempt.transport_error
        if attempt.producer_error is not None:
            if self._topology.rank == 0 and created_transaction is not None:
                try:
                    created_transaction.cleanup_owned()
                except BaseException as cleanup_error:
                    add_note = getattr(attempt.producer_error, "add_note", None)
                    if callable(add_note):
                        add_note("rank-zero checkpoint cleanup also failed: %s" % cleanup_error)
            raise attempt.producer_error

        selection_error = None
        transaction = None
        try:
            if self._topology.rank == 0:
                transaction = created_transaction
                if transaction is None or transaction.to_data() != attempt.value:
                    raise RuntimeError(
                        "rank-zero transaction receipt differs from its collective evidence"
                    )
            else:
                transaction = _CheckpointTransactionReceipt.observed(attempt.value)
            if transaction.parent != local_directory:
                raise ValueError(
                    "checkpoint staging directory differs across ranks: local %s, rank-0 %s"
                    % (local_directory, transaction.parent)
                )
        except BaseException as error:
            selection_error = error
        try:
            consensus(self._topology, "staging agreement", error=selection_error)
        except BaseException as error:
            if self._topology.rank == 0 and created_transaction is not None:
                try:
                    created_transaction.cleanup_owned()
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note("rank-zero checkpoint cleanup also failed: %s" % cleanup_error)
            raise
        if transaction is None:
            raise RuntimeError("checkpoint staging selection returned no transaction receipt")

        try:
            proof = runtime._checkpoint_payload(
                transaction.staging_path,
                transaction_receipt=transaction,
            )
        except _CheckpointTransportFailure:
            self._discarded = True
            raise
        except BaseException as error:
            if self._topology.rank == 0:
                try:
                    transaction.cleanup_owned()
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note("rank-zero checkpoint cleanup also failed: %s" % cleanup_error)
            self._discarded = True
            raise
        if type(proof) is not _CheckpointPayloadProof or proof.transaction is not transaction:
            error = RuntimeError("RuntimeInstance returned no exact checkpoint payload proof")
            if self._topology.rank == 0:
                failures = []
                if type(proof) is _CheckpointPayloadProof:
                    try:
                        proof.close()
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                try:
                    transaction.cleanup_owned()
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
                if failures:
                    _append_exception_note(
                        error,
                        "rank-zero checkpoint cleanup also failed: "
                        + "; ".join(str(item) for item in failures)
                    )
            self._discarded = True
            raise error
        self._proof = proof
        self._staging_owned = True

    @property
    def path(self) -> Path:
        if self._proof is None:
            raise RuntimeError("restart snapshot has no checkpoint payload proof")
        return self._proof.path

    @staticmethod
    def _target_directory_flags() -> int:
        return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def _close_published_root(self) -> list[BaseException]:
        failures = []
        entry = self._published_entry
        self._published_entry = None
        if entry is not None:
            try:
                entry.close()
            except BaseException as error:
                failures.append(error)
        descriptor = self._published_parent_fd
        self._published_parent_fd = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                failures.append(error)
        return failures

    def _quarantine_published_root(self, *, phase: str) -> None:
        entry = self._published_entry
        descriptor = self._published_parent_fd
        target = self._published_target
        self._published_entry = None
        self._published_parent_fd = None
        self._published_target = None
        if entry is None or descriptor is None or target is None:
            failures = []
            if entry is not None:
                try:
                    entry.close()
                except BaseException as error:
                    failures.append(error)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as error:
                    failures.append(error)
            _raise_cleanup_failures("published checkpoint authority cleanup failed", failures)
            return
        from ._writers.common import _StagedOutputFile

        primary: BaseException | None = None
        try:
            _StagedOutputFile._quarantine_owned_path(
                target,
                entry.owner,
                replaced_message=(
                    "checkpoint %s refuses to delete replaced target %s" % (phase, target)
                ),
                directory_fd=descriptor,
            )
        except BaseException as error:
            primary = error
        finally:
            failures = []
            try:
                entry.close()
            except BaseException as error:
                failures.append(error)
            try:
                os.close(descriptor)
            except BaseException as error:
                failures.append(error)
        if primary is not None:
            if failures:
                add_note = getattr(primary, "add_note", None)
                if callable(add_note):
                    add_note(
                        "published checkpoint descriptor cleanup also failed: "
                        + "; ".join(str(error) for error in failures)
                    )
            raise primary
        _raise_cleanup_failures("published checkpoint descriptor cleanup failed", failures)

    def _cleanup_root(self, *, include_published: bool) -> None:
        failures = []
        proof = self._proof
        if self._staging_owned and proof is not None:
            self._staging_owned = False
            try:
                proof.transaction.quarantine_entry_at(
                    proof.entry,
                    phase="snapshot staging cleanup",
                )
            except BaseException as error:
                failures.append(error)
        elif proof is not None:
            try:
                proof.close()
            except BaseException as error:
                failures.append(error)
        if include_published and self._published_target is not None:
            try:
                self._quarantine_published_root(phase="snapshot rollback")
            except BaseException as error:
                failures.append(error)
        if proof is not None:
            try:
                proof.transaction.cleanup_empty()
            except BaseException as error:
                failures.append(error)
        if include_published:
            failures.extend(self._close_published_root())
        _raise_cleanup_failures("checkpoint snapshot cleanup failed", failures)

    def publish(self, target: Any) -> Path:
        from ._checkpoint_collective import root_attempt

        if self._discarded:
            raise RuntimeError("discarded restart snapshot cannot be published")
        if self._proof is None:
            raise RuntimeError("restart snapshot has no checkpoint payload proof")
        local_target = canonical_checkpoint_path(target)
        target_error = None
        try:
            rows = consensus(
                self._topology,
                "target agreement",
                value=str(local_target),
            )
            if any(row["value"] != str(local_target) for row in rows):
                raise ValueError("checkpoint target differs across ranks")
            if self._published_target is not None and self._published_target != local_target:
                raise ValueError("restart snapshot was already published to another target")
        except BaseException as error:
            target_error = error
        if target_error is not None:
            if self._topology.rank == 0:
                try:
                    self._cleanup_root(include_published=True)
                except BaseException as cleanup_error:
                    add_note = getattr(target_error, "add_note", None)
                    if callable(add_note):
                        add_note("rank-zero checkpoint cleanup also failed: %s" % cleanup_error)
            self._discarded = True
            raise target_error
        if self._published_target is not None:
            return self._published_target

        def publish_root() -> dict[str, Any]:
            proof = self._proof
            if proof is None or not self._staging_owned:
                raise RuntimeError("restart snapshot has no owned staging proof")
            local_target.parent.mkdir(parents=True, exist_ok=True)
            parent_fd = os.open(local_target.parent, self._target_directory_flags())
            linked = False
            primary: BaseException | None = None

            def authenticate_target_at() -> None:
                named = os.stat(local_target.name, dir_fd=parent_fd, follow_symlinks=False)
                retained = os.fstat(proof.entry.fileno())
                if (
                    not stat.S_ISREG(named.st_mode)
                    or _owner(named) != proof.owner
                    or _owner(retained) != proof.owner
                ):
                    raise RuntimeError("checkpoint publication differs from its retained proof")

            try:
                proof.transaction.authenticate_entry_at(proof.entry)
                os.link(
                    proof.entry.name,
                    local_target.name,
                    src_dir_fd=proof.transaction.directory_fileno(),
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                linked = True
                authenticate_target_at()
                proof.transaction.quarantine_entry_at(
                    proof.entry,
                    phase="successful staging cleanup",
                    close_entry=False,
                )
                self._staging_owned = False
                proof.transaction.cleanup_empty()
                # Re-authenticate immediately before handing the still-open inode to the
                # compensatable published state; publication never reopens the target by path.
                authenticate_target_at()
                self._published_entry = proof.entry.transfer(local_target.name)
                self._published_parent_fd = parent_fd
                parent_fd = -1
                self._published_target = local_target
            except FileExistsError as error:
                primary = FileExistsError("checkpoint target collision: %s" % local_target)
                raise primary from error
            except BaseException as error:
                primary = error
                if linked:
                    try:
                        from ._writers.common import _StagedOutputFile

                        _StagedOutputFile._quarantine_owned_path(
                            local_target,
                            proof.owner,
                            replaced_message=(
                                "checkpoint failed publication refuses replaced target %s"
                                % local_target
                            ),
                            directory_fd=parent_fd,
                        )
                    except BaseException as cleanup_error:
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note("failed checkpoint publication cleanup: %s" % cleanup_error)
                raise
            finally:
                if parent_fd >= 0:
                    try:
                        os.close(parent_fd)
                    except BaseException as close_error:
                        if primary is None:
                            raise
                        add_note = getattr(primary, "add_note", None)
                        if callable(add_note):
                            add_note(
                                "checkpoint publication descriptor cleanup also failed: %s"
                                % close_error
                            )
            return {
                "target": str(local_target),
                "entry_owner": list(self._published_entry.owner),
            }

        attempt = root_attempt(self._topology, "publication", publish_root)
        if attempt.transport_error is not None:
            error = _CheckpointTransportFailure(
                "checkpoint transport failed during publication: %s" % attempt.transport_error
            )
            if attempt.producer_error is not None:
                _append_exception_note(
                    error, "rank-zero producer also failed: %s" % attempt.producer_error
                )
            if self._topology.rank == 0:
                try:
                    self._cleanup_root(include_published=True)
                except BaseException as cleanup_error:
                    _append_exception_note(
                        error,
                        "rank-zero checkpoint cleanup also failed: %s" % cleanup_error,
                    )
            self._discarded = True
            raise error from attempt.transport_error
        if attempt.producer_error is not None:
            error = attempt.producer_error
            cleanup = root_attempt(
                self._topology,
                "failed publication cleanup",
                lambda: self._cleanup_root(include_published=True),
            )
            cleanup_errors = tuple(
                item
                for item in (cleanup.producer_error, cleanup.transport_error)
                if item is not None
            )
            if cleanup_errors:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "checkpoint publication cleanup also failed: "
                        + "; ".join(str(item) for item in cleanup_errors)
                    )
            self._discarded = True
            raise error
        publication = attempt.value
        publication_error = None
        owner = None
        try:
            if not isinstance(publication, dict) or set(publication) != {
                "target",
                "entry_owner",
            }:
                raise RuntimeError("checkpoint publication returned invalid ownership evidence")
            if Path(publication["target"]) != local_target:
                raise RuntimeError("checkpoint publication returned a different target")
            owner = _validate_owner(publication["entry_owner"], where="published checkpoint owner")
        except BaseException as error:
            publication_error = error
        try:
            consensus(
                self._topology,
                "publication ownership proof",
                error=publication_error,
                value=publication,
            )
        except BaseException as error:
            if self._topology.rank == 0:
                try:
                    self._cleanup_root(include_published=True)
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note("rank-zero checkpoint cleanup also failed: %s" % cleanup_error)
            self._discarded = True
            raise
        if owner is None:
            raise RuntimeError("checkpoint publication proof validation returned no owner")
        if self._topology.rank != 0:
            self._published_target = local_target
            self._published_entry = _CheckpointEntryAuthority(local_target.name, owner, None)
        return local_target

    def _finish_cleanup(self, *, phase: str, include_published: bool) -> None:
        from ._checkpoint_collective import root_attempt

        attempt = root_attempt(
            self._topology,
            phase,
            lambda: self._cleanup_root(include_published=include_published),
        )
        self._discarded = True
        if attempt.transport_error is not None:
            error = _CheckpointTransportFailure(
                "checkpoint transport failed during %s: %s" % (phase, attempt.transport_error)
            )
            if attempt.producer_error is not None:
                _append_exception_note(
                    error, "rank-zero cleanup also failed: %s" % attempt.producer_error
                )
            raise error from attempt.transport_error
        if attempt.producer_error is not None:
            raise attempt.producer_error

    def discard(self) -> None:
        if self._discarded or self._published_target is not None:
            return
        self._finish_cleanup(phase="discard", include_published=False)

    def rollback(self) -> None:
        if self._discarded:
            return
        self._finish_cleanup(phase="rollback", include_published=True)

    def finalize(self) -> None:
        failures = []
        if self._proof is not None:
            try:
                self._proof.close()
            except BaseException as error:
                failures.append(error)
            try:
                self._proof.transaction.close()
            except BaseException as error:
                failures.append(error)
        failures.extend(self._close_published_root())
        _raise_cleanup_failures("checkpoint snapshot finalization failed", failures)

    def __del__(self) -> None:
        try:
            self.finalize()
        except BaseException:
            pass


@dataclass(frozen=True, slots=True)
class RestartV3:
    """Compatibility-named adapter over strict Uniform v9 / AMR v12 accepted-state payloads."""

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
        from ._checkpoint_collective import _bounded_checkpoint_path_bytes, root_bytes
        from ._checkpoint_contract import require_checkpoint_resource_budget

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
        archive_budget = require_checkpoint_resource_budget(runtime).max_archive_bytes

        def read_and_authenticate_root() -> dict[str, Any]:
            nonlocal root_payload
            root_payload = _bounded_checkpoint_path_bytes(target, archive_budget)
            cursors = runtime._inspect_checkpoint_payload(root_payload)
            return cursors.to_data()

        cursor_data = root_value(
            topology, "restart read and authentication", read_and_authenticate_root
        )
        payload = root_bytes(
            topology,
            "restart payload broadcast",
            lambda: root_payload,
            max_bytes=archive_budget,
        )
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
