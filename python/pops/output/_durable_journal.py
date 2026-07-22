"""Crash-recoverable storage for accepted observer frames.

The journal deliberately has a tiny three-state protocol::

    prepared -> pending -> delivered

Each transition first creates a hard link in the destination directory, syncs
that directory, and only then removes the source link.  A crash can therefore
leave two links to the same archive, but it cannot turn a committed frame back
into an uncommitted one.  Automatic recovery validates both copies before
removing the older state.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pops.identity import Identity
from pops.output._observer_archive import (
    decode_observer_frame,
    encode_observer_frame,
    observer_archive_identity,
)
from pops.output.observers import ObserverFrame


_FORMAT = "pops-durable-observer-journal"
_SCHEMA_VERSION = 1
_MARKER = "journal.json"
_DELIVERY_AUTHORITY = "delivery.json"
_STATES = ("prepared", "pending", "delivered")
_ARCHIVE_NAME = re.compile(r"^([0-9a-f]{64})\.pfa$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


_MARKER_BYTES = _canonical_json({
    "format": _FORMAT,
    "schema_version": _SCHEMA_VERSION,
})


@dataclass(frozen=True, slots=True)
class _JournalRecord:
    """Authenticated immutable view of one archive in a journal state."""

    frame: ObserverFrame
    frame_identity: Identity
    archive_identity: Identity
    file_sha256: str
    path: Path
    state: str
    _root: Path = field(repr=False)


@dataclass(frozen=True, slots=True)
class DurableJournal:
    """Immutable configuration for one filesystem-backed observer journal.

    ``sync="fsync"`` is the durable mode.  ``sync="none"`` retains atomic
    no-clobber transitions but is intended only for filesystems or tests whose
    durability is managed externally.  Automatic recovery rolls back orphaned
    preparations and completes the cleanup side of committed transitions.
    """

    root: Path
    sync: str = "fsync"
    recover: str = "automatic"
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if type(self.sync) is not str or self.sync not in {"fsync", "none"}:
            raise ValueError("DurableJournal.sync must be exactly 'fsync' or 'none'")
        if type(self.recover) is not str or self.recover not in {"automatic", "manual"}:
            raise ValueError(
                "DurableJournal.recover must be exactly 'automatic' or 'manual'")
        try:
            root = Path(self.root).expanduser().resolve()
        except TypeError as error:
            raise TypeError("DurableJournal.root must be path-like") from error
        object.__setattr__(self, "root", root)
        self._initialize()

    def _directory(self, state_name: str) -> Path:
        return self.root / state_name

    def to_data(self) -> dict[str, Any]:
        """Return the canonical placement/durability policy used by a consumer manifest."""

        return {
            "schema_version": 1,
            "kind": "durable_observer_journal",
            "root": self.root.as_posix(),
            "sync": self.sync,
            "recover": self.recover,
            "delivery": "at_least_once_after_handoff",
        }

    def bind_delivery_authority(self, authority: Any) -> None:
        """Durably bind replay to one exact consumer placement authority.

        A pending archive contains the scientific frame, while this root marker authenticates
        where that frame was handed off.  Reopening the same journal with a different output root
        is rejected instead of silently replaying an old frame into a new location.
        """

        expected_keys = {
            "schema_version", "consumer_id", "manifest_identity", "target_uri",
            "resolved_target",
        }
        if type(authority) is not dict or set(authority) != expected_keys:
            raise TypeError("journal delivery authority has an unsupported schema")
        if authority["schema_version"] != 1:
            raise ValueError("journal delivery authority schema_version must be 1")
        for name in ("consumer_id", "manifest_identity", "target_uri", "resolved_target"):
            value = authority[name]
            if not isinstance(value, str) or not value or value.strip() != value:
                raise TypeError(
                    "journal delivery authority %s must be non-empty canonical text" % name)
        if not Path(authority["resolved_target"]).is_absolute():
            raise ValueError("journal delivery authority resolved_target must be absolute")
        payload = _canonical_json(authority)
        target = self.root / _DELIVERY_AUTHORITY
        try:
            current = self._regular_bytes(target)
        except FileNotFoundError:
            self._publish_exact(target, payload)
            current = self._regular_bytes(target)
        if current != payload:
            raise RuntimeError(
                "DurableJournal is already bound to a different consumer/output authority")

    def _sync_directory(self, directory: Path) -> None:
        if self.sync != "fsync":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _regular_bytes(path: Path) -> bytes:
        try:
            before = os.lstat(path)
        except FileNotFoundError:
            raise
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("journal entries must be regular files: %s" % path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) \
                    or not stat.S_ISREG(opened.st_mode):
                raise RuntimeError("journal entry changed while it was opened: %s" % path)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        return b"".join(chunks)

    def _write_temporary(self, directory: Path, basename: str, payload: bytes) -> Path:
        temporary = directory / (".%s.%s.tmp" % (basename, secrets.token_hex(12)))
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("journal archive write made no progress")
                written += count
            if self.sync == "fsync":
                os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        return temporary

    def _publish_exact(self, target: Path, payload: bytes) -> None:
        temporary = self._write_temporary(target.parent, target.name, payload)
        try:
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                if self._regular_bytes(target) != payload:
                    raise FileExistsError(
                        "journal target already contains different authenticated bytes: %s"
                        % target) from None
            # Sync even when another exact publisher won the no-clobber race:
            # its link may be visible while its own directory fsync is still pending.
            self._sync_directory(target.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            else:
                self._sync_directory(temporary.parent)

    @staticmethod
    def _require_directory(path: Path) -> None:
        information = os.lstat(path)
        if not stat.S_ISDIR(information.st_mode):
            raise ValueError("journal state path is not a directory: %s" % path)

    def _initialize(self) -> None:
        missing = []
        cursor = self.root
        while not cursor.exists():
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                pass
            self._require_directory(directory)
            # Persist every newly introduced path component, not only the leaf. Runtime journals
            # add manifest/rank directories beneath the user root and a crash must not lose one of
            # those parent-directory entries after the leaf itself was fsynced.
            self._sync_directory(directory.parent)
        self._require_directory(self.root)
        created_state = False
        for state_name in _STATES:
            directory = self._directory(state_name)
            if not directory.exists():
                try:
                    directory.mkdir()
                except FileExistsError:
                    pass
                created_state = True
            self._require_directory(directory)
        if created_state:
            self._sync_directory(self.root)

        marker = self.root / _MARKER
        try:
            current = self._regular_bytes(marker)
        except FileNotFoundError:
            self._publish_exact(marker, _MARKER_BYTES)
        else:
            if current != _MARKER_BYTES:
                raise ValueError("DurableJournal root uses an unsupported journal schema")
        if self.recover == "automatic":
            for path in self.root.glob(".%s.*.tmp" % _MARKER):
                self._remove(path)
            self._recover()

    @staticmethod
    def _archive_filename(frame_identity: Identity) -> str:
        if type(frame_identity) is not Identity \
                or frame_identity.domain != "post-commit-observer-frame":
            raise TypeError("journal frame identity has the wrong domain")
        return frame_identity.hexdigest + ".pfa"

    def _load_record(self, path: Path, state_name: str) -> _JournalRecord:
        if state_name not in _STATES or path.parent != self._directory(state_name):
            raise ValueError("journal record path does not belong to its declared state")
        match = _ARCHIVE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("journal archive has a non-canonical filename: %s" % path.name)
        payload = self._regular_bytes(path)
        frame = decode_observer_frame(payload)
        if frame.identity.hexdigest != match.group(1):
            raise ValueError("journal filename does not authenticate its observer frame")
        return _JournalRecord(
            frame,
            frame.identity,
            observer_archive_identity(frame),
            hashlib.sha256(payload).hexdigest(),
            path,
            state_name,
            self.root,
        )

    @staticmethod
    def _same_record(left: _JournalRecord, right: _JournalRecord) -> bool:
        return (
            left.frame_identity == right.frame_identity
            and left.archive_identity == right.archive_identity
            and left.file_sha256 == right.file_sha256
        )

    def _remove(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        self._sync_directory(path.parent)

    def _state_entries(self, state_name: str, *, remove_temporary: bool) -> dict[str, Path]:
        result = {}
        directory = self._directory(state_name)
        for path in directory.iterdir():
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                if remove_temporary:
                    self._remove(path)
                    continue
                raise ValueError("journal contains an unresolved temporary entry: %s" % path)
            if _ARCHIVE_NAME.fullmatch(path.name) is None:
                raise ValueError("journal contains an unknown state entry: %s" % path)
            information = os.lstat(path)
            if not stat.S_ISREG(information.st_mode):
                raise ValueError("journal entries must be regular files: %s" % path)
            result[path.name] = path
        return result

    def _recover(self) -> None:
        entries = {
            state_name: self._state_entries(state_name, remove_temporary=True)
            for state_name in _STATES
        }
        names = set().union(*(set(items) for items in entries.values()))
        for name in sorted(names):
            present = [
                self._load_record(entries[state_name][name], state_name)
                for state_name in _STATES if name in entries[state_name]
            ]
            reference = present[0]
            if any(not self._same_record(reference, item) for item in present[1:]):
                raise ValueError(
                    "journal crash states disagree for observer frame %s" % name)
            states = {item.state: item for item in present}
            if "delivered" in states:
                for stale in ("pending", "prepared"):
                    if stale in states:
                        self._remove(states[stale].path)
            elif "pending" in states:
                if "prepared" in states:
                    self._remove(states["prepared"].path)
            else:
                # Preparation is compensatable.  Without a committed pending
                # link it must be rolled back after a process crash.
                self._remove(states["prepared"].path)

    def _expected(self, record: _JournalRecord) -> tuple[str, _JournalRecord]:
        if type(record) is not _JournalRecord or record._root != self.root:
            raise TypeError("journal transition requires one record from this DurableJournal")
        if record.frame.identity != record.frame_identity \
                or observer_archive_identity(record.frame) != record.archive_identity \
                or _SHA256.fullmatch(record.file_sha256) is None:
            raise ValueError("journal record authentication is inconsistent")
        return self._archive_filename(record.frame_identity), record

    def _existing_exact(
        self,
        state_name: str,
        filename: str,
        expected: _JournalRecord,
    ) -> _JournalRecord | None:
        path = self._directory(state_name) / filename
        try:
            actual = self._load_record(path, state_name)
        except FileNotFoundError:
            return None
        if not self._same_record(actual, expected):
            raise FileExistsError(
                "journal identity collision contains different archive bytes: %s" % path)
        return actual

    def _promote(
        self,
        source_state: str,
        target_state: str,
        filename: str,
        expected: _JournalRecord,
    ) -> _JournalRecord:
        source = self._existing_exact(source_state, filename, expected)
        if source is None:
            raise FileNotFoundError(
                "journal has no %s archive for %s" % (source_state, filename))
        target_path = self._directory(target_state) / filename
        try:
            os.link(source.path, target_path, follow_symlinks=False)
        except FileExistsError:
            target = self._existing_exact(target_state, filename, expected)
            if target is None:  # pragma: no cover - target disappeared during exact inspection
                raise RuntimeError(
                    "journal target disappeared during promotion") from None
        else:
            target = self._load_record(target_path, target_state)
            if not self._same_record(target, expected):  # defensive against external mutation
                raise RuntimeError("journal promotion changed authenticated archive bytes")
        # Establish the destination link durably before removing the source,
        # including the idempotent/racing case where that exact link pre-existed.
        self._sync_directory(target_path.parent)
        current_source = self._existing_exact(source_state, filename, expected)
        if current_source is not None:
            self._remove(current_source.path)
        return target

    def prepare(self, frame: ObserverFrame) -> _JournalRecord:
        """Durably prepare ``frame`` without making it eligible for replay."""
        if type(frame) is not ObserverFrame:
            raise TypeError("DurableJournal.prepare requires an exact ObserverFrame")
        payload = encode_observer_frame(frame)
        expected = _JournalRecord(
            frame,
            frame.identity,
            observer_archive_identity(frame),
            hashlib.sha256(payload).hexdigest(),
            self._directory("prepared") / self._archive_filename(frame.identity),
            "prepared",
            self.root,
        )
        filename = expected.path.name
        for state_name in ("delivered", "pending", "prepared"):
            existing = self._existing_exact(state_name, filename, expected)
            if existing is not None:
                return existing
        self._publish_exact(expected.path, payload)
        return self._existing_exact("prepared", filename, expected) or expected

    def discard_prepared(self, record: _JournalRecord) -> bool:
        """Roll back an uncommitted preparation; repeat calls are harmless."""
        filename, expected = self._expected(record)
        for state_name in ("delivered", "pending"):
            if self._existing_exact(state_name, filename, expected) is not None:
                raise RuntimeError("a committed journal record cannot be discarded")
        prepared = self._existing_exact("prepared", filename, expected)
        if prepared is None:
            return False
        self._remove(prepared.path)
        return True

    def commit(self, record: _JournalRecord) -> _JournalRecord:
        """Atomically make a prepared frame eligible for crash replay."""
        filename, expected = self._expected(record)
        delivered = self._existing_exact("delivered", filename, expected)
        if delivered is not None:
            stale = self._existing_exact("prepared", filename, expected)
            if stale is not None:
                self._remove(stale.path)
            return delivered
        pending = self._existing_exact("pending", filename, expected)
        if pending is not None:
            stale = self._existing_exact("prepared", filename, expected)
            if stale is not None:
                self._remove(stale.path)
            return pending
        return self._promote("prepared", "pending", filename, expected)

    def delivered(self, record: _JournalRecord) -> _JournalRecord:
        """Atomically acknowledge one committed frame; repeat calls are exact."""
        filename, expected = self._expected(record)
        acknowledged = self._existing_exact("delivered", filename, expected)
        if acknowledged is None:
            acknowledged = self._promote("pending", "delivered", filename, expected)
        for stale_state in ("pending", "prepared"):
            stale = self._existing_exact(stale_state, filename, expected)
            if stale is not None:
                self._remove(stale.path)
        return acknowledged

    @staticmethod
    def _record_order(record: _JournalRecord) -> tuple[Any, ...]:
        snapshot = record.frame.snapshot
        clock = snapshot.clock
        return (
            float.fromhex(clock.time_hex),
            clock.macro_step,
            clock.tick,
            clock.level,
            clock.substep,
            clock.stage_index,
            snapshot.provenance.run_identity.token,
            record.frame_identity.token,
        )

    def list_pending(self) -> tuple[_JournalRecord, ...]:
        """Return every committed, unacknowledged frame in deterministic clock order."""

        records = tuple(
            self._load_record(path, "pending")
            for path in self._state_entries("pending", remove_temporary=False).values()
        )
        return tuple(sorted(records, key=self._record_order))

    def list_committed(self) -> tuple[_JournalRecord, ...]:
        """Return pending and acknowledged records for collective replay reconciliation."""

        records = tuple(
            self._load_record(path, state_name)
            for state_name in ("pending", "delivered")
            for path in self._state_entries(state_name, remove_temporary=False).values()
        )
        return tuple(sorted(records, key=self._record_order))


__all__ = ["DurableJournal"]
