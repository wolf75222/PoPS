"""Backend-independent scientific-output identity and publication transaction."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pops.identity import Identity, make_identity

from pops.output.data import OutputRequest, OutputSnapshot, array_evidence
from pops._native_collectives import (
    barrier as native_barrier,
    broadcast_value,
    rank as native_rank,
    require_world,
    size as native_size,
)


OUTPUT_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _exception_text(error: BaseException) -> str:
    text = "%s: %s" % (type(error).__name__, error)
    notes = tuple(
        note for note in getattr(error, "__notes__", ())
        if isinstance(note, str) and note
    )
    return text if not notes else text + "; " + "; ".join(notes)


def _rename_no_replace(
    source: str,
    destination: str,
    *,
    src_dir_fd: int,
    dst_dir_fd: int,
) -> None:
    """Atomically rename without replacement on the supported Linux/macOS contract."""
    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source, encoded_destination = os.fsencode(source), os.fsencode(destination)
    syscall_number = None
    if sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        flags = 0x00000004  # RENAME_EXCL from <sys/stdio.h>
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        flags = 0x00000001  # RENAME_NOREPLACE from <linux/fs.h>
        if operation is None:
            # Older glibc releases may omit the wrapper even when the running kernel implements
            # renameat2.  These are the two Linux architectures supported by the PoPS release
            # wheels; use the kernel ABI directly before declaring the guarantee unavailable.
            syscall_number = {
                "x86_64": 316,
                "amd64": 316,
                "aarch64": 276,
                "arm64": 276,
            }.get(os.uname().machine.lower())
            operation = getattr(libc, "syscall", None) if syscall_number is not None else None
    else:  # pragma: no cover - the package contract currently targets Linux and macOS.
        operation = None
        flags = 0
    if operation is None:
        raise RuntimeError(
            "scientific output atomic quarantine requires renameat2 or renameatx_np")
    if syscall_number is None:
        operation.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        )
        operation.restype = ctypes.c_int
        result = operation(
            src_dir_fd, encoded_source, dst_dir_fd, encoded_destination, flags)
    else:
        operation.argtypes = (
            ctypes.c_long, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
        )
        operation.restype = ctypes.c_long
        result = operation(
            syscall_number, src_dir_fd, encoded_source,
            dst_dir_fd, encoded_destination, flags)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), source, None, destination)


def writer_execution_capability(
    execution_context: Any,
    mode: Any,
    *,
    provider_id: str,
) -> dict[str, Any]:
    """Authenticate one writer's topology without consulting process-global MPI state."""
    from pops._platform_contracts import ExecutionContext
    from pops.output._consumer_contracts import ParallelMode

    if type(execution_context) is not ExecutionContext:
        raise TypeError("writer preflight requires an exact ExecutionContext")
    if type(mode) is not ParallelMode:
        raise TypeError("writer preflight requires an exact ParallelMode")
    if not isinstance(provider_id, str) or not provider_id or provider_id.strip() != provider_id:
        raise TypeError("writer preflight provider_id must be canonical text")
    communicator = execution_context.communicator
    serial = communicator.identity == "serial"
    handle = communicator.handle
    if serial:
        if handle is not None:
            raise ValueError("serial writer preflight rejects a hidden communicator handle")
        size = 1
    else:
        native = require_world(handle)
        size = native_size(native)
    if mode is ParallelMode.SERIAL and not serial:
        raise ValueError("SERIAL scientific output requires a serial ExecutionContext")
    if mode is not ParallelMode.SERIAL and serial:
        raise ValueError(
            "%s scientific output requires a distributed ExecutionContext" % mode.name)
    return {
        "schema_version": 1,
        "provider_id": provider_id,
        "parallel_mode": mode.value,
        "communicator": communicator.identity,
        "size": size,
    }


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def identity_from_token(token: Any, domain: str, where: str) -> Identity:
    try:
        result = Identity.from_token(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s has an invalid identity" % where) from exc
    if result.domain != domain:
        raise ValueError("%s must use the %r identity domain" % (where, domain))
    return result


def deterministic_target(
    directory: Any,
    prefix: Any,
    request: OutputRequest,
    snapshot: OutputSnapshot,
    extension: str,
) -> Path:
    """Return the sole deterministic, filesystem-bounded output filename.

    Human-readable prefixes are deliberately bounded, while the digest covers every full
    identity-bearing input.  Long consumer or clock identities therefore cannot exceed common
    ``NAME_MAX`` limits and cannot collide merely because their readable prefixes are equal.
    """
    root = Path(directory)
    clean_prefix = _SAFE_NAME.sub("-", str(prefix)).strip("-")
    clean_consumer = _SAFE_NAME.sub("-", request.consumer_id).strip("-")
    clean_clock = _SAFE_NAME.sub("-", snapshot.clock.clock_id).strip("-")
    if not clean_prefix or not clean_consumer or not clean_clock:
        raise ValueError("output filename parts must contain a safe non-empty token")
    if (not extension.startswith(".") or "/" in extension or "\\" in extension
            or len(extension.encode("utf-8")) > 32):
        raise ValueError("output extension must be a simple suffix")
    target_identity = make_identity("scientific-output-target", {
        "prefix": str(prefix),
        "consumer_id": request.consumer_id,
        "clock": snapshot.clock.to_data(),
        "publication_selection": request.publication_data(),
        "extension": extension,
    })
    from pops.output._consumer_contracts import ParallelMode

    rank_part = (
        "__r%06d" % request.rank
        if request.parallel_mode is ParallelMode.PER_RANK else ""
    )
    name = "%s__%s__s%09d%s__%s%s" % (
        clean_prefix[:40],
        clean_consumer[:40],
        snapshot.clock.macro_step,
        rank_part,
        target_identity.hexdigest,
        extension,
    )
    return root / name


def manifest(
    format_name: str,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    arrays: dict[str, Any],
    *,
    snapshot_data: dict[str, Any] | None = None,
    datasets: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Identity]:
    base = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "format": format_name,
        "snapshot": snapshot_data if snapshot_data is not None else snapshot.to_data(request),
        "datasets": datasets or {},
        "arrays": {name: arrays[name] for name in sorted(arrays)},
    }
    identity = make_identity("scientific-output", base)
    return dict(base, output_identity=identity.token), identity


def authenticate_manifest(
    value: Any,
    format_name: str,
) -> tuple[dict[str, Any], Identity]:
    if not isinstance(value, dict):
        raise TypeError("scientific output manifest must be a mapping")
    required = {
        "schema_version", "format", "snapshot", "datasets", "arrays", "output_identity",
    }
    if set(value) != required:
        raise ValueError("scientific output manifest keys are not exact")
    if value["schema_version"] != OUTPUT_SCHEMA_VERSION or value["format"] != format_name:
        raise ValueError("scientific output schema/format mismatch")
    supplied = identity_from_token(
        value["output_identity"], "scientific-output", "output_identity")
    base = {key: value[key] for key in required - {"output_identity"}}
    expected = make_identity("scientific-output", base)
    if supplied != expected:
        raise ValueError("scientific output manifest identity mismatch")
    return value, expected


@dataclass(frozen=True, slots=True)
class OutputPublicationReceipt:
    path: Path
    format: str
    output_identity: Identity
    selection_identity: Identity


class WriterSession(Protocol):
    """Public structural transaction returned by a scientific-output writer.

    Implementations need not inherit from this protocol.  Construction and ``authority`` access
    must be effect-free; the runtime authenticates every rank before calling ``stage``.
    """

    @property
    def authority(self) -> dict[str, Any]: ...

    @property
    def identity(self) -> Identity: ...

    def stage(self) -> None: ...

    def abort_prepare(self) -> None: ...

    def publish(self) -> OutputPublicationReceipt | None: ...

    def rollback(self) -> None: ...

    def finalize(self) -> None: ...


class ScientificWriter(Protocol):
    """Public structural writer protocol implemented by format providers."""

    def preflight(self, execution_context: Any) -> dict[str, Any]: ...

    def prepare_session(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> WriterSession: ...


_SESSION_AUTHORITY_KEYS = frozenset({
    "schema_version", "session_identity", "format", "parallel_mode", "rank", "size",
    "role", "target", "selection_identity",
})


def writer_session_authority(
    format_name: str,
    request: OutputRequest,
    target: Any,
) -> dict[str, Any]:
    """Build canonical effect-free authority for one rank's writer session."""
    from pops.output._consumer_contracts import ParallelMode

    if not isinstance(format_name, str) or not format_name \
            or format_name.strip() != format_name:
        raise TypeError("writer session format must be canonical text")
    if type(request) is not OutputRequest:
        raise TypeError("writer session authority requires an exact OutputRequest")
    path = Path(target).expanduser().resolve()
    mode = request.parallel_mode
    role = {
        ParallelMode.SERIAL: "serial",
        ParallelMode.ROOT: "root" if request.rank == 0 else "participant",
        ParallelMode.COLLECTIVE: "collective",
        ParallelMode.PER_RANK: "local",
    }[mode]
    base = {
        "schema_version": 1,
        "format": format_name,
        "parallel_mode": mode.value,
        "rank": request.rank,
        "size": request.size,
        "role": role,
        "target": path.as_posix(),
        "selection_identity": request.publication_identity.token,
    }
    identity = make_identity("scientific-output-writer-session", base)
    return dict(base, session_identity=identity.token)


def authenticate_writer_session(session: Any) -> dict[str, Any]:
    """Validate a writer session structurally without naming a concrete backend type."""
    required_methods = ("stage", "abort_prepare", "publish", "rollback", "finalize")
    if any(not callable(getattr(session, name, None)) for name in required_methods):
        raise TypeError(
            "scientific output session must implement stage(), abort_prepare(), "
            "publish(), rollback(), and finalize()")
    first = getattr(session, "authority", None)
    second = getattr(session, "authority", None)
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("writer session authority must be one deterministic dict")
    if set(first) != _SESSION_AUTHORITY_KEYS:
        raise ValueError("writer session authority keys are not exact")
    if first["schema_version"] != 1:
        raise ValueError("writer session authority schema_version must be 1")
    if first["parallel_mode"] not in {"serial", "root", "collective", "per_rank"}:
        raise ValueError("writer session authority has an invalid parallel mode")
    rank, size = first["rank"], first["size"]
    if type(rank) is not int or type(size) is not int or size < 1 or rank not in range(size):
        raise ValueError("writer session authority has an invalid rank topology")
    expected_role = {
        "serial": "serial",
        "root": "root" if rank == 0 else "participant",
        "collective": "collective",
        "per_rank": "local",
    }[first["parallel_mode"]]
    if first["role"] != expected_role:
        raise ValueError("writer session role differs from its parallel mode and rank")
    for key in ("format", "target", "selection_identity"):
        value = first[key]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise TypeError("writer session %s must be canonical text" % key)
    supplied = identity_from_token(
        first["session_identity"],
        "scientific-output-writer-session",
        "writer session identity",
    )
    base = {key: first[key] for key in _SESSION_AUTHORITY_KEYS - {"session_identity"}}
    expected = make_identity("scientific-output-writer-session", base)
    if supplied != expected:
        raise ValueError("writer session identity does not authenticate its authority")
    if getattr(session, "identity", None) != expected:
        raise ValueError("writer session identity property differs from its authority")
    return first


class _StagingAuthority:
    """Exact inode authority retained from the creating ``mkstemp`` descriptor."""

    __slots__ = ("path", "owner", "_fd")

    def __init__(self, path: Any, owner: tuple[int, int], fd: int | None) -> None:
        if type(owner) is not tuple or len(owner) != 2 \
                or any(type(item) is not int or item < 0 for item in owner):
            raise ValueError("staging authority requires an exact inode owner")
        if fd is not None and (isinstance(fd, bool) or not isinstance(fd, int) or fd < 0):
            raise ValueError("staging authority descriptor must be an open integer fd")
        self.path, self.owner, self._fd = Path(path), owner, fd
        if fd is not None:
            actual = os.fstat(fd)
            if (int(actual.st_dev), int(actual.st_ino)) != owner:
                raise ValueError("staging descriptor differs from its inode authority")

    @classmethod
    def created(
        cls,
        target: Path,
        *,
        suffix: str = ".prepared",
    ) -> _StagingAuthority:
        if not isinstance(suffix, str) or not suffix.startswith(".") \
                or "/" in suffix or "\x00" in suffix:
            raise ValueError("staging authority suffix must be one local filename suffix")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".%s." % target.name,
            suffix=suffix,
            dir=str(target.parent),
        )
        try:
            created = os.fstat(descriptor)
            return cls(
                Path(name),
                (int(created.st_dev), int(created.st_ino)),
                descriptor,
            )
        except BaseException:
            os.close(descriptor)
            raise

    @classmethod
    def observed(cls, path: Any, owner: tuple[int, int]) -> _StagingAuthority:
        """Non-owning peer view of rank zero's retained staging descriptor."""
        return cls(path, owner, None)

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def fileno(self) -> int:
        if self._fd is None:
            raise RuntimeError("this rank does not own the staging descriptor")
        return self._fd

    def duplicate(self) -> int:
        return os.dup(self.fileno())

    def authenticate_path(self) -> None:
        try:
            current = self.path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(
                "scientific output staging path disappeared before authority transfer") from error
        if (int(current.st_dev), int(current.st_ino)) != self.owner:
            raise RuntimeError(
                "scientific output staging path was replaced before authority transfer")
        if self._fd is not None:
            descriptor = os.fstat(self._fd)
            if (int(descriptor.st_dev), int(descriptor.st_ino)) != self.owner:
                raise RuntimeError("scientific output staging descriptor authority changed")

    def close(self) -> None:
        descriptor = self._fd
        if descriptor is None:
            return
        self._fd = None
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _QuarantineRecovery:
    """Explicit lifecycle for one unauthenticated inode retained after a race."""

    public_path: Path
    quarantine_path: Path
    owner: tuple[int, int]
    directory_owner: tuple[int, int]

    def restore(self) -> None:
        """Restore the retained inode without overwriting a path created after the failure."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory = self.quarantine_path.parent
        parent_descriptor = os.open(directory.parent, flags)
        descriptor = os.open(directory.name, flags, dir_fd=parent_descriptor)
        try:
            current_directory = os.fstat(descriptor)
            if (int(current_directory.st_dev), int(current_directory.st_ino)) \
                    != self.directory_owner:
                raise RuntimeError("scientific output recovery directory authority changed")
            retained = os.stat(
                self.quarantine_path.name, dir_fd=descriptor, follow_symlinks=False)
            if (int(retained.st_dev), int(retained.st_ino)) != self.owner:
                raise RuntimeError("scientific output recovery inode authority changed")
            try:
                os.link(
                    self.quarantine_path.name,
                    self.public_path.name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                public = self.public_path.lstat()
                if (int(public.st_dev), int(public.st_ino)) != self.owner:
                    raise RuntimeError(
                        "scientific output recovery refuses to overwrite the public path") from None
        finally:
            os.close(descriptor)
            os.close(parent_descriptor)

    def cleanup_restored(self) -> None:
        """Drop the retained hardlink only while the same inode remains restored publicly."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory = self.quarantine_path.parent
        parent_descriptor = os.open(directory.parent, flags)
        descriptor = os.open(directory.name, flags, dir_fd=parent_descriptor)
        try:
            current_directory = os.fstat(descriptor)
            if (int(current_directory.st_dev), int(current_directory.st_ino)) \
                    != self.directory_owner:
                raise RuntimeError("scientific output recovery directory authority changed")
            public = self.public_path.lstat()
            if (int(public.st_dev), int(public.st_ino)) != self.owner:
                raise RuntimeError(
                    "scientific output recovery refuses cleanup before exact restoration")
            try:
                retained = os.stat(
                    self.quarantine_path.name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                # A previous call may have unlinked the retained hardlink before rmdir failed.
                # The authenticated public inode is then the remaining recovery authority.
                pass
            else:
                retained_owner = (int(retained.st_dev), int(retained.st_ino))
                if retained_owner != self.owner:
                    raise RuntimeError("scientific output recovery inode authority changed")
                os.unlink(self.quarantine_path.name, dir_fd=descriptor)
            os.rmdir(directory.name, dir_fd=parent_descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_descriptor)


class _OutputRecoveryRequired(RuntimeError):
    def __init__(self, message: str, recovery: _QuarantineRecovery) -> None:
        super().__init__(message)
        self.recovery = recovery


class _StagedOutputFile:
    """Verified temporary scientific file, not yet attached to a consumer effect."""

    __slots__ = (
        "temporary", "target", "format", "output_identity", "selection_identity",
        "_verify", "_published", "_discarded", "_created_target", "_target_owner",
        "_staging", "_temporary_owner", "_communicator", "_recoveries",
    )

    def __init__(
        self,
        temporary: _StagingAuthority,
        target: Any,
        *,
        format: str,
        output_identity: Identity,
        selection_identity: Identity,
        verify: Callable[[Any], Any],
        communicator: Any = None,
    ) -> None:
        if type(temporary) is not _StagingAuthority:
            raise TypeError("staged output requires its exact mkstemp authority")
        self._staging = temporary
        self.temporary, self.target = temporary.path, Path(target)
        self.format = format
        self.output_identity, self.selection_identity = output_identity, selection_identity
        self._verify, self._communicator = verify, communicator
        self._published = self._discarded = False
        self._created_target = False
        self._target_owner: tuple[int, int] | None = None
        self._temporary_owner = temporary.owner
        self._recoveries: list[Any] = []
        if communicator is None and not temporary.is_open:
            raise ValueError("serial scientific output requires its retained mkstemp descriptor")
        try:
            temporary.authenticate_path()
        except BaseException:
            temporary.close()
            raise

    def __del__(self) -> None:
        try:
            self._staging.close()
        except BaseException:
            pass

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return tuple(self._recoveries)

    def cleanup_recoveries(self) -> None:
        failures = []
        remaining = []
        for recovery in self._recoveries:
            try:
                recovery.cleanup_restored()
            except BaseException as error:
                failures.append("%s: %s" % (type(error).__name__, error))
                remaining.append(recovery)
        self._recoveries = remaining
        if failures:
            raise RuntimeError(
                "scientific output recovery cleanup failed: " + "; ".join(failures))

    def _rank(self) -> int:
        return 0 if self._communicator is None else native_rank(self._communicator)

    def _barrier(self) -> None:
        if self._communicator is not None:
            native_barrier(self._communicator)

    @staticmethod
    def _quarantine_owned_path(
        path: Path,
        expected_owner: tuple[int, int] | None,
        *,
        replaced_message: str,
    ) -> None:
        """Atomically detach one path, then delete only from a private quarantine.

        The rename is the authority boundary: a concurrent replacement is moved, authenticated in
        the private directory, and retained for explicit recovery instead of being unlinked.  The
        quarantine directory is mode 0700, randomly named, and all lookups after its creation are
        anchored by directory descriptors.  This removes the public-path ``lstat -> unlink`` race
        on both Linux and macOS.
        """
        if expected_owner is None:
            raise RuntimeError(replaced_message + "; no authenticated inode authority exists")
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        parent_fd = os.open(path.parent, directory_flags)
        quarantine_name = ""
        quarantine_fd: int | None = None
        retain_quarantine = False
        cleanup_error: BaseException | None = None
        try:
            for _attempt in range(8):
                quarantine_name = ".pops-quarantine-%s" % os.urandom(16).hex()
                try:
                    os.mkdir(quarantine_name, 0o700, dir_fd=parent_fd)
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError("scientific output could not allocate a private quarantine")
            quarantine_fd = os.open(
                quarantine_name, directory_flags, dir_fd=parent_fd)
            parent_stat = os.fstat(parent_fd)
            directory_stat = os.fstat(quarantine_fd)
            named_directory = os.stat(
                quarantine_name, dir_fd=parent_fd, follow_symlinks=False)
            directory_owner = (int(directory_stat.st_dev), int(directory_stat.st_ino))
            if not stat.S_ISDIR(directory_stat.st_mode) \
                    or stat.S_IMODE(directory_stat.st_mode) & 0o077 \
                    or directory_owner != (
                        int(named_directory.st_dev), int(named_directory.st_ino)) \
                    or int(directory_stat.st_dev) != int(parent_stat.st_dev):
                retain_quarantine = True
                raise RuntimeError(
                    "scientific output private quarantine failed directory authentication")
            try:
                os.stat("owned", dir_fd=quarantine_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                retain_quarantine = True
                raise RuntimeError(
                    "scientific output private quarantine destination already exists")
            try:
                _rename_no_replace(
                    path.name,
                    "owned",
                    src_dir_fd=parent_fd,
                    dst_dir_fd=quarantine_fd,
                )
            except FileNotFoundError:
                return
            except FileExistsError as error:
                # The earlier lookup is only diagnostic.  The rename itself is the atomic
                # no-clobber boundary, so retain an independently-created destination verbatim.
                retain_quarantine = True
                raise RuntimeError(
                    "scientific output private quarantine destination appeared concurrently at %s"
                    % (path.parent / quarantine_name / "owned")
                ) from error
            # From this point on, every exception must retain the detached inode for recovery.
            retain_quarantine = True
            recovery = path.parent / quarantine_name / "owned"
            try:
                quarantined = os.stat("owned", dir_fd=quarantine_fd, follow_symlinks=False)
                owner = (int(quarantined.st_dev), int(quarantined.st_ino))
                if owner != expected_owner:
                    recovery_authority = _QuarantineRecovery(
                        path, recovery, owner, directory_owner)
                    try:
                        os.link(
                            "owned",
                            path.name,
                            src_dir_fd=quarantine_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        recovery_note = (
                            "the public path is occupied; replacement retained for recovery at %s"
                            % recovery)
                    except OSError as restore_error:
                        recovery_note = (
                            "replacement retained for recovery at %s; restoration failed: %s"
                            % (recovery, restore_error))
                    else:
                        recovery_note = (
                            "replacement restored and recovery authority retained at %s" % recovery)
                    raise _OutputRecoveryRequired(
                        replaced_message + "; " + recovery_note,
                        recovery_authority,
                    )

                # Only the authenticated inode reaches this private, descriptor-anchored unlink.
                os.unlink("owned", dir_fd=quarantine_fd)
                retain_quarantine = False
            except _OutputRecoveryRequired:
                raise
            except BaseException as error:
                recovery_authority = _QuarantineRecovery(
                    path, recovery, expected_owner, directory_owner)
                raise _OutputRecoveryRequired(
                    "%s; cleanup failed after atomic quarantine; recovery authority retained at %s: %s"
                    % (replaced_message, recovery, error),
                    recovery_authority,
                ) from error
        finally:
            try:
                if quarantine_fd is not None:
                    os.close(quarantine_fd)
                if quarantine_name and not retain_quarantine:
                    os.rmdir(quarantine_name, dir_fd=parent_fd)
            except BaseException as error:
                cleanup_error = error
            try:
                os.close(parent_fd)
            except BaseException as error:
                cleanup_error = cleanup_error or error
            if cleanup_error is not None:
                primary = sys.exc_info()[1]
                if primary is not None:
                    add_note = getattr(primary, "add_note", None)
                    if callable(add_note):
                        add_note("quarantine cleanup also failed: %s" % cleanup_error)
                else:
                    raise cleanup_error

    def _remove_temporary_owned(self) -> None:
        try:
            self._quarantine_owned_path(
                self.temporary,
                self._temporary_owner,
                replaced_message=(
                    "scientific output refuses to delete a replaced temporary at %s"
                    % self.temporary),
            )
        except _OutputRecoveryRequired as error:
            self._recoveries.append(error.recovery)
            raise

    def publish(self) -> OutputPublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded output cannot be published")
        if self._published:
            return OutputPublicationReceipt(
                self.target, self.format, self.output_identity, self.selection_identity)
        self._barrier()
        failure = None
        if self._rank() == 0:
            try:
                self._verify(self.temporary)
                self.target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    staged = self.temporary.lstat()
                    owner = (int(staged.st_dev), int(staged.st_ino))
                    if owner != self._temporary_owner:
                        raise RuntimeError(
                            "scientific output staging inode changed before publication")
                    os.link(self.temporary, self.target)
                    self._created_target = True
                    self._target_owner = self._temporary_owner
                    linked = self.target.lstat()
                    linked_owner = (int(linked.st_dev), int(linked.st_ino))
                    if linked_owner != self._target_owner:
                        raise RuntimeError(
                            "scientific output staging inode changed during publication")
                except FileExistsError:
                    if hashlib.sha256(self.temporary.read_bytes()).digest() != hashlib.sha256(
                            self.target.read_bytes()).digest():
                        raise FileExistsError(
                            "scientific output collision at deterministic target %s" % self.target
                        ) from None
                self._remove_temporary_owned()
            except BaseException as exc:
                failure = _exception_text(exc)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        if failure is not None:
            if self._communicator is None and failure.startswith("FileExistsError:"):
                raise FileExistsError(failure.split(": ", 1)[1])
            raise RuntimeError("collective output publication failed: %s" % failure)
        self._barrier()
        self._published = True
        return OutputPublicationReceipt(
            self.target, self.format, self.output_identity, self.selection_identity)

    def discard(self) -> None:
        if self._published or self._discarded:
            return
        self._barrier()
        failure = None
        if self._rank() == 0:
            try:
                self._remove_temporary_owned()
            except BaseException as exc:
                failure = _exception_text(exc)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        self._barrier()
        if failure is not None:
            raise RuntimeError("collective output discard failed: %s" % failure)
        self._staging.close()
        self._discarded = True

    def rollback(self) -> None:
        """Compensate a staged or published output without deleting a pre-existing artifact."""
        if self._discarded:
            return
        self._barrier()
        failure = None
        if self._rank() == 0:
            failures = []
            if self._created_target:
                try:
                    self._quarantine_owned_path(
                        self.target,
                        self._target_owner,
                        replaced_message=(
                            "scientific output rollback refused a replaced target at %s"
                            % self.target),
                    )
                except _OutputRecoveryRequired as exc:
                    self._recoveries.append(exc.recovery)
                    failures.append(_exception_text(exc))
                except BaseException as exc:
                    failures.append(_exception_text(exc))
            try:
                self._remove_temporary_owned()
            except BaseException as exc:
                failures.append(_exception_text(exc))
            if failures:
                failure = "; ".join(failures)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        self._barrier()
        if failure is not None:
            raise RuntimeError("collective output rollback failed: %s" % failure)
        self._staging.close()
        self._published = False
        self._discarded = True

    def finalize(self) -> None:
        """Release rollback inode authority after the outer transaction commits."""
        if not self._published or self._discarded:
            raise RuntimeError("only a published output can release rollback authority")
        self._staging.close()
        return None


class OutputWriterSession:
    """Built-in file-session implementation; custom writers may remain fully structural."""

    __slots__ = (
        "_authority", "_identity", "_stage_file", "_staged", "_aborted", "_finalized",
    )

    def __init__(
        self,
        authority: dict[str, Any],
        stage_file: Callable[[], _StagedOutputFile] | None,
    ) -> None:
        self._authority = dict(authority)
        self._identity = identity_from_token(
            self._authority.get("session_identity"),
            "scientific-output-writer-session",
            "writer session identity",
        )
        self._stage_file = stage_file
        self._staged: _StagedOutputFile | None = None
        self._aborted = False
        self._finalized = False
        authenticate_writer_session(self)

    @property
    def authority(self) -> dict[str, Any]:
        return dict(self._authority)

    @property
    def identity(self) -> Identity:
        return self._identity

    @property
    def temporary(self) -> Path | None:
        return None if self._staged is None else self._staged.temporary

    @property
    def target(self) -> Path:
        return Path(self._authority["target"])

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return () if self._staged is None else self._staged.recoveries

    def cleanup_recoveries(self) -> None:
        if self._staged is not None:
            self._staged.cleanup_recoveries()

    def stage(self) -> None:
        if self._aborted or self._finalized:
            raise RuntimeError("aborted writer session cannot be staged")
        if self._staged is not None or self._stage_file is None:
            return
        staged = self._stage_file()
        if type(staged) is not _StagedOutputFile:
            raise TypeError("built-in writer stage must return its private staged file")
        self._staged = staged

    def abort_prepare(self) -> None:
        if self._aborted:
            return
        if self._staged is not None:
            self._staged.discard()
        self._aborted = True

    def publish(self) -> OutputPublicationReceipt | None:
        if self._stage_file is None:
            return None
        if self._staged is None:
            raise RuntimeError("writer session must be staged before publication")
        return self._staged.publish()

    def rollback(self) -> None:
        if self._finalized:
            raise RuntimeError("finalized writer session cannot be rolled back")
        if self._staged is not None:
            self._staged.rollback()
        self._aborted = True

    def finalize(self) -> None:
        if self._finalized:
            return None
        if self._staged is not None:
            result = self._staged.finalize()
            if result is not None:
                raise TypeError("built-in writer finalize() must return None")
        self._finalized = True
        return None


@dataclass(frozen=True, slots=True)
class ReopenedOutput:
    manifest: dict[str, Any]
    arrays: dict[str, Any]
    output_identity: Identity

    def require_selection(self, request: OutputRequest) -> ReopenedOutput:
        recorded = self.manifest["snapshot"]["selection"]
        if recorded != request.publication_data():
            raise ValueError("reopened output selection differs from the requested selection")
        return self


def temporary_path(target: Path) -> _StagingAuthority:
    """Create and retain the exact staging inode authority from ``mkstemp``."""
    return _StagingAuthority.created(target)


def _cleanup_staging_authority(
    authority: _StagingAuthority,
    *,
    replaced_message: str,
) -> None:
    """Remove a failed staging inode through the same quarantine protocol, then close its fd."""
    try:
        _StagedOutputFile._quarantine_owned_path(
            authority.path,
            authority.owner,
            replaced_message=replaced_message,
        )
    finally:
        authority.close()


def piece_payload(
    snapshot: OutputSnapshot,
    request: OutputRequest,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Encode exact pieces without padding sparse AMR cells into physical values.

    Each piece remains a distinct identity-bearing array and its half-open global bounds are
    retained in the dataset map.  The same representation therefore works for complete serial/root
    snapshots and sparse collective/per-rank snapshots without a second storage convention.
    """
    arrays: dict[str, Any] = {}
    datasets: dict[str, Any] = {"fields": {}, "geometries": {}}
    fields = snapshot.select(request)
    for field_index, field in enumerate(fields):
        pieces = []
        for piece_index, piece in enumerate(field.pieces):
            name = "field_%04d_piece_%04d" % (field_index, piece_index)
            arrays[name] = piece.values
            pieces.append({
                "name": name,
                "lower": list(piece.lower),
                "upper": list(piece.upper),
                "global_box_index": piece.global_box_index,
                "owner_rank": piece.owner_rank,
                "replicated": piece.replicated,
            })
        datasets["fields"][field.key.identity.token] = {
            "global_shape": list(field.global_shape),
            "dtype": field.array_dtype,
            "pieces": pieces,
        }
    geometries = selected_geometries(snapshot, request, fields)
    for index, geometry in enumerate(sorted(geometries.values(), key=lambda item: item.key)):
        coverage = "geometry_%04d_coverage" % index
        valid = "geometry_%04d_valid" % index
        volumes = "geometry_%04d_volumes" % index
        arrays[coverage], arrays[valid], arrays[volumes] = (
            geometry.coverage, geometry.valid_cells, geometry.cell_volumes)
        datasets["geometries"]["%s#%d" % geometry.key] = {
            "coverage": coverage,
            "valid_cells": valid,
            "cell_volumes": volumes,
        }
    evidence = {name: array_evidence(value) for name, value in arrays.items()}
    return arrays, datasets, evidence


def field_values_on_mask(field: Any, mask: Any, *, require_piece_subset: bool) -> Any:
    """Return values in row-major mask order from exact sparse pieces.

    No value is synthesized outside the mask.  This is the common sparse projection used by VTK
    and composite diagnostics; missing or duplicate ownership fails before publication.
    """
    import numpy as np

    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != field.global_shape:
        raise ValueError("field selection mask differs from global_shape")
    ordinals = np.full(field.global_shape, -1, dtype=np.int64)
    ordinals[selected] = np.arange(int(np.count_nonzero(selected)), dtype=np.int64)
    components = len(field.component_names)
    shape = (int(np.count_nonzero(selected)), components) if components else \
        (int(np.count_nonzero(selected)),)
    result = np.empty(shape, dtype=np.dtype(field.array_dtype))
    written = np.zeros(shape[0], dtype=np.bool_)
    for piece in field.pieces:
        jlo, ilo = piece.lower
        jhi, ihi = piece.upper
        local_mask = selected[jlo:jhi, ilo:ihi]
        if require_piece_subset and not np.all(local_mask):
            raise ValueError("field piece extends outside its exact valid geometry mask")
        target = ordinals[jlo:jhi, ilo:ihi][local_mask]
        if np.any(written[target]):
            raise ValueError("field pieces overlap on the selected geometry mask")
        if components:
            result[target, :] = piece.values[:, local_mask].T
        else:
            result[target] = piece.values[local_mask]
        written[target] = True
    if not np.all(written):
        raise ValueError("field pieces do not cover the exact selected geometry mask")
    result.setflags(write=False)
    return result


def validate_field_pieces(
    field: Any,
    geometry: Any,
    *,
    complete: bool,
    rank: int | None = None,
    size: int | None = None,
) -> None:
    """Prove that pieces are a non-overlapping subset/partition of valid geometry boxes."""
    pieces = sorted(field.pieces, key=lambda piece: (piece.lower, piece.upper))
    boxes = tuple(geometry.boxes)
    active: list[Any] = []
    covered = 0
    for piece in pieces:
        jlo, ilo = piece.lower
        jhi, ihi = piece.upper
        if piece.global_box_index >= len(boxes):
            raise ValueError("field global_box_index lies outside exact geometry boxes")
        if (jlo, ilo, jhi, ihi) != boxes[piece.global_box_index]:
            raise ValueError("field piece differs from its indexed exact geometry box")
        if rank is not None and piece.owner_rank != rank:
            raise ValueError("field piece owner_rank differs from its publication rank")
        if size is not None and piece.owner_rank >= size:
            raise ValueError("field piece owner_rank lies outside publication topology")
        if complete and piece.replicated and piece.owner_rank != 0:
            raise ValueError("complete field uses a non-root replicated piece authority")
        active = [other for other in active if other.upper[0] > jlo]
        if any(not (ihi <= other.lower[1] or other.upper[1] <= ilo) for other in active):
            raise ValueError("field pieces overlap")
        active.append(piece)
        covered += (jhi - jlo) * (ihi - ilo)
    if complete:
        expected = sum(
            (jhi - jlo) * (ihi - ilo)
            for jlo, ilo, jhi, ihi in boxes
        )
        if covered != expected:
            raise ValueError("field pieces do not exactly cover the valid geometry boxes")
        if {piece.global_box_index for piece in pieces} != set(range(len(boxes))):
            raise ValueError("field pieces do not authenticate every exact geometry box")


def selected_geometries(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    fields: Any,
) -> dict[Any, Any]:
    geometries = {
        snapshot.geometry(field.key).key: snapshot.geometry(field.key)
        for field in fields
    }
    diagnostic_layouts = {item.layout_identity.token for item in request.diagnostics}
    geometries.update({
        item.key: item
        for item in snapshot.geometries
        if item.layout_identity.token in diagnostic_layouts
    })
    return geometries


__all__ = [
    "OUTPUT_SCHEMA_VERSION", "OutputPublicationReceipt", "OutputWriterSession",
    "ScientificWriter", "WriterSession", "ReopenedOutput",
    "authenticate_writer_session",
    "deterministic_target", "field_values_on_mask", "piece_payload",
    "validate_field_pieces", "writer_execution_capability", "writer_session_authority",
]
