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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from pops.identity import Identity, make_identity
from pops._frozen_data import freeze_data, thaw_data

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
_FILE_EVIDENCE_SIZE = 5


def _file_evidence(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return immutable evidence that detects replacement and ordinary in-place mutation."""
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("scientific output entry must be a regular file")
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _validated_file_evidence(value: Any, *, where: str) -> tuple[int, int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != _FILE_EVIDENCE_SIZE \
            or any(isinstance(item, bool) or type(item) is not int or item < 0 for item in value):
        raise TypeError("%s must be exact regular-file evidence" % where)
    return tuple(value)  # type: ignore[return-value]


def _path_file_evidence(path: Path) -> tuple[int, int, int, int, int]:
    return _file_evidence(path.lstat())


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


def _series_representation_data(format_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return artifact-representation evidence without the catalogue toggle itself."""
    data = thaw_data(format_data)
    if type(data) is not dict:
        raise TypeError("scientific output representation must be canonical mapping data")
    options = data.get("options")
    if isinstance(options, Mapping):
        options = dict(options)
        options.pop("series", None)
        data["options"] = options
    return data


def output_series_family_identity(
    format_data: Mapping[str, Any],
    *,
    format_name: str,
    selection: Mapping[str, Any],
    run_identity: str,
) -> Identity:
    """Identify one format/selection/run timeline independently of its sample clock."""
    if not isinstance(format_name, str) or not format_name \
            or format_name.strip() != format_name:
        raise TypeError("scientific output series format must be canonical text")
    if not isinstance(selection, Mapping):
        raise TypeError("scientific output series selection must be canonical data")
    if not isinstance(run_identity, str) or not run_identity \
            or run_identity.strip() != run_identity:
        raise TypeError("scientific output series run identity must be canonical text")
    return make_identity("scientific-output-series-family", {
        "format": format_name,
        "representation": _series_representation_data(format_data),
        "selection": thaw_data(selection),
        "run_identity": run_identity,
    })


def _snapshot_series_family_identity(
    format_data: Mapping[str, Any],
    format_name: str,
    snapshot: OutputSnapshot,
    request: OutputRequest,
) -> Identity:
    return output_series_family_identity(
        format_data,
        format_name=format_name,
        selection=request.publication_data(),
        run_identity=snapshot.provenance.run_identity.token,
    )


def deterministic_target(
    directory: Any,
    prefix: Any,
    request: OutputRequest,
    snapshot: OutputSnapshot,
    extension: str,
    *,
    format_data: Mapping[str, Any] | None = None,
    format_name: str | None = None,
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
        "provenance": snapshot.provenance.to_data(),
        "publication_selection": request.publication_data(),
        "extension": extension,
    })
    if format_data is None:
        format_data = {
            "schema_version": 1,
            "provider_id": "pops.output.unspecified-representation.v1",
            "extension": extension,
            "parallel_mode": request.parallel_mode.value,
            "options": {},
        }
    if format_name is None:
        format_name = "extension:%s" % extension
    family = _snapshot_series_family_identity(
        format_data, format_name, snapshot, request)
    from pops.output._consumer_contracts import ParallelMode

    rank_part = (
        "__r%06d" % request.rank
        if request.parallel_mode is ParallelMode.PER_RANK else ""
    )
    name = "%s__%s__f%s__s%09d%s__%s%s" % (
        clean_prefix[:24],
        clean_consumer[:24],
        family.hexdigest,
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
    file_evidence: tuple[int, int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.file_evidence is not None:
            object.__setattr__(self, "file_evidence", _validated_file_evidence(
                self.file_evidence, where="output publication file_evidence"))


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


class ScientificSeriesCatalog(Protocol):
    """Small structural extension for publishing and reopening one output timeline."""

    def catalog_data(self) -> dict[str, Any]: ...

    def prepare(
        self,
        target: Any,
        snapshot: OutputSnapshot,
        request: OutputRequest,
    ) -> ScientificSeriesPublication: ...

    def reopen(self, path: Any) -> ReopenedSeries: ...


class ScientificSeriesPublication(Protocol):
    """Lightweight retry owner detached from field arrays and writer rollback state."""

    @property
    def authority(self) -> dict[str, Any]: ...

    def publish(self, artifact: Mapping[str, Any]) -> None: ...


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
        "_target_evidence", "_staging", "_temporary_owner", "_communicator", "_recoveries",
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
        self._target_evidence: tuple[int, int, int, int, int] | None = None
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
                self.target, self.format, self.output_identity, self.selection_identity,
                self._target_evidence)
        self._barrier()
        failure = None
        target_evidence = None
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
                target_evidence = _path_file_evidence(self.target)
                self._target_owner = target_evidence[:2]
            except BaseException as exc:
                failure = _exception_text(exc)
        if self._communicator is not None:
            failure = broadcast_value(self._communicator, failure, root=0)
        if failure is not None:
            if self._communicator is None and failure.startswith("FileExistsError:"):
                raise FileExistsError(failure.split(": ", 1)[1])
            raise RuntimeError("collective output publication failed: %s" % failure)
        if self._communicator is not None:
            target_evidence = broadcast_value(
                self._communicator, target_evidence, root=0)
        self._target_evidence = _validated_file_evidence(
            target_evidence, where="published scientific output")
        self._barrier()
        self._published = True
        return OutputPublicationReceipt(
            self.target, self.format, self.output_identity, self.selection_identity,
            self._target_evidence)

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


@dataclass(frozen=True, slots=True)
class SeriesSample:
    """One lazy member; reopening authenticates its content and exact series family."""

    path: Path
    time: float
    macro_step: int
    format: str
    family_scope: str
    _format_data: dict[str, Any] = dataclass_field(repr=False, compare=False)
    _reopen: Callable[[Any], ReopenedOutput] = dataclass_field(repr=False, compare=False)

    def reopen(self) -> ReopenedOutput:
        output = self._reopen(self.path)
        if type(output) is not ReopenedOutput or output.manifest.get("format") != self.format:
            raise TypeError("scientific output series member has the wrong format")
        snapshot = output.manifest.get("snapshot")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("clock"), dict):
            raise ValueError("scientific output series member has no exact clock")
        provenance = snapshot.get("provenance")
        selection = snapshot.get("selection")
        if not isinstance(provenance, dict) or not isinstance(selection, dict):
            raise ValueError("scientific output series member has no exact family evidence")
        run_identity = provenance.get("run_identity")
        if not isinstance(run_identity, str) or not run_identity \
                or run_identity.strip() != run_identity:
            raise ValueError("scientific output series member has no exact family evidence")
        family = output_series_family_identity(
            self._format_data,
            format_name=self.format,
            selection=selection,
            run_identity=run_identity,
        )
        if family.hexdigest != self.family_scope:
            raise ValueError("scientific output series member belongs to another family")
        clock = snapshot["clock"]
        try:
            physical_time = float.fromhex(clock["time"])
            macro_step = clock["macro_step"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("scientific output series member clock is malformed") from exc
        if physical_time != self.time or macro_step != self.macro_step:
            raise ValueError("scientific output series member differs from its recorded clock")
        return output


@dataclass(frozen=True, slots=True)
class ReopenedSeries:
    """Lazy chronological catalogue; ``latest`` or ``verify`` authenticates members."""

    path: Path
    format: str
    samples: tuple[SeriesSample, ...]
    family_scope: str

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(sample.time for sample in self.samples)

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(sample.path for sample in self.samples)

    @property
    def latest(self) -> ReopenedOutput:
        if not self.samples:
            raise RuntimeError("scientific output series has no samples")
        return self.samples[-1].reopen()

    def verify(self) -> ReopenedSeries:
        """Reopen every member exactly while retaining no historical field arrays."""
        for sample in self.samples:
            sample.reopen()
        return self


def _validate_series_extension(extension: Any) -> str:
    if not isinstance(extension, str) or not extension.startswith(".") \
            or "/" in extension or "\\" in extension or extension.endswith(".series"):
        raise TypeError("output series extension must be a canonical file suffix")
    return extension


def _series_filename(extension: str, family_scope: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", family_scope) is None:
        raise ValueError("scientific output series family scope must be one full digest")
    return "series__f%s%s.series" % (family_scope, extension)


def _series_scope_from_path(path: Path, extension: str) -> str:
    match = re.fullmatch(
        r"series__f([0-9a-f]{64})%s\.series" % re.escape(extension),
        path.name,
    )
    if match is None:
        raise ValueError("scientific output series filename has no exact family scope")
    return match.group(1)


def output_series_path(
    directory: Any,
    extension: str,
    family: Identity | None = None,
) -> Path:
    """Return one scoped companion index, discovering it only when unambiguous."""
    extension = _validate_series_extension(extension)
    unresolved_root = Path(directory).expanduser()
    if unresolved_root.is_symlink():
        raise ValueError("logical output target must not be a symbolic link")
    root = unresolved_root.resolve()
    if family is not None:
        if type(family) is not Identity or family.domain != "scientific-output-series-family":
            raise TypeError("output series family must be an exact series-family Identity")
        return root / _series_filename(extension, family.hexdigest)
    candidates = tuple(sorted(
        path for path in root.glob("series__f*%s.series" % extension)
        if not path.is_symlink()
        if re.fullmatch(
            r"series__f[0-9a-f]{64}%s\.series" % re.escape(extension),
            path.name,
        ) is not None
    ))
    if not candidates:
        raise FileNotFoundError("logical output target has no %s time series" % extension)
    if len(candidates) != 1:
        raise ValueError(
            "logical output target has multiple %s time series; pass one exact index path"
            % extension)
    return candidates[0]


def _series_rows(
    document: Any,
    *,
    extension: str,
    family_scope: str,
) -> tuple[dict[str, Any], ...]:
    member_marker = "__f%s__" % family_scope
    if not isinstance(document, dict) or set(document) != {
            "file-series-version", "files"}:
        raise ValueError("scientific output series has an unknown schema")
    if document["file-series-version"] != "1.0" or not isinstance(document["files"], list):
        raise ValueError("scientific output series version/files are invalid")
    rows = []
    for index, row in enumerate(document["files"]):
        if not isinstance(row, dict) or set(row) != {"name", "time"}:
            raise ValueError("scientific output series file %d is malformed" % index)
        name, time = row["name"], row["time"]
        member = Path(name) if isinstance(name, str) else Path()
        if not isinstance(name, str) or not name or member.name != name \
                or not name.endswith(extension) or member_marker not in name:
            raise ValueError(
                "scientific output series file %d is not a local member of its family"
                % index)
        if isinstance(time, bool) or not isinstance(time, (int, float)):
            raise TypeError("scientific output series time %d must be binary64" % index)
        time = float(time)
        if not (float("-inf") < time < float("inf")):
            raise ValueError("scientific output series time %d must be finite" % index)
        rows.append({"name": name, "time": time})
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("scientific output series contains duplicate files")
    if any(right["time"] < left["time"]
           for left, right in zip(rows, rows[1:], strict=False)):
        raise ValueError("scientific output series times are not chronological")
    return tuple(rows)


def reopen_output_series(
    path: Any,
    *,
    extension: str,
    format_name: str,
    format_data: Mapping[str, Any],
    reopen: Callable[[Any], ReopenedOutput],
) -> ReopenedSeries:
    """Open an ``extension.series`` catalogue without materializing historical arrays."""
    extension = _validate_series_extension(extension)
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("scientific output series path must not be a symbolic link")
    if unresolved.is_dir():
        series_path = output_series_path(unresolved, extension)
    else:
        if unresolved.parent.is_symlink():
            raise ValueError("scientific output series parent must not be a symbolic link")
        series_path = unresolved.resolve()
    family_scope = _series_scope_from_path(series_path, extension)
    if series_path.is_symlink() or not series_path.is_file():
        raise ValueError("scientific output series path must be a regular file")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(series_path.parent, directory_flags)
    try:
        owner = os.fstat(directory_fd)
        visible = series_path.parent.lstat()
        if (int(owner.st_dev), int(owner.st_ino)) != (
                int(visible.st_dev), int(visible.st_ino)):
            raise RuntimeError("scientific output series directory authority changed")
        rows, _index_evidence = _read_series_at(
            directory_fd,
            series_path.name,
            extension=extension,
            family_scope=family_scope,
        )
        if not rows:
            raise ValueError("scientific output series contains no samples")
        samples = []
        for row in rows:
            member = series_path.parent / row["name"]
            try:
                _entry_evidence_at(directory_fd, row["name"])
            except ValueError as exc:
                raise ValueError(
                    "scientific output series member must not be a symbolic link"
                ) from exc
            step_match = re.search(r"__s([0-9]+)(?:__r[0-9]+)?__", member.name)
            if step_match is None:
                raise ValueError(
                    "scientific output series member has no deterministic macro step")
            macro_step = int(step_match.group(1))
            physical_time = row["time"]
            samples.append(SeriesSample(
                member,
                physical_time,
                macro_step,
                format_name,
                family_scope,
                _series_representation_data(format_data),
                reopen,
            ))
    finally:
        os.close(directory_fd)
    return ReopenedSeries(series_path, format_name, tuple(samples), family_scope)


def _series_clock_data(snapshot: Mapping[str, Any]) -> tuple[float, tuple[Any, ...]]:
    clock = snapshot.get("clock")
    if not isinstance(clock, Mapping):
        raise ValueError("scientific output series member has no exact clock")
    try:
        physical_time = float.fromhex(clock["time"])
        fraction = clock["fraction"]
        if not isinstance(fraction, list) or len(fraction) != 2 \
                or any(isinstance(item, bool) or type(item) is not int for item in fraction) \
                or fraction[0] < 0 or fraction[1] <= 0 or fraction[0] > fraction[1]:
            raise TypeError
        integer_clock = tuple(clock[name] for name in (
            "tick", "macro_step", "level", "substep", "stage_index",
        ))
        if any(isinstance(item, bool) or type(item) is not int or item < 0
               for item in integer_clock):
            raise TypeError
        stage = clock["stage"]
        if not isinstance(stage, str) or not stage:
            raise TypeError
        exact_fraction = Fraction(fraction[0], fraction[1])
        if (exact_fraction.numerator, exact_fraction.denominator) != tuple(fraction):
            raise TypeError
        order = (
            physical_time,
            integer_clock[0],
            integer_clock[1],
            integer_clock[2],
            integer_clock[3],
            exact_fraction,
            integer_clock[4],
            stage,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("scientific output series member clock is malformed") from exc
    if not (float("-inf") < physical_time < float("inf")):
        raise ValueError("scientific output series member time must be finite")
    return physical_time, order


def _authenticated_series_member(
    path: Path,
    *,
    family_scope: str,
    format_name: str,
    format_data: Mapping[str, Any],
    reopen: Callable[[Any], ReopenedOutput],
) -> tuple[float, tuple[Any, ...]]:
    output = reopen(path)
    if type(output) is not ReopenedOutput or output.manifest.get("format") != format_name:
        raise TypeError("scientific output series received an incompatible artifact")
    snapshot = output.manifest.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("scientific output series member has no exact snapshot")
    provenance = snapshot.get("provenance")
    selection = snapshot.get("selection")
    if not isinstance(provenance, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("scientific output series member has no exact family evidence")
    run_identity = provenance.get("run_identity")
    if not isinstance(run_identity, str) or not run_identity \
            or run_identity.strip() != run_identity:
        raise ValueError("scientific output series member has no exact family evidence")
    family = output_series_family_identity(
        format_data,
        format_name=format_name,
        selection=selection,
        run_identity=run_identity,
    )
    if family.hexdigest != family_scope:
        raise ValueError("scientific output series mixes different format/selection/run families")
    return _series_clock_data(snapshot)


def _regular_entry_at(directory_fd: int, name: str) -> tuple[int, int]:
    value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("scientific output series entry must be a regular file")
    return int(value.st_dev), int(value.st_ino)


def _entry_evidence_at(
    directory_fd: int, name: str,
) -> tuple[int, int, int, int, int]:
    value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    return _file_evidence(value)


def _read_series_at(
    directory_fd: int,
    name: str,
    *,
    extension: str,
    family_scope: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = _file_evidence(os.fstat(descriptor))
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as stream:
            try:
                document = json.load(stream)
            except json.JSONDecodeError as exc:
                raise ValueError("scientific output series is not valid JSON") from exc
        rows = _series_rows(
            document, extension=extension, family_scope=family_scope)
        after = _file_evidence(os.fstat(descriptor))
        if after != before:
            raise RuntimeError("scientific output series changed while it was read")
        return rows, after
    finally:
        os.close(descriptor)


def _write_series_at(
    directory_fd: int,
    series_name: str,
    document: Mapping[str, Any],
    previous_evidence: tuple[int, int, int, int, int] | None,
) -> None:
    temporary_name = None
    temporary_owner = None
    descriptor = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        for _ in range(32):
            candidate = ".%s.%s.tmp" % (series_name, os.urandom(12).hex())
            try:
                descriptor = os.open(candidate, flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            info = os.fstat(descriptor)
            temporary_owner = (int(info.st_dev), int(info.st_ino))
            break
        if descriptor is None or temporary_name is None or temporary_owner is None:
            raise RuntimeError("could not allocate a unique scientific output series staging file")
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if _regular_entry_at(directory_fd, temporary_name) != temporary_owner:
            raise RuntimeError("scientific output series staging inode was replaced")
        try:
            current_evidence = _entry_evidence_at(directory_fd, series_name)
        except FileNotFoundError:
            current_evidence = None
        if current_evidence != previous_evidence:
            raise RuntimeError("scientific output series changed during its atomic update")
        os.replace(
            temporary_name,
            series_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None and temporary_owner is not None:
            try:
                if _regular_entry_at(directory_fd, temporary_name) == temporary_owner:
                    os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def publish_output_series(
    target: Any,
    *,
    extension: str,
    format_name: str,
    format_data: Mapping[str, Any],
    family: Identity,
    clock_data: Mapping[str, Any],
    selection_identity: Identity,
    artifact: Mapping[str, Any],
    reopen: Callable[[Any], ReopenedOutput],
) -> None:
    """Atomically refresh one family-scoped catalogue after accepted publication."""
    extension = _validate_series_extension(extension)
    if type(family) is not Identity or family.domain != "scientific-output-series-family":
        raise TypeError("scientific output series requires its exact family identity")
    if type(selection_identity) is not Identity \
            or selection_identity.domain != "output-publication-selection":
        raise TypeError("scientific output series requires its exact selection identity")
    required_artifact = {
        "output_identity", "selection_identity", "path", "format", "file_evidence",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != required_artifact:
        raise TypeError("scientific output series artifact authority has an unknown schema")
    target = Path(target).expanduser().resolve()
    marker = "__f%s__" % family.hexdigest
    if not target.name.endswith(extension) or marker not in target.name:
        raise ValueError("scientific output target differs from its exact series family")
    if Path(artifact["path"]).expanduser().resolve() != target \
            or artifact["format"] != format_name:
        raise ValueError("scientific output artifact authority differs from its series target")
    if identity_from_token(
            artifact["selection_identity"], "output-publication-selection",
            "series artifact selection") != selection_identity:
        raise ValueError("scientific output artifact selection differs from its series")
    identity_from_token(
        artifact["output_identity"], "scientific-output", "series artifact output")
    expected_target_evidence = _validated_file_evidence(
        artifact["file_evidence"], where="series artifact file_evidence")
    physical_time, current_order = _series_clock_data({"clock": dict(clock_data)})
    series_path = output_series_path(target.parent, extension, family)
    target.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(target.parent, directory_flags)
    lock_fd = None
    try:
        directory_owner = os.fstat(directory_fd)
        visible_directory = target.parent.lstat()
        if (int(visible_directory.st_dev), int(visible_directory.st_ino)) != (
                int(directory_owner.st_dev), int(directory_owner.st_ino)):
            raise RuntimeError("scientific output series directory authority changed")
        if _entry_evidence_at(directory_fd, target.name) != expected_target_evidence:
            raise RuntimeError(
                "accepted scientific output artifact changed before series indexing")
        lock_name = ".%s.lock" % series_path.name
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(lock_name, lock_flags, 0o600, dir_fd=directory_fd)
        lock_info = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_info.st_mode):
            raise ValueError("scientific output series lock must be a regular file")
        lock_owner = (int(lock_info.st_dev), int(lock_info.st_ino))
        try:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runtime is updating this scientific output series") from exc
        if _regular_entry_at(directory_fd, lock_name) != lock_owner:
            raise RuntimeError("scientific output series lock inode was replaced")

        try:
            rows, previous_evidence = _read_series_at(
                directory_fd,
                series_path.name,
                extension=extension,
                family_scope=family.hexdigest,
            )
        except FileNotFoundError:
            rows, previous_evidence = (), None
        indexed = {row["name"]: row["time"] for row in rows}
        scoped_names = []
        for name in os.listdir(directory_fd):
            if marker not in name or not name.endswith(extension):
                continue
            _regular_entry_at(directory_fd, name)
            scoped_names.append(name)
        scoped_names = sorted(scoped_names)
        if target.name not in scoped_names:
            raise RuntimeError("accepted scientific output artifact disappeared before indexing")
        if not set(indexed).issubset(scoped_names):
            raise ValueError("scientific output series references a missing family member")

        unexpected = set(scoped_names) - set(indexed) - {target.name}
        if unexpected:
            recovered = []
            for name in scoped_names:
                member_time, order = _authenticated_series_member(
                    target.parent / name,
                    family_scope=family.hexdigest,
                    format_name=format_name,
                    format_data=format_data,
                    reopen=reopen,
                )
                recovered.append((order, name, member_time))
            recovered.sort(key=lambda item: (item[0], item[1]))
            if len({item[0] for item in recovered}) != len(recovered):
                raise ValueError("scientific output series contains duplicate exact clocks")
            ordered_rows = [
                {"name": name, "time": member_time}
                for _order, name, member_time in recovered
            ]
        else:
            previous = indexed.get(target.name)
            if previous is not None and previous != physical_time:
                raise ValueError("scientific output series file changed its physical time")
            ordered_rows = list(rows)
            if previous is None:
                if ordered_rows and physical_time < ordered_rows[-1]["time"]:
                    raise ValueError("scientific output series cannot append an earlier clock")
                ordered_rows.append({"name": target.name, "time": physical_time})
            del current_order
        document = {"file-series-version": "1.0", "files": ordered_rows}
        if _regular_entry_at(directory_fd, lock_name) != lock_owner:
            raise RuntimeError("scientific output series lock inode changed before commit")
        _write_series_at(directory_fd, series_path.name, document, previous_evidence)
        visible_directory = target.parent.lstat()
        if (int(visible_directory.st_dev), int(visible_directory.st_ino)) != (
                int(directory_owner.st_dev), int(directory_owner.st_ino)):
            raise RuntimeError("scientific output series directory changed during commit")
    finally:
        if lock_fd is not None:
            try:
                import fcntl

                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


_SERIES_PUBLICATION_KEYS = frozenset({
    "schema_version", "catalog_identity", "target", "selection_identity",
    "family_identity", "clock", "publication_identity",
})


def _catalog_identity(catalog_data: Mapping[str, Any]) -> Identity:
    return make_identity("scientific-output-series-catalog", dict(catalog_data))


def _series_publication_authority(
    catalog_data: Mapping[str, Any],
    *,
    target: Path,
    request: OutputRequest,
    family: Identity,
    clock: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "catalog_identity": _catalog_identity(catalog_data).token,
        "target": target.expanduser().resolve().as_posix(),
        "selection_identity": request.publication_identity.token,
        "family_identity": family.token,
        "clock": dict(clock),
    }
    identity = make_identity("scientific-output-series-publication", base)
    return dict(base, publication_identity=identity.token)


class FileSeriesPublication:
    """Immutable, array-free publication retry for one accepted scientific artifact."""

    __slots__ = (
        "_authority", "_target", "_extension", "_format_name", "_format_data",
        "_family", "_clock_data", "_selection_identity", "_reopen",
    )

    def __init__(
        self,
        catalog_data: Mapping[str, Any],
        format_data: Mapping[str, Any],
        *,
        format_name: str,
        target: Any,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        reopen: Callable[[Any], ReopenedOutput],
    ) -> None:
        if type(snapshot) is not OutputSnapshot or type(request) is not OutputRequest:
            raise TypeError("file series preparation requires exact snapshot/request values")
        target_path = Path(target).expanduser().resolve()
        family = _snapshot_series_family_identity(
            format_data, format_name, snapshot, request)
        clock_data = snapshot.clock.to_data()
        object.__setattr__(self, "_authority", freeze_data(_series_publication_authority(
            catalog_data,
            target=target_path,
            request=request,
            family=family,
            clock=clock_data,
        ), "scientific output series publication authority"))
        object.__setattr__(self, "_target", target_path)
        object.__setattr__(self, "_extension", format_data["extension"])
        object.__setattr__(self, "_format_name", format_name)
        object.__setattr__(self, "_format_data", freeze_data(
            dict(format_data), "scientific output series representation"))
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_clock_data", freeze_data(
            clock_data, "scientific output series clock"))
        object.__setattr__(self, "_selection_identity", request.publication_identity)
        object.__setattr__(self, "_reopen", reopen)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("file series publications are immutable")

    @property
    def authority(self) -> dict[str, Any]:
        return thaw_data(self._authority)

    def publish(self, artifact: Mapping[str, Any]) -> None:
        publish_output_series(
            self._target,
            extension=self._extension,
            format_name=self._format_name,
            format_data=thaw_data(self._format_data),
            family=self._family,
            clock_data=thaw_data(self._clock_data),
            selection_identity=self._selection_identity,
            artifact=artifact,
            reopen=self._reopen,
        )


class FileSeriesCatalog:
    """Built-in structural policy for official extension.series catalogues."""

    __slots__ = ("_format_data", "_format_name", "_reopen")

    def __init__(
        self,
        format_data: Mapping[str, Any],
        *,
        format_name: str,
        reopen: Callable[[Any], ReopenedOutput],
    ) -> None:
        if not callable(reopen):
            raise TypeError("file series catalogue requires an exact artifact reader")
        representation = _series_representation_data(format_data)
        extension = representation.get("extension")
        _validate_series_extension(extension)
        if not isinstance(format_name, str) or not format_name:
            raise TypeError("file series catalogue format must be canonical text")
        object.__setattr__(self, "_format_data", freeze_data(
            dict(format_data), "scientific output series representation"))
        object.__setattr__(self, "_format_name", format_name)
        object.__setattr__(self, "_reopen", reopen)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("file series catalogue policies are immutable")

    def catalog_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "catalog_id": "pops.output.file-series.v1",
            "format": self._format_name,
            "representation": _series_representation_data(
                thaw_data(self._format_data)),
        }

    def prepare(
        self,
        target: Any,
        snapshot: OutputSnapshot,
        request: OutputRequest,
    ) -> FileSeriesPublication:
        return FileSeriesPublication(
            self.catalog_data(),
            thaw_data(self._format_data),
            format_name=self._format_name,
            target=target,
            snapshot=snapshot,
            request=request,
            reopen=self._reopen,
        )

    def reopen(self, path: Any) -> ReopenedSeries:
        return reopen_output_series(
            path,
            extension=self._format_data["extension"],
            format_name=self._format_name,
            format_data=thaw_data(self._format_data),
            reopen=self._reopen,
        )


def authenticate_series_catalog(
    catalog: Any,
    format_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a small structural catalogue capability without class branching."""
    for method in ("prepare", "reopen", "catalog_data"):
        if not callable(getattr(catalog, method, None)):
            raise TypeError("scientific output series catalogue lacks %s()" % method)
    first, second = catalog.catalog_data(), catalog.catalog_data()
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("scientific output series catalog_data() must be deterministic")
    if set(first) != {"schema_version", "catalog_id", "format", "representation"} \
            or first["schema_version"] != 1:
        raise ValueError("scientific output series catalogue authority is not exact")
    if not isinstance(first["catalog_id"], str) or not first["catalog_id"] \
            or not isinstance(first["format"], str) or not first["format"]:
        raise TypeError("scientific output series catalogue ids must be canonical text")
    if first["format"] != format_data.get("format_name"):
        raise ValueError(
            "scientific output series catalogue format differs from its format provider")
    if first["representation"] != _series_representation_data(format_data):
        raise ValueError("scientific output series catalogue differs from its format provider")
    json_text(first)
    return first


def authenticate_series_publication(
    publication: Any,
    catalog_data: Mapping[str, Any],
    *,
    target: Any,
    snapshot: OutputSnapshot,
    request: OutputRequest,
) -> dict[str, Any]:
    """Authenticate an effect-free, lightweight series retry before writer I/O."""
    if not callable(getattr(publication, "publish", None)):
        raise TypeError("scientific output series publication lacks publish()")
    first, second = getattr(publication, "authority", None), getattr(
        publication, "authority", None)
    if type(first) is not dict or type(second) is not dict or first != second:
        raise TypeError("scientific output series publication authority must be deterministic")
    if set(first) != _SERIES_PUBLICATION_KEYS or first["schema_version"] != 1:
        raise ValueError("scientific output series publication authority is not exact")
    if first["catalog_identity"] != _catalog_identity(catalog_data).token:
        raise ValueError("scientific output series publication names another catalogue")
    if first["target"] != Path(target).expanduser().resolve().as_posix() \
            or first["selection_identity"] != request.publication_identity.token \
            or first["clock"] != snapshot.clock.to_data():
        raise ValueError("scientific output series publication differs from its accepted sample")
    identity_from_token(
        first["family_identity"], "scientific-output-series-family",
        "series publication family")
    supplied = identity_from_token(
        first["publication_identity"], "scientific-output-series-publication",
        "series publication identity")
    base = {key: first[key] for key in _SERIES_PUBLICATION_KEYS - {"publication_identity"}}
    if supplied != make_identity("scientific-output-series-publication", base):
        raise ValueError("scientific output series publication identity mismatch")
    json_text(first)
    return first


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
    "OUTPUT_SCHEMA_VERSION", "FileSeriesCatalog", "FileSeriesPublication",
    "OutputPublicationReceipt", "OutputWriterSession", "ScientificSeriesCatalog",
    "ScientificSeriesPublication", "ScientificWriter", "WriterSession", "ReopenedOutput",
    "ReopenedSeries", "SeriesSample",
    "authenticate_series_catalog", "authenticate_series_publication",
    "authenticate_writer_session",
    "deterministic_target", "output_series_family_identity", "output_series_path",
    "field_values_on_mask", "piece_payload",
    "validate_field_pieces", "writer_execution_capability", "writer_session_authority",
]
