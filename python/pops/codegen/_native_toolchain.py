"""Exact metadata authority for an explicitly selected native platform SDK.

An SDK directory is never an include authority.  Darwin's compiler reports its
SDKSettings.json alongside headers in a depfile; authenticate that one declared
input before compiling, include its bytes in artifact identity, and check again
before publication.  Other platforms retain their existing compiler flags.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any


def _platform_name() -> str:
    return sys.platform


@dataclass(frozen=True, slots=True)
class CapturedNativeToolchain:
    """Immutable file authority captured from the caller's declared SDKROOT."""

    sysroot: str
    metadata_path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if type(self.sysroot) is not str or not os.path.isabs(self.sysroot):
            raise ValueError("captured native sysroot must be an absolute path")
        if type(self.metadata_path) is not str or self.metadata_path != os.path.join(
            self.sysroot, "SDKSettings.json"
        ):
            raise ValueError("captured native metadata must be the declared SDK's exact file")
        if type(self.sha256) is not str or len(self.sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.sha256
        ):
            raise ValueError("captured native metadata requires a lowercase SHA-256 digest")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("captured native metadata requires an exact non-negative size")

    def verify(self) -> frozenset[str]:
        root = Path(self.sysroot)
        metadata = Path(self.metadata_path)
        if (not root.is_dir() or os.path.realpath(root) != self.sysroot
                or metadata.is_symlink() or not metadata.is_file()
                or os.path.realpath(metadata) != self.metadata_path):
            raise RuntimeError("declared native SDK metadata path changed or is unavailable")
        try:
            data = metadata.read_bytes()
        except OSError as exc:
            raise RuntimeError("declared native SDK metadata is unavailable") from exc
        if len(data) != self.size or hashlib.sha256(data).hexdigest() != self.sha256:
            raise RuntimeError("declared native SDK metadata changed after capture")
        return frozenset((self.metadata_path,))

    def to_data(self) -> dict[str, Any]:
        self.verify()
        return {
            "contract": "pops.native-toolchain-metadata@1",
            "sysroot": self.sysroot,
            "files": [{"path": "SDKSettings.json", "sha256": self.sha256, "size": self.size}],
        }


def capture_native_toolchain(environment: Any = None) -> CapturedNativeToolchain | None:
    """Capture declared SDK metadata, never infer authority from observed dependencies."""
    if _platform_name() != "darwin":
        return None
    source = os.environ if environment is None else environment
    declared = source.get("SDKROOT")
    if (type(declared) is not str or not declared or "\0" in declared
            or not os.path.isabs(declared)):
        raise ValueError("Darwin native compilation requires an explicitly declared absolute SDKROOT")
    root = Path(os.path.realpath(declared))
    metadata = root / "SDKSettings.json"
    if not root.is_dir() or metadata.is_symlink() or not metadata.is_file():
        raise ValueError("declared native SDK metadata must be an existing regular file")
    try:
        data = metadata.read_bytes()
    except OSError as exc:
        raise ValueError("declared native SDK metadata is unavailable") from exc
    capture = CapturedNativeToolchain(
        str(root), str(metadata), hashlib.sha256(data).hexdigest(), len(data)
    )
    capture.verify()
    return capture


def verify_native_toolchain(capture: Any) -> frozenset[str]:
    if capture is None:
        return frozenset()
    if type(capture) is not CapturedNativeToolchain:
        raise TypeError("native toolchain authority must be an exact captured SDK proof")
    return capture.verify()


def native_toolchain_flags(capture: Any, flags: Any) -> list[str]:
    """Pin the captured sysroot explicitly without opening an SDK include directory."""
    if type(flags) not in (list, tuple) or any(type(flag) is not str for flag in flags):
        raise TypeError("native compiler flags must be an exact text sequence")
    verify_native_toolchain(capture)
    if capture is None:
        return list(flags)
    for index, token in enumerate(flags):
        if token == "-Xclang":
            if index + 1 == len(flags):
                raise ValueError("native compiler forwarding flag has no argument")
            forwarded = flags[index + 1]
        elif token.startswith("-Xclang="):
            forwarded = token[len("-Xclang="):]
        else:
            continue
        # cc1 receives these *after* the driver's direct -isysroot, irrespective of
        # their order in the driver argv. Never let native MPI forwarding replace it.
        if forwarded.startswith(("-isysroot", "--sysroot", "@")):
            raise ValueError("forwarded native compiler sysroot/response selectors are not supported")
    index = 0
    selected: list[str] = []
    while index < len(flags):
        token = flags[index]
        path = None
        if token in ("-isysroot", "--sysroot"):
            index += 1
            if index == len(flags):
                raise ValueError("native compiler sysroot flag has no path")
            path = flags[index]
        elif token.startswith("--sysroot="):
            path = token[len("--sysroot="):]
        elif token.startswith("-isysroot"):
            path = token[len("-isysroot"):].removeprefix("=")
        if path is not None:
            if not os.path.isabs(path) or os.path.realpath(path) != capture.sysroot:
                raise ValueError("native compiler sysroot differs from the captured SDKROOT")
        else:
            selected.append(token)
        index += 1
    return [*selected, "-isysroot", capture.sysroot]
