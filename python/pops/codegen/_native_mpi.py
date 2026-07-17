"""Authenticated replay of the host ``MPI::MPI_CXX`` contract."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset({
    "schema_version",
    "abi_sha256",
    "compiler",
    "standard",
    "include_dirs",
    "compile_options",
    "compile_definitions",
    "link_options",
    "link_libraries",
    "header_paths",
    "header_sha256",
    "library_paths",
    "library_sha256",
})
_RESERVED_DEFINITIONS = frozenset({"POPS_HAS_MPI", "POPS_MPI_ABI"})


@dataclass(frozen=True, slots=True)
class NativeMpiContract:
    abi_sha256: str
    compiler: str
    standard: str
    include_dirs: tuple[str, ...]
    compile_options: tuple[str, ...]
    compile_definitions: tuple[str, ...]
    link_options: tuple[str, ...]
    link_libraries: tuple[str, ...]
    header_paths: tuple[str, ...]
    header_sha256: tuple[str, ...]
    library_paths: tuple[str, ...]
    library_sha256: tuple[str, ...]


def _text_tuple(data: Mapping[str, Any], name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    value = data[name]
    if type(value) is not tuple or any(not isinstance(item, str) or not item for item in value):
        raise RuntimeError("pops._pops.__mpi_contract__[%r] must be a tuple of text" % name)
    if nonempty and not value:
        raise RuntimeError("pops._pops.__mpi_contract__[%r] must not be empty" % name)
    return value


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _abi_material(contract: NativeMpiContract) -> bytes:
    lines = [
        "compiler=%s" % contract.compiler,
        "standard=%s" % contract.standard,
        *("compile_option=%s" % value for value in contract.compile_options),
        *("compile_definition=%s" % value for value in contract.compile_definitions),
        *("link_option=%s" % value for value in contract.link_options),
    ]
    headers = dict(zip(contract.header_paths, contract.header_sha256, strict=True))
    for include in contract.include_dirs:
        lines.append("include=%s" % include)
        header = str(Path(include) / "mpi.h")
        if header in headers:
            lines.append("header=%s;sha256=%s" % (header, headers[header]))
    lines.extend(
        "library=%s;sha256=%s" % pair
        for pair in zip(contract.library_paths, contract.library_sha256, strict=True)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _native_mpi_contract(module: Any) -> NativeMpiContract | None:
    """Validate and reauthenticate the module's concrete MPI development manifest."""
    if module is None:
        return None
    enabled = getattr(module, "__has_mpi__", None)
    raw = getattr(module, "__mpi_contract__", None)
    if enabled is False and raw is None:
        return None
    if enabled is not True:
        raise RuntimeError("loaded pops._pops exposes no exact __has_mpi__ boolean contract")
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
        raise RuntimeError(
            "MPI-enabled pops._pops must expose the exact __mpi_contract__ schema")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise RuntimeError("unsupported pops._pops.__mpi_contract__ schema_version")
    abi = raw["abi_sha256"]
    compiler, standard = raw["compiler"], raw["standard"]
    if not isinstance(abi, str) or _SHA256.fullmatch(abi) is None:
        raise RuntimeError("MPI contract abi_sha256 must be a lowercase SHA-256")
    if not isinstance(compiler, str) or not isinstance(standard, str):
        raise RuntimeError("MPI contract compiler/standard facts must be text")

    contract = NativeMpiContract(
        abi_sha256=abi,
        compiler=compiler,
        standard=standard,
        include_dirs=_text_tuple(raw, "include_dirs", nonempty=True),
        compile_options=_text_tuple(raw, "compile_options"),
        compile_definitions=_text_tuple(raw, "compile_definitions"),
        link_options=_text_tuple(raw, "link_options"),
        link_libraries=_text_tuple(raw, "link_libraries", nonempty=True),
        header_paths=_text_tuple(raw, "header_paths", nonempty=True),
        header_sha256=_text_tuple(raw, "header_sha256", nonempty=True),
        library_paths=_text_tuple(raw, "library_paths", nonempty=True),
        library_sha256=_text_tuple(raw, "library_sha256", nonempty=True),
    )
    serialized_values = (
        contract.compiler, contract.standard, *contract.include_dirs,
        *contract.compile_options, *contract.compile_definitions,
        *contract.link_options, *contract.link_libraries,
        *contract.header_paths, *contract.library_paths,
    )
    if any(any(token in value for token in ("|", ";", "\r", "\n"))
           for value in serialized_values):
        raise RuntimeError("MPI contract contains an ambiguous serialized delimiter")
    if len(set(contract.include_dirs)) != len(contract.include_dirs):
        raise RuntimeError("MPI contract include_dirs must be unique")
    if len(contract.header_paths) != len(contract.header_sha256):
        raise RuntimeError("MPI contract header path/hash arity mismatch")
    if len(contract.library_paths) != len(contract.library_sha256):
        raise RuntimeError("MPI contract library path/hash arity mismatch")
    if contract.link_libraries != contract.library_paths:
        raise RuntimeError("every replayed MPI link library must have an authenticated file hash")
    if any(_SHA256.fullmatch(value) is None
           for value in (*contract.header_sha256, *contract.library_sha256)):
        raise RuntimeError("MPI contract file hashes must be lowercase SHA-256 values")
    reserved = {
        value.split("=", 1)[0] for value in contract.compile_definitions
    } & _RESERVED_DEFINITIONS
    if reserved:
        raise RuntimeError("MPI target contract overrides PoPS-owned definitions: %s"
                           % sorted(reserved))

    for include in contract.include_dirs:
        if not Path(include).is_absolute() or not Path(include).is_dir():
            raise RuntimeError("MPI include directory is unavailable: %s" % include)
    current_headers = tuple(
        str(Path(include) / "mpi.h")
        for include in contract.include_dirs
        if (Path(include) / "mpi.h").is_file()
    )
    if current_headers != contract.header_paths:
        raise RuntimeError("MPI header manifest no longer matches the selected include directories")
    for path in contract.library_paths:
        if not Path(path).is_absolute() or not Path(path).is_file():
            raise RuntimeError("MPI link library is unavailable: %s" % path)
        if Path(path).suffix.lower() == ".a":
            raise RuntimeError(
                "MPI link library is a static archive and cannot be relinked into dynamic PoPS "
                "plugins without creating a second MPI runtime: %s" % path)

    for path, expected in zip(contract.header_paths, contract.header_sha256, strict=True):
        if _sha256(path) != expected:
            raise RuntimeError(
                "MPI header changed in place after pops._pops was built: %s; rebuild PoPS" % path)
    for path, expected in zip(contract.library_paths, contract.library_sha256, strict=True):
        if _sha256(path) != expected:
            raise RuntimeError(
                "MPI library changed in place after pops._pops was built: %s; rebuild PoPS" % path)
    computed_abi = hashlib.sha256(_abi_material(contract)).hexdigest()
    if computed_abi != contract.abi_sha256:
        raise RuntimeError("MPI contract payload does not authenticate its abi_sha256")
    return contract


def native_mpi_build_flags(module: Any) -> tuple[list[str], list[str]]:
    """Replay the exact authenticated ``MPI::MPI_CXX`` compile and link contract."""
    contract = _native_mpi_contract(module)
    if contract is None:
        return [], []
    compile_flags = [
        *contract.compile_options,
        *("-D" + value for value in contract.compile_definitions),
        "-DPOPS_HAS_MPI",
        '-DPOPS_MPI_ABI="%s"' % contract.abi_sha256,
    ]
    for include in contract.include_dirs:
        compile_flags.extend(("-I", include))
    return compile_flags, [*contract.link_options, *contract.link_libraries]


def native_mpi_compile_flags(module: Any) -> list[str]:
    return native_mpi_build_flags(module)[0]


def native_mpi_link_flags(module: Any) -> list[str]:
    return native_mpi_build_flags(module)[1]


def native_mpi_abi_key(module: Any) -> str:
    contract = _native_mpi_contract(module)
    return "mpi=off" if contract is None else "mpi=on;mabi=%s" % contract.abi_sha256


def native_mpi_communicator(module: Any) -> str:
    return "MPI_COMM_WORLD" if _native_mpi_contract(module) is not None else "serial"


__all__ = [
    "NativeMpiContract",
    "native_mpi_abi_key",
    "native_mpi_build_flags",
    "native_mpi_communicator",
    "native_mpi_compile_flags",
    "native_mpi_link_flags",
]
