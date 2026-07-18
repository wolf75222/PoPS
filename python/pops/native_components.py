"""Authenticated build inputs for arbitrary prepared native providers.

The public extension contract deliberately supports a small, real native build surface rather
than an arbitrary list of compiler or linker flags.  A header-only component snapshots every file
below one include root, records the content identities (never the machine-local root path), and is
staged from verified bytes into the compiler's private temporary directory.  The generated Program
therefore cannot accidentally compile a different header tree from the one named by its artifact
identity.

PoPS-owned providers use :meth:`PreparedNativeComponent.pops_builtin`; those headers
are already authenticated by the PoPS header signature carried by every generated loader.
Numerical execution remains entirely in the resulting C++ shared object.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Any


_COMPONENT_SCHEMA_VERSION = 2
_NATIVE_INTERFACE = "pops.prepared-native-component@2"
_INCLUDE_DIRECTIVE = re.compile(rb"^[ \t]*#[ \t]*include\b(.*)$", re.MULTILINE)


def _nonempty_string(value: Any, *, where: str) -> str:
    if type(value) is not str or not value:
        raise TypeError("%s must be a non-empty exact string" % where)
    return value


def _relative_header(value: Any, *, where: str) -> str:
    value = _nonempty_string(value, where=where)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in (".", "..")
        or value != path.as_posix()
        or ".." in path.parts
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError("%s must be a normalized relative POSIX path" % where)
    return value


def _file_digest(path: Path) -> tuple[str, int, bytes]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), data


@dataclass(frozen=True, slots=True)
class PreparedNativeHeaderFile:
    """One exact file in a header-only native component tree."""

    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_header(self.path, where="header file path"))
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("header file sha256 must be a lowercase SHA-256 hex digest")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("header file size must be a non-negative exact integer")

    def to_data(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


def _snapshot_tree(root: Path) -> tuple[PreparedNativeHeaderFile, ...]:
    """Snapshot every regular file below *root* and reject ambiguous filesystem entries."""
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("header-only component include_root must be an existing absolute directory")
    files: list[PreparedNativeHeaderFile] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        names.sort()
        filenames.sort()
        directory_path = Path(directory)
        for name in names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError("header-only component trees cannot contain symlink directories")
        for name in filenames:
            candidate = directory_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("header-only component trees must contain only regular files")
            relative = candidate.relative_to(root).as_posix()
            digest, size, _ = _file_digest(candidate)
            files.append(PreparedNativeHeaderFile(relative, digest, size))
    if not files:
        raise ValueError("header-only component include_root contains no files")
    return tuple(files)


def _validate_component_includes(
    root: Path, files: tuple[PreparedNativeHeaderFile, ...]
) -> None:
    """Refuse non-literal or escaping includes in every component-owned source file.

    Angle includes may name PoPS, Kokkos or standard-toolchain headers; those are authenticated by
    the generated loader ABI/toolchain. Quoted includes must resolve to this exact component tree
    (or to the PoPS SDK). No component can smuggle a machine-local absolute header or macro-selected
    include into a supposedly content-addressed build.
    """
    owned = frozenset(item.path for item in files)
    for record in files:
        data = (root / record.path).read_bytes()
        for match in _INCLUDE_DIRECTIVE.finditer(data):
            value = match.group(1).strip()
            if not value or value[:1] not in (b'"', b"<"):
                raise ValueError(
                    "header-only components require literal #include directives in %r"
                    % record.path
                )
            closing = b'"' if value[:1] == b'"' else b">"
            end = value.find(closing, 1)
            if end < 0:
                raise ValueError("malformed #include directive in %r" % record.path)
            try:
                included = value[1:end].decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("component #include paths must be ASCII") from exc
            path = PurePosixPath(included)
            if path.is_absolute() or ".." in path.parts or "\\" in included or "\0" in included:
                raise ValueError(
                    "header-only component include %r escapes its authenticated build inputs"
                    % included
                )
            if value[:1] == b'"':
                resolved = posixpath.normpath(
                    posixpath.join(posixpath.dirname(record.path), included)
                )
                if resolved not in owned and not included.startswith("pops/"):
                    raise ValueError(
                        "quoted component include %r is absent from its authenticated file tree"
                        % included
                    )


def compiler_include_roots(flags: Any) -> tuple[str, ...]:
    """Extract only explicit compiler include roots from a closed toolchain flag sequence.

    Callers must pass the authenticated PoPS/Kokkos/MPI compiler flags, never user optimisation
    flags. The resulting roots are an allowlist for compiler-reported dependencies, not another
    search path.
    """
    if not isinstance(flags, (list, tuple)) or any(type(item) is not str for item in flags):
        raise TypeError("native compiler flags must be an exact text sequence")
    roots: list[str] = []
    index = 0
    paired = frozenset({"-I", "-isystem", "-iquote", "-idirafter", "-iframework", "-F"})
    joined = ("-I", "-isystem", "-iquote", "-idirafter", "-iframework", "-F")
    while index < len(flags):
        token = flags[index]
        value: str | None = None
        if token in paired:
            index += 1
            if index >= len(flags):
                raise ValueError("native compiler include flag %r has no path" % token)
            value = flags[index]
        else:
            for prefix in joined:
                if token.startswith(prefix) and token != prefix:
                    value = token[len(prefix):]
                    break
        if value:
            path = os.path.realpath(value)
            if not os.path.isdir(path):
                raise ValueError("native compiler include root is unavailable: %s" % value)
            if path not in roots:
                roots.append(path)
        index += 1
    return tuple(roots)


def _make_dependency_paths(path: Any, *, working_directory: Any = None) -> tuple[str, ...]:
    """Parse one GCC/Clang ``-MMD`` make depfile into canonical dependency paths."""
    dependency_file = Path(path)
    try:
        data = dependency_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("native compiler did not publish its dependency file") from exc
    # A backslash-newline is a Make continuation, not part of a dependency path.
    data = re.sub(r"\\\r?\n[ \t]*", "", data)
    escaped = False
    separator = -1
    for index, character in enumerate(data):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            separator = index
            break
    if separator < 0:
        raise RuntimeError("native compiler dependency file has no target separator")

    tokens: list[str] = []
    current: list[str] = []
    escaped = False
    for character in data[separator + 1:]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character.isspace():
            if current:
                tokens.append("".join(current).replace("$$", "$"))
                current = []
        else:
            current.append(character)
    if escaped:
        raise RuntimeError("native compiler dependency file ends in an escape")
    if current:
        tokens.append("".join(current).replace("$$", "$"))
    if not tokens:
        raise RuntimeError("native compiler dependency file contains no dependencies")

    base = os.path.realpath(os.fspath(working_directory or os.getcwd()))
    result: list[str] = []
    for token in tokens:
        candidate = token if os.path.isabs(token) else os.path.join(base, token)
        canonical = os.path.realpath(candidate)
        if canonical not in result:
            result.append(canonical)
    return tuple(result)


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def verify_prepared_native_dependencies(
    dependency_file: Any,
    *,
    generated_source: Any,
    pops_include_root: Any,
    staged_components: Any,
    toolchain_include_roots: Any,
    working_directory: Any = None,
) -> tuple[str, ...]:
    """Verify the compiler-observed transitive closure before artifact publication.

    A dependency is accepted only when it is the generated translation unit, an exact file covered
    by the PoPS SDK signature, an exact file in a staged component manifest, or a header below an
    explicitly replayed Kokkos/MPI/toolchain include root. Ambient compiler paths are not authorities.
    """
    from pops.codegen.toolchain import pops_authenticated_header_paths

    generated = os.path.realpath(os.fspath(generated_source))
    dependencies = _make_dependency_paths(
        dependency_file, working_directory=working_directory)
    if generated not in dependencies:
        raise RuntimeError("native compiler dependency closure omits the generated source")

    sdk_headers = pops_authenticated_header_paths(pops_include_root)
    component_headers: set[str] = set()
    required_entries: set[str] = set()
    for item in staged_components:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("staged native components require (component, root) pairs")
        component, root_value = item
        if type(component) is not PreparedNativeComponent or component.kind != "header_only":
            raise TypeError("staged native dependency authority must be a header-only component")
        root = os.path.realpath(os.fspath(root_value))
        if _snapshot_tree(Path(root)) != component.files:
            raise RuntimeError(
                "staged native component %r changed before artifact publication"
                % component.component_id
            )
        for record in component.files:
            component_headers.add(os.path.realpath(os.path.join(root, record.path)))
        for entry in component.entry_headers:
            required_entries.add(os.path.realpath(os.path.join(root, entry)))

    roots: list[str] = []
    for value in toolchain_include_roots:
        root = os.path.realpath(os.fspath(value))
        if not os.path.isdir(root):
            raise ValueError("authenticated toolchain include root is unavailable: %s" % value)
        if root not in roots:
            roots.append(root)

    observed = set(dependencies)
    missing_entries = sorted(required_entries - observed)
    if missing_entries:
        raise RuntimeError(
            "native compiler dependency closure omits component entry headers: %s"
            % ", ".join(missing_entries)
        )
    unexpected = sorted(
        dependency
        for dependency in dependencies
        if dependency != generated
        and dependency not in sdk_headers
        and dependency not in component_headers
        and not any(_is_within(dependency, root) for root in roots)
    )
    if unexpected:
        raise RuntimeError(
            "native compiler used dependencies outside the authenticated component/SDK/toolchain "
            "closure: %s" % ", ".join(unexpected)
        )
    return dependencies


@dataclass(frozen=True, slots=True)
class PreparedNativeComponent:
    """Typed native build input owned by one prepared provider.

    Use :meth:`header_only` for external components.  Its include root is a source location only:
    it is deliberately excluded from :meth:`manifest` and all artifact identities.  Fresh builds
    call :meth:`stage_verified` and compile only the staged, content-checked copy.
    """

    component_id: str
    abi_version: int
    kind: str
    entry_headers: tuple[str, ...]
    files: tuple[PreparedNativeHeaderFile, ...] = ()
    include_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_id", _nonempty_string(self.component_id, where="component_id")
        )
        if type(self.abi_version) is not int or self.abi_version < 1:
            raise ValueError("native component abi_version must be a positive exact integer")
        if self.kind not in ("pops_builtin", "header_only"):
            raise ValueError("native component kind must be 'pops_builtin' or 'header_only'")
        if type(self.entry_headers) is not tuple:
            raise TypeError("native component entry_headers must be an exact tuple")
        headers = tuple(
            _relative_header(header, where="native component entry header")
            for header in self.entry_headers
        )
        if len(set(headers)) != len(headers):
            raise ValueError("native component entry_headers contains a duplicate")
        object.__setattr__(self, "entry_headers", headers)
        if type(self.files) is not tuple or any(
            type(item) is not PreparedNativeHeaderFile for item in self.files
        ):
            raise TypeError("native component files must contain exact header-file records")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("native component files must be uniquely sorted by relative path")
        if self.kind == "pops_builtin":
            if self.include_root is not None or self.files:
                raise ValueError("PoPS builtin components cannot carry external source files")
        else:
            if type(self.include_root) is not str or not os.path.isabs(self.include_root):
                raise ValueError("header-only component include_root must be an absolute path")
            if not self.files:
                raise ValueError("header-only component must authenticate its complete file tree")
            if not headers:
                raise ValueError("header-only component must declare at least one entry header")
            missing = set(headers) - set(paths)
            if missing:
                raise ValueError(
                    "native component entry headers are absent from its file manifest: %s"
                    % sorted(missing)
                )

    @classmethod
    def pops_builtin(
        cls,
        component_id: str,
        *,
        entry_headers: tuple[str, ...] = (),
        abi_version: int = 1,
    ) -> PreparedNativeComponent:
        """Describe PoPS-owned code covered by the generated loader's header signature."""
        return cls(component_id, abi_version, "pops_builtin", entry_headers)

    @classmethod
    def header_only(
        cls,
        component_id: str,
        *,
        include_root: Any,
        entry_headers: tuple[str, ...],
        abi_version: int = 1,
    ) -> PreparedNativeComponent:
        """Snapshot a complete external header tree as an immutable native component manifest."""
        root = Path(os.path.realpath(os.fspath(include_root)))
        files = _snapshot_tree(root)
        _validate_component_includes(root, files)
        return cls(
            component_id,
            abi_version,
            "header_only",
            entry_headers,
            files,
            str(root),
        )

    def manifest(self) -> dict[str, Any]:
        """Return path-free canonical data entering the Program artifact identity."""
        return {
            "schema_version": _COMPONENT_SCHEMA_VERSION,
            "interface": _NATIVE_INTERFACE,
            "component_id": self.component_id,
            "abi_version": self.abi_version,
            "kind": self.kind,
            "entry_headers": list(self.entry_headers),
            "files": [item.to_data() for item in self.files],
        }

    @property
    def manifest_sha256(self) -> str:
        encoded = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def authority(self) -> dict[str, Any]:
        """Compact identity embedded in the immutable solve IR."""
        return {
            "schema_version": _COMPONENT_SCHEMA_VERSION,
            "component_id": self.component_id,
            "abi_version": self.abi_version,
            "manifest_sha256": self.manifest_sha256,
        }

    def verify_builtin_headers(self, pops_include_root: Any) -> None:
        """Ensure every PoPS-owned entry header exists below the authenticated SDK root."""
        if self.kind != "pops_builtin":
            return
        root = Path(os.path.realpath(os.fspath(pops_include_root)))
        for header in self.entry_headers:
            candidate = root / header
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(
                    "PoPS builtin native component %r is missing authenticated header %r"
                    % (self.component_id, header)
                )

    def stage_verified(self, destination: Any) -> str | None:
        """Materialize verified external bytes and return their private compiler include root."""
        if self.kind == "pops_builtin":
            return None
        root = Path(self.include_root or "")
        current = _snapshot_tree(root)
        if current != self.files:
            raise ValueError(
                "header-only native component %r changed after registration"
                % self.component_id
            )
        _validate_component_includes(root, current)
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=False)
        expected = {item.path: item for item in self.files}
        for relative, record in expected.items():
            source = root / relative
            digest, size, data = _file_digest(source)
            if digest != record.sha256 or size != record.size:
                raise ValueError(
                    "header-only native component %r changed while staging %r"
                    % (self.component_id, relative)
                )
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        if _snapshot_tree(root) != self.files:
            raise ValueError(
                "header-only native component %r changed while its tree was staged"
                % self.component_id
            )
        return str(target)


__all__ = [
    "PreparedNativeHeaderFile",
    "PreparedNativeComponent",
]
