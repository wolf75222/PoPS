"""Select exactly one compile-time native spatial specialization per process."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
from types import ModuleType
from typing import Any


_UNSELECTED = "unselected"
_LOADING = "loading"
_SELECTED = "selected"
_POISONED = "poisoned"

_LOCK = threading.RLock()
_STATE = _UNSELECTED
_MODULE: ModuleType | None = None
_DIMENSION: int | None = None
_FAILURE: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _Variant:
    dimension: int
    path: Path
    sha256: str
    version: str
    abi_key: str
    has_mpi: bool
    has_kokkos: bool


def _exact_dimension(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in (1, 2, 3):
        raise ValueError("%s must be exactly 1, 2, or 3" % where)
    return value


def _native_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    explicit = os.environ.get("POPS_NATIVE_VARIANTS_ROOT")
    if explicit:
        roots.append(Path(explicit).absolute())
    package = sys.modules.get("pops")
    roots.extend((Path(item).absolute() / "_native")
                 for item in getattr(package, "__path__", ()))
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_from_manifest(dimension: int) -> _Variant:
    manifests = tuple(root / "variants.json" for root in _native_roots()
                      if (root / "variants.json").is_file()
                      and not (root / "variants.json").is_symlink())
    if len(manifests) != 1:
        raise RuntimeError(
            "expected exactly one PoPS native variants manifest, found %d under %s"
            % (len(manifests), ", ".join(str(root) for root in _native_roots())))
    manifest = manifests[0]
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read native variants manifest %s" % manifest) from exc
    if type(document) is not dict or set(document) != {"schema_version", "variants"} \
            or document["schema_version"] != 1 or type(document["variants"]) is not list:
        raise RuntimeError("unsupported native variants manifest schema")
    expected_keys = {
        "dimension", "path", "sha256", "version", "abi_key", "has_mpi", "has_kokkos",
    }
    rows: dict[int, _Variant] = {}
    root = manifest.parent.resolve()
    for raw in document["variants"]:
        if type(raw) is not dict or set(raw) != expected_keys:
            raise RuntimeError("native variant row has an invalid schema")
        row_dimension = _exact_dimension(raw["dimension"], where="variant dimension")
        if type(raw["path"]) is not str or not raw["path"]:
            raise RuntimeError("native variant path must be canonical relative text")
        relative = PurePosixPath(raw["path"])
        if relative.is_absolute() or str(relative) != raw["path"] \
                or any(part in ("", ".", "..") for part in relative.parts):
            raise RuntimeError("native variant path must be a canonical relative POSIX path")
        if len(relative.parts) != 2 or relative.parts[0] != "dim%d" % row_dimension \
                or relative.parts[1] not in {
                    "_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES
                }:
            raise RuntimeError(
                "native variant path must be dimN/_pops<EXT_SUFFIX> for its dimension")
        lexical_path = root.joinpath(*relative.parts)
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise RuntimeError("native variant path is absent or is a symlink")
        path = lexical_path.resolve()
        try:
            contained = os.path.commonpath((str(root), str(path))) == str(root)
        except ValueError:
            contained = False
        if not contained or not path.is_file():
            raise RuntimeError("native variant path is absent or escapes its package root")
        sha256 = raw["sha256"]
        if type(sha256) is not str or len(sha256) != 64 \
                or any(char not in "0123456789abcdef" for char in sha256):
            raise RuntimeError("native variant sha256 must be lowercase hexadecimal")
        if type(raw["version"]) is not str or not raw["version"] \
                or type(raw["abi_key"]) is not str or not raw["abi_key"]:
            raise RuntimeError("native variant version and abi_key must be non-empty text")
        if type(raw["has_mpi"]) is not bool or type(raw["has_kokkos"]) is not bool:
            raise RuntimeError("native variant backend facts must be exact booleans")
        if row_dimension in rows:
            raise RuntimeError("native variants manifest repeats dimension %d" % row_dimension)
        rows[row_dimension] = _Variant(
            row_dimension, path, sha256, raw["version"], raw["abi_key"],
            raw["has_mpi"], raw["has_kokkos"])
    if dimension not in rows:
        raise RuntimeError(
            "installed PoPS distribution has no native specialization for Dim=%d" % dimension)
    selected = rows[dimension]
    # Hash only the leaf that this process will load.  Hashing all three large extensions on every
    # MPI rank would multiply cold shared-filesystem traffic without strengthening the selected
    # process boundary; the other leaves are authenticated when their own dimension is selected.
    if _sha256(selected.path) != selected.sha256:
        raise RuntimeError(
            "native variant bytes differ from variants.json: %s" % selected.path)
    return selected


def _load_global(path: Path) -> ModuleType:
    """Load one exact leaf as ``pops._pops`` with global native symbol visibility."""
    name = "pops._pops"

    def load() -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot create an extension loader for %s" % path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(name) is module:
                del sys.modules[name]
            raise
        return module

    if not (hasattr(sys, "setdlopenflags") and hasattr(sys, "getdlopenflags")):
        return load()
    previous = sys.getdlopenflags()
    flags = previous
    if hasattr(os, "RTLD_NOW"):
        flags |= os.RTLD_NOW
    if hasattr(os, "RTLD_GLOBAL"):
        flags |= os.RTLD_GLOBAL
    sys.setdlopenflags(flags)
    try:
        return load()
    finally:
        sys.setdlopenflags(previous)


def _verify_module(module: ModuleType, variant: _Variant) -> None:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not Path(origin).samefile(variant.path):
        raise RuntimeError("loaded native module origin differs from variants.json")
    if getattr(module, "__native_dimension__", None) != variant.dimension:
        raise RuntimeError("loaded native module dimension differs from variants.json")
    if getattr(module, "__version__", None) != variant.version:
        raise RuntimeError("loaded native module version differs from variants.json")
    from pops._version import authenticate_native_version

    authenticate_native_version(module)
    abi = getattr(module, "abi_key", None)
    if not callable(abi) or abi() != variant.abi_key:
        raise RuntimeError("loaded native module ABI differs from variants.json")
    if getattr(module, "__has_mpi__", None) is not variant.has_mpi \
            or getattr(module, "__has_kokkos__", None) is not variant.has_kokkos:
        raise RuntimeError("loaded native module backend differs from variants.json")


def _verify_collective_identity(module: ModuleType, variant: _Variant) -> None:
    """Require every MPI rank to load byte-identical specialization and ABI authority."""
    world_provider = getattr(module, "mpi_world", None)
    if not callable(world_provider):
        raise RuntimeError("loaded native module has no MPI world authority")
    world = world_provider()
    gather = getattr(world, "allgather_bytes", None)
    size = getattr(world, "size", None)
    if not callable(gather) or type(size) is not int or size < 1:
        raise RuntimeError("loaded native module has an invalid MPI world authority")
    payload = json.dumps(
        {
            "abi_key": variant.abi_key,
            "dimension": variant.dimension,
            "sha256": variant.sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    gathered = gather(payload)
    if type(gathered) is not tuple or len(gathered) != size \
            or any(type(item) is not bytes for item in gathered):
        raise RuntimeError("native specialization MPI consensus returned an invalid rank set")
    mismatches = tuple(rank for rank, item in enumerate(gathered) if item != payload)
    if mismatches:
        raise RuntimeError(
            "MPI ranks selected different PoPS native dimensions, ABIs, or variant bytes; "
            "mismatching ranks=%s" % (mismatches,))


def select_native_dimension(dimension: Any) -> ModuleType:
    """Load and freeze the specialization derived from one verified resolved plan."""
    selected_dimension = _exact_dimension(dimension, where="resolved native dimension")
    global _STATE, _MODULE, _DIMENSION, _FAILURE

    with _LOCK:
        if _STATE == _POISONED:
            raise RuntimeError(
                "native specialization loading previously failed; this process is sealed"
            ) from _FAILURE
        if _STATE == _LOADING:
            raise RuntimeError("recursive native specialization loading is forbidden")
        if _STATE == _SELECTED:
            if _DIMENSION != selected_dimension:
                raise RuntimeError(
                    "this process already selected PoPS Dim=%d and cannot switch to Dim=%d; "
                    "use a separate process" % (_DIMENSION, selected_dimension))
            assert _MODULE is not None
            return _MODULE
        if "pops._pops" in sys.modules:
            error = RuntimeError(
                "pops._pops was loaded outside the native selector; dimension provenance is lost")
            _STATE, _FAILURE = _POISONED, error
            raise error

        variant = _variant_from_manifest(selected_dimension)
        _STATE = _LOADING
        module: ModuleType | None = None
        try:
            module = _load_global(variant.path)
            _verify_module(module, variant)
            _verify_collective_identity(module, variant)
            package = sys.modules.get("pops")
            if package is not None:
                package._pops = module
            _MODULE = module
            _DIMENSION = selected_dimension
            _STATE = _SELECTED
            return module
        except BaseException as exc:
            # A failed extension cannot be safely unloaded, so the process remains poisoned.  Still
            # remove every Python-visible reference: no caller may accidentally consume a module
            # that failed provenance or ABI verification after dlopen succeeded.
            if module is not None and sys.modules.get("pops._pops") is module:
                del sys.modules["pops._pops"]
            package = sys.modules.get("pops")
            if package is not None and module is not None \
                    and getattr(package, "_pops", None) is module:
                delattr(package, "_pops")
            _FAILURE = exc
            _STATE = _POISONED
            raise


def selected_native_module(*, required: bool = False) -> ModuleType | None:
    """Return the selected module without loading or inferring a dimension."""
    with _LOCK:
        if _STATE == _SELECTED:
            assert _MODULE is not None
            return _MODULE
        if _STATE == _POISONED:
            raise RuntimeError("native specialization selection is poisoned") from _FAILURE
        if required:
            raise RuntimeError(
                "no PoPS native dimension is selected; call pops.compile(resolved_plan) first")
        return None


def selected_native_dimension() -> int | None:
    with _LOCK:
        if _STATE == _SELECTED:
            return _DIMENSION
        if _STATE == _POISONED:
            raise RuntimeError("native specialization selection is poisoned") from _FAILURE
        return None


__all__ = [
    "select_native_dimension",
    "selected_native_dimension",
    "selected_native_module",
]
