#!/usr/bin/env python3
"""The receipt protocol of a PoPS qualification (ROM-17, CF14, K6, CC11).

A receipt records what one process actually established about one requirement, stage by stage, so
that every K6 level can be read from it.  It names the source commit, the installed SDK, every DSO
mapped in the process, the parameters, the backend and the exact ranks.  The receiver passes the
real lifecycle objects (``pops.resolve``, ``pops.compile``, ``pops.bind``, ``pops.run``); the receipt
checks that each one is an instance of the SDK it names and re-checks each claim from files and
process state.  Some facts are reported by the qualified code itself and are named as such: the
program hash a System installed, the program hash a mapped image exports, the ABI key and the
runtime environment report of the native leaf.

Stages and levels (K6):

``E``         the case sources, hashed and committed unchanged in the source;
``G``         the SDK's ``ResolvedSimulationPlan`` built by the real frontend;
``C``         the SDK's ``CompiledSimulationArtifact`` of that plan: every DSO hashed with its sidecar;
``loaded``    the SDK's native leaf mapped with its exact bytes, the System a class of that leaf, the
              runtime an SDK object, and every artifact DSO mapped in this process by content;
``advanced``  the SDK's ``RunReport`` of ``pops.run`` on that runtime and artifact;
``S``         a committed, independent oracle that the receipt itself calls on the run.

The level is the highest consecutive rung of ``E < G < C < R < S``; ``R`` needs both ``loaded`` and
``advanced``.  A standalone C++ receipt is ``C-finite`` or ``R-finite`` and never ``R`` (K6).  The
process starts from a committed file of the source or from ``python -m pytest``, and the code that
drives the receipt descends from that entry; that code and every module the process imported from
the source must be committed unchanged, and the interpreter may drive it only through its standard
library and pytest's own packages.  Code compiled from a string or read from standard input, and
code from anywhere but the source, the SDK and the interpreter's standard library and site
directories, is an overlay.
A receipt is *admitted* to a cohort only when it names the cohort's clean source and installed SDK,
held the installation lease, saw no overlay, and reached the required level (``admit``).

Commands::

  check RECEIPT [--cohort FILE] [--require LEVEL] [--rehash]
  record-standalone --requirement TEXT --source ROOT --binary PATH --directory DIR -- CMD...
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import site
import socket
import subprocess
import sys
import sysconfig
import traceback
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import sdk_cohort  # noqa: E402
from sdk_cohort import CohortError, digest, sha256_file, utc_now  # noqa: E402


RECEIPT_SCHEMA = "pops.receipt@4"
KINDS = ("pops-runtime", "standalone-cpp")
LADDER = ("E", "G", "C", "R", "S")
STAGES = ("E", "G", "C", "loaded", "advanced", "S")
_FINITE = ("C-finite", "R-finite")
_RECEIPT_KEYS = {
    "schema", "receipt_id", "kind", "requirement", "status", "opened_utc", "closed_utc",
    "source", "sdk", "coherence", "cohort", "lease", "process", "backend", "ranks",
    "parameters", "stages", "dsos", "overlay", "levels", "level", "limits", "failures",
}
#: Environment variables that select what a PoPS process loads or how it runs.
_ENVIRONMENT = re.compile(
    r"^(POPS_.*|PYTHONPATH|PYTHONNOUSERSITE|PYTHONHOME|Kokkos_ROOT|KOKKOS_.*|OMP_.*|CC|CXX|"
    r"LD_LIBRARY_PATH|LD_PRELOAD|DYLD_.*|CONDA_PREFIX|VIRTUAL_ENV|OMPI_.*|MPICH_.*|PMI_.*)$")
_SECRET = re.compile(r"(TOKEN|SECRET|PASSWORD|CREDENTIAL|_KEY)$", re.IGNORECASE)


class ReceiptError(RuntimeError):
    """A receipt is malformed, or an observation contradicts the claim it was given."""


# --- canonical parameter values ----------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """The JSON image a receipt stores: tuples become lists, foreign objects their type and repr."""
    def fallback(item: Any) -> Any:
        return {"object": type(item).__module__ + "." + type(item).__qualname__,
                "repr": repr(item)[:200]}
    return json.loads(json.dumps(value, default=fallback))


def _key(value: Any) -> str:
    if isinstance(value, str):
        return value
    for attribute in ("qualified_id", "path", "name"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str) and candidate:
            return "%s:%s" % (type(value).__name__, candidate)
    return "%s:%s" % (type(value).__name__, str(value))


def parameter_value(value: Any) -> Any:
    """A canonical, JSON-ready image of one parameter; arrays are named by their exact bytes."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) \
            else {"float": repr(value)}
    if isinstance(value, Mapping):
        return {_key(key): parameter_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [parameter_value(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None and hasattr(value, "tobytes"):
        import numpy

        array = numpy.ascontiguousarray(value)
        return {"array": {"shape": list(array.shape), "dtype": str(array.dtype),
                          "sha256": hashlib.sha256(array.tobytes()).hexdigest()}}
    return {"object": type(value).__module__ + "." + type(value).__qualname__,
            "repr": repr(value)[:200]}


# --- process observations ----------------------------------------------------------------------


def mapped_files() -> list[dict[str, Any]]:
    """Every shared object mapped in this process, with device and inode where the OS gives them."""
    rows: dict[str, dict[str, Any]] = {}
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split(None, 5)
            if len(fields) < 6:
                continue
            path = fields[5].strip()
            deleted = path.endswith(" (deleted)")
            if deleted:
                path = path[: -len(" (deleted)")]
            if not path.startswith("/") or ".so" not in Path(path).name:
                continue
            major, _, minor = fields[3].partition(":")
            rows.setdefault(path, {"path": path, "device": "%s:%s" % (major, minor),
                                   "inode": int(fields[4]), "deleted": deleted})
        return sorted(rows.values(), key=lambda row: row["path"])
    if sys.platform == "darwin":
        import ctypes

        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        for index in range(dyld._dyld_image_count()):
            raw = dyld._dyld_get_image_name(index)
            if raw:
                path = raw.decode("utf-8", errors="replace")
                rows.setdefault(path, {"path": path, "device": None, "inode": None,
                                       "deleted": False})
        return sorted(rows.values(), key=lambda row: row["path"])
    raise ReceiptError("this platform exposes no list of mapped shared objects")


def _same_file(row: Mapping[str, Any], path: Path) -> bool | None:
    if row.get("inode") is None:
        return None
    try:
        status = path.stat()
    except OSError:
        return False
    major, _, minor = str(row["device"]).partition(":")
    return (status.st_ino == row["inode"]
            and os.major(status.st_dev) == int(major, 16)
            and os.minor(status.st_dev) == int(minor, 16))


def _command_line() -> list[str] | None:
    """The exact command line of this process (``python -c`` code included).

    The OS's, where it has one (``/proc``); else the arguments the interpreter was given.
    """
    try:
        raw = Path("/proc/self/cmdline").read_bytes()
    except OSError:
        given = getattr(sys, "orig_argv", None)
        return None if given is None else [str(item)[:4000] for item in given]
    return [item.decode("utf-8", errors="replace")[:4000] for item in raw.split(b"\0") if item]


def process_observation() -> dict[str, Any]:
    environment = {key: ("<redacted>" if _SECRET.search(key) else value)
                   for key, value in sorted(os.environ.items()) if _ENVIRONMENT.match(key)}
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "command_line": _command_line(),
        "cwd": os.getcwd(),
        "environment": environment,
        "sys_path_head": list(sys.path[:8]),
    }


def _native_module() -> Any:
    return sys.modules.get("pops._pops")


def backend_observation() -> dict[str, Any]:
    """Backend facts reported by the loaded native leaf; absent when no leaf is loaded."""
    native = _native_module()
    if native is None:
        return {"native_loaded": False}
    def attribute(name: str) -> Any:
        value = getattr(native, name, None)
        return value if isinstance(value, (bool, int, str)) or value is None else str(value)
    compiler = attribute("__cxx_compiler__")
    compiler_sha256 = sha256_file(Path(compiler)) if compiler and Path(compiler).is_file() \
        else None
    def contract_digest(name: str) -> Any:
        contract = getattr(native, name, None)
        return contract.get("abi_sha256") if isinstance(contract, dict) else None
    def call(name: str) -> Any:
        function = getattr(native, name, None)
        if not callable(function):
            return None
        try:
            return function()
        except Exception as exc:  # the backend says it cannot report: record why
            return {"error": "%s: %s" % (type(exc).__name__, exc)}
    report = call("runtime_environment_report")
    abi = call("abi_key")
    return {
        "native_loaded": True,
        "native_file": attribute("__file__"),
        "dimension": attribute("__native_dimension__"),
        "version": attribute("__version__"),
        "abi_key": abi,
        "abi_version": attribute("__abi_version__"),
        "release_contract_sha256": attribute("__release_contract_sha256__"),
        "has_kokkos": attribute("__has_kokkos__"),
        "has_mpi": attribute("__has_mpi__"),
        "has_parallel_hdf5": attribute("__has_parallel_hdf5__"),
        "cxx_compiler": compiler,
        "cxx_compiler_sha256": compiler_sha256,
        "cxx_std": attribute("__cxx_std__"),
        "kokkos_abi_sha256": contract_digest("__kokkos_contract__"),
        "mpi_abi_sha256": contract_digest("__mpi_contract__"),
        "runtime_environment": dict(report) if isinstance(report, Mapping) else None,
    }


# --- where code comes from ----------------------------------------------------------------------


#: The interpreter modules that may start a receiver process: ``python -m pytest``.  Any other
#: entry of the interpreter (``cProfile``, ``trace``, ``pdb``, ``code``...) runs whatever code it
#: is given, which no receipt judges.
ENTRY_MODULES = ("pytest.__main__",)
#: The packages of a site directory that may drive a receipt: pytest's own.  Any other module of a
#: site directory on the driver's frames (a plugin placed there) is a driver no receipt judges.
DRIVER_PACKAGES = ("pytest", "_pytest", "pluggy")


def _standard_library() -> tuple[Path, ...]:
    """The base interpreter's standard library (a venv has none of its own)."""
    library = sysconfig.get_paths(vars={
        "base": sys.base_prefix, "platbase": sys.base_exec_prefix,
        "installed_base": sys.base_prefix, "installed_platbase": sys.base_exec_prefix})
    return tuple(sorted({Path(library[name]) for name in ("stdlib", "platstdlib")
                         if library.get(name)}))


def _interpreter_roots() -> tuple[Path, ...]:
    """The interpreter's own code: its standard library and its site directories, nothing else.

    The site directories are ``site.getsitepackages()`` and the user site when it is enabled.  The
    rest of a prefix (its root, ``bin/``) is not the interpreter: a script placed there is foreign.
    """
    roots = {*_standard_library(), *(Path(item) for item in site.getsitepackages() if item)}
    if site.ENABLE_USER_SITE:
        roots.add(Path(site.getusersitepackages()))
    return tuple(sorted(roots))


def _depth_within(path: Path, root: Path) -> int | None:
    """How specific ``root`` is for ``path`` (its depth), lexically or once resolved; else None."""
    for candidate, base in ((Path(os.path.abspath(path)), Path(os.path.abspath(root))),
                            (Path(path).resolve(), Path(root).resolve())):
        if candidate == base or candidate.is_relative_to(base):
            return len(base.parts)
    return None


def code_origin(filename: str, source_root: Path, package: Path,
                interpreter: Sequence[Path] | None = None) -> str:
    """Where one code file comes from: ``sdk``, ``source``, ``interpreter``, ``internal``, ``foreign``.

    ``internal`` is code without a file (``<frozen ...>``, ``<string>``, ``<stdin>``).  The most
    specific root wins, so a virtual environment inside a checkout is the interpreter and a
    checkout inside a site directory is the source.  ``interpreter`` defaults to this
    interpreter's standard library and site directories (``_interpreter_roots``).
    """
    if not filename or filename.startswith("<"):
        return "internal"
    roots = [(package, "sdk"), (source_root, "source")] + [
        (root, "interpreter")
        for root in (_interpreter_roots() if interpreter is None else interpreter)]
    matches = [(depth, origin) for root, origin in roots
               if (depth := _depth_within(Path(filename), root)) is not None]
    return max(matches)[1] if matches else "foreign"


def interpreter_part(filename: str, roots: Sequence[Path], standard: Sequence[Path]
                     ) -> dict[str, Any]:
    """Which part of the interpreter one of its files is: its library and its top-level package.

    The most specific root wins, so a site directory inside the standard library (a conda
    environment) is a site directory.  ``library`` is ``stdlib`` for a module the interpreter
    ships (``sys.stdlib_module_names``), ``site`` for a file of a site directory, and None for a
    file placed beside the standard library under a name the interpreter does not ship.
    """
    best: tuple[int, Path, Path] | None = None
    for root in roots:
        for candidate, base in ((Path(os.path.abspath(filename)), Path(os.path.abspath(root))),
                                (Path(filename).resolve(), Path(root).resolve())):
            if candidate != base and candidate.is_relative_to(base):
                if best is None or len(base.parts) > best[0]:
                    best = (len(base.parts), root, candidate.relative_to(base))
                break
    if best is None:
        return {"library": None, "package": None}
    _, root, relative = best
    top = relative.parts[0]
    name = top if len(relative.parts) > 1 else top.split(".", 1)[0]
    if root not in standard:
        return {"library": "site", "package": name}
    return {"library": "stdlib" if name in sys.stdlib_module_names else None, "package": name}


def code_observation(source: Mapping[str, Any], package: Path, driver: Sequence[str], *,
                     interpreter: Sequence[Path] | None = None,
                     standard: Sequence[Path] | None = None) -> dict[str, Any]:
    """Judge the code of this process: its entry, the frames that drive the receipt, every module.

    The entry is the file the interpreter started (``__main__``): a process started by
    ``python -c``, standard input or an interactive prompt has none, and the code it ran first is
    never judged.  Code from the source must be committed unchanged; the SDK is judged by its own
    file table and the interpreter's standard library and site directories are the interpreter;
    each frame of the interpreter records its library and package (``interpreter_part``).
    ``driver`` lists the frames that created the receipt, outermost first.
    """
    root = Path(source["root"])
    roots = _interpreter_roots() if interpreter is None else tuple(interpreter)
    library = _standard_library() if standard is None else tuple(standard)
    frames = [{"file": name, "origin": code_origin(name, root, package, roots)} for name in driver]
    for frame in frames:
        if frame["origin"] == "interpreter":
            frame.update(interpreter_part(frame["file"], roots, library))
    main = sys.modules.get("__main__")
    started = getattr(main, "__file__", None)
    started = started if isinstance(started, str) and started else None
    entry = {"file": started,
             "module": getattr(getattr(main, "__spec__", None), "name", None),
             "origin": "internal" if started is None else code_origin(started, root, package,
                                                                      roots)}
    modules, foreign, interpreter_modules = [], [], 0
    for name, module in sorted(list(sys.modules.items()), key=lambda item: item[0]):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            continue
        kind = code_origin(origin, root, package, roots)
        if kind == "source":
            modules.append((name, Path(origin).resolve()))
        elif kind == "foreign":
            foreign.append({"module": name, "path": origin})
        elif kind == "interpreter":
            interpreter_modules += 1
    paths = sorted({path for _, path in modules} | {
        Path(frame["file"]).resolve() for frame in frames if frame["origin"] == "source"})
    judged = {Path(root / item["path"]).resolve(): item for item in committed_files(
        source, [path for path in paths if path.is_file()])}
    for frame in frames:
        if frame["origin"] == "source":
            frame["committed"] = bool(judged.get(Path(frame["file"]).resolve(), {}).get(
                "committed"))
    return {
        "entry": entry,
        "driver": frames,
        "source_modules": [{"module": name, **judged.get(path, {"path": str(path),
                                                               "committed": False})}
                           for name, path in modules],
        "foreign_modules": foreign,
        "interpreter_modules": interpreter_modules,
        "interpreter_roots": [str(item) for item in roots],
    }


def _driver_files(frame: Any) -> list[str]:
    files = []
    while frame is not None:
        files.append(frame.f_code.co_filename)
        frame = frame.f_back
    return files[::-1]


def _qualified_type(value: Any) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


# --- the receipt -------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:48] or "receipt"


def _identity_token(value: Any) -> str | None:
    token = getattr(value, "token", None)
    return token if isinstance(token, str) else None


def _dso_file(path: Any, role: str) -> dict[str, Any]:
    resolved = Path(str(path)).resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ReceiptError("%s DSO is missing: %s" % (role, resolved))
    entry: dict[str, Any] = {"role": role, "path": str(resolved),
                             "sha256": sha256_file(resolved)}
    sidecar = Path(str(resolved) + ".pops-artifact.json")
    if sidecar.is_file():
        entry["artifact_sidecar_sha256"] = sha256_file(sidecar)
        try:
            entry["artifact_sidecar"] = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReceiptError("artifact sidecar of %s is invalid JSON" % resolved) from exc
    compile_record = Path(str(resolved) + ".pops-model-compile.json")
    generated = Path(str(resolved) + ".pops-model.cpp")
    if compile_record.is_file() and generated.is_file():
        record = json.loads(compile_record.read_text(encoding="utf-8"))
        entry["generated_cpp"] = {
            "path": str(generated),
            "sha256": sha256_file(generated),
            "recorded_source_sha256": record.get("source_sha256"),
            "recorded_binary_sha256": record.get("binary_sha256"),
            "compiler_sha256": record.get("compiler_sha256"),
            "command": record.get("command"),
        }
    return entry


class Receipt:
    """Record one qualification of one requirement, stage by stage, in one process.

    Use it as a context manager.  Opening identifies the source, takes the shared qualification
    lease of the installation for this source when the SDK is a pip installation (a checkout is
    never leased), then identifies the installed SDK; closing observes the process, derives the
    levels, writes the receipt and releases the lease.  An exception inside the block is recorded
    as the failure of the stage in progress and the receipt is still written.  The code that
    creates the receipt is its driver: it must be a committed file of the source, in a process
    started from a committed file or by ``python -m pytest``.
    """

    def __init__(self, requirement: str, *, source_root: Path, directory: Path | None = None,
                 domain: str = "", package: Path | None = None, lease: bool = True,
                 gather: Callable[[Any], Sequence[Any]] | None = None) -> None:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ReceiptError("a receipt names the requirement it qualifies")
        self._directory = None if directory is None else Path(directory)
        self._gather = gather
        self._want_lease = lease
        self._lease: sdk_cohort.Lease | None = None
        self._artifact_dsos: list[dict[str, Any]] = []
        self._artifact_identity: str | None = None
        self._plan_identity: str | None = None
        self._loaded = False
        self._runtime: Any = None
        self._report: Any = None
        self._hashes: dict[Path, str | None] = {}
        self._sdk_files: dict[str, str] = {}
        self._driver = _driver_files(sys._getframe(1))
        self._current = "open"
        self.path: Path | None = None
        self.data: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "receipt_id": None,
            "kind": "pops-runtime",
            "requirement": {"statement": requirement, "domain": domain},
            "status": "open",
            "opened_utc": utc_now(),
            "closed_utc": None,
            "source": None, "sdk": None, "coherence": None, "cohort": None, "lease": None,
            "process": None, "backend": None, "ranks": None,
            "parameters": {}, "stages": [], "dsos": [], "overlay": None,
            "levels": {}, "level": "none", "limits": [], "failures": [],
        }
        self._source_root = Path(source_root).resolve()
        self._package = package

    # -- lifecycle ------------------------------------------------------------------------------

    def __enter__(self) -> Receipt:
        self.open()
        return self

    def __exit__(self, kind: Any, error: BaseException | None, trace: Any) -> None:
        if error is not None:
            self.fail(self._current, error)
        self.close()

    def open(self) -> None:
        source = sdk_cohort.source_identity(self._source_root)
        self.data["source"] = source
        package_dir = sdk_cohort.locate_package(self._package)
        installation = package_dir.parent
        # The lease comes before the SDK identity, so no build that respects it can replace the
        # installation between the identity and the lease.
        self.data["lease"] = {"held": False, "installation_root": str(installation)}
        if self._want_lease and sdk_cohort._distribution(package_dir) is None:
            # A checkout is never admitted; leasing it would write lock files into the source
            # under qualification.
            self.data["lease"]["skipped"] = "the SDK is no pip installation (a checkout)"
        elif self._want_lease:
            try:
                self._lease = sdk_cohort.acquire_qualification_lease(
                    installation, source["commit"],
                    purpose=self.data["requirement"]["statement"])
            except sdk_cohort.InstallationBusy:
                raise
            except CohortError as exc:
                self.data["lease"]["error"] = str(exc)
            else:
                self.data["lease"] = {**self._lease.record, "held": True}
        try:
            sdk = sdk_cohort.sdk_identity(package_dir)
            self.data["sdk"] = sdk
            self._sdk_files = sdk_cohort.sdk_file_table(package_dir)
            try:
                self.data["coherence"] = sdk_cohort.coherence(source["root"], package_dir)
            except CohortError as exc:
                self.data["coherence"] = {"coherent": False, "error": str(exc)}
            try:
                record = sdk_cohort.read_cohort(installation)
            except CohortError:
                record = None
        except BaseException:
            if self._lease is not None:
                self._lease.release()
            raise
        self.data["cohort"] = None if record is None else {
            "cohort_id": record["cohort_id"], "source_commit": record["source"]["commit"],
            "sdk_id": record["sdk"]["sdk_id"]}
        self._current = "E"

    def close(self) -> dict[str, Any]:
        if self.data["closed_utc"] is not None:
            return self.data
        observations = (
            ("process", process_observation),
            ("backend", backend_observation),
            ("dsos", lambda: self._observe_dsos(final=True)),
        )
        try:
            for key, observe in observations:
                try:
                    self.data[key] = _jsonable(observe())
                except Exception as exc:  # an observation that fails is recorded, never hidden
                    self.fail("close:" + key, exc)
            if self.data["process"] is not None:
                try:
                    self.data["process"]["harness"] = committed_files(
                        self.data["source"], HARNESS, contents=_HARNESS_BYTES)
                    self.data["process"]["code"] = code_observation(
                        self.data["source"], Path(self.data["sdk"]["package_dir"]), self._driver)
                    now = sdk_cohort.source_identity(self._source_root)
                    self.data["process"]["source_at_close"] = {
                        "commit": now["commit"], "tree": now["tree"], "clean": now["clean"],
                        "modified": now["modified"][:20],
                        "untracked_sdk_inputs": now["untracked_sdk_inputs"][:20]}
                except (CohortError, OSError) as exc:
                    self.fail("close:source", exc)
            self.data["overlay"] = overlay_findings(self.data, self._sdk_files)
            try:
                self.data["ranks"] = _jsonable(self._ranks())
            except Exception as exc:
                self.fail("close:ranks", exc)
        finally:
            if self._lease is not None:
                self._lease.release()
                self.data["lease"]["released_utc"] = utc_now()
        self.data["closed_utc"] = utc_now()
        self.data["levels"], self.data["level"], self.data["limits"] = derive_levels(self.data)
        self.data["status"] = "failed" if self.data["failures"] else "closed"
        self.data["receipt_id"] = receipt_id(self.data)
        validate_receipt(self.data)
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
            self.path = self._directory / ("%s-%s.json" % (
                _slug(self.data["requirement"]["statement"]), self.data["receipt_id"][:16]))
            sdk_cohort.write_json_atomic(self.path, self.data)
        return self.data

    # -- stages ---------------------------------------------------------------------------------

    def _stage(self, name: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        row = {"stage": name, "at_utc": utc_now(), "evidence": _jsonable(dict(evidence))}
        self.data["stages"].append(row)
        return row

    def parameters(self, **values: Any) -> None:
        """Record the parameters of the run: bind values, run controls, data."""
        for name, value in values.items():
            self.data["parameters"][name] = parameter_value(value)

    def equations(self, *sources: Any) -> dict[str, Any]:
        """E: the Python sources that write the case, committed unchanged in the source.

        Pass the imported modules (their real ``__file__`` is hashed) or file paths.
        """
        self._current = "E"
        if not sources:
            raise ReceiptError("E names at least one case source")
        files = committed_files(self.data["source"], [_source_path(item) for item in sources])
        return self._stage("E", {"files": files})

    def plan(self, plan: Any) -> dict[str, Any]:
        """G: the SDK's resolved plan that the real frontend built."""
        self._current = "G"
        verify = getattr(plan, "verify", None)
        if not callable(verify):
            raise ReceiptError("G needs the ResolvedSimulationPlan that pops.resolve returns")
        verify()
        identity = _identity_token(getattr(plan, "plan_identity", None))
        semantic = _identity_token(getattr(getattr(plan, "snapshot", None),
                                           "semantic_identity", None))
        self._plan_identity = semantic
        return self._stage("G", {
            "plan_type": self._type_evidence(plan, "ResolvedSimulationPlan"),
            "verified": True,
            "plan_identity": identity,
            "semantic_identity": semantic,
            "resolved_dimension": getattr(plan, "resolved_dimension", None),
            "target": getattr(plan, "target", None),
            "backend": getattr(plan, "backend", None),
        })

    def compiled(self, artifact: Any) -> dict[str, Any]:
        """C: every DSO of the artifact hashed, with its identity sidecar and generated C++."""
        self._current = "C"
        artifact.verify()
        identity = _identity_token(getattr(artifact, "artifact_identity", None))
        if identity is None:
            raise ReceiptError("C needs a compiled artifact with an artifact identity")
        dsos = []
        for layout, path in sorted(dict(getattr(artifact, "layout_program_paths", {})).items()):
            dsos.append({**_dso_file(path, "program"), "layout": layout})
        for block in getattr(artifact, "blocks", ()):
            path = getattr(getattr(block, "model", None), "so_path", None)
            if path:
                dsos.append({**_dso_file(path, "model"), "block": block.name})
        for component in getattr(artifact, "component_artifacts", ()):
            path = getattr(component, "so_path", None) or getattr(component, "path", None)
            if path:
                dsos.append({**_dso_file(path, "component"),
                             "component": getattr(component, "component_id", None)})
        if not any(row["role"] == "program" for row in dsos):
            raise ReceiptError("C needs at least one compiled program DSO")
        semantic = _identity_token(getattr(artifact, "semantic_identity", None))
        self._artifact_dsos = dsos
        self._artifact_identity = identity
        return self._stage("C", {
            "artifact_type": self._type_evidence(artifact, "CompiledSimulationArtifact"),
            "artifact_identity": identity,
            "semantic_identity": semantic,
            "same_plan_as_G": self._plan_identity is not None and semantic == self._plan_identity,
            "target": getattr(artifact, "target", None),
            "backend": getattr(artifact, "backend", None),
            "abi_key": str(getattr(artifact, "abi_key", "")),
            "dsos": dsos,
        })

    def loaded(self, runtime: Any) -> dict[str, Any]:
        """Loaded: the SDK's native leaf and every artifact DSO mapped here, found by content.

        The System must be an instance of a class of that leaf and the runtime an SDK object; the
        program hash the System installed is compared with the one the mapped image exports.
        """
        self._current = "loaded"
        if not self._artifact_dsos:
            raise ReceiptError("a DSO is loaded only after the stage C named it")
        mapped = self._observe_dsos(final=False)
        found = []
        for dso in self._artifact_dsos:
            matches = [row for row in mapped if row.get("sha256") == dso["sha256"]]
            found.append({"role": dso["role"], "path": dso["path"], "sha256": dso["sha256"],
                          "mapped_as": [row["path"] for row in matches]})
        executor = getattr(runtime, "_executor", None)
        system = getattr(executor, "_s", None)
        installed = getattr(system, "installed_program_hash", None)
        installed_hash = installed() if callable(installed) else None
        symbols = self._program_hash_symbols(found)
        self._loaded = True
        self._runtime = runtime
        return self._stage("loaded", {
            "runtime": _qualified_type(runtime),
            "runtime_type": self._type_evidence(runtime),
            "executor_type": self._type_evidence(executor),
            "native": self._native_evidence(system, mapped),
            "system": None if system is None else type(system).__qualname__,
            "installed_program_hash": installed_hash,
            "program_hash_symbols": symbols,
            "installed_program_matches": installed_hash is not None and any(
                row["pops_program_hash"] == installed_hash for row in symbols),
            "artifact_dsos": found,
            "all_mapped": all(item["mapped_as"] for item in found),
        })

    def advanced(self, report: Any, runtime: Any) -> dict[str, Any]:
        """Advanced: ``pops.run`` accepted steps on the runtime of the stage loaded.

        ``runtime`` is the runtime that ``pops.run`` advanced: R needs it to be the loaded one.
        """
        self._current = "advanced"
        if not self._loaded:
            raise ReceiptError("a run advances the artifact only after the stage loaded saw it")
        data = report.to_data() if callable(getattr(report, "to_data", None)) else None
        if not isinstance(data, Mapping):
            raise ReceiptError("advanced needs the RunReport that pops.run returns")
        artifact = _identity_token(getattr(report, "artifact_identity", None))
        self._report = report
        evidence = {
            "report_type": self._type_evidence(report, "RunReport"),
            "same_runtime_as_loaded": runtime is self._runtime,
            "accepted_steps": data.get("accepted_steps"),
            "rejected_steps": data.get("rejected_steps"),
            "final_time": data.get("final_time"),
            "final_macro_step": data.get("final_macro_step"),
            "stop_reason": data.get("stop_reason"),
            "run_identity": _identity_token(getattr(report, "run_identity", None)),
            "bind_identity": _identity_token(getattr(report, "bind_identity", None)),
            "execution_identity": _identity_token(getattr(report, "execution_identity", None)),
            "artifact_identity": artifact,
            "same_artifact_as_C": artifact is not None and artifact == self._artifact_identity,
        }
        time_of = getattr(runtime, "time", None)
        step_of = getattr(runtime, "macro_step", None)
        evidence["runtime_time"] = time_of() if callable(time_of) else None
        evidence["runtime_macro_step"] = step_of() if callable(step_of) else None
        return self._stage("advanced", evidence)

    def property(self, name: str, check: Callable[..., Any], *arguments: Any,
                 domain: str) -> dict[str, Any]:
        """S: call one committed, independent oracle on the run and record what it returns.

        The receipt calls ``check(*arguments)`` itself; each argument must be the runtime of the
        stage loaded or the report of the stage advanced.  The verdict is the oracle's boolean (or
        its ``"passed"`` entry), never a value the caller passes; the oracle's file and the bounds
        it closes over are recorded.  An oracle of the SDK is not independent.
        """
        self._current = "S"
        if not isinstance(name, str) or not name or not domain or not callable(check):
            raise ReceiptError("a property names itself and its domain and gives the oracle")
        oracle = self._oracle_evidence(check)
        given = [{"type": _qualified_type(item),
                  "run_object": "runtime" if item is not None and item is self._runtime else
                  "report" if item is not None and item is self._report else None}
                 for item in arguments]
        value = check(*arguments)
        passed = value if isinstance(value, bool) else (
            value.get("passed") if isinstance(value, Mapping) else None)
        if type(passed) is not bool:
            raise ReceiptError("oracle %s returned no boolean verdict" % oracle["qualname"])
        return self._stage("S", {"name": name, "oracle": oracle, "arguments": given,
                                 "value": parameter_value(value), "passed": passed,
                                 "domain": domain})

    def fail(self, stage: str, error: BaseException) -> None:
        self.data["failures"].append({
            "stage": stage, "type": type(error).__module__ + "." + type(error).__qualname__,
            "message": str(error)[:4000],
            "traceback": "".join(traceback.format_exception(error))[-8000:],
        })

    # -- observations ---------------------------------------------------------------------------

    def _file_sha256(self, path: Path) -> str | None:
        if path not in self._hashes:
            self._hashes[path] = sha256_file(path) if path.is_file() else None
        return self._hashes[path]

    def _defined_in_sdk(self, cls: type) -> bool:
        """Whether ``cls`` is defined by an SDK file that still has its installed bytes."""
        module = sys.modules.get(getattr(cls, "__module__", None) or "")
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            return False
        path = Path(origin).resolve()
        package = Path(self.data["sdk"]["package_dir"])
        if not _inside(path, package):
            return False
        expected = self._sdk_files.get(path.relative_to(package).as_posix())
        return expected is not None and self._file_sha256(path) == expected

    def _type_evidence(self, value: Any, expected: str | None = None) -> dict[str, Any]:
        cls = type(value)
        bases = [base.__qualname__ for base in cls.__mro__ if self._defined_in_sdk(base)]
        evidence: dict[str, Any] = {"type": _qualified_type(value),
                                    "in_sdk": self._defined_in_sdk(cls)}
        if expected is not None:
            evidence["expected"] = expected
            evidence["is_expected"] = expected in bases
        return evidence

    def _native_evidence(self, system: Any, mapped: Sequence[Mapping[str, Any]]
                         ) -> dict[str, Any]:
        """The native leaf this process loaded, and whether the System is one of its classes."""
        native = sys.modules.get("pops._pops")
        origin = getattr(native, "__file__", None)
        package = Path(self.data["sdk"]["package_dir"])
        rows = {str((package / "_native" / row["path"]).resolve()): row
                for row in self.data["sdk"]["native"]}
        row = rows.get(str(Path(origin).resolve())) if isinstance(origin, str) and origin \
            else None
        cls = type(system)
        return {
            "module": getattr(native, "__name__", None),
            "file": origin,
            "sdk_leaf": row is not None,
            "leaf_sha256": None if row is None else row["sha256"],
            "leaf_mapped": row is not None and any(
                item.get("sha256") == row["sha256"] and not item.get("deleted")
                for item in mapped),
            "system_type": _qualified_type(system),
            "system_of_leaf": native is not None and system is not None
            and cls.__module__ == getattr(native, "__name__", None)
            and getattr(native, cls.__name__, None) is cls,
        }

    def _oracle_evidence(self, check: Callable[..., Any]) -> dict[str, Any]:
        """The oracle's own code, its origin and the values it closes over (its bounds)."""
        function: Any = check
        partial = None
        if isinstance(function, functools.partial):
            partial = {"args": parameter_value(list(function.args)),
                       "keywords": parameter_value(dict(function.keywords))}
            function = function.func
        function = inspect.unwrap(getattr(function, "__func__", function))
        code = getattr(function, "__code__", None)
        filename = getattr(code, "co_filename", None) or ""
        root = Path(self.data["source"]["root"])
        origin = code_origin(filename, root, Path(self.data["sdk"]["package_dir"]))
        judged = committed_files(self.data["source"], [Path(filename).resolve()])[0] \
            if origin == "source" else None
        closure: dict[str, Any] = {}
        if code is not None:
            with contextlib.suppress(TypeError, ValueError):
                closure = dict(inspect.getclosurevars(function).nonlocals)
        return {
            "module": getattr(function, "__module__", None),
            "qualname": getattr(function, "__qualname__", repr(function)),
            "file": judged["path"] if judged else filename,
            "sha256": None if judged is None else judged["sha256"],
            "origin": origin,
            "committed": bool(judged and judged["committed"]),
            "closure": {key: parameter_value(value) for key, value in sorted(closure.items())
                        if not inspect.ismodule(value) and not callable(value)},
            "partial": partial,
        }

    def _program_hash_symbols(self, found: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """``pops_program_hash()`` of each mapped program image, read without loading anything."""
        import ctypes

        mode = getattr(os, "RTLD_NOLOAD", 0) | getattr(os, "RTLD_NOW", 2)
        rows = []
        for item in found:
            if item["role"] != "program":
                continue
            value = None
            for mapped in item["mapped_as"]:
                try:
                    symbol = ctypes.CDLL(mapped, mode=mode).pops_program_hash
                except (OSError, AttributeError):
                    continue
                symbol.restype = ctypes.c_char_p
                symbol.argtypes = []
                value = symbol().decode()
                break
            rows.append({"path": item["path"], "pops_program_hash": value})
        return rows

    def _observe_dsos(self, *, final: bool) -> list[dict[str, Any]]:
        sdk = self.data["sdk"]
        package = Path(sdk["package_dir"])
        cache = os.environ.get("POPS_CACHE_DIR")
        roots = [package] + ([Path(cache).resolve()] if cache else [])
        artifact_paths = {dso["path"] for dso in self._artifact_dsos}
        rows = []
        for row in mapped_files():
            path = Path(row["path"])
            relevant = (row["path"] in artifact_paths or "pops" in path.name.lower()
                        or path.name.startswith("image-") or any(
                            _inside(path, root) for root in roots))
            entry = dict(row)
            if relevant and not row["deleted"]:
                try:
                    entry["sha256"] = sha256_file(path)
                except OSError:
                    entry["sha256"] = None  # unmapped or replaced since /proc/self/maps was read
                entry["same_inode"] = _same_file(row, path)
            role = "other"
            if _inside(path, package):
                role = "sdk"
            for dso in self._artifact_dsos:
                if entry.get("sha256") == dso["sha256"]:
                    role = dso["role"]
            entry["role"] = role
            if relevant or final:
                rows.append(entry)
        return rows

    def _ranks(self) -> dict[str, Any]:
        backend = self.data["backend"] or {}
        report = backend.get("runtime_environment") or {}
        if backend.get("native_loaded"):
            size, rank = report.get("mpi_ranks"), report.get("mpi_rank")
        else:
            size, rank = 1, 0  # no native leaf, hence no communicator: one process
        _levels, level, _limits = derive_levels(self.data)
        local = digest("pops.receipt-rank", {
            "sdk": self.data["sdk"]["sdk_id"], "source": self.data["source"]["commit"],
            "artifact": self._artifact_identity,
            "dsos": sorted(row["sha256"] for row in self._artifact_dsos),
            "overlay": sorted((self.data["overlay"] or {}).get("findings", ["unobserved"])),
            "level": level, "failures": len(self.data["failures"])})
        consensus = None
        gathered = None
        if self._gather is not None:
            gathered = list(self._gather(local))
            consensus = len(gathered) == size and all(item == local for item in gathered)
        elif size == 1:
            consensus = True  # one rank: its own observation is the whole world
        return {"size": size, "rank": rank, "communicator": report.get("communicator"),
                "local_identity": local, "gathered": gathered, "consensus": consensus}


def _source_path(item: Any) -> Path:
    origin = getattr(item, "__file__", None) if not isinstance(item, (str, os.PathLike)) else item
    if not origin:
        raise ReceiptError("case source %r has no file" % (item,))
    return Path(origin).resolve()


def committed_files(source: Mapping[str, Any], paths: Sequence[Path], *,
                    contents: Mapping[Path, bytes] | None = None) -> list[dict[str, Any]]:
    """Hash each file and say whether it is committed unchanged at the source's HEAD.

    ``contents`` gives the bytes to judge instead of the file's current bytes (the harness is
    judged on the bytes it was imported from).
    """
    root = Path(source["root"])
    inside = [path.relative_to(root).as_posix() for path in paths if _inside(path, root)]
    tracked = {} if not inside else {
        path: object_id for _mode, kind, object_id, path in sdk_cohort._tree_rows(root, inside)
        if kind == "blob"}
    files = []
    for path in paths:
        data = contents[path] if contents is not None and path in contents else path.read_bytes()
        relative = path.relative_to(root).as_posix() if _inside(path, root) else None
        files.append({
            "path": relative or str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "committed": relative is not None and tracked.get(relative) == sdk_cohort.git_blob_id(
                data, source["object_format"]),
        })
    return files


#: The code that observes and judges a qualification; it must itself be the committed source.
#: It is judged on the bytes read when this module is imported, the code that actually runs.
HARNESS = (Path(__file__).resolve(), SCRIPTS / "sdk_cohort.py",
           SCRIPTS / "write_native_variant_manifest.py")
_HARNESS_BYTES = {path: path.read_bytes() for path in HARNESS}


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


# --- overlay, levels and admission -------------------------------------------------------------


def overlay_findings(data: Mapping[str, Any], sdk_files: Mapping[str, str]) -> dict[str, Any]:
    """Every way this process may run code that is not the named source and SDK."""
    findings = []
    sdk = data["sdk"]
    package = Path(sdk["package_dir"])
    source = data["source"]
    if not source["clean"]:
        findings.append("source %s is not clean" % source["commit"])
    if sdk["installation"]["kind"] != "wheel":
        findings.append("installation is %s, not a wheel authenticated by RECORD"
                        % sdk["installation"]["kind"])
    coherence = data.get("coherence") or {}
    if not coherence.get("coherent"):
        findings.append("installed SDK is not coherent with the source")
    lease = data.get("lease") or {}
    if not lease.get("held"):
        findings.append("the installation lease was not held")
    at_close = (data.get("process") or {}).get("source_at_close")
    if at_close is not None and (at_close["commit"], at_close["tree"], at_close["clean"]) \
            != (source["commit"], source["tree"], source["clean"]):
        findings.append("the source changed during the qualification (now %s, clean=%s, "
                        "modified %s)" % (at_close["commit"], at_close["clean"],
                                          at_close["modified"][:5]))
    for item in (data.get("process") or {}).get("harness", ()):
        if not item["committed"]:
            findings.append("qualification harness %s is not committed unchanged in the source"
                            % item["path"])
    findings.extend(_code_findings((data.get("process") or {}).get("code")))
    module = sys.modules.get("pops")
    if module is not None:
        locations = [str(Path(item).resolve()) for item in getattr(module, "__path__", ())]
        if locations != [str(package)]:
            findings.append("pops resolves to %s, not the SDK %s" % (locations, package))
    for name, loaded in sorted(sys.modules.items()):
        if name != "pops" and not name.startswith("pops."):
            continue
        origin = getattr(loaded, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        if not _inside(path, package):
            findings.append("module %s loaded from %s, outside the SDK" % (name, path))
            continue
        relative = path.relative_to(package).as_posix()
        expected = sdk_files.get(relative)
        if expected is None or not path.is_file() or sha256_file(path) != expected:
            findings.append("module %s differs from the SDK file %s" % (name, relative))
    explicit = os.environ.get("POPS_NATIVE_VARIANTS_ROOT")
    if explicit and Path(explicit).resolve() != (package / "_native").resolve():
        findings.append("POPS_NATIVE_VARIANTS_ROOT selects native leaves outside the SDK: %s"
                        % explicit)
    native = sys.modules.get("pops._pops")
    if native is not None:
        leaves = {str((package / "_native" / row["path"]).resolve()) for row in sdk["native"]}
        origin = str(Path(getattr(native, "__file__", "")).resolve())
        if origin not in leaves:
            findings.append("native leaf %s is not a leaf of the SDK" % origin)
    for name in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        if os.environ.get(name):
            findings.append("%s injects code into the process: %s" % (name, os.environ[name]))
    include = os.environ.get("POPS_INCLUDE")
    if include:
        try:
            signature = sdk_cohort.header_signature(Path(include))
        except (CohortError, OSError) as exc:
            findings.append("POPS_INCLUDE=%s has no readable header manifest (%s)" % (include, exc))
        else:
            if signature != sdk["header_signature"]:
                findings.append("POPS_INCLUDE=%s carries other headers than the SDK" % include)
    for row in data.get("dsos", ()):
        if row.get("role") in ("program", "model", "component") and row.get("same_inode") is False:
            findings.append("mapped DSO %s is not the hashed file" % row["path"])
        if row.get("role") != "sdk":
            continue
        if row.get("deleted"):
            findings.append("SDK file %s is mapped but was deleted or replaced after it was loaded"
                            % row["path"])
            continue
        relative = Path(row["path"]).resolve().relative_to(package).as_posix() \
            if _inside(Path(row["path"]), package) else None
        if relative is not None and row.get("sha256") != sdk_files.get(relative):
            findings.append("mapped SDK file %s does not have the bytes of the SDK this receipt "
                            "names" % relative)
    findings = list(dict.fromkeys(findings))  # two frames of one driver file are one finding
    return {"clean": not findings, "findings": findings}


#: What a frame without a file ran; only the interpreter's frozen modules are admitted.
_FILELESS = {"<string>": "compiled from a string (python -c, exec)",
             "<stdin>": "read from standard input"}


def _runs_the_entry(frame: Mapping[str, Any]) -> bool:
    """A frame of ``runpy``, which runs the entry of ``python -m`` (frozen, or from the stdlib)."""
    return frame["file"] == "<frozen runpy>" or (
        frame.get("library") == "stdlib" and frame.get("package") == "runpy")


def _code_findings(code: Mapping[str, Any] | None) -> list[str]:
    """Overlay findings from the judged code of the process (``code_observation``)."""
    if code is None:
        return ["the code of the process was not judged"]
    findings = []
    entry = code.get("entry") or {}
    if not entry.get("file") or entry.get("origin") == "internal":  # standard input: "<stdin>"
        findings.append("the process was not started from a file (python -c, standard input or "
                        "an interactive prompt): the code it ran first is not judged")
    elif entry.get("origin") == "sdk":
        findings.append("the process was started by the SDK it qualifies: %s" % entry["file"])
    elif entry.get("origin") == "interpreter" and entry.get("module") not in ENTRY_MODULES:
        findings.append("the process was started by the interpreter's %s, which runs the code it "
                        "is given; a receiver is started from a committed file or by python -m "
                        "pytest" % (entry.get("module") or entry["file"]))
    frames = code.get("driver", [])
    for frame in frames:
        origin, name = frame["origin"], frame["file"]
        if origin == "foreign":
            findings.append("the receipt is driven by code outside the source, the SDK and the "
                            "interpreter: %s" % name)
        elif origin == "sdk":
            findings.append("the receipt is driven by the SDK it qualifies: %s" % name)
        elif origin == "source" and not frame.get("committed"):
            findings.append("driver %s is not committed unchanged in the source" % name)
        elif origin == "internal" and not name.startswith("<frozen "):
            findings.append("the receipt is driven by code %s: %s"
                            % (_FILELESS.get(name, "without a file"), name))
        elif origin == "interpreter" and not (
                frame.get("library") == "stdlib"
                or frame.get("library") == "site" and frame.get("package") in DRIVER_PACKAGES):
            findings.append("the receipt is driven by code of the interpreter that is neither its "
                            "standard library nor pytest (a file placed in a site directory or "
                            "beside the standard library): %s" % name)
    # The driver descends from the entry: a receiver called from atexit, a thread or another
    # callback of the interpreter has none of the code that set it up on its frames.
    started = entry.get("file") if entry.get("origin") != "internal" else None
    outermost = next((frame for frame in frames if not _runs_the_entry(frame)), None)
    if started and outermost is not None \
            and os.path.normpath(outermost["file"]) != os.path.normpath(started):
        findings.append("the receipt is not created under the process entry %s: its outermost "
                        "frame is %s (a receiver called from atexit, a thread or another callback "
                        "of the interpreter)" % (started, outermost["file"]))
    if not frames or frames[-1]["origin"] != "source":
        findings.append("the receipt is not created by a committed file of the source (%s)"
                        % (frames[-1]["file"] if frames else "no frame"))
    for item in code.get("source_modules", ()):
        if not item.get("committed"):
            findings.append("module %s (%s) is not committed unchanged in the source"
                            % (item["module"], item["path"]))
    for item in code.get("foreign_modules", ()):
        findings.append("module %s loaded from %s, outside the source, the SDK and the "
                        "interpreter" % (item["module"], item["path"]))
    return findings


def _stage_rows(data: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    return [row for row in data["stages"] if row["stage"] == name]


def derive_levels(data: Mapping[str, Any]) -> tuple[dict[str, bool], str, list[str]]:
    """Read the K6 levels off the recorded stages; return them, the level and its limits."""
    limits: list[str] = []
    if data["kind"] == "standalone-cpp":
        runs = _stage_rows(data, "executed")
        builds = _stage_rows(data, "compiled")
        level = "none"
        if builds:
            level = "C-finite"
        if builds and runs and all(row["evidence"]["returncode"] == 0 for row in runs):
            level = "R-finite"
        limits.append("a standalone C++ program is C/R-finite, never R-PoPS (K6)")
        return {name: False for name in LADDER}, level, limits
    stages_ok = {}
    equations = _stage_rows(data, "E")
    stages_ok["E"] = bool(equations) and all(
        item["committed"] for row in equations for item in row["evidence"]["files"])
    if equations and not stages_ok["E"]:
        limits.append("E: a case source is not committed unchanged in the source")
    plans = _stage_rows(data, "G")
    stages_ok["G"] = any(row["evidence"]["verified"] and row["evidence"]["plan_type"]["is_expected"]
                         for row in plans)
    if plans and not stages_ok["G"]:
        limits.append("G: the plan is not the SDK's verified ResolvedSimulationPlan")
    compiled = _stage_rows(data, "C")
    stages_ok["C"] = any(
        row["evidence"]["artifact_type"]["is_expected"]
        and row["evidence"]["same_plan_as_G"] and row["evidence"]["dsos"]
        and all("artifact_sidecar" in dso for dso in row["evidence"]["dsos"]
                if dso["role"] == "program")
        for row in compiled)
    if compiled and not stages_ok["C"]:
        limits.append("C: the artifact is not the SDK's CompiledSimulationArtifact of the G plan, "
                      "or lost its identity sidecar")
    loads = _stage_rows(data, "loaded")
    missing = [] if not loads else _loaded_missing(loads[-1]["evidence"])
    loaded = any(not _loaded_missing(row["evidence"]) for row in loads)
    advanced = any(row["evidence"]["same_artifact_as_C"]
                   and (row["evidence"]["accepted_steps"] or 0) >= 1
                   and row["evidence"]["report_type"]["is_expected"]
                   and row["evidence"]["same_runtime_as_loaded"] is True
                   for row in _stage_rows(data, "advanced"))
    stages_ok["R"] = loaded and advanced
    if compiled and not loaded:
        limits.append("R: not loaded by the SDK's runtime (%s)"
                      % ("; ".join(missing) if missing else "no loaded stage"))
    if loaded and not advanced:
        limits.append("R: the loaded artifact did not advance through pops.run (the SDK's "
                      "RunReport of the same runtime and artifact)")
    properties = _stage_rows(data, "S")
    problems = [problem for row in properties for problem in _property_problems(row["evidence"])]
    stages_ok["S"] = bool(properties) and not problems
    if problems:
        limits.append("S: " + problems[0])
    levels = {}
    reached = True
    level = "none"
    for name in LADDER:
        reached = reached and stages_ok[name]
        levels[name] = reached
        if reached:
            level = name
    if data["failures"]:
        limits.append("failed in stage %s: %s" % (data["failures"][0]["stage"],
                                                 data["failures"][0]["message"][:200]))
    return levels, level, limits


def _loaded_missing(evidence: Mapping[str, Any]) -> list[str]:
    """What the stage loaded lacks to show the SDK's runtime running the compiled artifact."""
    native = evidence["native"]
    checks = (
        (native["sdk_leaf"], "the SDK's native leaf is not the loaded pops._pops"),
        (native["leaf_mapped"], "the SDK's native leaf is not mapped with its installed bytes"),
        (native["system_of_leaf"], "the System is not a class of the SDK's native leaf"),
        (evidence["runtime_type"]["in_sdk"], "the runtime is not an SDK object"),
        (evidence["executor_type"]["in_sdk"], "the runtime's executor is not an SDK object"),
        (evidence["all_mapped"], "the compiled DSOs are not all mapped in the process"),
        (evidence["installed_program_matches"],
         "the System does not run the mapped program"),
    )
    return [message for passed, message in checks if not passed]


def _property_problems(evidence: Mapping[str, Any]) -> list[str]:
    oracle = evidence["oracle"]
    problems = []
    if oracle["origin"] != "source" or not oracle["committed"]:
        problems.append("oracle %s is not a committed file of the source (%s)"
                        % (oracle["qualname"], oracle["origin"]))
    if not evidence["arguments"] or not all(item["run_object"] for item in evidence["arguments"]):
        problems.append("oracle %s was not called on the runtime or report of the run"
                        % oracle["qualname"])
    if not evidence["passed"]:
        problems.append("property %s failed" % evidence["name"])
    return problems


def receipt_id(data: Mapping[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "receipt_id"}
    return digest("pops.receipt", body)


def validate_receipt(data: Any) -> None:
    if not isinstance(data, dict) or set(data) != _RECEIPT_KEYS:
        missing = sorted(_RECEIPT_KEYS - set(data or ())) if isinstance(data, dict) else []
        extra = sorted(set(data or ()) - _RECEIPT_KEYS) if isinstance(data, dict) else []
        raise ReceiptError("receipt has an unknown schema (missing %s, extra %s)"
                           % (missing, extra))
    if data["schema"] != RECEIPT_SCHEMA or data["kind"] not in KINDS:
        raise ReceiptError("receipt schema or kind is unsupported")
    if data["status"] not in ("closed", "failed"):
        raise ReceiptError("receipt was never closed")
    for row in data["stages"]:
        allowed = STAGES if data["kind"] == "pops-runtime" else ("compiled", "executed")
        if row.get("stage") not in allowed:
            raise ReceiptError("receipt stage %r is not a %s stage" % (row.get("stage"),
                                                                     data["kind"]))
    levels, level, limits = derive_levels(data)
    if (levels, level, limits) != (data["levels"], data["level"], data["limits"]):
        raise ReceiptError("receipt levels do not follow from its stages")


def read_receipt(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_receipt(data)
    if receipt_id(data) != data["receipt_id"]:
        raise ReceiptError("receipt %s is not self-consistent: its id is not the digest of its "
                           "content" % path)
    return data


def rehash(data: Mapping[str, Any]) -> list[str]:
    """Files named by the receipt whose current bytes differ from the recorded ones."""
    changed = []
    for row in data["stages"]:
        for dso in row["evidence"].get("dsos", ()) if row["stage"] == "C" else ():
            path = Path(dso["path"])
            if not path.is_file() or sha256_file(path) != dso["sha256"]:
                changed.append(dso["path"])
        if row["stage"] == "E":
            root = Path(data["source"]["root"])
            for item in row["evidence"]["files"]:
                path = root / item["path"]
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    changed.append(str(path))
    return changed


def admit(data: Mapping[str, Any], cohort: Mapping[str, Any] | None, *,
          require: str = "R") -> dict[str, Any]:
    """Decide whether a receipt qualifies a cohort at the required level."""
    if require not in LADDER:
        raise ReceiptError("required level must be one of %s" % (LADDER,))
    reasons = []
    if data["kind"] != "pops-runtime":
        reasons.append("a %s receipt is %s, never R-PoPS (K6); it qualifies no cohort"
                       % (data["kind"], data["level"]))
    elif not data["levels"].get(require):
        reasons.append("level %s does not reach %s" % (data["level"], require))
    if data["failures"]:
        reasons.append("the receipt records failures")
    overlay = data.get("overlay") or {}
    if not overlay.get("clean"):
        reasons.extend("overlay: " + item for item in overlay.get("findings", ["unobserved"]))
    ranks = data.get("ranks") or {}
    if data["kind"] == "pops-runtime" and ranks.get("consensus") is not True:
        reasons.append("ranks did not agree on the source, SDK, artifact, overlay and level")
    if data["status"] != "closed":
        reasons.append("the receipt is %s" % data["status"])
    if cohort is None:
        reasons.append("no cohort record names this installation")
    else:
        sdk_cohort.validate_cohort(cohort)
        source = data.get("source") or {}
        if (source.get("commit"), source.get("tree")) != (cohort["source"]["commit"],
                                                          cohort["source"]["tree"]):
            reasons.append("receipt source %s is not the cohort source %s"
                           % (source.get("commit"), cohort["source"]["commit"]))
        sdk = data.get("sdk") or {}
        if sdk.get("sdk_id") != cohort["sdk"]["sdk_id"]:
            reasons.append("receipt SDK %s is not the cohort SDK %s"
                           % (sdk.get("sdk_id"), cohort["sdk"]["sdk_id"]))
    return {"admitted": not reasons, "level": data["level"], "require": require,
            "cohort_id": None if cohort is None else cohort["cohort_id"],
            "receipt_id": data["receipt_id"], "reasons": reasons}


# --- standalone C++ receptions -----------------------------------------------------------------


def record_standalone(requirement: str, *, source_root: Path, binary: Path,
                      command: Sequence[str], directory: Path | None = None) -> dict[str, Any]:
    """Run a standalone native program and record it as the C/R-finite reception it is."""
    source = sdk_cohort.source_identity(source_root)
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise ReceiptError("standalone binary is missing: %s" % binary)
    data: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "receipt_id": None, "kind": "standalone-cpp",
        "requirement": {"statement": requirement, "domain": ""}, "status": "open",
        "opened_utc": utc_now(), "closed_utc": None, "source": source, "sdk": None,
        "coherence": None, "cohort": None, "lease": {"held": False}, "process": None,
        "backend": None, "ranks": None, "parameters": {"argv": list(command)}, "stages": [],
        "dsos": [], "overlay": None, "levels": {}, "level": "none", "limits": [], "failures": [],
    }
    data["stages"].append({"stage": "compiled", "at_utc": utc_now(),
                           "evidence": {"binary": str(binary), "sha256": sha256_file(binary)}})
    result = subprocess.run(list(command), capture_output=True, check=False)
    data["stages"].append({"stage": "executed", "at_utc": utc_now(), "evidence": {
        "argv": list(command), "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}})
    data["process"] = process_observation()
    data["overlay"] = {"clean": False, "findings": ["no PoPS runtime: standalone native program"]}
    data["closed_utc"] = utc_now()
    data["levels"], data["level"], data["limits"] = derive_levels(data)
    data["status"] = "closed"
    data["receipt_id"] = receipt_id(data)
    validate_receipt(data)
    if directory is not None:
        path = Path(directory) / ("%s-%s.json" % (_slug(requirement), data["receipt_id"][:16]))
        sdk_cohort.write_json_atomic(path, data)
    return data


# --- command line ------------------------------------------------------------------------------


def _load_cohort(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return sdk_cohort.validate_cohort(record, where="cohort record %s" % path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("receipt", type=Path)
    check.add_argument("--cohort", type=Path)
    check.add_argument("--require", default="R", choices=LADDER)
    check.add_argument("--rehash", action="store_true")
    standalone = commands.add_parser("record-standalone")
    standalone.add_argument("--requirement", required=True)
    standalone.add_argument("--source", type=Path, required=True)
    standalone.add_argument("--binary", type=Path, required=True)
    standalone.add_argument("--directory", type=Path)
    standalone.add_argument("argv", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        if arguments.command == "check":
            data = read_receipt(arguments.receipt)
            verdict = admit(data, _load_cohort(arguments.cohort), require=arguments.require)
            if arguments.rehash:
                verdict["changed_since_receipt"] = rehash(data)
            print(json.dumps(verdict, indent=2, sort_keys=True))
            return 0 if verdict["admitted"] else 1
        command = list(arguments.argv)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("record-standalone needs the command after --")
        data = record_standalone(arguments.requirement, source_root=arguments.source,
                                 binary=arguments.binary, command=command,
                                 directory=arguments.directory)
        print(json.dumps({"receipt_id": data["receipt_id"], "level": data["level"],
                          "limits": data["limits"]}, indent=2, sort_keys=True))
        return 0
    except (ReceiptError, CohortError, OSError, json.JSONDecodeError) as exc:
        print("qualification-receipt: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
