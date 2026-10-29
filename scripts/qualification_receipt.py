#!/usr/bin/env python3
"""The receipt protocol of a PoPS qualification (ROM-17, CF14, K6, CC11).

A receipt records what one process actually established about one requirement, stage by stage, so
that every K6 level can be read from it.  It names the source commit, the installed SDK, every DSO
mapped in the process, the parameters, the backend and the exact ranks.  The receiver observes the
real lifecycle objects (``pops.resolve``, ``pops.compile``, ``pops.bind``, ``pops.run``); the receipt
re-checks each claim from files and process state instead of trusting the code it qualifies.

Stages and levels (K6):

``E``         the case sources, hashed and committed unchanged in the source;
``G``         a resolved plan built by the real frontend (``pops.resolve`` of ``pops.validate``);
``C``         the artifact compiled from that plan: every DSO hashed, its identity sidecar kept;
``loaded``    every DSO of the artifact mapped in this process, identified by content;
``advanced``  the bound runtime advanced by ``pops.run`` with the loaded artifact;
``S``         a scientific property checked independently in its announced domain.

The level is the highest consecutive rung of ``E < G < C < R < S``; ``R`` needs both ``loaded`` and
``advanced``.  A standalone C++ receipt is ``C-finite`` or ``R-finite`` and never ``R`` (K6).  A
receipt is *admitted* to a cohort only when it names the cohort's clean source and installed SDK,
held the installation lease, saw no overlay, and reached the required level (``admit``).

Commands::

  check RECEIPT [--cohort FILE] [--require LEVEL] [--rehash]
  record-standalone --requirement TEXT --source ROOT --binary PATH --directory DIR -- CMD...
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import traceback
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import sdk_cohort  # noqa: E402
from sdk_cohort import CohortError, digest, sha256_file, utc_now  # noqa: E402


RECEIPT_SCHEMA = "pops.receipt@1"
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
    report = native.runtime_environment_report() \
        if callable(getattr(native, "runtime_environment_report", None)) else None
    abi = native.abi_key() if callable(getattr(native, "abi_key", None)) else None
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

    Use it as a context manager.  Opening identifies the source and the installed SDK and takes the
    shared qualification lease of that installation for this source; closing observes the process,
    derives the levels, writes the receipt and releases the lease.  An exception inside the block
    is recorded as the failure of the stage in progress and the receipt is still written.
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
        sdk = sdk_cohort.sdk_identity(self._package)
        self.data["source"] = source
        self.data["sdk"] = sdk
        self._sdk_files = sdk_cohort.sdk_file_table(Path(sdk["package_dir"]))
        try:
            self.data["coherence"] = sdk_cohort.coherence(source["root"], Path(sdk["package_dir"]))
        except CohortError as exc:
            self.data["coherence"] = {"coherent": False, "error": str(exc)}
        installation = Path(sdk["installation_root"])
        try:
            record = sdk_cohort.read_cohort(installation)
        except CohortError:
            record = None
        self.data["cohort"] = None if record is None else {
            "cohort_id": record["cohort_id"], "source_commit": record["source"]["commit"],
            "sdk_id": record["sdk"]["sdk_id"]}
        self.data["lease"] = {"held": False, "installation_root": str(installation)}
        if self._want_lease:
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
        self._current = "E"

    def close(self) -> dict[str, Any]:
        if self.data["status"] != "open" and self.data["closed_utc"] is not None:
            return self.data
        try:
            self.data["process"] = process_observation()
            self.data["process"]["harness"] = committed_files(self.data["source"], HARNESS)
            self.data["backend"] = backend_observation()
            self.data["dsos"] = self._observe_dsos(final=True)
            self.data["ranks"] = self._ranks()
            self.data["overlay"] = overlay_findings(self.data, self._sdk_files)
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
        row = {"stage": name, "at_utc": utc_now(), "evidence": dict(evidence)}
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
        """G: the resolved plan that the real frontend built."""
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
            "type": type(plan).__module__ + "." + type(plan).__qualname__,
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
            "artifact_identity": identity,
            "semantic_identity": semantic,
            "same_plan_as_G": self._plan_identity is not None and semantic == self._plan_identity,
            "target": getattr(artifact, "target", None),
            "backend": getattr(artifact, "backend", None),
            "abi_key": str(getattr(artifact, "abi_key", "")),
            "dsos": dsos,
        })

    def loaded(self, runtime: Any) -> dict[str, Any]:
        """Loaded: every artifact DSO mapped in this process, found by its exact content."""
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
        return self._stage("loaded", {
            "runtime": type(runtime).__module__ + "." + type(runtime).__qualname__,
            "system": None if system is None else type(system).__qualname__,
            "installed_program_hash": installed_hash,
            "program_hash_symbols": symbols,
            "installed_program_matches": installed_hash is not None and any(
                row["pops_program_hash"] == installed_hash for row in symbols),
            "artifact_dsos": found,
            "all_mapped": all(item["mapped_as"] for item in found),
        })

    def advanced(self, report: Any, runtime: Any = None) -> dict[str, Any]:
        """Advanced: ``pops.run`` accepted steps with the artifact this process loaded."""
        self._current = "advanced"
        if not self._loaded:
            raise ReceiptError("a run advances the artifact only after the stage loaded saw it")
        data = report.to_data() if callable(getattr(report, "to_data", None)) else None
        if not isinstance(data, Mapping):
            raise ReceiptError("advanced needs the RunReport that pops.run returns")
        artifact = _identity_token(getattr(report, "artifact_identity", None))
        evidence = {
            "report_type": type(report).__module__ + "." + type(report).__qualname__,
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
        if runtime is not None:
            time_of = getattr(runtime, "time", None)
            step_of = getattr(runtime, "macro_step", None)
            evidence["runtime_time"] = time_of() if callable(time_of) else None
            evidence["runtime_macro_step"] = step_of() if callable(step_of) else None
        return self._stage("advanced", evidence)

    def property(self, name: str, *, value: Any, bound: Any, passed: bool, oracle: str,
                 domain: str) -> dict[str, Any]:
        """S: one scientific property, its measured value, its bound and its independent oracle."""
        self._current = "S"
        if type(passed) is not bool or not oracle or not domain:
            raise ReceiptError("a property names its oracle and domain and has a boolean verdict")
        return self._stage("S", {"name": name, "value": parameter_value(value),
                                 "bound": parameter_value(bound), "passed": passed,
                                 "oracle": oracle, "domain": domain})

    def fail(self, stage: str, error: BaseException) -> None:
        self.data["failures"].append({
            "stage": stage, "type": type(error).__module__ + "." + type(error).__qualname__,
            "message": str(error)[:4000],
            "traceback": "".join(traceback.format_exception(error))[-8000:],
        })

    # -- observations ---------------------------------------------------------------------------

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
                entry["sha256"] = sha256_file(path) if path.is_file() else None
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
        size = report.get("mpi_ranks", 1)
        rank = report.get("mpi_rank", 0)
        local = digest("pops.receipt-rank", {
            "sdk": self.data["sdk"]["sdk_id"], "source": self.data["source"]["commit"],
            "artifact": self._artifact_identity,
            "dsos": sorted(row["sha256"] for row in self._artifact_dsos)})
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


def committed_files(source: Mapping[str, Any], paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Hash each file and say whether it is committed unchanged at the source's HEAD."""
    root = Path(source["root"])
    inside = [path.relative_to(root).as_posix() for path in paths if _inside(path, root)]
    tracked = {} if not inside else {
        path: object_id for _mode, kind, object_id, path in sdk_cohort._tree_rows(root, inside)
        if kind == "blob"}
    files = []
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix() if _inside(path, root) else None
        files.append({
            "path": relative or str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "committed": relative is not None and tracked.get(relative) == sdk_cohort.git_blob_id(
                data, source["object_format"]),
        })
    return files


#: The code that observes and judges a qualification; it must itself be the committed source.
HARNESS = (Path(__file__).resolve(), SCRIPTS / "sdk_cohort.py",
           SCRIPTS / "write_native_variant_manifest.py")


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
    for item in (data.get("process") or {}).get("harness", ()):
        if not item["committed"]:
            findings.append("qualification harness %s is not committed unchanged in the source"
                            % item["path"])
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
    return {"clean": not findings, "findings": findings}


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
    stages_ok["G"] = any(row["evidence"]["verified"] for row in _stage_rows(data, "G"))
    compiled = _stage_rows(data, "C")
    stages_ok["C"] = any(
        row["evidence"]["same_plan_as_G"] and row["evidence"]["dsos"]
        and all("artifact_sidecar" in dso for dso in row["evidence"]["dsos"]
                if dso["role"] == "program")
        for row in compiled)
    if compiled and not stages_ok["C"]:
        limits.append("C: the artifact is not tied to the G plan or lost its identity sidecar")
    loaded = any(row["evidence"]["all_mapped"] and row["evidence"]["system"]
                 and row["evidence"]["installed_program_matches"]
                 for row in _stage_rows(data, "loaded"))
    advanced = any(row["evidence"]["same_artifact_as_C"]
                   and (row["evidence"]["accepted_steps"] or 0) >= 1
                   for row in _stage_rows(data, "advanced"))
    stages_ok["R"] = loaded and advanced
    if compiled and not loaded:
        limits.append("R: the compiled DSOs were not all found mapped in the process, or the "
                      "System does not run the mapped program")
    if loaded and not advanced:
        limits.append("R: the loaded artifact did not advance through pops.run")
    properties = _stage_rows(data, "S")
    stages_ok["S"] = bool(properties) and all(row["evidence"]["passed"] for row in properties)
    if properties and not stages_ok["S"]:
        limits.append("S: a recorded property failed")
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
        raise ReceiptError("receipt %s does not authenticate its own identity" % path)
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
    if ranks.get("consensus") is not True:
        reasons.append("ranks did not agree on the source, SDK and artifact")
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
