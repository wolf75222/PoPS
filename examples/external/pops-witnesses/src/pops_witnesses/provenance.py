"""Read-only fingerprints and evidence integrity, separate from numerical acceptance."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tarfile
import subprocess

from .reference import BASE_REVISION, require

CORE_PATHS = (
    "python",
    "include",
    "src",
    "cmake",
    "scripts",
    "CMakeLists.txt",
    "CMakePresets.json",
    "pyproject.toml",
    "environment.yml",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_default(value):
    """Preserve native report byte payloads without pretending they are text."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "encoding": "hex", "value": bytes(value).hex()}
    raise TypeError("unsupported evidence value: " + type(value).__name__)


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, default=evidence_default) + "\n",
        encoding="utf-8",
    )


def tree_manifest(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def source_manifest(root):
    """Compare every selected tracked core byte with the frozen Git objects."""
    root = Path(root)
    archive = subprocess.check_output(
        ["git", "-C", str(root), "archive", BASE_REVISION, *CORE_PATHS]
    )
    expected = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                expected[item.name] = hashlib.sha256(stream.extractfile(item).read()).hexdigest()
    actual = {name: digest(root / name) for name in expected}
    require(actual == expected, "core source differs from retained revision")
    # Catch additional untracked core source as well as changed tracked bytes.
    allowed = set(expected)
    for name in CORE_PATHS:
        path = root / name
        for item in path.rglob("*") if path.is_dir() else (path,):
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                require(str(item.relative_to(root)) in allowed, "extra core file: " + str(item))
    return actual


def core_identity(source, wheel):
    """Private native introspection is evidence plumbing, never physical authoring."""
    import pops
    import sys
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(2)
    package = Path(pops.__file__).resolve().parent
    require(
        package.is_relative_to(Path(sys.prefix).resolve()), "PoPS must be installed in the venv"
    )
    require(
        not package.is_relative_to(Path(source).resolve()),
        "source import instead of installed PoPS",
    )
    require(native.__native_dimension__ == 2 and not native.__has_mpi__, "CPU Dim2 serial scope")
    require(native.__has_kokkos__, "Kokkos required")
    core = source_manifest(source)
    payload = tree_manifest(package)
    for name, expected in core.items():
        relative = None
        if name.startswith("python/pops/") and Path(name).suffix in {".py", ".pyi", ".typed"}:
            relative = name.removeprefix("python/pops/")
        elif name.startswith("include/") and name in {
            "include/" + line.split()[1]
            for line in (Path(source) / "include/pops_headers.manifest").read_text().splitlines()
            if line and not line.startswith("#") and line.split()[0] != "test-only"
        }:
            relative = name
        elif name == "include/pops_headers.manifest":
            relative = name
        if relative is not None:
            require(payload.get(relative) == expected, "installed source/SDK mismatch: " + name)
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature, pops_include

    require(
        module_header_signature() == pops_header_signature(package / "include"),
        "SDK/native ABI mismatch",
    )
    require(Path(pops_include()).resolve() == package / "include", "JIT uses another SDK")
    return {
        "base_revision": BASE_REVISION,
        "core_source": core,
        "installed_package": str(package),
        "installed_payload": payload,
        "native_path": str(Path(native.__file__).resolve()),
        "native_sha256": digest(native.__file__),
        "native_abi": native.abi_key(),
        "compiler": str(Path(native.__cxx_compiler__).resolve()),
        "compiler_sha256": digest(native.__cxx_compiler__),
        "wheel_path": str(Path(wheel).resolve()),
        "wheel_sha256": digest(wheel),
        "precision": "float64",
        "dimension": 2,
        "communicator": "serial",
    }


def loaded_paths():
    """Linux loader paths observed by the executing process, not an artifact catalogue."""
    return {
        line.split(None, 5)[5].strip()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].strip().startswith("/")
    }


@contextmanager
def exclusive_directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    yield path


def verify_capture(directory):
    """Integrity relative to the original execution log; not adversarial attestation."""
    directory = Path(directory)
    events = [json.loads(row) for row in (directory / "execution.jsonl").read_text().splitlines()]
    require(
        len(events) == 2 and events[0]["event"] == "start" and events[1]["event"] == "finish",
        "incomplete execution log",
    )
    start, finish = events
    require(
        start["pid"] == finish["pid"] and start["job_id"] == finish["job_id"], "producer identity"
    )
    require(finish["accepted_steps"] == 1, "no accepted native step")
    require(
        digest(directory / "actual-arrays.npz") == finish["output_sha256"],
        "original output replaced",
    )
    receipt = json.loads((directory / "capture.json").read_text())
    require(digest(directory / "capture.json") == finish["receipt_sha256"], "receipt replaced")
    require(receipt["core_before"] == receipt["core_after"], "core mutated during execution")
    for relative, expected in finish["files"].items():
        path = directory / relative
        require(path.resolve().is_relative_to(directory.resolve()), "evidence path escapes root")
        require(digest(path) == expected, "retained evidence changed: " + relative)
    require(
        digest(directory / "actual-bind-inputs.npz") == start["input_sha256"], "bind inputs changed"
    )
    records = list((directory / "compiler").glob("*/invocation.json"))
    successful = [json.loads(p.read_text()) for p in records]
    successful = [r for r in successful if r["returncode"] == 0 and r.get("output_sha256")]
    require(successful, "no actual compiler invocations")
    compiled = {r["output_sha256"] for r in successful}
    for row in receipt["loaded_binaries"]:
        require(row["original_path"] in receipt["loaded_paths"], "binary not observed loaded")
        require(
            digest(directory / row["retained"]) == row["sha256"], "retained loaded binary changed"
        )
        if row["role"] != "core":
            require(row["sha256"] in compiled, "loaded binary differs from compiler output")
    return receipt
