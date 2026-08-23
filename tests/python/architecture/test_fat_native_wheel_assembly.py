"""Source-only proofs for deterministic Dim=1/2/3 release-wheel assembly."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
WHEEL_NAME = "pops-0.3.0-cp312-cp312-macosx_11_0_arm64.whl"
RECORD = "pops-0.3.0.dist-info/RECORD"
MANIFEST = "pops/_native/variants.json"
BUILD_FINGERPRINT = "a" * 64


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assembler = _load(
    "_fat_native_wheel_assembly_test",
    SCRIPTS / "assemble_native_variant_wheel.py",
)


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _record(payloads: dict[str, bytes], *, corrupt: bool = False) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(payloads):
        digest = _record_digest(payloads[name])
        if corrupt and name == "pops/__init__.py":
            digest = "sha256=" + "A" * 43
        writer.writerow((name, digest, str(len(payloads[name]))))
    writer.writerow((RECORD, "", ""))
    return stream.getvalue().encode("utf-8")


def _mono_wheel(
    root: Path,
    dimension: int,
    *,
    abi_common: str = "clang=17;cxx=20;kokkos=4.4.1;mpi=off",
    build_fingerprint: str = BUILD_FINGERPRINT,
    common_payload: bytes = b"identical pure Python payload\n",
    has_mpi: bool = False,
    has_kokkos: bool = True,
    corrupt_record: bool = False,
    extra_native: bool = False,
) -> Path:
    wheel = root / f"dim{dimension}" / WHEEL_NAME
    wheel.parent.mkdir(parents=True)
    leaf = f"pops/_native/dim{dimension}/_pops.so"
    native = f"repaired-and-signed-Dim={dimension}".encode()
    # This deliberately records the pre-repair digest. The assembler must replace it with the
    # digest of ``native`` and prove those final repaired bytes, not reject a legitimate signature.
    row = {
        "dimension": dimension,
        "path": f"dim{dimension}/_pops.so",
        "sha256": hashlib.sha256(b"pre-repair image").hexdigest(),
        "version": "0.3.0",
        "abi_key": f"{abi_common};dim={dimension}",
        "build_fingerprint": build_fingerprint,
        "has_mpi": has_mpi,
        "has_kokkos": has_kokkos,
    }
    payloads = {
        "pops/__init__.py": common_payload,
        "pops-0.3.0.dist-info/METADATA": (
            b"Metadata-Version: 2.3\nName: PoPS\nVersion: 0.3.0\n"
        ),
        "pops-0.3.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            b"Tag: cp312-cp312-macosx_11_0_arm64\n"
        ),
        MANIFEST: (
            json.dumps(
                {"schema_version": 2, "variants": [row]},
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode(),
        leaf: native,
    }
    if extra_native:
        payloads["pops/_native/hidden/_pops.so"] = b"undeclared native leaf"
    payloads[RECORD] = _record(payloads, corrupt=corrupt_record)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    return wheel


def _inputs(
    root: Path,
    overrides: dict[int, dict] | None = None,
) -> dict[int, Path]:
    overrides = {} if overrides is None else overrides
    return {
        dimension: _mono_wheel(root, dimension, **overrides.get(dimension, {}))
        for dimension in (1, 2, 3)
    }


def _assemble(
    inputs: dict[int, Path],
    output: Path,
    calls: list[int],
) -> dict:
    def prove(_python: Path, _extracted: Path, row: dict) -> dict:
        calls.append(row["dimension"])
        return dict(row)

    return assembler.assemble_native_variant_wheel(
        inputs,
        output=output,
        python_executable=Path(sys.executable),
        expect_mpi=False,
        expect_kokkos=True,
        signature_policy="none",
        proof_runner=prove,
    )


def _assert_record(archive: zipfile.ZipFile) -> None:
    rows = list(csv.reader(io.StringIO(archive.read(RECORD).decode(), newline="")))
    by_name = {name: (digest, size) for name, digest, size in rows}
    files = {info.filename for info in archive.infolist() if not info.is_dir()}
    assert set(by_name) == files
    assert by_name[RECORD] == ("", "")
    for name in files - {RECORD}:
        payload = archive.read(name)
        assert by_name[name] == (_record_digest(payload), str(len(payload)))


def test_assembly_is_exact_deterministic_and_reproves_every_final_leaf(tmp_path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    calls: list[int] = []
    first = tmp_path / "out-a" / WHEEL_NAME
    second = tmp_path / "out-b" / WHEEL_NAME

    evidence = _assemble(inputs, first, calls)
    _assemble(inputs, second, calls)

    assert calls == [1, 2, 3, 1, 2, 3]
    assert first.read_bytes() == second.read_bytes()
    assert evidence["dimensions"] == [1, 2, 3]
    assert evidence["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        assert [row["dimension"] for row in manifest["variants"]] == [1, 2, 3]
        assert {
            row["path"] for row in manifest["variants"]
        } == {f"dim{dimension}/_pops.so" for dimension in (1, 2, 3)}
        for row in manifest["variants"]:
            member = "pops/_native/" + row["path"]
            assert row["sha256"] == hashlib.sha256(archive.read(member)).hexdigest()
            assert row["sha256"] != hashlib.sha256(b"pre-repair image").hexdigest()
            assert row["build_fingerprint"] == BUILD_FINGERPRINT
        _assert_record(archive)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({3: {"abi_common": "clang=18;cxx=20;kokkos=4.4.1;mpi=off"}}, "toolchain ABI"),
        ({3: {"build_fingerprint": "b" * 64}}, "build fingerprint"),
        ({3: {"common_payload": b"drifted package\n"}}, "outside native leaves"),
        ({3: {"has_mpi": True}}, "backend facts"),
        ({3: {"extra_native": True}}, "native members"),
        ({3: {"corrupt_record": True}}, "RECORD digest drifted"),
    ],
)
def test_assembly_rejects_inhomogeneous_or_unauthenticated_inputs(
    tmp_path, overrides, message,
) -> None:
    inputs = _inputs(tmp_path / "inputs", overrides)

    with pytest.raises(assembler.FatWheelAssemblyError, match=message):
        _assemble(inputs, tmp_path / "out" / WHEEL_NAME, [])


def test_assembly_requires_all_dimensions_and_an_absent_output(tmp_path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    output = tmp_path / "out" / WHEEL_NAME

    with pytest.raises(assembler.FatWheelAssemblyError, match="exactly dimensions"):
        _assemble({1: inputs[1], 2: inputs[2]}, output, [])
    output.parent.mkdir(parents=True)
    output.write_bytes(b"do not overwrite")
    with pytest.raises(assembler.FatWheelAssemblyError, match="absent .whl"):
        _assemble(inputs, output, [])


def test_cli_requires_explicit_dimensions_backend_signature_and_interpreter() -> None:
    source = (SCRIPTS / "assemble_native_variant_wheel.py").read_text(encoding="utf-8")

    for option in (
        "--variant",
        "--output",
        "--python",
        "--expect-mpi",
        "--expect-kokkos",
        "--signature-policy",
    ):
        assert f'parser.add_argument("{option}"' in source
    assert "required=True" in source
    assert "FAT_WHEEL_DIMENSIONS = (1, 2, 3)" in source
