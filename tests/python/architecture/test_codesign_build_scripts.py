"""Source-only contracts for manifest-driven post-install Darwin code-signing."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "codesign_pops_extensions.py"
SETUP = ROOT / "scripts" / "setup_env.sh"
BUILD = ROOT / "scripts" / "build_python.sh"
VERIFY_NATIVE = ROOT / "scripts" / "verify_installed_native.py"


def _helper():
    spec = importlib.util.spec_from_file_location("_pops_codesign_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _installed_variant(helper, root: Path, dimension: int = 2, payload: bytes = b"native"):
    native = root / "_native"
    extension = native / f"dim{dimension}" / (
        "_pops" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    extension.parent.mkdir(parents=True)
    extension.write_bytes(payload)
    row = {
        "dimension": dimension,
        "path": extension.relative_to(native).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "version": "1.0.0",
        "abi_key": f"abi-dim{dimension}",
        "has_mpi": True,
        "has_kokkos": True,
    }
    helper.write_manifest_atomic(native / "variants.json", [row])
    return helper.InstalledNativeVariant(dimension, extension.resolve(), row)


def test_locator_resolves_only_the_exact_manifest_leaf_without_importing_pops(
    tmp_path, monkeypatch,
):
    helper = _helper()
    package_root = tmp_path / "pops"
    variant = _installed_variant(helper, package_root)
    package = importlib.util.spec_from_loader("pops", loader=None, is_package=True)
    assert package is not None
    package.submodule_search_locations = [str(package_root)]
    monkeypatch.setattr(helper.importlib.util, "find_spec", lambda name: package)

    before = sys.modules.get("pops")
    located = helper.locate_installed_pops_variants((2,))

    assert [(item.dimension, item.path) for item in located] == [(2, variant.path)]
    assert sys.modules.get("pops") is before


def test_locator_rejects_unmanifested_root_or_dimension_leaf(tmp_path, monkeypatch):
    helper = _helper()
    package_root = tmp_path / "pops"
    _installed_variant(helper, package_root)
    package = importlib.util.spec_from_loader("pops", loader=None, is_package=True)
    assert package is not None
    package.submodule_search_locations = [str(package_root)]
    monkeypatch.setattr(helper.importlib.util, "find_spec", lambda name: package)
    stale = package_root / "_native" / (
        "_pops" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    stale.write_bytes(b"legacy root")

    with pytest.raises(helper.CodesignError, match="unmanifested native extension"):
        helper.locate_installed_pops_variants((2,))


def test_locator_can_select_one_explicit_leaf_from_a_fat_manifest(tmp_path, monkeypatch):
    helper = _helper()
    package_root = tmp_path / "pops"
    dim1 = _installed_variant(helper, package_root, dimension=1, payload=b"dim1")
    dim3 = _installed_variant(helper, package_root, dimension=3, payload=b"dim3")
    helper.write_manifest_atomic(
        package_root / "_native" / "variants.json", [dim1.row, dim3.row]
    )
    package = importlib.util.spec_from_loader("pops", loader=None, is_package=True)
    assert package is not None
    package.submodule_search_locations = [str(package_root)]
    monkeypatch.setattr(helper.importlib.util, "find_spec", lambda name: package)

    located = helper.locate_installed_pops_variants((3,))

    assert [(item.dimension, item.path) for item in located] == [(3, dim3.path)]


def test_non_darwin_never_locates_or_invokes_codesign(monkeypatch):
    helper = _helper()
    monkeypatch.setattr(helper.sys, "platform", "linux")
    monkeypatch.setattr(
        helper, "locate_installed_pops_variants",
        lambda *args, **kwargs: pytest.fail("non-Darwin must not inspect the extension"))
    monkeypatch.setattr(
        helper.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("non-Darwin must not invoke codesign"))

    assert helper.codesign_imported_extensions((2,)) == ()


def test_darwin_preserves_valid_signatures_and_refreshes_final_manifest_hash(
    tmp_path, monkeypatch,
):
    helper = _helper()
    variant = _installed_variant(helper, tmp_path / "pops", payload=b"signed extension")
    calls = []

    def run(command, **kwargs):
        calls.append(tuple(command))
        evidence = "Signature=adhoc\n" if "--display" in command else ""
        return subprocess.CompletedProcess(command, 0, "", evidence)

    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(
        helper, "locate_installed_pops_variants", lambda dimensions, **kwargs: (variant,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(helper.subprocess, "run", run)

    authenticated = helper.codesign_imported_extensions((2,))

    assert authenticated[0].row["sha256"] == hashlib.sha256(variant.path.read_bytes()).hexdigest()
    assert calls == [
        ("/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(variant.path)),
        ("/usr/bin/codesign", "--display", "--verbose=4", str(variant.path)),
    ]


def test_darwin_repairs_then_records_the_post_sign_bytes(tmp_path, monkeypatch):
    helper = _helper()
    variant = _installed_variant(helper, tmp_path / "pops", payload=b"unsigned")
    calls = []

    def run(command, **kwargs):
        calls.append(tuple(command))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, "", "unsigned")
        if "--force" in command:
            variant.path.write_bytes(b"signed bytes")
        evidence = "Signature=adhoc\n" if "--display" in command else ""
        return subprocess.CompletedProcess(command, 0, "", evidence)

    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(
        helper, "locate_installed_pops_variants", lambda dimensions, **kwargs: (variant,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(helper.subprocess, "run", run)

    authenticated = helper.codesign_imported_extensions((2,))

    final_digest = hashlib.sha256(b"signed bytes").hexdigest()
    assert authenticated[0].row["sha256"] == final_digest
    manifest_rows = helper.load_manifest(
        variant.path.parents[1] / "variants.json", expected_dimensions=(2,)
    )
    assert manifest_rows[0]["sha256"] == final_digest
    assert calls == [
        ("/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(variant.path)),
        ("/usr/bin/codesign", "--force", "--sign", "-", str(variant.path)),
        ("/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(variant.path)),
        ("/usr/bin/codesign", "--display", "--verbose=4", str(variant.path)),
    ]


def test_structured_evidence_binds_dimension_and_post_sign_bytes(tmp_path, monkeypatch):
    helper = _helper()
    variant = _installed_variant(helper, tmp_path / "pops", payload=b"signed extension")
    monkeypatch.setattr(helper.sys, "platform", "darwin")

    evidence = helper.codesign_evidence((variant,))

    assert evidence == {
        "schema_version": 2,
        "platform": "darwin",
        "extensions": [
            {
                "dimension": 2,
                "path": str(variant.path),
                "sha256": hashlib.sha256(variant.path.read_bytes()).hexdigest(),
                "signature": "adhoc",
            }
        ],
    }


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_darwin_codesign_or_verification_failure_is_explicit(
    tmp_path, monkeypatch, failure_call,
):
    helper = _helper()
    variant = _installed_variant(helper, tmp_path / "pops")
    calls = []

    def run(command, **kwargs):
        call = len(calls)
        calls.append(tuple(command))
        if call == 0:
            return subprocess.CompletedProcess(command, 1, "", "unsigned")
        if call == failure_call:
            return subprocess.CompletedProcess(command, 9, "", "signature failure")
        evidence = "Signature=adhoc\n" if "--display" in command else ""
        return subprocess.CompletedProcess(command, 0, "", evidence)

    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(
        helper, "locate_installed_pops_variants", lambda dimensions, **kwargs: (variant,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(helper.subprocess, "run", run)

    with pytest.raises(helper.CodesignError, match=r"failed \(exit 9\): signature failure"):
        helper.codesign_imported_extensions((2,))


def test_darwin_refuses_a_verified_non_ad_hoc_signature(tmp_path, monkeypatch):
    helper = _helper()
    variant = _installed_variant(helper, tmp_path / "pops")
    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(
        helper, "locate_installed_pops_variants", lambda dimensions, **kwargs: (variant,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(
        helper.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "", "Authority=Developer ID\n"))

    with pytest.raises(helper.CodesignError, match="signature is not ad hoc"):
        helper.codesign_imported_extensions((2,))


def test_if_present_skips_only_an_absent_package(monkeypatch):
    helper = _helper()
    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper, "locate_installed_pops_variants", lambda *args, **kwargs: ())

    assert helper.codesign_imported_extensions((2,), if_present=True) == ()

    def missing_manifest(*args, **kwargs):
        raise helper.CodesignError("package has no manifest")

    monkeypatch.setattr(helper, "locate_installed_pops_variants", missing_manifest)
    with pytest.raises(helper.CodesignError, match="package has no manifest"):
        helper.codesign_imported_extensions((2,), if_present=True)


def test_scripts_pass_exact_dimension_before_every_native_import_or_doctor():
    setup = SETUP.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")
    helper_call = "codesign_pops_extensions.py"
    verifier_call = "verify_installed_native.py"

    assert setup.index(helper_call) < setup.index(verifier_call)
    assert 'python -c "import pops"' not in setup
    assert "--expect-dim \"$NATIVE_DIM\"" in setup
    assert build.index('python -m pip "${pip_args[@]}"') \
        < build.index(helper_call) \
        < build.index(verifier_call) \
        < build.index("select_native_dimension($POPS_NATIVE_DIM)")
    assert "--expect-dim \"$POPS_NATIVE_DIM\"" in build
    assert "--expect-mpi --expect-parallel-hdf5" in build
    assert "PYTHONPATH= PYTHONNOUSERSITE=1" in build
    assert VERIFY_NATIVE.is_file()


def test_codesign_command_is_reachable_only_after_the_darwin_guard():
    helper = HELPER.read_text(encoding="utf-8")
    assert helper.index('if sys.platform != "darwin"') \
        < helper.index('shutil.which("codesign")')
