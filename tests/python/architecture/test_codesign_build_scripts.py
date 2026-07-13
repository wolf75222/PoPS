"""ADC-647 source-only tests for post-install Darwin code-signing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "codesign_pops_extensions.py"
SETUP = ROOT / "scripts" / "setup_env.sh"
BUILD = ROOT / "scripts" / "build_python.sh"


def _helper():
    spec = importlib.util.spec_from_file_location("_pops_codesign_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locator_resolves_the_exact_child_extension_without_importing_pops(tmp_path, monkeypatch):
    helper = _helper()
    package = importlib.util.spec_from_loader("pops", loader=None, is_package=True)
    assert package is not None
    package.submodule_search_locations = [str(tmp_path)]
    extension = tmp_path / ("_pops" + helper.importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.touch()
    child = importlib.util.spec_from_file_location("pops._pops", extension)
    monkeypatch.setattr(helper.importlib.util, "find_spec", lambda name: package)
    monkeypatch.setattr(
        helper.importlib.machinery.PathFinder, "find_spec",
        staticmethod(lambda name, locations: child))

    before = sys.modules.get("pops")
    assert helper.locate_imported_pops_extensions() == (extension.resolve(),)
    assert sys.modules.get("pops") is before


def test_non_darwin_never_locates_or_invokes_codesign(monkeypatch):
    helper = _helper()
    monkeypatch.setattr(helper.sys, "platform", "linux")
    monkeypatch.setattr(
        helper, "locate_imported_pops_extensions",
        lambda: pytest.fail("non-Darwin must not inspect the extension"))
    monkeypatch.setattr(
        helper.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("non-Darwin must not invoke codesign"))

    assert helper.codesign_imported_extensions() == ()


def test_darwin_signs_then_verifies_and_authenticates_ad_hoc_signature(tmp_path, monkeypatch):
    helper = _helper()
    extension = tmp_path / "_pops.so"
    extension.touch()
    calls = []

    def run(command, **kwargs):
        calls.append(tuple(command))
        evidence = "Signature=adhoc\n" if "--display" in command else ""
        return subprocess.CompletedProcess(command, 0, "", evidence)

    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper, "locate_imported_pops_extensions", lambda: (extension,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(helper.subprocess, "run", run)

    assert helper.codesign_imported_extensions() == (extension,)
    assert calls == [
        ("/usr/bin/codesign", "--force", "--sign", "-", str(extension)),
        ("/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(extension)),
        ("/usr/bin/codesign", "--display", "--verbose=4", str(extension)),
    ]


@pytest.mark.parametrize("failure_call", [0, 1])
def test_darwin_codesign_or_verification_failure_is_explicit(
    tmp_path, monkeypatch, failure_call,
):
    helper = _helper()
    extension = tmp_path / "_pops.so"
    extension.touch()
    calls = []

    def run(command, **kwargs):
        call = len(calls)
        calls.append(tuple(command))
        if call == failure_call:
            return subprocess.CompletedProcess(command, 9, "", "signature failure")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper, "locate_imported_pops_extensions", lambda: (extension,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(helper.subprocess, "run", run)

    with pytest.raises(helper.CodesignError, match=r"failed \(exit 9\): signature failure"):
        helper.codesign_imported_extensions()


def test_darwin_refuses_a_verified_non_ad_hoc_signature(tmp_path, monkeypatch):
    helper = _helper()
    extension = tmp_path / "_pops.so"
    extension.touch()

    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper, "locate_imported_pops_extensions", lambda: (extension,))
    monkeypatch.setattr(helper.shutil, "which", lambda command: "/usr/bin/codesign")
    monkeypatch.setattr(
        helper.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "", "Authority=Developer ID\n"))

    with pytest.raises(helper.CodesignError, match="signature is not ad hoc"):
        helper.codesign_imported_extensions()


def test_if_present_skips_only_an_absent_package(monkeypatch):
    helper = _helper()
    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper, "locate_imported_pops_extensions", lambda: ())

    assert helper.codesign_imported_extensions(if_present=True) == ()
    with pytest.raises(helper.CodesignError, match="was not found after build/install"):
        helper.codesign_imported_extensions(if_present=False)
    def missing_extension():
        raise helper.CodesignError("package has no extension")

    monkeypatch.setattr(helper, "locate_imported_pops_extensions", missing_extension)
    with pytest.raises(helper.CodesignError, match="package has no extension"):
        helper.codesign_imported_extensions(if_present=True)


def test_scripts_run_the_shared_helper_before_every_import_or_doctor():
    setup = SETUP.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")
    helper_call = "codesign_pops_extensions.py"

    assert setup.index(helper_call) < setup.index('python -c "import pops"')
    assert build.index('python -m pip "${pip_args[@]}"') \
        < build.index(helper_call) \
        < build.index('python -c "import pops;')
    assert "PYTHONPATH= PYTHONNOUSERSITE=1" in build
    assert setup.count("PYTHONPATH= PYTHONNOUSERSITE=1") == 3


def test_codesign_command_is_reachable_only_after_the_darwin_guard():
    helper = HELPER.read_text(encoding="utf-8")
    assert helper.index('if sys.platform != "darwin"') \
        < helper.index('shutil.which("codesign")')
