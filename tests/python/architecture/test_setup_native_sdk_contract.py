"""Official setup must persist the SDK declaration consumed by native admission."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _darwin_setup():
    setup = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")
    return setup.split('case "$(uname)" in\n  Darwin)\n', 1)[1].split("\n  Linux)", 1)[0]


def test_setup_preserves_explicit_sdk_and_discovers_only_selected_developer_sdk():
    setup = _darwin_setup()
    provided = 'if [[ -n "${SDKROOT+x}" ]]; then'
    explicit = 'declared_sdk_root="$SDKROOT"'
    discover = '/usr/bin/xcrun --sdk macosx --show-sdk-path'
    validate = 'SDKROOT="$declared_sdk_root" python - "$HERE"'
    assert setup.index(provided) < setup.index(explicit) < setup.index(discover) < setup.index(validate)
    assert 'conda env config vars set -n "$ENV_NAME"' in setup
    assert 'sdk_config_vars=("SDKROOT=$canonical_sdk_root")' in setup
    assert 'sdk_config_vars+=("DEVELOPER_DIR=$DEVELOPER_DIR")' in setup
    assert "-license accept" not in setup
    assert "xcode-select --switch" not in setup


def test_setup_validates_metadata_with_same_capture_before_env_mutation():
    setup = _darwin_setup()
    stream = 'conda run --no-capture-output -n "$ENV_NAME" env -u PYTHONPATH'
    capture = '"capture_native_toolchain"'
    root_guard = 'if [[ -z "$canonical_sdk_root" || "$canonical_sdk_root" != /*'
    persist = 'conda env config vars set -n "$ENV_NAME"'
    assert setup.index(stream) < setup.index(capture) < setup.index(root_guard) < setup.index(persist)
    assert '"python/pops/codegen/_native_toolchain.py"' in setup
    assert "print(capture.sysroot)" in setup


def test_setup_capture_imports_pops_from_the_checkout_on_a_fresh_env():
    # The capture runs before pops is installed and with PYTHONPATH unset, so the heredoc must put
    # the checkout's python/ directory on sys.path before it loads _native_toolchain.py.
    setup = _darwin_setup()
    checkout = 'sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "python"))'
    run = "capture = runpy.run_path("
    assert setup.index(checkout) < setup.index(run)


def _capture_heredoc():
    setup = _darwin_setup()
    return setup.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0] + "\n"


def _run_capture_as_fresh_env(tmp_path, **extra_env):
    # A fresh env has no installed pops: -I drops PYTHONPATH and the user site, -S drops
    # site-packages, and the working directory is outside the checkout, as in setup_env.sh.
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.pop("SDKROOT", None)
    environment.update(extra_env)
    return subprocess.run(
        [sys.executable, "-I", "-S", "-", str(ROOT)],
        input=_capture_heredoc(), text=True, capture_output=True, cwd=tmp_path,
        env=environment, timeout=120, check=False,
    )


def test_setup_capture_loads_the_checkout_without_an_installed_pops(tmp_path):
    result = _run_capture_as_fresh_env(tmp_path)
    assert "ModuleNotFoundError" not in result.stderr, result.stderr


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("xcrun") is None,
                    reason="the Darwin SDK capture needs the macOS developer tools")
def test_setup_capture_returns_the_sdk_on_a_fresh_env(tmp_path):
    declared = subprocess.run(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
                              text=True, capture_output=True, check=True).stdout.strip()
    result = _run_capture_as_fresh_env(tmp_path, SDKROOT=declared)
    assert result.returncode == 0, result.stderr
    sysroot = result.stdout.strip().splitlines()[-1]
    assert os.path.isabs(sysroot) and os.path.isdir(sysroot)
