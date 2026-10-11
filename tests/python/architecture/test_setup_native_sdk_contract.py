"""Official setup must persist the SDK declaration consumed by native admission."""
from pathlib import Path


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
