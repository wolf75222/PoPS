"""Pure Python origin checks usable with checkout and installed CI packages."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def assert_current_python_sources(package, *modules):
    """Authenticate the inspected modules against the actual checkout bytes.

    This is Python source authority only; it loads no extension and makes no
    claim about the native binary's readiness or provenance.
    """
    source = (ROOT / "python" / "pops").resolve()
    observed = Path(package.__file__).resolve().parent
    assert observed == source or observed.is_relative_to(Path(sys.prefix).resolve())
    for module in (package, *modules):
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(observed), "mixed Python package origin: " + str(origin)
        expected = source / origin.relative_to(observed)
        assert expected.is_file(), "untracked Python module origin: " + str(origin)
        assert hashlib.sha256(origin.read_bytes()).digest() == hashlib.sha256(expected.read_bytes()).digest(), (
            "loaded Python source differs from checkout: " + str(origin))


def native_module_bindings():
    """Snapshot bindings to prove a source-only operation adds no native module."""
    return {name: id(module) for name, module in sys.modules.items()
            if name == "_pops" or name.endswith("._pops")}
