"""Source guards for the changed internal C++ point; no Native module is loaded."""
import hashlib
from pathlib import Path
import subprocess
import pytest
from pops.codegen import abi, toolchain

ROOT = Path(__file__).resolve().parents[4]
BASE = '1e579f3b90f752b8d9341f659265d3c517e582f0'
CHANGED = ('pops/runtime/system/derived_aux_provider.hpp',
           'pops/runtime/system/auxiliary_checkpoint.hpp')


def header_signatures():
    current = toolchain.pops_header_signature(ROOT / 'include')
    entries = []
    rows = toolchain._installed_pops_header_rows(ROOT / 'include')
    assert all(('sdk-support', path) in rows for path in CHANGED)
    for category, relative in rows:
        data = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            BASE + ':include/' + relative]) if relative in CHANGED else (ROOT/'include'/relative).read_bytes()
        entries.append('%s %s\n%s\n' % (category, relative, hashlib.sha256(data).hexdigest()))
    previous = hashlib.sha256(''.join(entries).encode()).hexdigest()
    assert previous != current
    return previous, current


def test_signature_authenticates_both_changed_sdk_support_headers(monkeypatch):
    previous, current = header_signatures()
    monkeypatch.setattr(abi, 'module_header_signature', lambda: previous)
    with pytest.raises(RuntimeError, match='DO NOT MATCH'):
        toolchain._check_headers_match_module(ROOT/'include')
    monkeypatch.setattr(abi, 'module_header_signature', lambda: current)
    assert toolchain._check_headers_match_module(ROOT/'include') == current


def test_cached_package_wiring_rejects_old_headers_before_dlopen(monkeypatch):
    previous, current = header_signatures()
    monkeypatch.setattr(abi, 'loader_native_dimension', lambda: 2)
    monkeypatch.setattr(abi, 'module_header_signature', lambda: current)
    with pytest.raises(RuntimeError, match='headers DIFFERENT'):
        abi.check_compiled_matches_module(previous+'|host-compiler|c++20|dim=2')
    abi.check_compiled_matches_module(current+'|host-compiler|c++20|dim=2')
    with pytest.raises(RuntimeError, match='dimension'):
        abi.check_compiled_matches_module(current+'|host-compiler|c++20|dim=3')
