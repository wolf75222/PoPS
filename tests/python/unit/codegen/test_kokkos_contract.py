from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(root: Path, *, core: bytes = b"core", config: bytes = b"config"):
    include = root / "include"
    include.mkdir(parents=True)
    paths = (include / "Kokkos_Core.hpp", include / "KokkosCore_config.h")
    paths[0].write_bytes(core)
    paths[1].write_bytes(config)
    hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    material = (
        f"include={include}\n"
        f"header={paths[0]};sha256={hashes[0]}\n"
        f"header={paths[1]};sha256={hashes[1]}\n"
    ).encode()
    return SimpleNamespace(
        __has_kokkos__=True,
        __kokkos_contract__={
            "schema_version": 1,
            "abi_sha256": hashlib.sha256(material).hexdigest(),
            "include_dirs": (str(include),),
            "header_paths": tuple(map(str, paths)),
            "header_sha256": hashes,
        },
    )


def _clear_overrides(monkeypatch):
    for name in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_baked_kokkos_contract_needs_no_shell_environment(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)

    root, includes, abi = toolchain._native_kokkos_selection()
    assert root == str(tmp_path / "built")
    assert includes == (str(tmp_path / "built" / "include"),)
    assert abi == module.__kokkos_contract__["abi_sha256"]

    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda _cxx: "c++")
    compile_flags, _ = toolchain._native_kokkos_flags()
    assert compile_flags[:4] == ["-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", includes[0]]


def test_explicit_relocated_root_must_match_baked_headers(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _module(tmp_path / "relocated")
    shutil.rmtree(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("POPS_KOKKOS_ROOT", str(tmp_path / "relocated"))
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)

    root, includes, _ = toolchain._native_kokkos_selection()
    assert root == str(tmp_path / "relocated")
    assert includes == (str(tmp_path / "relocated" / "include"),)


def test_explicit_different_kokkos_root_is_rejected(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _module(tmp_path / "other", config=b"different backend")
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("Kokkos_ROOT", str(tmp_path / "other"))
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)

    with pytest.raises(RuntimeError, match="differs from the installation"):
        toolchain._native_kokkos_selection()


def test_kokkos_contract_fails_closed_when_header_changes(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    (tmp_path / "built" / "include" / "KokkosCore_config.h").write_text("upgraded")

    with pytest.raises(RuntimeError, match="changed in place"):
        toolchain._native_kokkos_selection()
