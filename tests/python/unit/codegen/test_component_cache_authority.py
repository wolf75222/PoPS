from pathlib import Path

from pops.codegen.cache import component_store_dir, pops_cache_dir


def test_component_store_uses_explicit_pops_cache_authority(tmp_path, monkeypatch):
    root = tmp_path / "configured-cache"
    monkeypatch.setenv("POPS_CACHE_DIR", str(root))

    assert Path(pops_cache_dir()) == root
    assert Path(component_store_dir()) == root / "component-store-v1"
    assert (root / "component-store-v1").is_dir()


def test_component_store_and_dsl_cache_share_xdg_pops_root(tmp_path, monkeypatch):
    monkeypatch.delenv("POPS_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    root = tmp_path / "xdg" / "pops"
    assert Path(pops_cache_dir()) == root / "dsl"
    assert Path(component_store_dir()) == root / "component-store-v1"
