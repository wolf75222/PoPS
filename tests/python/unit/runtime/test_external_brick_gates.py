"""ADC-544 compile-time validation gates for a CompiledBrickRef.

A :class:`pops.external.CompiledBrickRef` is VALIDATED before any use, at resolve / compile time,
and every refusal RAISES (never warns). This suite exercises the four gates against FAKE manifests
(pure Python, no compiler): every gate, valid + each invalid variant, plus the no-false-positive
SKIPs (a gate that is not checkable in the given scope must NOT reject). An INCOMPLETE manifest is
REFUSED (the strict versioned schema), never warned.

A compiler-gated sibling (:func:`test_gated_brick_so_*`) reuses the ``_compile_brick_so`` idiom from
``test_external_bricks.py`` to build a real ``.so`` and drive the G4 dlsym probe on a loaded handle;
it self-skips when no C++ compiler / pops headers are present -- the fake-manifest tests above are the
authoritative local proof, and its pre-compile call path still executes here so the gate wiring is
locally exercised.

Pure functions + real classes are used; pops is never faked.
"""
import json
import os
import shutil
import subprocess

import pytest

_desc = pytest.importorskip("pops.descriptors")
_gates = pytest.importorskip("pops.external._brick_gates")
_bricks = pytest.importorskip("pops.external.bricks")
CompiledBrickRef = _bricks.CompiledBrickRef

from tests.python.support.requirements import repo_include
_INCLUDE = repo_include()


@pytest.fixture(autouse=True)
def _clean_catalog():
    _desc._clear_external_catalog()
    yield
    _desc._clear_external_catalog()


def _record(**over):
    """A parsed per-brick record (parse_brick_manifest output shape) with sane defaults."""
    rec = {"id": "my_ext", "native_id": "my_ext", "category": "riemann",
           "requirements": [], "capabilities": [], "supported_layouts": [],
           "supported_platforms": [], "params": [], "options": [], "exported_symbols": []}
    rec.update(over)
    return rec


def _write_manifest(tmp_path, entry, *, schema_version=None, name="bricks.json", **top):
    """Write a strict current brick manifest .json carrying @p entry, return its path."""
    row = {
        "native_id": entry.get("id", "my_ext"), "supported_layouts": "",
        "supported_platforms": "", "params": "", "options": "", "exported_symbols": "",
    }
    row.update(entry)
    doc = {
        "schema_version": (_desc.BRICK_MANIFEST_SCHEMA_VERSION
                           if schema_version is None else schema_version),
        "abi_key": _gates._module_abi_key(), "annotations": {}, "bricks": [row],
    }
    doc.update(top)
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


# ---- G1 ABI mismatch (RuntimeError) -----------------------------------------------------------

def test_g1_abi_mismatch_raises_runtime_error():
    with pytest.raises(RuntimeError) as exc:
        _gates.check_abi(_record(), "bogus_key=1", module_abi_key="real_key=2")
    msg = str(exc.value)
    assert "was compiled with an ABI key DIFFERENT from the loaded module" in msg
    assert "bogus_key=1" in msg and "real_key=2" in msg
    assert "recompile" in msg and "SAME toolchain" in msg


def test_g1_matching_abi_passes():
    _gates.check_abi(_record(), "same_key=1", module_abi_key="same_key=1")  # no raise


def test_g1_skips_when_manifest_has_no_key():
    # A .json manifest with no abi_key is not checkable -> SKIP, never a false reject.
    _gates.check_abi(_record(), None, module_abi_key="real_key=2")


def test_g1_skips_when_module_key_unavailable():
    # _pops absent -> module key is the placeholder -> not checkable -> SKIP.
    _gates.check_abi(_record(), "bogus_key=1", module_abi_key="abi_key=unavailable")
    _gates.check_abi(_record(), "bogus_key=1", module_abi_key="")


# ---- G2 missing capability (ValueError) -------------------------------------------------------

def test_g2_missing_capability_raises_value_error():
    rec = _record(requirements=["pressure", "wave_speeds"])
    with pytest.raises(ValueError) as exc:
        _gates.check_capabilities(rec, {"capabilities": ["density"]})
    msg = str(exc.value)
    assert "requires capability" in msg
    assert "not provided by the model" in msg
    assert "available capabilities" in msg


def test_g2_all_capabilities_provided_passes():
    rec = _record(requirements=["pressure", "wave_speeds"])
    _gates.check_capabilities(rec, {"capabilities": ["pressure", "wave_speeds", "density"]})


def test_g2_skips_without_capability_info():
    # No model / no explicit capability set -> not checkable -> SKIP (no false positive).
    rec = _record(requirements=["pressure"])
    _gates.check_capabilities(rec, {})
    _gates.check_capabilities(rec, None)


def test_g2_no_requirements_passes():
    _gates.check_capabilities(_record(requirements=[]), {"capabilities": []})


# ---- G3 unsupported layout (ValueError) -------------------------------------------------------

def test_g3_unsupported_layout_raises_value_error():
    rec = _record(supported_layouts=["uniform"])
    with pytest.raises(ValueError) as exc:
        _gates.check_layout(rec, {"layout": "amr"})
    msg = str(exc.value)
    assert "does not support layout=amr" in msg
    assert "supported layouts are" in msg


def test_g3_supported_layout_passes():
    rec = _record(supported_layouts=["uniform", "amr"])
    _gates.check_layout(rec, {"layout": "amr"})


def test_g3_skips_when_layouts_unconstrained():
    # Empty supported_layouts = unconstrained/unknown -> NOT a rejection (no false positive).
    _gates.check_layout(_record(supported_layouts=[]), {"layout": "amr"})


def test_g3_skips_without_requested_layout():
    rec = _record(supported_layouts=["uniform"])
    _gates.check_layout(rec, {})  # no requested layout -> not checkable -> SKIP


# ---- G4 missing symbol (ValueError) -----------------------------------------------------------

class _FakeHandle:
    """A ctypes-like handle: getattr(name) succeeds for present symbols, raises AttributeError else."""

    def __init__(self, symbols):
        self._symbols = set(symbols)

    def __getattr__(self, name):
        if name in self._symbols:
            return lambda *a, **k: None
        raise AttributeError(name)


def test_g4_missing_symbol_raises_value_error():
    rec = _record(exported_symbols=["pops_brick_residual", "pops_brick_missing"])
    handle = _FakeHandle(["pops_brick_residual"])  # second symbol absent
    with pytest.raises(ValueError) as exc:
        _gates.check_symbols(rec, handle)
    msg = str(exc.value)
    assert "does not export symbol pops_brick_missing()" in msg
    assert "rebuild the .so with the expected entry point" in msg


def test_g4_present_symbols_pass():
    rec = _record(exported_symbols=["pops_brick_residual"])
    _gates.check_symbols(rec, _FakeHandle(["pops_brick_residual", "unrelated"]))


def test_g4_json_manifest_skips_probe():
    # A .json-only manifest has no .so to probe (handle None) -> G4 SKIPPED (honest note).
    rec = _record(exported_symbols=["pops_brick_residual"])
    _gates.check_symbols(rec, None)


def test_g4_no_symbols_declared_passes():
    _gates.check_symbols(_record(exported_symbols=[]), _FakeHandle([]))


# ---- validate_ref ordering: G1 fires before the others ----------------------------------------

def test_validate_ref_raises_on_first_failing_gate():
    # A record that fails G1 (ABI) AND G2 (capability): the RuntimeError (G1) wins the ordering.
    rec = _record(requirements=["pressure"], supported_layouts=["uniform"])
    with pytest.raises(RuntimeError):
        _gates.validate_ref(rec, manifest_abi_key="bogus=1",
                            context={"capabilities": [], "layout": "amr"},
                            module_abi_key="real=2")


# ---- CompiledBrickRef.resolve() happy path + gate integration ---------------------------------

def test_resolve_happy_path_from_fake_json(tmp_path):
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "", "capabilities": ""})
    ref = CompiledBrickRef(manifest=path, native_id="my_ext", expect_category="riemann")
    d = ref.resolve()
    assert d.brick_type == "external_cpp"
    assert d.native_id == "my_ext"


def test_resolve_raises_g2_from_fake_json(tmp_path):
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "pressure", "capabilities": ""})
    ref = CompiledBrickRef(manifest=path, native_id="my_ext")
    with pytest.raises(ValueError) as exc:
        ref.resolve(context={"capabilities": ["density"]})
    assert "requires capability" in str(exc.value)


def test_resolve_raises_g3_from_fake_json_with_expect_layout(tmp_path):
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "", "capabilities": "",
                                      "supported_layouts": "uniform"})
    # The ref pins the route context (expect_layouts) so the layout gate fires at resolve.
    ref = CompiledBrickRef(manifest=path, native_id="my_ext", expect_layouts=["amr"])
    with pytest.raises(ValueError) as exc:
        ref.resolve()
    assert "does not support layout=amr" in str(exc.value)


def test_resolve_skips_g4_for_json_manifest(tmp_path):
    # A .json manifest listing exported_symbols has no .so -> G4 skipped -> resolve succeeds.
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "", "capabilities": "",
                                      "exported_symbols": "pops_brick_residual"})
    ref = CompiledBrickRef(manifest=path, native_id="my_ext")
    assert ref.resolve().native_id == "my_ext"


def test_available_degrades_to_no_carrying_the_gate_reason(tmp_path):
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "pressure", "capabilities": ""})
    ref = CompiledBrickRef(manifest=path, native_id="my_ext")
    status = ref.available(context={"capabilities": ["density"]})
    assert not status.ok
    assert "requires capability" in status.reason


# ---- incomplete manifest is REFUSED, not warned -----------------------------------------------

def test_incomplete_manifest_missing_required_field_is_refused(tmp_path):
    # An entry missing the required 'capabilities' field is REFUSED by the strict parser (via resolve).
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "pressure"})  # no capabilities
    ref = CompiledBrickRef(manifest=path, native_id="my_ext")
    with pytest.raises(ValueError) as exc:
        ref.resolve()
    assert "capabilities" in str(exc.value) and "missing" in str(exc.value)


def test_v1_manifest_is_refused(tmp_path):
    # A pre-v2 manifest is refused (refuse-never-warn on the versioned wire format).
    path = _write_manifest(tmp_path, {"id": "my_ext", "category": "riemann",
                                      "requirements": "", "capabilities": ""},
                           schema_version=1)
    ref = CompiledBrickRef(manifest=path, native_id="my_ext")
    with pytest.raises(ValueError) as exc:
        ref.resolve()
    assert "schema_version" in str(exc.value)


# ---- compiler-gated .so / dlsym lane ----------------------------------------------------------

# A brick .so whose manifest declares an exported symbol. G4 is exercised against the REAL dlopened
# ctypes handle the loader returns (manifests.register_and_capture) so the dlsym probe runs on THIS
# .so's own symbols (the STB_GNU_UNIQUE / ADC-622 contract). G4 is driven DIRECTLY here rather than
# through resolve(): a header-light registry .so (no Kokkos, no matching POPS_HEADER_SIG) carries a
# deliberately mismatched abi_key, so resolve()'s G1 gate would fire first -- correctly. Isolating G4
# on the real handle proves the dlsym contract with a real .so; the fake-manifest tests above are the
# authoritative local proof of G1 and of the full resolve() ordering.
_BRICK_SRC_TMPL = """
#include <pops/runtime/program/external_brick.hpp>
#include <string>
namespace {{
::pops::runtime::program::BrickManifestEntry make_entry() {{
  return {{"gated_brick", "riemann", "", "", "gated_brick", "uniform", "cpu", "", "", "{symbols}"}};
}}
const bool registered = [] {{
  ::pops::runtime::program::BrickRegistry::instance().register_brick(make_entry());
  return true;
}}();
}}
POPS_DEFINE_BRICK_MANIFEST();
"""


def _compile_gated_brick_so(workdir, exported_symbols):
    """Compile a brick .so whose manifest declares @p exported_symbols; None if toolchain unusable."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(_INCLUDE):
        return None
    src = os.path.join(workdir, "gated_brick.cpp")
    so = os.path.join(workdir, "gated_brick.so")
    with open(src, "w") as f:
        f.write(_BRICK_SRC_TMPL.format(symbols=exported_symbols))
    flags = ["-shared", "-fPIC", "-std=c++20", "-O0", "-I", _INCLUDE]
    if os.uname().sysname == "Darwin":
        flags += ["-undefined", "dynamic_lookup"]
    try:
        subprocess.run([cxx, *flags, src, "-o", so], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError):
        return None
    return so


def _capture_gated_record(so, native_id="gated_brick"):
    """Dlopen the gated .so via the loader, returning (record, handle) for the G4 probe."""
    from pops.external.manifests import register_and_capture
    records, _abi, handle = register_and_capture(so)
    record = next((r for r in records if r["native_id"] == native_id), None)
    return record, handle


def test_gated_brick_so_g4_passes_for_present_symbol(tmp_path):
    # The .so ALWAYS exports pops_brick_manifest() (POPS_DEFINE_BRICK_MANIFEST): declare it and the
    # dlsym probe on the REAL handle finds it -> G4 passes.
    so = _compile_gated_brick_so(str(tmp_path), "pops_brick_manifest")
    if so is None:
        pytest.skip("no C++ compiler or pops headers to build the gated brick .so")
    record, handle = _capture_gated_record(so)
    assert record is not None and handle is not None
    _gates.check_symbols(record, handle)  # probes pops_brick_manifest on THIS handle -> present


def test_gated_brick_so_g4_raises_for_missing_symbol(tmp_path):
    # The manifest declares a symbol the .so does NOT export -> the dlsym probe (G4) on the real
    # handle raises. This is the compiler-gated proof of the G4 contract with a real .so.
    so = _compile_gated_brick_so(str(tmp_path), "pops_brick_absent_symbol")
    if so is None:
        pytest.skip("no C++ compiler or pops headers to build the gated brick .so")
    record, handle = _capture_gated_record(so)
    assert record is not None and handle is not None
    with pytest.raises(ValueError) as exc:
        _gates.check_symbols(record, handle)
    assert "does not export symbol pops_brick_absent_symbol()" in str(exc.value)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
