"""Spec 3 external C++ bricks (ADC-463, criterion 20).

A Spec 3 brick is native / generated / macro / external-C++. These tests cover
the last category: ``pops.descriptors.load_cpp_library(path)`` dlopens a user ``.so``,
reads its JSON manifest (over the C++ ``BrickRegistry``), and registers the ids
in an in-process catalog. Riemann rows additionally need the authenticated v2 numerical ABI;
manifest-only Riemann identities are rejected rather than published as executable descriptors.
An id that was never loaded raises a CLEAR error.

The manifest-parsing seam (``_register_manifest``) is exercised directly so the
test needs no compiled ``.so``; ``load_cpp_library`` is the real ``.so`` path on
top of it. The real functions are used -- pops is never faked.
"""
import os
import json
import subprocess
import types

import pytest

# Spec 5 (sec.4): the brick-loader + generic external() live in pops.descriptors, and
# the riemann ``User`` selector in pops.numerics.riemann (formerly all under pops.lib).
import pops.descriptors as _desc
from pops.numerics import riemann

lib = types.SimpleNamespace(
    riemann=riemann,
    external=_desc.external,
    load_cpp_library=_desc.load_cpp_library,
    _register_manifest=_desc._register_manifest,
    _clear_external_catalog=_desc._clear_external_catalog,
)

from tests.python.support.requirements import default_cxx, repo_include, require_native_or_skip
_INCLUDE = repo_include()

# A minimal non-numerical external brick .so exercises the generic manifest path. The real Riemann
# static-dispatch ABI is compiled and executed by test_external_riemann_dispatch.cpp.
_BRICK_SRC = """
#include <pops/runtime/program/external_brick.hpp>
#include <string>
POPS_REGISTER_BRICK("my_so_preconditioner", "preconditioner", "linear_operator");
POPS_DEFINE_BRICK_MANIFEST();
"""

_LEGACY_RIEMANN_SRC = """
#include <pops/runtime/program/external_brick.hpp>
POPS_REGISTER_BRICK("legacy_riemann", "riemann", "physical_flux");
POPS_DEFINE_BRICK_MANIFEST();
extern "C" void pops_brick_residual() {}
"""


def _compile_brick_so(workdir, source=_BRICK_SRC, stem="external_brick"):
    """Compile the minimal brick to a .so; a compiler failure is never converted to a skip."""
    cxx = default_cxx()
    if not cxx or not os.path.isdir(_INCLUDE):
        require_native_or_skip(
            f"native brick prerequisites unavailable: compiler={cxx!r}, include={_INCLUDE}",
            optional_skip=pytest.skip,
        )
        raise AssertionError("require_native_or_skip must not return")
    src = os.path.join(workdir, stem + ".cpp")
    so = os.path.join(workdir, stem + ".so")
    with open(src, "w") as f:
        f.write(source)
    flags = ["-shared", "-fPIC", "-std=c++20", "-O0", "-I", _INCLUDE]
    # ADC-622: on GCC compile the brick .so with -fno-gnu-unique so the header-only BrickRegistry
    # singleton is never emitted STB_GNU_UNIQUE (the loader would otherwise unify it across every
    # dlopen'd brick .so). Belt-and-suspenders behind the hidden-visibility BrickRegistry; harmless on
    # a compiler that already isolates (Clang / AppleClang reject the flag, so gate it on g++).
    cxx_name = os.path.basename(cxx)
    if "g++" in cxx_name and "clang" not in cxx_name:
        flags.append("-fno-gnu-unique")
    if os.uname().sysname == "Darwin":
        flags.append("-undefined")
        flags.append("dynamic_lookup")
    subprocess.run([cxx, *flags, src, "-o", so], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return so


@pytest.fixture(autouse=True)
def _clean_catalog():
    """Reset the in-process external-brick catalog around each test (no leakage)."""
    lib._clear_external_catalog()
    yield
    lib._clear_external_catalog()


def _manifest(*entries, schema_version=None):
    """Build a STRICT versioned manifest (ADC-611 / ADC-544): stamp schema_version and fill each entry's
    four required fields (id / category / requirements / capabilities) with defaults so the happy-path
    tests exercise a valid payload. The ADC-544 v2 fields (native_id / supported_layouts / ...) are
    optional -- an entry may add them, but the required set is unchanged. A test that probes a MISSING
    field passes the raw dict via _register_manifest directly rather than through this helper."""
    normed = []
    for e in entries:
        row = {
            "category": "brick", "requirements": "", "capabilities": "",
            "supported_layouts": "", "supported_platforms": "", "params": "", "options": "",
            "exported_symbols": "",
        }
        row.update(e)
        if "id" in row:
            row.setdefault("native_id", row["id"])
        normed.append(row)
    return json.dumps({
        "schema_version": (_desc.BRICK_MANIFEST_SCHEMA_VERSION
                           if schema_version is None else schema_version),
        "abi_key": "test-abi", "annotations": {}, "bricks": normed,
    })


def test_unknown_external_id_raises_clear_error():
    # Not loaded -> a clear, actionable error naming the id and load_cpp_library.
    with pytest.raises(LookupError) as exc:
        lib.riemann.User("my_hllc")
    msg = str(exc.value)
    assert "my_hllc" in msg
    assert "not loaded" in msg
    assert "load_cpp_library" in msg


def test_generic_external_unknown_id_raises():
    with pytest.raises(LookupError) as exc:
        lib.external("nope")
    assert "nope" in str(exc.value)
    assert "load_cpp_library" in str(exc.value)


def test_register_manifest_then_user_surfaces_external_descriptor():
    n = lib._register_manifest(_manifest(
        {"id": "my_hllc", "category": "riemann",
         "requirements": "pressure,wave_speeds", "capabilities": "physical_flux"}))
    assert n == 1
    d = lib.riemann.User("my_hllc")
    assert d.brick_type == "external_cpp"
    assert d.category == "riemann"
    assert d.native_id == "my_hllc"
    # The CSV requirements/capabilities become list metadata on the descriptor.
    assert d.requirements == {"capabilities": ["pressure", "wave_speeds"]}
    assert d.capabilities == {"provides": ["physical_flux"]}


def test_generic_external_surfaces_descriptor_with_its_category():
    lib._register_manifest(_manifest(
        {"id": "my_precond", "category": "preconditioner", "requirements": ""}))
    d = lib.external("my_precond")
    assert d.brick_type == "external_cpp"
    assert d.category == "preconditioner"
    assert d.native_id == "my_precond"
    # No requirements -> empty metadata, never a fabricated capability.
    assert d.requirements == {}
    assert d.capabilities == {}


def test_user_category_must_match_when_registered_elsewhere():
    # Registered as a preconditioner; selecting it via riemann.User is a loud mismatch.
    lib._register_manifest(_manifest(
        {"id": "x", "category": "preconditioner", "requirements": ""}))
    with pytest.raises(ValueError) as exc:
        lib.riemann.User("x")
    assert "preconditioner" in str(exc.value)
    assert "riemann" in str(exc.value)


def test_manifest_must_be_well_formed():
    with pytest.raises(ValueError):
        lib._register_manifest("not json")
    with pytest.raises(ValueError):
        # An entry missing its id is rejected (no silently-dropped brick).
        lib._register_manifest(_manifest({"category": "riemann"}))


def test_load_cpp_library_rejects_a_missing_path():
    with pytest.raises(FileNotFoundError):
        lib.load_cpp_library("/no/such/brick.so")


def test_load_cpp_library_dlopens_a_real_non_numerical_brick_so(tmp_path):
    """The deferred half: compile a REAL brick .so, dlopen it via load_cpp_library, and assert
    riemann.User surfaces its manifest. Missing prerequisites may skip only in an explicitly optional
    source-only lane; once compilation starts, every compiler error fails the test."""
    so = _compile_brick_so(str(tmp_path))
    # The registry .so is header-light (only external_brick.hpp): plain flags, no Kokkos needed.
    n = lib.load_cpp_library(so)
    # ADC-622: this .so's manifest describes exactly ITS OWN one brick. The BrickRegistry singleton is
    # now hidden-visibility (POPS_BRICK_LOCAL, + -fno-gnu-unique on the GCC compile above), so it is
    # per-image: a sibling brick .so loaded earlier in this process no longer leaks its ids into this
    # manifest (before the fix, gcc emitted the registry STB_GNU_UNIQUE and glibc unified it across the
    # dlopen'd .so, so n was the process-wide count). The C++ two-fixture proof is
    # test_external_brick_isolation.cpp.
    assert n == 1
    d = lib.external("my_so_preconditioner")
    assert d.brick_type == "external_cpp"
    assert d.category == "preconditioner"
    assert d.native_id == "my_so_preconditioner"
    assert d.requirements == {"capabilities": ["linear_operator"]}


def test_load_cpp_library_rejects_legacy_unversioned_riemann_abi(tmp_path):
    so = _compile_brick_so(
        str(tmp_path), source=_LEGACY_RIEMANN_SRC, stem="legacy_riemann"
    )
    with pytest.raises(ValueError) as exc:
        lib.load_cpp_library(so)
    message = str(exc.value)
    assert "legacy_riemann" in message
    assert "legacy/unversioned" in message
    with pytest.raises(LookupError):
        lib.riemann.User("legacy_riemann")


def test_load_cpp_library_rejects_a_non_brick_so(tmp_path):
    """A loadable library that does NOT export pops_brick_manifest() is rejected clearly (it is not an
    pops brick .so), never silently treated as carrying zero bricks."""
    cxx = default_cxx()
    if not cxx:
        require_native_or_skip("no C++ compiler to build the non-brick .so",
                               optional_skip=pytest.skip)
        raise AssertionError("require_native_or_skip must not return")
    src = os.path.join(str(tmp_path), "not_a_brick.cpp")
    so = os.path.join(str(tmp_path), "not_a_brick.so")
    with open(src, "w") as f:
        f.write('extern "C" int unrelated_symbol() { return 0; }\n')
    flags = ["-shared", "-fPIC", "-O0"]
    if os.uname().sysname == "Darwin":
        flags += ["-undefined", "dynamic_lookup"]
    subprocess.run([cxx, *flags, src, "-o", so], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with pytest.raises(ValueError) as exc:
        lib.load_cpp_library(so)
    assert "pops_brick_manifest" in str(exc.value)
