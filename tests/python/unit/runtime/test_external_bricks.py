"""Spec 3 external C++ bricks (ADC-463, criterion 20).

A Spec 3 brick is native / generated / macro / external-C++. These tests cover
the last category: ``pops.descriptors.load_cpp_library(path)`` dlopens a user ``.so``,
reads its JSON manifest (over the C++ ``BrickRegistry``), and registers the ids
in an in-process catalog; ``pops.numerics.riemann.User(id)`` / ``pops.descriptors.external(id)``
then surface an ``external_cpp`` descriptor carrying the manifest's requirements.
An id that was never loaded raises a CLEAR error.

The manifest-parsing seam (``_register_manifest``) is exercised directly so the
test needs no compiled ``.so``; ``load_cpp_library`` is the real ``.so`` path on
top of it. The real functions are used -- pops is never faked.
"""
import os
import json
import shutil
import subprocess
import types

import pytest

# Spec 5 (sec.4): the brick-loader + generic external() live in pops.descriptors, and
# the riemann ``User`` selector in pops.numerics.riemann (formerly all under pops.lib).
_desc = pytest.importorskip("pops.descriptors")
lib = types.SimpleNamespace(
    riemann=pytest.importorskip("pops.numerics.riemann").riemann,
    external=_desc.external,
    load_cpp_library=_desc.load_cpp_library,
    _register_manifest=_desc._register_manifest,
    _clear_external_catalog=_desc._clear_external_catalog,
)

from tests.python.support.requirements import repo_include
_INCLUDE = repo_include()

# A minimal external brick .so: POPS_REGISTER_BRICK populates the host registry at static-init time and
# POPS_DEFINE_BRICK_MANIFEST exports the C reader load_cpp_library dlopens. No numerics here -- the
# manifest path only needs the identity + requirements (the static-dispatch ABI is the C++ test).
_BRICK_SRC = """
#include <pops/runtime/program/external_brick.hpp>
#include <string>
POPS_REGISTER_BRICK("my_so_riemann", "riemann", "pressure,wave_speeds");
POPS_DEFINE_BRICK_MANIFEST();
"""


def _compile_brick_so(workdir):
    """Compile the minimal brick to a .so; return its path, or None if the toolchain is unusable."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(_INCLUDE):
        return None
    src = os.path.join(workdir, "my_so_riemann.cpp")
    so = os.path.join(workdir, "my_so_riemann.so")
    with open(src, "w") as f:
        f.write(_BRICK_SRC)
    flags = ["-shared", "-fPIC", "-std=c++20", "-O0", "-I", _INCLUDE]
    if os.uname().sysname == "Darwin":
        flags.append("-undefined")
        flags.append("dynamic_lookup")
    try:
        subprocess.run([cxx, *flags, src, "-o", so], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError):
        return None
    return so


@pytest.fixture(autouse=True)
def _clean_catalog():
    """Reset the in-process external-brick catalog around each test (no leakage)."""
    lib._clear_external_catalog()
    yield
    lib._clear_external_catalog()


def _manifest(*entries, schema_version=2):
    """Build a STRICT versioned manifest (ADC-611 / ADC-544): stamp schema_version and fill each entry's
    four required fields (id / category / requirements / capabilities) with defaults so the happy-path
    tests exercise a valid payload. The ADC-544 v2 fields (native_id / supported_layouts / ...) are
    optional -- an entry may add them, but the required set is unchanged. A test that probes a MISSING
    field passes the raw dict via _register_manifest directly rather than through this helper."""
    normed = []
    for e in entries:
        row = {"category": "brick", "requirements": "", "capabilities": ""}
        row.update(e)
        normed.append(row)
    return json.dumps({"schema_version": schema_version, "bricks": normed})


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
    with pytest.raises((OSError, ValueError)):
        lib.load_cpp_library("/no/such/brick.so")


def test_load_cpp_library_dlopens_a_real_so_and_surfaces_the_descriptor(tmp_path):
    """The deferred half: compile a REAL brick .so, dlopen it via load_cpp_library, and assert
    riemann.User surfaces its manifest. Self-skips if no C++ compiler / pops headers are present
    (the registry-seam tests above cover the parsing without a toolchain)."""
    so = _compile_brick_so(str(tmp_path))
    if so is None:
        pytest.skip("no C++ compiler or pops headers to build the brick .so")
    # The registry .so is header-light (only external_brick.hpp): plain flags, no Kokkos needed.
    n = lib.load_cpp_library(so)
    # NOT n == 1: the BrickRegistry singleton is a function-local static in a header-only class,
    # emitted STB_GNU_UNIQUE by gcc, so on Linux it UNIFIES across every brick .so dlopen'd by this
    # process -- when a sibling test in the same pytest process loaded brick libraries first, this
    # .so's pops_brick_manifest() lists THEIR bricks too (ADC-622 tracks the unification itself).
    # The load contract this test locks is proven by the surfaced descriptor below.
    assert n >= 1
    d = lib.riemann.User("my_so_riemann")
    assert d.brick_type == "external_cpp"
    assert d.category == "riemann"
    assert d.native_id == "my_so_riemann"
    assert d.requirements == {"capabilities": ["pressure", "wave_speeds"]}


def test_load_cpp_library_rejects_a_non_brick_so(tmp_path):
    """A loadable library that does NOT export pops_brick_manifest() is rejected clearly (it is not an
    pops brick .so), never silently treated as carrying zero bricks."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler to build the non-brick .so")
    src = os.path.join(str(tmp_path), "not_a_brick.cpp")
    so = os.path.join(str(tmp_path), "not_a_brick.so")
    with open(src, "w") as f:
        f.write('extern "C" int unrelated_symbol() { return 0; }\n')
    flags = ["-shared", "-fPIC", "-O0"]
    if os.uname().sysname == "Darwin":
        flags += ["-undefined", "dynamic_lookup"]
    try:
        subprocess.run([cxx, *flags, src, "-o", so], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError):
        pytest.skip("could not build the non-brick .so")
    with pytest.raises(ValueError) as exc:
        lib.load_cpp_library(so)
    assert "pops_brick_manifest" in str(exc.value)
