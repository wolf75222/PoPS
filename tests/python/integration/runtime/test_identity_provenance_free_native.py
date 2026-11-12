"""N11 lot 2a (spec 0.3 CC2, G07 criterion): a comment does not change the compiled identity.

The same case module, written to two directories (two checkouts, gap C48) and compiled once as
written and once with comment lines inserted before the case (every line of the case moves), gives
one compiled ``artifact_identity``, one native program hash, one cache key and one shared object.
Before the lot, the source spans of the time program and of the module manifest entered the plan
identity and the binary evidence, so the compiled identity moved with the file and the line.
"""

import pytest

import pops
from tests.python.support.provenance_case import COMMENT, resolve_written_case


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_comment_keeps_the_compiled_artifact_identity(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    plain, _ = resolve_written_case(tmp_path / "plain")
    commented, _ = resolve_written_case(tmp_path / "commented", comment=COMMENT)
    assert plain.snapshot.hash != commented.snapshot.hash

    first = pops.compile(plain)
    second = pops.compile(commented)

    assert first.artifact_identity == second.artifact_identity
    assert first.program_hash == second.program_hash
    assert first.cache_key == second.cache_key
    assert first.so_path == second.so_path
