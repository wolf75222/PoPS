#!/usr/bin/env python3
"""ADC-558 acceptance: a compiled artifact is validated-or-absent (no post-compile check).

Every structural check happens DURING ``pops.compile(...)``; a returned ``CompiledProblem`` is
always fully valid, so there is no post-compile ``check()`` step. This module pins, at the pure
metadata level (no compiler / no ``.so``):

  1  ``CompiledProblem`` exposes NO public ``check()`` method (the validity is guaranteed by the
     handle existing; the inspectable surface is inspect / requirements / manifest);
  2  the ``CompiledArtifact`` protocol names NO ``check`` (public code never calls one);
  3  the single validity signal is the ``inspect().status`` line ("compiled, waiting for
     pops.bind(...)"), so a valid handle is self-describing without a check step;
  4  no advanced ``_assert_invariants`` seam is advertised (not in ``__all__`` / not public).

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.compiled_artifact import CompiledArtifact  # noqa: E402
from pops.codegen.loader import CompiledProblem  # noqa: E402


def _bare_handle():
    """A minimal handle (no program / model) -- enough to probe the surface, not to bind."""
    return CompiledProblem("/tmp/none.so", None, None, "SIG|c++|c++23", "c++", "c++23")


# --- 1 + 4: no public check(), no advertised _assert_invariants ---------------------------------

def test_compiled_problem_has_no_public_check():
    handle = _bare_handle()
    assert not hasattr(handle, "check"), "a compiled artifact must not carry a post-compile check()"


def test_no_assert_invariants_is_advertised():
    import pops.codegen.loader as loader_mod
    # No advanced re-validation seam is public: not in __all__ (if the module defines one).
    exported = getattr(loader_mod, "__all__", None)
    if exported is not None:
        assert "_assert_invariants" not in exported


# --- 2: the protocol names no check -------------------------------------------------------------

def test_protocol_has_no_check():
    assert not hasattr(CompiledArtifact, "check"), "the CompiledArtifact protocol must not name check"
    # The surface it DOES name is the inspectable reports (ADC-536 widening).
    for method in ("inspect", "requirements", "manifest", "arguments", "capability_matrix"):
        assert hasattr(CompiledArtifact, method)


# --- 3: the status line is the single validity signal ------------------------------------------

def test_status_is_the_single_validity_signal():
    report = _bare_handle().inspect()
    assert report.status == "compiled, waiting for pops.bind(...)", report.status
    # The status is serialized in the report dict too (no separate check() needed to read validity).
    assert report.to_dict()["status"] == "compiled, waiting for pops.bind(...)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
