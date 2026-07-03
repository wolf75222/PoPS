#!/usr/bin/env python3
"""ADC-536 acceptance: the CompiledArtifact Protocol names the full inspectable surface.

ADC-536 states the public inspectable surface of a compiled handle is
``inspect`` / ``requirements`` / ``manifest`` / ``arguments`` / ``capability_matrix``. The
``CompiledArtifact`` :class:`typing.Protocol` (re-exported at ``pops.CompiledArtifact``) must NAME
all five so public code annotating against it depends on the full contract, and the concrete
``CompiledProblem`` loader class must SATISFY it structurally.

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops.codegen.compiled_artifact import CompiledArtifact  # noqa: E402
from pops.codegen.loader import CompiledProblem  # noqa: E402


def test_protocol_names_the_full_surface():
    for method in ("so_path", "inspect", "requirements", "manifest", "arguments",
                   "capability_matrix"):
        assert hasattr(CompiledArtifact, method), "the Protocol must name %r" % method


def test_compiled_problem_satisfies_the_widened_protocol():
    # The concrete loader class implements every callable method the Protocol names (so_path is an
    # instance attribute set in __init__, verified separately below via a runtime_checkable check).
    for method in ("inspect", "requirements", "manifest", "arguments", "capability_matrix"):
        assert callable(getattr(CompiledProblem, method, None)), \
            "CompiledProblem must implement %r" % method


def test_a_real_handle_is_a_compiled_artifact_instance():
    # A minimal handle (built outside compile_problem, so_path set) is structurally a
    # CompiledArtifact: the runtime_checkable Protocol accepts it (so_path + the report methods).
    handle = CompiledProblem("/tmp/none.so", None, None, "SIG|c++|c++23", "c++", "c++23")
    assert isinstance(handle, CompiledArtifact), \
        "a compiled handle satisfies the widened CompiledArtifact protocol"


def test_pops_reexports_the_protocol():
    import pops
    assert pops.CompiledArtifact is CompiledArtifact, "pops.CompiledArtifact is the same Protocol"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
