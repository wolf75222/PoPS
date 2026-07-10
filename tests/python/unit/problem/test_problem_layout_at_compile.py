"""ADC-526: the SAME Problem compiles under layout=Uniform AND layout=AMR (the two-layouts proof).

The Problem owns no layout; the layout is chosen at pops.compile(problem, layout=...). This is the
core acceptance: one assembly, two compile targets. The real .so compile is Kokkos-gated, so the
Uniform driver (compile_problem) and the per-block AMR loader (.dsl.compile) are MONKEYPATCHED to
assert the ROUTING WITHOUT a compile. It also checks the layout-free errors: no layout given, a
layout= that disagrees with a constructor layout, and a Uniform compile that would drop recorded AMR
criteria.

Pure Python; needs only `import pops`.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import orchestration  # noqa: E402
import pops.codegen.compile_drivers as compile_drivers  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402


class _StubCompiledModel:
    def __init__(self, name="stub"):
        self.name = name
        self.so_path = "/tmp/%s_amr.so" % name
        self.target = "amr_system"
        self.adder = "add_native_block"


class _StubDsl:
    def __init__(self, name="stub"):
        self.name = name

    def compile(self, *, backend, target, **kw):
        return _StubCompiledModel(self.name)


class _StubModel:
    def __init__(self, name="stub"):
        self.name = name
        self.dsl = _StubDsl(name)


class _StubCompiled:
    def __init__(self, target="system", model=None):
        self.so_path = "/tmp/stub.so"
        self.model = model
        self._target = target


class _StubTime:
    """A structural time test double; opaque ``object()`` is not a cache identity."""

    def __init__(self, name="stub-time"):
        self.name = name


def _fresh_problem():
    """One Problem, no layout -- the subject of the two-layouts proof."""
    return pops.Problem(name="plasma").block(
        "ne", physics=_StubModel("ne")).program(_StubTime())


def _patched_uniform(captured):
    def _fake(*, time, model, backend, target, **kw):
        captured.update(target=target, model=model)
        return _StubCompiled(target=target, model=model)
    return _fake


def test_same_problem_compiles_uniform_and_amr():
    saved = compile_drivers.compile_problem
    captured = {}
    compile_drivers.compile_problem = _patched_uniform(captured)
    try:
        prob = _fresh_problem()
        # Under Uniform -> target='system' (the whole-system Program is compiled once).
        compiled_u = orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_StubTime())
        assert captured["target"] == "system"
        assert compiled_u._target == "system"
        # The SAME Problem under AMR -> target='amr_system' (per-block native loader; no Program).
        compiled_a = orchestration.compile(prob, layout=AMR(base=CartesianMesh()))
        assert compiled_a._target == "amr_system"
        assert hasattr(compiled_a, "_block_compiled_models")
        assert set(compiled_a._block_compiled_models) == {"ne"}
    finally:
        compile_drivers.compile_problem = saved


def test_compile_without_layout_raises_pointing_at_layout_kwarg():
    saved = compile_drivers.compile_problem
    compile_drivers.compile_problem = _patched_uniform({})
    try:
        with pytest.raises(ValueError, match=r"pops\.compile\(problem, layout="):
            orchestration.compile(_fresh_problem(), time=_StubTime())
    finally:
        compile_drivers.compile_problem = saved


def test_constructor_layout_still_works_for_back_compat():
    saved = compile_drivers.compile_problem
    captured = {}
    compile_drivers.compile_problem = _patched_uniform(captured)
    try:
        prob = pops.Problem(layout=Uniform(CartesianMesh())).block(
            "ne", physics=_StubModel()).program(_StubTime())
        compiled = orchestration.compile(
            prob, time=_StubTime())  # no layout= : uses the constructor one
        assert compiled._target == "system"
    finally:
        compile_drivers.compile_problem = saved


def test_explicit_layout_disagreeing_with_constructor_is_refused():
    prob = pops.Problem(layout=Uniform(CartesianMesh())).block(
        "ne", physics=_StubModel()).program(_StubTime())
    with pytest.raises(ValueError, match="disagrees"):
        orchestration.compile(prob, layout=AMR(base=CartesianMesh()), time=_StubTime())


def test_recorded_amr_criteria_apply_to_the_amr_layout():
    from pops.mesh.amr import RegridEvery
    prob = _fresh_problem()
    prob.amr.refine(regrid=RegridEvery(7))
    layout = AMR(base=CartesianMesh())
    saved = compile_drivers.compile_problem
    compile_drivers.compile_problem = _patched_uniform({})
    try:
        compiled = orchestration.compile(prob, layout=layout)
        assert compiled._target == "amr_system"
        # The recorded regrid criterion was applied to the AMR layout at compile.
        assert layout.regrid is not None and layout.regrid.steps == 7
    finally:
        compile_drivers.compile_problem = saved


def test_recorded_amr_criteria_refused_on_a_uniform_compile():
    from pops.mesh.amr import Refine
    prob = _fresh_problem()
    prob.amr.refine(Refine.on("rho").above(0.1))
    with pytest.raises(ValueError, match="no level to refine onto"):
        orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_StubTime())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
