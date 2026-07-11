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
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.codegen._compiled_model_identity import model_compile_identity  # noqa: E402
import pops.codegen.compile_drivers as compile_drivers  # noqa: E402
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402
from pops.model import DeclarationIndex, Handle, OwnerKind, OwnerPath  # noqa: E402


class _StubCompiledModel(CompiledModel):
    def __init__(self, source, target="amr_system"):
        super().__init__(
            "/tmp/%s_%s.so" % (source.name, target), "production",
            "add_native_block" if target == "amr_system" else "add_dynamic_block",
            (), (), (), 0, None, 0, {}, {"cpu": True, "amr": target == "amr_system"},
            "abi", source._model_hash(), "c++", "c++20", target=target,
            definition_identity=model_compile_identity(source))
        self.name = source.name

    @property
    def sealed(self):
        return getattr(self, "_sealed", False)


class _StubDsl:
    def __init__(self, name="stub"):
        self.name = name

    def compile(self, *, backend, target, **kw):
        return _StubCompiledModel(self, target=target)

    def _model_hash(self):
        return "model-hash:%s" % self.name


class _StubModel:
    def __init__(self, name="stub"):
        self.name = name
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        self.rho = Handle("rho", kind="state", owner=self.owner_path)
        self.dsl = _StubDsl(name)

    def declaration_index(self):
        return DeclarationIndex(owner=self.owner_path, handles=(self.rho,))


class _StubCompiled:
    def __init__(self, target="system", model=None):
        self.so_path = "/tmp/stub.so"
        self.model = model
        self.install_plan = None


def _stub_time():
    """An exact final Program value; opaque subclasses are intentionally not freeze-trusted."""
    return pops.Program("stub-time")


def _fresh_problem():
    """One Problem, no layout -- the subject of the two-layouts proof."""
    return pops.Problem(name="plasma").block(
        "ne", physics=_StubModel("ne")).program(_stub_time())


def _rho(problem):
    return problem._blocks.spec("ne")["model"].rho


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
        compiled_u = orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_stub_time())
        assert captured["target"] == "system"
        assert compiled_u.install_plan.target == "system"
        # The SAME Problem under AMR -> target='amr_system' (per-block native loader; no Program).
        compiled_a = orchestration.compile(prob, layout=AMR(base=CartesianMesh()))
        assert compiled_a.install_plan.target == "amr_system"
        assert set(compiled_a.install_plan.block_models) == {"ne"}
        assert all(child.sealed for child in compiled_a.install_plan.block_models.values())
        assert not hasattr(compiled_a, "_block_compiled_models")
    finally:
        compile_drivers.compile_problem = saved


def test_compile_without_layout_raises_pointing_at_layout_kwarg():
    saved = compile_drivers.compile_problem
    compile_drivers.compile_problem = _patched_uniform({})
    try:
        with pytest.raises(ValueError, match=r"pops\.compile\(problem, layout="):
            orchestration.compile(_fresh_problem(), time=_stub_time())
    finally:
        compile_drivers.compile_problem = saved


def test_constructor_layout_still_works_for_back_compat():
    saved = compile_drivers.compile_problem
    captured = {}
    compile_drivers.compile_problem = _patched_uniform(captured)
    try:
        prob = pops.Problem(layout=Uniform(CartesianMesh())).block(
            "ne", physics=_StubModel()).program(_stub_time())
        compiled = orchestration.compile(
            prob, time=_stub_time())  # no layout= : uses the constructor one
        assert compiled.install_plan.target == "system"
    finally:
        compile_drivers.compile_problem = saved


def test_explicit_layout_disagreeing_with_constructor_is_refused():
    prob = pops.Problem(layout=Uniform(CartesianMesh())).block(
        "ne", physics=_StubModel()).program(_stub_time())
    with pytest.raises(ValueError, match="disagrees"):
        orchestration.compile(prob, layout=AMR(base=CartesianMesh()), time=_stub_time())


def test_recorded_amr_criteria_apply_to_the_amr_layout():
    from pops.mesh.amr import RegridEvery
    prob = _fresh_problem()
    prob.amr.refine(regrid=RegridEvery(7))
    layout = AMR(base=CartesianMesh())
    saved = compile_drivers.compile_problem
    compile_drivers.compile_problem = _patched_uniform({})
    try:
        compiled = orchestration.compile(prob, layout=layout)
        assert compiled.install_plan.target == "amr_system"
        # Compile owns a detached merged layout; the caller's reusable descriptor is untouched.
        assert compiled.install_plan.layout is not layout
        assert compiled.install_plan.layout.regrid.steps == 7
        assert layout.regrid is None
    finally:
        compile_drivers.compile_problem = saved


def test_recorded_amr_criteria_refused_on_a_uniform_compile():
    from pops.mesh.amr import Refine
    prob = _fresh_problem()
    prob.amr.refine(Refine.on(_rho(prob)).above(0.1))
    with pytest.raises(ValueError, match="no level to refine onto"):
        orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_stub_time())


def test_layout_and_problem_cannot_both_author_the_same_amr_slot():
    from pops.mesh.amr import RegridEvery

    prob = _fresh_problem()
    prob.amr.refine(regrid=RegridEvery(7))
    layout = AMR(base=CartesianMesh(), regrid=RegridEvery(7))
    with pytest.raises(ValueError, match="competing authorities"):
        orchestration._resolve_layout(prob, layout)
    assert layout.regrid.steps == 7


def test_reusing_one_layout_across_problems_does_not_leak_criteria():
    from pops.mesh.amr import RegridEvery

    layout = AMR(base=CartesianMesh())
    tagged = _fresh_problem()
    tagged.amr.refine(regrid=RegridEvery(5))
    plain = _fresh_problem()

    tagged_layout = orchestration._resolve_layout(tagged, layout)
    plain_layout = orchestration._resolve_layout(plain, layout)
    assert tagged_layout is not layout and plain_layout is not layout
    assert tagged_layout.regrid.steps == 5
    assert plain_layout.regrid is None
    assert layout.regrid is None


def test_direct_layout_refine_and_output_handles_are_resolved_on_the_detached_copy():
    from pops.mesh.amr import AMROutput, Refine
    from pops.ir import ValueExpr
    from pops.ir.ops import dx, dy, sqrt

    prob = _fresh_problem()
    rho = _rho(prob)
    indicator = sqrt(dx(ValueExpr(rho)) ** 2 + dy(ValueExpr(rho)) ** 2)
    layout = AMR(
        base=CartesianMesh(),
        refine=Refine.on(indicator).above(0.1),
        output=AMROutput(fields=(rho,)),
    )
    resolved = orchestration._resolve_layout(prob, layout)

    assert resolved is not layout
    assert resolved.refine.subject.a.a.a.field.handle.is_resolved
    assert resolved.output.fields[0].is_resolved
    assert not layout.refine.subject.a.a.a.field.handle.is_resolved
    assert not layout.output.fields[0].is_resolved


def test_expression_indicator_snapshot_is_deterministic_and_leaks_no_authoring_token():
    import json
    from pops.ir import ValueExpr
    from pops.ir.ops import dx, dy, sqrt
    from pops.mesh.amr import Refine
    from pops.problem._snapshot import build_problem_snapshot

    def authored_problem():
        model = _StubModel("same-model")
        value = ValueExpr(model.rho)
        indicator = sqrt(dx(value) ** 2 + dy(value) ** 2)
        layout = AMR(
            base=CartesianMesh(),
            refine=Refine.on(indicator).above(0.1),
        )
        return pops.Problem(name="same-case", layout=layout).block("ne", physics=model)

    left = build_problem_snapshot(authored_problem())
    right = build_problem_snapshot(authored_problem())
    assert left.hash == right.hash
    encoded = json.dumps(left.to_dict(), sort_keys=True)
    assert "#authoring=" not in encoded


def test_amr_output_rejects_a_canonical_ghost_during_layout_resolution():
    from pops.mesh.amr import AMROutput
    from pops.model import MissingOwnershipError

    prob = _fresh_problem()
    model = prob._blocks.spec("ne")["model"]
    ghost = Handle("ghost", kind="state", owner=model.owner_path.canonical())
    layout = AMR(base=CartesianMesh(), output=AMROutput(fields=(ghost,)))
    with pytest.raises(MissingOwnershipError):
        orchestration._resolve_layout(prob, layout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
