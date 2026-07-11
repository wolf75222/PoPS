"""ADC-479 Spec 5 C3: multi-block Problems on a Uniform layout lower.

PURE-PYTHON tests of the assembly + orchestration: a Problem with more than one block validates, and
pops.compile resolves EACH block's physics into the immutable InstallPlan so bind() installs each
block with its own model. The real .so
compile is Kokkos-gated, so compile_problem is MONKEYPATCHED to assert the wiring WITHOUT a compile.

Runs both under pytest and as a plain script; the CI runner executes it via the __main__ guard.
"""
import sys

try:
    import pops
    from pops.codegen import orchestration
    from pops.codegen._compiled_model_identity import model_compile_identity
    from pops.codegen.loader import CompiledModel
    import pops.codegen.compile_drivers as compile_drivers
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform
    from pops.model import DeclarationIndex, OwnerKind, OwnerPath
except Exception as exc:  # noqa: BLE001
    print("skip test_case_multiblock_uniform (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check


class _StubModel:
    """A physics stand-in exposing the .dsl engine model pops.compile resolves."""

    def __init__(self, name="stub"):
        self.name = name
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        self.dsl = _StubDescriptor("dsl")

    def declaration_index(self):
        return DeclarationIndex(owner=self.owner_path, handles=())


class _StubDescriptor:
    def __init__(self, name):
        self.name = name

    def to_data(self):
        return {"kind": "stub", "name": self.name}

    def _model_hash(self):
        return "model-hash:%s" % self.name

    def compile(self, *, backend, target, **kw):
        """Implement the final install-model protocol without invoking a native compiler."""
        return CompiledModel(
            so_path="/tmp/%s_%s.so" % (self.name, target),
            backend=backend,
            adder="add_native_block",
            cons_names=(),
            cons_roles=(),
            prim_names=(),
            n_vars=0,
            gamma=None,
            n_aux=0,
            params={},
            caps={"cpu": True},
            abi_key="abi",
            model_hash=self._model_hash(),
            cxx="c++",
            std="c++20",
            target=target,
            definition_identity=model_compile_identity(self),
        )


def _stub_time():
    """Return the exact final Program type required by Uniform compilation."""
    return pops.Program("stub-time")


class _StubCompiled:
    def __init__(self, target="system", model=None, problem_snapshot=None):
        self.so_path = "/tmp/stub.so"
        self.model = model
        self.install_plan = None
        self._problem_snapshot = problem_snapshot

    @property
    def authoring_snapshot(self):
        return self._problem_snapshot


_SAVED = []


def _patch_compile_problem(fn):
    _SAVED.append(compile_drivers.compile_problem)
    compile_drivers.compile_problem = fn


def _unpatch():
    while _SAVED:
        compile_drivers.compile_problem = _SAVED.pop()


def test_multi_block_uniform_validates():
    """C3: a two-block Uniform Problem validates (the >1-block reject is removed)."""
    case = (pops.Problem().block("ne", physics=_StubModel("ne"))
            .block("ni", physics=_StubModel("ni")))
    _check(case.validate() is True, "a 2-block Uniform Problem validates")
    _check(case.options()["n_blocks"] == 2, "options() reports two blocks")
    print("ok test_multi_block_uniform_validates")


def test_multi_block_uniform_compiles_one_handle_per_block():
    """C3: compile lowers a multi-block Uniform Problem into one immutable InstallPlan."""
    captured = {}

    def _fake(*, time, model, backend, target, **kw):
        captured.update(model=model, target=target, backend=backend)
        return _StubCompiled(
            target=target, model=model, problem_snapshot=kw.get("problem_snapshot"))

    _patch_compile_problem(_fake)
    try:
        case = (pops.Problem().block("ne", physics=_StubModel("ne"))
                .block("ni", physics=_StubModel("ni")))
        compiled = orchestration.compile(
            case, layout=Uniform(CartesianMesh()), time=_stub_time())
        _check(captured["target"] == "system", "Uniform multi-block routes to target='system'")
        plan = compiled.install_plan
        _check(plan.target == "system", "the InstallPlan carries the Uniform target")
        _check(set(plan.block_models) == {"ne", "ni"},
               "one resolved model per block (got %r)" % sorted(plan.block_models))
        _check(compiled.authoring_snapshot.hash == plan.snapshot_hash,
               "snapshot and InstallPlan share one compile identity")
        _check(not hasattr(compiled, "_problem"), "the artifact retains no live Problem")
        _check(not hasattr(compiled, "_block_specs"), "the artifact retains no authoring block specs")
    finally:
        _unpatch()
    print("ok test_multi_block_uniform_compiles_one_handle_per_block")


def test_single_block_uniform_still_lowers():
    """C3 regression: a single-block Uniform Problem still lowers unchanged."""
    def _fake(*, time, model, backend, target, **kw):
        return _StubCompiled(
            target=target, model=model, problem_snapshot=kw.get("problem_snapshot"))

    _patch_compile_problem(_fake)
    try:
        case = pops.Problem().block("ne", physics=_StubModel())
        compiled = orchestration.compile(
            case, layout=Uniform(CartesianMesh()), time=_stub_time())
        _check(compiled.install_plan.target == "system",
               "single-block Uniform routes to target='system'")
        _check(set(compiled.install_plan.block_models) == {"ne"}, "one block carried")
    finally:
        _unpatch()
    print("ok test_single_block_uniform_still_lowers")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
