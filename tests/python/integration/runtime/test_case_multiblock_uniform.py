"""ADC-479 Spec 5 C3: multi-block Problems on a Uniform layout lower.

PURE-PYTHON tests of the assembly + orchestration: a Problem with more than one block validates, and
pops.compile resolves EACH block's physics and carries the {block: model} table (_block_models) on
the handle so bind()'s _assemble_instances installs each block with its own model. The real .so
compile is Kokkos-gated, so compile_problem is MONKEYPATCHED to assert the wiring WITHOUT a compile.

Runs both under pytest and as a plain script; the CI runner executes it via the __main__ guard.
"""
import sys

try:
    import pops
    from pops.codegen import orchestration
    import pops.codegen.compile_drivers as compile_drivers
except Exception as exc:  # noqa: BLE001
    print("skip test_case_multiblock_uniform (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check


class _StubModel:
    """A physics stand-in exposing the .dsl engine model pops.compile resolves."""

    def __init__(self, name="stub"):
        self.name = name
        self.dsl = object()


class _StubCompiled:
    def __init__(self, target="system", model=None):
        self.so_path = "/tmp/stub.so"
        self.model = model
        self._target = target


_SAVED = []


def _patch_compile_problem(fn):
    _SAVED.append(compile_drivers.compile_problem)
    compile_drivers.compile_problem = fn


def _unpatch():
    while _SAVED:
        compile_drivers.compile_problem = _SAVED.pop()


def test_multi_block_uniform_validates():
    """C3: a two-block Uniform Problem validates (the >1-block reject is removed)."""
    case = (pops.Problem().block("ne", physics=_StubModel())
            .block("ni", physics=_StubModel()))
    _check(case.validate() is True, "a 2-block Uniform Problem validates")
    _check(case.options()["n_blocks"] == 2, "options() reports two blocks")
    print("ok test_multi_block_uniform_validates")


def test_multi_block_uniform_compiles_one_handle_per_block():
    """C3: compile lowers a multi-block Uniform Problem, carrying a model per block (_block_models)."""
    captured = {}

    def _fake(*, time, model, backend, target, **kw):
        captured.update(model=model, target=target, backend=backend)
        return _StubCompiled(target=target, model=model)

    _patch_compile_problem(_fake)
    try:
        case = (pops.Problem().block("ne", physics=_StubModel())
                .block("ni", physics=_StubModel()))
        compiled = orchestration.compile(case, time=object())
        _check(captured["target"] == "system", "Uniform multi-block routes to target='system'")
        _check(hasattr(compiled, "_block_models"), "the handle carries _block_models (C3)")
        _check(set(compiled._block_models) == {"ne", "ni"},
               "one resolved model per block (got %r)" % sorted(compiled._block_models))
        _check(compiled._problem is case, "the assembly is carried for bind()")
    finally:
        _unpatch()
    print("ok test_multi_block_uniform_compiles_one_handle_per_block")


def test_single_block_uniform_still_lowers():
    """C3 regression: a single-block Uniform Problem still lowers unchanged."""
    def _fake(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target, model=model)

    _patch_compile_problem(_fake)
    try:
        case = pops.Problem().block("ne", physics=_StubModel())
        compiled = orchestration.compile(case, time=object())
        _check(compiled._target == "system", "single-block Uniform routes to target='system'")
        _check(set(compiled._block_models) == {"ne"}, "one block carried")
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
