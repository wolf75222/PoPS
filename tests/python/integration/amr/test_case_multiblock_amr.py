"""ADC-503 Spec 6 sec.11: multi-block on an AMR layout lowers via the native per-block path.

ADC-503 lifts the C3 boundary: a multi-block (and single-block) AMR Case now lowers. Each block's
resolved physics is compiled to a target='amr_system' production CompiledModel (the native AMR .so
loader, add_native_block), the {block: CompiledModel} table is carried on the handle, and bind
installs through the native path (_install_compiled(compiled=None, instances=...)). There is NO
whole-system time Program on AMR, so compile_problem is NOT called and time= is not required. The
real .so compile is Kokkos-gated (ROMEO), so the block model's .compile is MONKEYPATCHED here to
assert the ROUTING only.

Runs both under pytest and as a plain script; the CI runner executes it via the __main__ guard.
"""
import sys

try:
    import pops
    from pops.codegen import orchestration
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    import pops.codegen.compile_drivers as compile_drivers
except Exception as exc:  # noqa: BLE001
    print("skip test_case_multiblock_amr (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check


class _StubCompiledModel:
    """A target='amr_system' CompiledModel stand-in (the AMR route compiles each block to one)."""

    def __init__(self, name="stub"):
        self.name = name
        self.so_path = "/tmp/%s_amr.so" % name
        self.target = "amr_system"
        self.adder = "add_native_block"


class _StubDsl:
    """The .dsl engine model the compile route resolves to; its .compile records the call."""

    def __init__(self, name="stub"):
        self.name = name
        self.compiled = []

    def compile(self, *, backend, target, **kw):
        self.compiled.append((backend, target))
        return _StubCompiledModel(self.name)


class _StubModel:
    def __init__(self, name="stub"):
        self.name = name
        self.dsl = _StubDsl(name)


class _StubCompiled:
    def __init__(self, target="amr_system", model=None):
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


def test_multi_block_amr_lowers_natively():
    """ADC-503: a multi-block AMR Case lowers -- each block compiled for target='amr_system', the
    {block: CompiledModel} table carried, and compile_problem NEVER called (a tripwire proves it)."""
    called = {"hit": False}

    def _tripwire(*a, **kw):
        called["hit"] = True
        return _StubCompiled(target="amr_system")

    _patch_compile_problem(_tripwire)
    try:
        m_ne, m_ni = _StubModel("ne"), _StubModel("ni")
        case = (pops.Case(layout=AMR(CartesianMesh())).block("ne", physics=m_ne)
                .block("ni", physics=m_ni))
        compiled = orchestration.compile(case)  # no time= : the AMR route does not need one
        _check(called["hit"] is False, "multi-block AMR does NOT call compile_problem (no .so built)")
        _check(m_ne.dsl.compiled == [("production", "amr_system")],
               "block 'ne' compiled once for backend='production', target='amr_system'")
        _check(m_ni.dsl.compiled == [("production", "amr_system")],
               "block 'ni' compiled once for backend='production', target='amr_system'")
        _check(set(compiled._block_compiled_models) == {"ne", "ni"},
               "the {block: CompiledModel} table carries both blocks")
        _check(all(cm.target == "amr_system" for cm in compiled._block_compiled_models.values()),
               "every carried CompiledModel targets the AMR system")
        _check(compiled._target == "amr_system", "amr_system target carried on the handle")
    finally:
        _unpatch()
    print("ok test_multi_block_amr_lowers_natively")


def test_single_block_amr_still_lowers():
    """ADC-503 regression: a single-block AMR Case still lowers (now via the native per-block path,
    not compile_problem)."""
    called = {"hit": False}

    def _tripwire(*a, **kw):
        called["hit"] = True
        return _StubCompiled(target="amr_system")

    _patch_compile_problem(_tripwire)
    try:
        layout = AMR(CartesianMesh())
        model = _StubModel("ne")
        case = pops.Case(layout=layout).block("ne", physics=model)
        compiled = orchestration.compile(case)
        _check(called["hit"] is False, "single-block AMR does NOT call compile_problem")
        _check(model.dsl.compiled == [("production", "amr_system")],
               "the single block compiled for target='amr_system'")
        _check(set(compiled._block_compiled_models) == {"ne"},
               "the handle carries the single block's CompiledModel")
        _check(compiled._layout is layout, "AMR layout carried on the handle for bind()")
    finally:
        _unpatch()
    print("ok test_single_block_amr_still_lowers")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
