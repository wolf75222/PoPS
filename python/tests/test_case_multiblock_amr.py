"""ADC-479 Spec 5 C3 boundary: multi-block on an AMR layout is still deferred.

C3 lowers multi-block Cases on a Uniform layout; the whole-system AMR multi-block seam is NOT wired,
so a multi-block AMR Case is still rejected LOUD at compile (a HONEST deferral, not a silent
truncation). A SINGLE-block AMR Case keeps lowering to target='amr_system' unchanged (no early
whole-system-Program reject is introduced -- that compile-time AMR concern stays at master). The real
.so compile is Kokkos-gated, so compile_problem is MONKEYPATCHED.

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


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _StubModel:
    def __init__(self, name="stub"):
        self.name = name
        self.dsl = object()


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


def test_multi_block_amr_deferred():
    """C3 boundary: a multi-block AMR Case is rejected at compile, BEFORE any .so is built."""
    called = {"hit": False}

    def _tripwire(*a, **kw):
        called["hit"] = True
        return _StubCompiled(target="amr_system")

    _patch_compile_problem(_tripwire)
    try:
        case = (pops.Case(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
                .block("ni", physics=_StubModel()))
        try:
            orchestration.compile(case, time=object())
            raise AssertionError("a multi-block AMR Case must be rejected")
        except NotImplementedError as exc:
            _check("AMR" in str(exc), "the reject names the AMR layout")
            _check("multi-block" in str(exc), "the reject is the multi-block deferral")
        _check(called["hit"] is False, "the reject fires BEFORE compile_problem (no .so built)")
    finally:
        _unpatch()
    print("ok test_multi_block_amr_deferred")


def test_single_block_amr_still_lowers():
    """C3 boundary regression: a single-block AMR Case still lowers to target='amr_system' (no
    whole-system-Program compile-time reject is introduced)."""
    captured = {}

    def _fake(*, time, model, backend, target, **kw):
        captured.update(target=target)
        return _StubCompiled(target=target, model=model)

    _patch_compile_problem(_fake)
    try:
        layout = AMR(CartesianMesh())
        case = pops.Case(layout=layout).block("ne", physics=_StubModel())
        compiled = orchestration.compile(case, time=object())
        _check(captured["target"] == "amr_system", "single-block AMR routes to target='amr_system'")
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
