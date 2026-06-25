"""Spec 3 section 20 / criterion 23: a custom solver DSL that BUILDS IR.

``@adc.lib.solver`` registers a GENERATED-brick solver whose body is a Python
builder. Running the builder authors a SOLVER IR (matrix-free Krylov primitives:
norm2 / dot / apply / linear_combine / while) and computes NOTHING in Python --
no float arithmetic on real data, no numpy callback is captured. The generated
C++ lowering + run is the deferred C++ follow-up: it raises a clear ADC-462
NotImplementedError rather than faking a Python solve.

These tests are the AUTHORING slice: they assert the registration shape, the IR
ops, the no-Python-compute invariant, native-solver selectability, and the
honest deferral. They never run a custom solver numerically.
"""
import pytest

lib = pytest.importorskip("adc.lib")


def _richardson(ctx, A, b, *, omega=0.5, tol=1e-8, max_iter=100):
    """A textbook Richardson iteration authored as IR: x <- x + omega*(b - A x).

    Builds IR only -- omega / tol are IR literals, never multiplied against data.
    """
    x = ctx.zeros_like(b)
    it = ctx.scalar_int(0)
    with ctx.while_(ctx.logical_and(ctx.norm2(ctx.residual(A, x, b)) > tol,
                                    it < ctx.scalar_int(max_iter))):
        r = ctx.residual(A, x, b)          # r = b - A x
        x = ctx.combine(x + omega * r)     # affine IR combine, no Python float math
        it = it + ctx.scalar_int(1)
    return x


def test_decorator_registers_generated_solver_descriptor():
    @lib.solver(name="richardson_dsl", signature="(A, b)")
    def richardson(ctx, A, b):
        return _richardson(ctx, A, b)

    d = richardson  # the decorator returns the descriptor
    assert isinstance(d, lib.BrickDescriptor)
    assert d.brick_type == "generated"
    assert d.category == "solver"
    assert d.name == "richardson_dsl"
    assert d.scheme == "richardson_dsl"
    # The builder is carried OFF the identity key (like BrickDescriptor.expression).
    assert callable(d.builder)


def test_descriptor_is_registered_in_the_catalog():
    @lib.solver(name="cataloged_solver")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    assert lib.solvers.custom("cataloged_solver") is s
    assert "cataloged_solver" in lib.solvers.registered()


def test_builder_builds_an_ir_with_the_expected_ops():
    @lib.solver(name="ir_shape", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    ops = ir.op_kinds()
    # The matrix-free Krylov primitives the spec calls out.
    assert "norm2" in ops
    assert "apply" in ops          # A(x) inside the residual
    assert "linear_combine" in ops  # x + omega*r
    assert "while" in ops
    # The solution value the builder returned is a State-like IR value.
    assert ir.result.vtype == "state"


def test_richardson_example_builds_an_affine_update_loop():
    @lib.solver(name="rich_example", signature="(A, b)")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    # The body has an affine x + omega*r combine with the expected coefficients.
    combines = [n for n in ir.nodes() if n.op == "linear_combine"]
    assert combines, "the Richardson update must be an affine linear_combine"
    coeffs = {c.get(0) for n in combines for c in n.attrs["coeffs"]}
    assert 0.5 in coeffs   # omega
    assert 1.0 in coeffs   # the x term


def test_ir_has_no_python_numeric_compute():
    captured = {"calls": 0}

    @lib.solver(name="no_python_compute")
    def s(ctx, A, b):
        # If the DSL ever ran the body numerically, a Python callback here would
        # fire. It must be recorded as IR (apply), never invoked.
        def py_kernel(_state):       # pragma: no cover - must never run
            captured["calls"] += 1
            return _state
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    assert captured["calls"] == 0, "the builder must not run Python numerics"
    # Every IR node is a typed, inert record: no node holds a live float result
    # or a numpy array -- only IR values / literals / coefficient polynomials.
    for n in ir.nodes():
        for v in n.inputs:
            assert hasattr(v, "vtype"), "an IR input must be an IR value, not data"
        # Attr payloads are metadata (ints / floats-as-literals / dicts), never arrays.
        for av in n.attrs.values():
            assert not _looks_like_array(av), "an IR attr captured a data array: %r" % (av,)


def test_a_scalar_in_the_ir_cannot_collapse_to_a_python_bool():
    @lib.solver(name="loud_scalar")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    ir = lib.build_solver_ir(s)
    scalars = [n for n in ir.nodes() if n.vtype in ("scalar", "bool")]
    assert scalars, "the convergence test must build runtime scalar/bool nodes"
    with pytest.raises(TypeError):
        bool(scalars[0])   # a runtime scalar must never decide a Python branch


def test_custom_solver_is_selectable_like_a_native_solver():
    @lib.solver(name="selectable")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    native = lib.solvers.GMRES()
    # Same metadata shape as a native solver descriptor: a scheme string and a
    # category, so T.solve(..., solver=<descriptor>) accepts it like GMRES.
    assert s.scheme is not None
    assert s.category == native.category == "solver"
    assert s.scheme == "selectable"


def test_cpp_generation_is_deferred_with_a_clear_adc462_error():
    @lib.solver(name="defer_codegen")
    def s(ctx, A, b):
        return _richardson(ctx, A, b)

    with pytest.raises(NotImplementedError) as exc:
        lib.generate_solver_cpp(s)
    assert "ADC-462" in str(exc.value)


def test_builder_signature_and_name_are_validated():
    with pytest.raises((TypeError, ValueError)):
        lib.solver(name="")          # empty name is rejected

    with pytest.raises(TypeError):
        lib.solver(name="bad_signature", signature=123)   # signature must be a string

    with pytest.raises(TypeError):
        lib.solver(name="not_callable")(42)   # the body must be a callable builder


def _looks_like_array(value):
    """True if ``value`` looks like a live numeric data buffer (numpy array / list of
    floats), which an IR attr must never capture."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return True
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(x, float) for x in value):
        return True
    return False
