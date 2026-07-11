"""ADC-637: a compiled condensed-implicit Program's .so carries NO coupling/schur include.

The runtime-half sibling of tests/python/architecture/test_no_schur_header_leak.py (which pins the
source-parse hygiene of program_context.hpp without importing pops): these checks lower a real
Forward-Euler and a real condensed-implicit Program through the codegen, so they need the compiled
_pops extension and live in the unit/codegen tier the CI shards run WITH the module built -- in the
source-only architecture lane they would never execute.

The condensed-Schur Program brick is retired (ADC-637): the sole route is the generic condensed_*
solve, emitted inline via pops::detail::block_inverse (block_inverse.hpp) with no coupling/schur/**
in the generated .so.
"""
from pops.params import ConstParam
import pytest

# import pops loads _pops; skip honestly on a source-only checkout (the bootstrap wraps the raw
# ModuleNotFoundError in a plain ImportError, hence exc_type).
pops_time = pytest.importorskip("pops.time", exc_type=ImportError)
pops_lib_time = pytest.importorskip("pops.lib.time", exc_type=ImportError)
from typed_program_support import state_refs  # noqa: E402


def _lorentz_model(name):
    """A rho/mx/my block carrying the electrostatic-Lorentz linearization J the generic condensed
    route resolves (the canonical condensed block: rho / mx / my + grad_x / grad_y / B_z + J)."""
    from pops.ir.ops import sqrt
    from pops.lib.models import author_electrostatic_lorentz
    from pops.physics.facade import Model
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.value(m.param(ConstParam("cs2", 0.5)))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)
    m.aux("grad_x")
    m.aux("grad_y")
    m.aux("B_z")
    author_electrostatic_lorentz(m)
    return m


def _linear_handle(model):
    from pops.model import OperatorHandle
    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    return OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)


def test_forward_euler_program_excludes_schur_includes():
    """A condensed-free (Forward-Euler) Program's generated .so excludes coupling/schur/** AND
    block_inverse.hpp (both gated on the condensed ops, ADC-637)."""
    P = pops_time.Program("fe")
    block, state = state_refs(P, "gas")
    pops_lib_time.forward_euler(P, block, state)
    src = P.emit_cpp_program()
    assert "program/program_context.hpp" in src, "the .so must always include the runtime facade"
    assert "coupling/schur" not in src, (
        "a condensed-free Program's generated .so must not include coupling/schur/**"
    )
    assert "block_inverse.hpp" not in src, (
        "a condensed-free Program must not include block_inverse.hpp (the include is condensed-gated)"
    )


def test_condensed_program_includes_block_inverse_and_no_schur():
    """A condensed-implicit Program's generated .so pulls the block_inverse intrinsic and the re-homed
    coeff-free apply, and carries NO coupling/schur token -- neither the include path NOR the C++
    namespace (ADC-637: the brick is retired, the generic route is the sole route)."""
    model = _lorentz_model("cs_model")
    P = pops_time.Program("cs").bind_operators(model)
    block, state = state_refs(P, "blk", model=model)
    pops_lib_time.condensed_schur(
        P, block, state, alpha=1.0, theta=1.0,
        linear_operator=_linear_handle(model))
    src = P.emit_cpp_program(model=model)
    assert "numerics/linalg/block_inverse.hpp" in src, (
        "a condensed-implicit Program must include the closed-form block-inverse intrinsic"
    )
    assert "coupling/schur" not in src, (
        "the generated .so must not include coupling/schur/** (the brick is retired)"
    )
    assert "coupling::schur" not in src, (
        "the generated .so must not name the pops::coupling::schur:: free functions (retired)"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
