"""ADC-605 release-active validation errors on public runtime paths."""

import pytest
from pops.runtime._system import System  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.physics import Density  # noqa: E402
from pops.physics._facade import Model  # noqa: E402


def _make_system():
    sim = System(n=4, L=1.0, periodicity=(True, True))
    model = Model("validation_scalar")
    (density,) = model.conservative_vars("n", roles=(Density(),))
    model.flux(x=[0.3 * density], y=[0.2 * density])
    model.eigenvalues(x=[0.3 + 0.0 * density], y=[0.2 + 0.0 * density])
    model.primitive_vars(density)
    model.conservative_from([density])
    sim.add_equation(
        "ne",
        model.compile(
            backend="production", target="system", name="validation_scalar",
            consumer_owner_qid="tests.public-validation.scalar",
        ),
        spatial=engine.Spatial(none=True),
        time=engine.Explicit(),
    )
    return sim


def test_direct_coupled_source_rejects_stack_underflow_before_kernel():
    sim = _make_system()
    with pytest.raises(RuntimeError) as exc:
        sim._s._add_coupled_source(
            in_blocks=["ne"],
            in_roles=["density"],
            consts=[],
            out_blocks=["ne"],
            out_roles=["density"],
            prog_ops=[1],  # Add with an empty stack.
            prog_args=[0],
            prog_lens=[1],
        )

    msg = str(exc.value)
    assert "System::add_coupled_source term 0" in msg
    assert "stack underflow" in msg
    assert "expected" in msg and "received" in msg


def test_direct_coupled_frequency_rejects_unused_stack_result():
    sim = _make_system()
    with pytest.raises(RuntimeError) as exc:
        sim._s._add_coupled_source(
            in_blocks=["ne"],
            in_roles=["density"],
            consts=[],
            out_blocks=["ne"],
            out_roles=["density"],
            prog_ops=[0],
            prog_args=[0],
            prog_lens=[1],
            freq_prog_ops=[0, 0],  # PushReg, PushReg leaves two results.
            freq_prog_args=[0, 0],
        )

    msg = str(exc.value)
    assert "System::add_coupled_source frequency" in msg
    assert "exactly one result" in msg
    assert "final stack_depth=2" in msg
