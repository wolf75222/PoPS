"""Native path evolution against an independent array finite-volume oracle."""
import numpy as np
import pytest
import pops
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.symbolic_path_case import make_case

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


@pytest.mark.parametrize("scale,reverse", [(1., False), (2.5, True)])
def test_authored_product_executes_on_native_faces(isolated_native_cache, native_cxx, kokkos_root,
                                                 scale, reverse):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, dt = 8, 1.e-3
    case, layout = make_case(scale=scale, reverse=reverse, cells=cells, dt=dt)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    x = (np.arange(cells) + 0.5) / cells
    u = np.broadcast_to(1. + 0.1*np.sin(2*np.pi*x), (cells, cells)).copy()
    v = np.broadcast_to(0.3 + 0.1*np.cos(2*np.pi*x), (cells, cells)).copy()
    initial = np.stack((v, u) if reverse else (u, v))
    runtime = pops.bind(artifact, initial_state={"transport": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    # Independent exact integration of v(s)^2 du(s), separate from the Python
    # Simpson body and generated Expr tree. The two side terms share a sign;
    # only the conservative flux is an oppositely oriented transfer.
    ui, vi = (1, 0) if reverse else (0, 1)
    right = np.roll(initial, -1, axis=2)
    speed = 0.5 + abs(scale)*np.maximum(initial[vi]**2, right[vi]**2)
    flux = 0.25*(initial + right) - 0.5*speed*(right - initial)
    integral = np.zeros_like(initial)
    integral[ui] = (scale/3.) * (initial[vi]**2 + initial[vi]*right[vi] + right[vi]**2) * (right[ui]-initial[ui])
    rhs = cells * (np.roll(flux, 1, axis=2) - flux
                   - 0.5*(integral + np.roll(integral, 1, axis=2)))
    expected = initial + dt*rhs
    report = pops.run(runtime, t_end=dt, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("transport")).reshape(initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=3.e-14)
    np.testing.assert_allclose(actual[vi].sum(), initial[vi].sum(), rtol=0., atol=3.e-14)
    assert not np.allclose(actual[ui], initial[ui], rtol=0., atol=1.e-8)
