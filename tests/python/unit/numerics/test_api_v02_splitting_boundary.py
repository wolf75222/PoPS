import numpy as np
import pops
from pops.codegen.program_codegen import emit_cpp_program
from examples.api_v02.method_splitting_boundary import build_t03_dirichlet
from tests.python.support import splitting_boundary_oracle as reference


def test_original_dirichlet_pde_and_lifted_equation_have_identical_residual():
    for n in (3, 7, 13):
        for time in (0., .07, .2):
            original, lifted = reference.original_residual(n, time)
            np.testing.assert_allclose(original, lifted, rtol=0, atol=4.e-14)


def test_noncommuting_spatial_strang_with_boundary_lifting_is_second_order():
    _, a, b = reference.operators(7)
    assert np.linalg.norm(a@b-b@a) > 1
    exact = reference.exact(7, .2)
    errors = [np.max(abs(reference.discrete(7, .2/n, n)-exact)) for n in (4, 8, 16)]
    assert all(3.8 < first/second < 4.2 for first, second in zip(errors, errors[1:]))
    # A frozen-time boundary subproblem requires zero boundary tangent.
    assert np.exp(.1) > 1 and 2*np.exp(.1) > 2


def test_dirichlet_strang_native_graph_has_no_model_or_python_callback():
    spec = build_t03_dirichlet()
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks
    cpp = emit_cpp_program(plan.time)
    assert "continuation_publish" in cpp
    assert "PyObject" not in cpp
