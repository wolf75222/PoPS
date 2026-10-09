"""Closed corpus H05 only; independent scalar M1 closure checks are separate."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pops
import pytest


def _load(name):
    folder = Path(__file__).resolve().parents[4]/"examples/migration/scientific"
    previous = sys.path[:]
    sys.path.insert(0,str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name,folder/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


oracle = _load("api040_m22_h05_oracle")
case_module = _load("api040_m22_h05")


def test_exact_corpus_linear_exchange_not_a_T4_substitute():
    result = oracle.backward_euler((2.,.5),.8,.4,1)
    np.testing.assert_allclose(result,(70/41,65/82),rtol=0,atol=5e-16)
    assert abs(result.sum()-2.5) < 5e-16
    assert np.min(result) > 0
    # This residual names the exact normalized linear H05 equation.
    assert abs(result[0]-2.+.4*.8*(result[0]-result[1])) < 5e-16
    assert abs(result[1]-.5-.4*.8*(result[0]-result[1])) < 5e-16


def test_analytic_continuum_and_predeclared_time_order():
    from scipy.linalg import expm
    initial=np.array((2.,.5))
    exact=oracle.continuous(initial,.8,.4)
    np.testing.assert_allclose(exact,expm(.4*np.array([[-.8,.8],[.8,-.8]]))@initial,rtol=0,atol=5e-16)
    errors=[np.max(np.abs(oracle.backward_euler(initial,.8,.4/n,n)-exact))
            for n in case_module.STEPS]
    orders=np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders >= case_module.CRITERIA["temporal_order_min"])
    assert np.all(orders <= case_module.CRITERIA["temporal_order_max"])


@pytest.mark.parametrize("reverse",[False,True])
def test_public_h05_resolves_original_product_and_runtime_capture(reverse):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case,layout,_,coefficient=case_module.build_case(reverse=reverse)
    validated=pops.validate(case)
    assert validated.resolve(coefficient) is not None
    resolved=pops.resolve(validated,layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    emitted=emit_cpp_program(resolved.time,model=graph)
    assert "prepare_local_nonlinear_problem<2>" in emitted
    assert "local product requires co-located" in emitted
    assert emitted.index(".consume(pops::SolveConsumption::kAccept)") < emitted.index("ctx.commit_many(")


def test_chi_endpoints_near_beam_rotations_and_permutations():
    for reduced in (0.,.2,.7,1.-1e-12,1.):
        quotient=(3+4*reduced**2)/(5+2*np.sqrt(4-3*reduced**2))
        values=[]
        for vector in ((reduced,0.),(0.,reduced),(reduced/np.sqrt(2),reduced/np.sqrt(2))):
            value=oracle.chi(2.,2*np.array(vector))
            values.append(value)
            assert 1/3-1e-15 <= value <= 1.
            assert value == pytest.approx(quotient,abs=1e-15)
        np.testing.assert_allclose(values,values[0],rtol=0,atol=1e-15)
    assert oracle.chi(2.,(0.,0.)) == pytest.approx(1/3)
    assert oracle.chi(2.,(2.,0.)) == 1


@pytest.mark.parametrize("energy,flux,c",[(0.,(0.,0.),1.),(-1.,(0.,0.),1.),
    (1.,(1.000001,0.),1.),(1.,(0.,0.),0.),(1.,(np.nan,0.),1.)])
def test_chi_refuses_invalid_states_without_clipping(energy,flux,c):
    with pytest.raises(ValueError):
        oracle.chi(energy,flux,c)


@pytest.mark.parametrize("initial,k",[((2.,.5),-.8),((-1.,.5),.8),((np.nan,.5),.8)])
def test_h05_oracle_refuses_invalid_data(initial,k):
    with pytest.raises(ValueError):
        oracle.backward_euler(initial,k,.4,1)
