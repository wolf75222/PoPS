"""Authoring/oracle checks; none establish native reception."""
import numpy as np
import pops
import pytest
from examples.api_v02.models_M18_entropy import build_m18_entropy, reference_closure, reference_rk4
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize("order",(2,3))
def test_entropy_MN_closure_and_space_are_emitted_and_false_translation_is_rejected(order):
    spec=build_m18_entropy(order=order)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert cpp.count("ctx.spatial_interaction(")==4
    assert "std::exp(" in cpp
    initial=next(iter(spec.initial_values.values()))
    nodes,weights=np.polynomial.legendre.leggauss(6)
    expected=reference_rk4(initial,nodes,weights,1/16,spec.dt,2)
    shifted=np.roll(expected,1,axis=-1)
    alpha,flux,residual=reference_closure(shifted,nodes,weights)
    assert residual<2.e-14
    class FalseTranslation:
        def state_global(self,block):
            assert block=="moments"
            return shifted
        def history_global(self,label,depth):
            assert depth==1
            if label=="accepted_alpha":return alpha
            if label=="accepted_flux":return flux
            return np.zeros_like(shifted)
    metrics=spec.invariants(FalseTranslation())
    assert metrics["inventory_error"]<1.e-13
    assert metrics["all_moment_reconstruction_error"]<2.e-14
    assert metrics["independent_flux_error"]<2.e-14
    assert metrics["native_discrete_error"]>1.e-3
    assert not metrics["passed"]
