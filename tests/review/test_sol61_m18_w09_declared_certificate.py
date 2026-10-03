from fractions import Fraction
import pytest
from pops.moments.closures import DiscreteEntropyQuadrature,DiscreteEntropyCertificate

@pytest.mark.parametrize('basis,covector,target',[
    (((1.,1.,1.),(-.5,0.,.5)),(.5,-1.),(1.,.9)),
    (((-.5,0.,.5),(1.,1.,1.)),(-1.,.5),(.9,1.)),
])
def test_permuted_basis_certificate_has_exact_original_separator(basis,covector,target):
    q=DiscreteEntropyQuadrature((-.5,0.,.5),(1.,2.,3.),basis)
    c=DiscreteEntropyCertificate(q,covector,label='declared_support')
    assert c.margin(target)==-.4
    assert sum(Fraction(a)*Fraction(b) for a,b in zip(covector,target,strict=True))<0
    assert c.to_data()['basis']==[list(r) for r in basis]

@pytest.mark.parametrize('vector',[(.25,-1.),(0.,0.),(True,0.),(float('nan'),1.),(1.,)])
def test_forged_support_certificate_refused(vector):
    q=DiscreteEntropyQuadrature((-.5,0.,.5),(1.,1.,1.),((1.,1.,1.),(-.5,0.,.5)))
    with pytest.raises(ValueError):DiscreteEntropyCertificate(q,vector,label='support')

def test_true_public_graph_resolve_emit_has_two_distinct_diagnostics():
    import pops
    from examples.migration.scientific.api040_m18_w09 import make_case
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    case,layout,_=make_case()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    source=emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),target='system')
    assert source.index('upper_support_target_infeasible')<source.index('upper_support_no_finite_exponential_dual')<source.index('solve_prepared_local_nonlinear')
    assert 'AcceptAllLocalCandidates' in source  # declared guard, not a hidden solver recipe
