"""Real Program/SolveRequest authority, Source only; no Native substitute."""
import pytest
from pops.time import Program,SolveRequest,SolveUnknown,FailRun,SolveRequestError
from pops.linalg import LinearProblem,LinearOperatorProperties
from pops.solvers import CG
from pops.time._program.solve_request import validate_solve_request_node

def prepared():
    program=Program('typed immutable equation')
    rhs=program.scalar_field('bound RHS')
    operator=program.matrix_free_operator('linear',domain='scalar',range_='scalar',ncomp=1)
    program.set_apply(operator,lambda p,out,value:2*value)
    request=SolveRequest(LinearProblem(operator,rhs,nullspace=None,properties=LinearOperatorProperties.symmetric_positive_definite()),unknowns=(SolveUnknown('solution',rhs),),equation_inputs={'operator':operator,'rhs':rhs},seeds={'solution':None})
    program.solve(request,solver=CG(max_iter=4)).consume(action=FailRun())
    token=next(node for node in program._values if node.op=='solve_linear')
    return program,rhs,operator,token

def test_named_later_image_never_retargets_sealed_rhs():
    p,rhs,operator,token=prepared()
    before=dict(token.attrs['solve_request'])
    original_point=rhs.point
    image=p.value('later named image',rhs,at=p.stage('same time distinct stage',c=0))
    assert image.id!=rhs.id
    assert p._canonical_value(rhs).name=='bound RHS' and p._canonical_value(rhs).point==original_point
    validate_solve_request_node(p,token)
    assert token.attrs['solve_request']==before
    rebuilt=p._rebuild(lambda value:True,transformation='normalize')
    validate_solve_request_node(rebuilt,next(node for node in rebuilt._values if node.op=='solve_linear'))

@pytest.mark.parametrize('drift',('rhs','operator'))
def test_real_equation_drift_still_refused(drift):
    p,rhs,operator,token=prepared()
    if drift=='rhs':p._replace_value(rhs,attrs={**rhs.attrs,'ncomp':2})
    else:
        current=p._canonical_value(operator)
        p._replace_value(current,attrs={**current.attrs,'apply_result':3*current.attrs['apply_in']})
    with pytest.raises(SolveRequestError,match='equation_identity_drift'):validate_solve_request_node(p,token)

def test_unchanged_name_point_preserves_existing_ir():
    p=Program('no materialization needed')
    rhs=p.scalar_field('unchanged')
    before=p._ir_hash()
    same=p.value('unchanged',rhs,at=rhs.point)
    assert same is rhs and p._ir_hash()==before
