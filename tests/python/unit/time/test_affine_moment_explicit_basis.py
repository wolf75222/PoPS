"""Public explicit monomial/State binding; genuine Source authoring and lowering."""
import sys
from pathlib import Path
import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.moments import CartesianMonomialBasis
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_affine_moments import emit_affine_moment_kernel


def authored(order=2):
    indices = tuple((p,q) for q in range(order+1) for p in range(order+1-q))
    basis = CartesianMonomialBasis(indices[::-1])
    names = tuple('coordinate_'+str(i) for i in range(len(indices)))
    binding = dict(zip(indices, names[::-1]))
    frame = Rectangle('box',(0,0),(1,1)).frame(Cartesian2D())
    model = pops.Model('unlabelled_moments',frame=frame)
    state = model.state('other_species',components=names)
    matrix = [[0.] * len(names) for _ in names]
    x,y = names.index(binding[(1,0)]),names.index(binding[(0,1)])
    matrix[x][y],matrix[y][x] = 3.,-3.
    operator=model.operator('skew',returns=model.local_linear_operator('skew',on=state,matrix=tuple(map(tuple,matrix))))
    case=pops.Case('explicit_case'); block=case.block('other',model=model)
    program=pops.Program('explicit'); value=program.state(block[state])
    updated=program.affine_moment_update(value.n,value.n,linear_operator=operator,theta_dt=program.dt/2,basis=basis,components=binding)
    program.commit(value.next,program.value('accepted',updated,at=value.next.point))
    return model,program,updated,basis,binding,names


def test_source_origin_and_public_permuted_binding():
    assert Path(pops.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[4]/'python')
    assert 'pops._pops' not in sys.modules
    model,program,updated,basis,binding,names=authored()
    assert tuple(updated.attrs['component_binding']) == tuple(binding[i] for i in basis.indices)
    lowered,_=lower_and_validate(model,facade=model)
    source=emit_cpp_program(program,model=lowered)
    canonical=tuple((p,q) for q in range(3) for p in range(3-q))
    for i,index in enumerate(canonical):
        slot=names.index(binding[index])
        assert f'old_moments[{i}] = u0A(index, {slot});' in source
        assert f'outA(index, {slot}) = valid ? mapped[{i}] : old_moments[{i}];' in source
    assert program._serialize(include_provenance=False)['version']==23


def test_order_five_public_emission_no_demonstrator_ceiling():
    model,program,*_=authored(5)
    lowered,_=lower_and_validate(model,facade=model)
    assert 'affine_velocity_push_forward<5>' in emit_cpp_program(program,model=lowered)


@pytest.mark.parametrize('indices',[((False,0),(1,0),(0,1)),((0,0),(1,0)),((0,0),(0,0)),((0,),(1,0))])
def test_basis_refuses_nonexact_or_incomplete(indices):
    with pytest.raises(ValueError): CartesianMonomialBasis(indices)


def test_compiler_refuses_absent_or_foreign_binding_before_kernel():
    model,program,updated,*_=authored()
    lowered,_=lower_and_validate(model,facade=model)
    attrs=dict(updated.attrs)
    for mutation in ({k:v for k,v in attrs.items() if k!='basis'},dict(attrs,component_binding=('foreign',)*6)):
        with pytest.raises(ValueError,match='basis|binding'):
            emit_affine_moment_kernel(lowered,mutation,'u0','u0','u1','status','active',0,provider_plans=None,consumer_qid='test')


@pytest.mark.parametrize('order',[65535,2**31-1])
def test_public_huge_order_refuses_before_compatibility_expansion(monkeypatch,order):
    from pops.moments import model_builder
    program=pops.Program('oversized')
    def forbidden_expansion(*args,**kwargs):
        pytest.fail('moment names expanded before native cardinality refusal')
    monkeypatch.setattr(model_builder,'moment_names',forbidden_expansion)
    with pytest.raises(ValueError,match='cardinality exceeds native component index'):
        program.affine_moment_update(None,None,linear_operator=None,theta_dt=1,order=order)
