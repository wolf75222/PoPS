"""Installed joint primitive reconstruction against an independent NumPy FV oracle."""
import numpy as np
import pops
import pytest
from tests.python.support.principal_primitive_case import primitive_case,CELLS,DT
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile,_root_check,_assert_rejected

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _initial(count):
    x,y=np.meshgrid((np.arange(CELLS)+.5)/CELLS,(np.arange(CELLS)+.5)/CELLS)
    return np.ascontiguousarray(np.stack([2.+.4*j+.17*np.sinc(1/CELLS)*np.cos(2*np.pi*x+.2*j)
                                         +.09*np.sinc(1/CELLS)*np.sin(2*np.pi*y-.1*j)
                                         for j in range(count)]))


def _minmod(a,b,c):
    same=(np.signbit(a)==np.signbit(b)) & (np.signbit(b)==np.signbit(c))
    return np.where(same,np.copysign(np.minimum(np.minimum(abs(a),abs(b)),abs(c)),a),0.)


def _oracle(initial,matrices,widths,betas,sample_offsets=None):
    coefficients=np.repeat(betas,widths)[:,None,None]
    primitive=initial/initial[0:1]+coefficients*initial[0:1]**2
    primitive[0]=initial[0]
    result=initial.copy()
    def conservative(value):
        state=value[0:1]*(value-coefficients*value[0:1]**2)
        state[0]=value[0]
        return state
    for matrix,axis in zip(matrices,(2,1)):
        lower=primitive-np.roll(primitive,1,axis=axis)
        upper=np.roll(primitive,-1,axis=axis)-primitive
        slope=_minmod(2*lower,.5*(lower+upper),2*upper)
        if sample_offsets is None:
            left=conservative(primitive+.5*slope)
            right=conservative(np.roll(primitive-.5*slope,-1,axis=axis))
        else:
            offsets=np.repeat(sample_offsets,widths)
            left=conservative(np.stack([np.roll(value,-int(offset),axis=axis-1)
                for value,offset in zip(primitive,offsets)]))
            right=conservative(np.stack([np.roll(value,int(offset)-1,axis=axis-1)
                for value,offset in zip(primitive,offsets)]))
        speed=np.max(np.sum(abs(matrix),axis=1))
        flux=.5*np.einsum('ij,jyx->iyx',matrix,left+right)-.5*speed*(right-left)
        result += DT*CELLS*(np.roll(flux,1,axis=axis)-flux)
    return result


def _bind(artifact,subjects,handles,widths,initial,betas):
    offsets=np.cumsum((0,*widths))
    return pops.bind(artifact,params=dict(zip(handles,betas)),
        initial_values={state:initial[offsets[i]:offsets[i+1]].copy() for i,state in enumerate(subjects)},
        resources={'execution_context':artifact_execution_context(artifact)})


def _gather(runtime,widths,world):
    rows=[runtime.state_global('block%d'%i) for i in range(len(widths))]
    if world is None or world.rank==0:
        return np.concatenate([np.asarray(row).reshape(width,CELLS,CELLS) for row,width in zip(rows,widths)])
    return None


@pytest.mark.parametrize('widths,reverse',(((1,1),False),((1,1),True),((2,3),False),((2,3),True),((1,4),True)))
def test_joint_primitive_reconstruction_keeps_cross_fields_parameters_and_permutation(
        isolated_native_cache,native_cxx,kokkos_root,widths,reverse):
    case,layout,subjects,matrices,handles=primitive_case(widths,reverse=reverse)
    artifact,world=_compile(case,layout,'principal-primitive')
    initial=_initial(sum(widths))
    previous=None
    for betas in ((.02,-.03),(-.08,.11)):
        runtime=_bind(artifact,subjects,handles,widths,initial,betas)
        pops.run(runtime,t_end=DT,max_steps=1,console=False)
        actual=_gather(runtime,widths,world)
        def check():
            expected=_oracle(initial,matrices,widths,betas)
            np.testing.assert_allclose(actual,expected,rtol=3.e-12,atol=3.e-12)
            np.testing.assert_allclose(actual.sum(axis=(1,2)),initial.sum(axis=(1,2)),rtol=0,atol=5.e-11)
            if previous is not None:
                assert np.max(abs(actual-previous))>1.e-8
        _root_check(world,check)
        previous=actual


def test_joint_primitive_domain_failure_rejects_all_fields_and_time(
        isolated_native_cache,native_cxx,kokkos_root):
    widths=(2,3)
    case,layout,subjects,_,handles=primitive_case(widths)
    artifact,world=_compile(case,layout,'principal-primitive-domain')
    initial=_initial(sum(widths))
    # Finite values but an invalid coordinate domain in a localized cell.
    initial[0,1,1]=-.25
    runtime=_bind(artifact,subjects,handles,widths,initial,(.02,-.03))
    before=_gather(runtime,widths,world)
    _assert_rejected(runtime,world,DT)
    after=_gather(runtime,widths,world)
    _root_check(world,lambda:np.testing.assert_array_equal(after,before))


def test_joint_primitive_user_sampling_allocates_the_widest_halo_for_all_rows(
        isolated_native_cache,native_cxx,kokkos_root):
    widths,offsets,betas=(2,3),(0,2),(.02,-.03)
    case,layout,subjects,matrices,handles=primitive_case(widths,sample_offsets=offsets)
    artifact,world=_compile(case,layout,'principal-primitive-halo')
    initial=_initial(sum(widths))
    runtime=_bind(artifact,subjects,handles,widths,initial,betas)
    pops.run(runtime,t_end=DT,max_steps=1,console=False)
    actual=_gather(runtime,widths,world)
    _root_check(world,lambda:np.testing.assert_allclose(
        actual,_oracle(initial,matrices,widths,betas,offsets),rtol=3.e-12,atol=3.e-12))
