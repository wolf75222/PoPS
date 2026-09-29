"""Installed Dim2 C11 AMR: real partial refinement, stages and conserved row faces."""
import numpy as np
import pops
import pytest
from tests.python.support.principal_amr_case import principal_amr_case, CELLS, DT
from tests.python.support.amr_snapshots import composite_active_mask, level_valid_mask
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _bind(artifact, handles, gates, slopes, dissipation, *, threshold=1.e6):
    params = {}
    for owner, tables in enumerate(handles):
        for row, (slope_handle, dissipation_handle) in enumerate(tables):
            params[slope_handle] = slopes[row] if row == owner else -.37-owner
            params[dissipation_handle] = dissipation[row] if row == owner else 3.7+owner
    params.update({handle: threshold if i == 0 else 1.e6 for i,handle in enumerate(gates)})
    return pops.bind(artifact, params=params,
                     resources={"execution_context": artifact_execution_context(artifact)})


def _snapshot(runtime, widths, levels, world):
    result = []
    for level in range(levels):
        rows = [runtime.block_level_state_global("block%d" % i, level) for i in range(len(widths))]
        if world is None or world.rank == 0:
            n = CELLS*2**level
            result.append(np.concatenate([np.asarray(row).reshape(width,n,n).copy()
                                          for row,width in zip(rows,widths)]))
    return result


def _mass(runtime, arrays):
    return sum((value[:, composite_active_mask(runtime,level,refinement_ratio=2)].sum(axis=1)
                /(CELLS*2**level)**2 for level,value in enumerate(arrays)))


def _rhs(value, matrices, widths, slopes, dissipation):
    slope = np.repeat(slopes,widths)[:,None,None]
    d = np.repeat(dissipation,widths)[:,None,None]
    result = np.zeros_like(value)
    for matrix, axis in zip(matrices,(2,1)):
        right_cell = np.roll(value,-1,axis=axis)
        left = value+slope*(right_cell-np.roll(value,1,axis=axis))
        right = right_cell+slope*(value-np.roll(value,-2,axis=axis))
        speed = np.max(np.sum(abs(matrix),axis=1))
        flux = .5*np.einsum('ij,jyx->iyx',matrix,left+right)-d*speed*(right-left)
        result += value.shape[-1]*(np.roll(flux,1,axis=axis)-flux)
    return result


@pytest.mark.parametrize("widths,reverse,levels,stages", (
    ((1,1),False,1,False), ((1,1),False,1,True), ((1,1),True,2,True),
    ((2,3),False,2,False), ((2,3),True,2,True)))
def test_principal_amr_joint_faces_preserve_composite_components_and_rebind(
        isolated_native_cache,native_cxx,kokkos_root,widths,reverse,levels,stages):
    case,layout,handles,gates,matrices = principal_amr_case(widths,levels=levels,reverse=reverse,stages=stages)
    artifact,world = _compile(case,layout,"principal-amr-joint")
    previous = None
    for slopes,dissipation in (((.07,.19),(.6,.9)),((.23,-.04),(1.1,.55))):
        runtime = _bind(artifact,handles,gates,slopes,dissipation)
        before = _snapshot(runtime,widths,levels,world)
        boxes = tuple(runtime.patch_boxes())
        def check_initial():
            assert runtime.n_levels() == levels
            # patch_boxes exposes refined patches; the complete coarse domain
            # is the native level-zero array, checked separately below.
            assert {int(row[0]) for row in boxes} == set(range(1,levels))
            assert before[0].shape == (sum(widths),CELLS,CELLS)
            order = tuple(reversed(range(sum(widths)))) if reverse else tuple(range(sum(widths)))
            for level,value in enumerate(before):
                n = CELLS*2**level
                x,y = np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n)
                expected = np.stack([2.+k+.13*(1+.05*k)*np.sinc(1/n)*np.cos(2*np.pi*(x-.5))
                                     +.07*(1+.03*k)*np.sinc(1/n)*np.cos(2*np.pi*(y-.5)) for k in order])
                valid = level_valid_mask(runtime,level,refinement_ratio=2)
                np.testing.assert_allclose(value[:,valid],expected[:,valid],rtol=0,atol=5.e-12)
            if levels == 2:
                valid = level_valid_mask(runtime,1,refinement_ratio=2)
                assert 0 < valid.sum() < valid.size, "qualification requires a genuine coarse/fine interface"
        _root_check(world,check_initial)
        report = pops.run(runtime,t_end=DT,max_steps=1,console=False)
        after = _snapshot(runtime,widths,levels,world)
        def check_result():
            assert report.accepted_steps == 1 and runtime.time == pytest.approx(DT)
            assert tuple(runtime.patch_boxes()) == boxes
            np.testing.assert_allclose(_mass(runtime,after),_mass(runtime,before),rtol=0,atol=2.e-11)
            for level,values in enumerate(after):
                valid = level_valid_mask(runtime,level,refinement_ratio=2)
                assert np.isfinite(values[:,valid]).all()
                assert np.max(abs(values[:,valid]-before[level][:,valid])) > 1.e-7
            if levels == 1:
                derivative = _rhs(before[0],matrices,widths,slopes,dissipation)
                expected = (before[0]+.5*DT*(derivative+_rhs(before[0]+DT*derivative,matrices,widths,slopes,dissipation))
                            if stages else before[0]+DT*derivative)
                np.testing.assert_allclose(after[0],expected,rtol=3.e-12,atol=3.e-12)
            if levels == 2:
                # Independent stencil oracle only on fine cells whose complete
                # two-stage dependency cone lies inside the saved fine support.
                valid = level_valid_mask(runtime,1,refinement_ratio=2)
                interior = valid.copy()
                for axis in (0,1):
                    for offset in range(1, (4 if stages else 2)+1):
                        interior &= np.roll(valid,offset,axis=axis) & np.roll(valid,-offset,axis=axis)
                assert interior.any(), "fine oracle requires an interior dependency cone"
                derivative = _rhs(before[1],matrices,widths,slopes,dissipation)
                expected = (before[1]+.5*DT*(derivative+_rhs(before[1]+DT*derivative,matrices,widths,slopes,dissipation))
                            if stages else before[1]+DT*derivative)
                np.testing.assert_allclose(after[1][:,interior],expected[:,interior],rtol=3.e-12,atol=3.e-12)
            if previous is not None:
                assert np.max(abs(after[0]-previous[0])) > 1.e-8
        _root_check(world,check_result)
        previous = after


def test_principal_amr_active_singular_flux_rolls_back_all_rows_levels_and_time(
        isolated_native_cache,native_cxx,kokkos_root):
    widths,levels=(1,1),2
    case,layout,handles,gates,_=principal_amr_case(widths,levels=levels,singular=True,stages=True)
    artifact,world=_compile(case,layout,"principal-amr-invalid")
    # One artifact verifies inactive invalid algebra before the active-failure run.
    valid=_bind(artifact,handles,gates,(.07,.19),(.6,.9))
    pops.run(valid,t_end=DT,max_steps=1,console=False)
    runtime=_bind(artifact,handles,gates,(.07,.19),(.6,.9),threshold=2.12)
    before=_snapshot(runtime,widths,levels,world)
    boxes,time,step=tuple(runtime.patch_boxes()),runtime.time,runtime.macro_step
    error=None
    try:
        pops.run(runtime,t_end=DT,max_steps=1,console=False)
    except Exception as exc:
        error=str(exc)
    if world is not None:
        from pops._native_collectives import allgather_value
        failures=allgather_value(world,error)
    else:
        failures=[error]
    assert all(failures), failures
    after=_snapshot(runtime,widths,levels,world)
    assert runtime.time == time and runtime.macro_step == step
    def check():
        assert tuple(runtime.patch_boxes()) == boxes
        for actual,expected in zip(after,before):
            np.testing.assert_array_equal(actual,expected)
    _root_check(world,check)
