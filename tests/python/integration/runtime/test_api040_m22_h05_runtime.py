"""Prepared installed H05 acceptance; never a spatial radiation qualification."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pops
import pytest
from tests.python.support.native_execution_context import artifact_execution_context
from test_user_numerical_bodies_runtime import _compile, _root_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _example():
    folder=Path(__file__).resolve().parents[4]/"examples/migration/scientific"
    old=sys.path[:]
    sys.path.insert(0,str(folder))
    try:
        spec=importlib.util.spec_from_file_location("api040_m22_h05_runtime",folder/"api040_m22_h05.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:]=old


@pytest.mark.parametrize("reverse",[False,True])
def test_h05_original_residual_rebind_and_domain_rollback(
        isolated_native_cache,native_cxx,kokkos_root,tmp_path,reverse):
    example=_example()
    case,layout,subjects,parameter=example.build_case(reverse=reverse)
    validated=pops.validate(case)
    parameter=validated.resolve(parameter)
    artifact,world=_compile(case,layout,"M22-H05-%s" % reverse)
    for k in (.8,0.,-.8):
        initial=np.broadcast_to(np.array((2.,.5))[:,None,None],(2,4,4)).copy()
        runtime=pops.bind(artifact,
            initial_values={subject:initial[i:i+1] for i,subject in enumerate(subjects)},
            params={parameter:k},resources={"execution_context":artifact_execution_context(artifact)})
        before=[np.asarray(runtime.state_global(name)).copy() for name in ("radiation","matter")]
        failure=""
        try:
            pops.run(runtime,t_end=.4,max_steps=1,console=False)
        except RuntimeError as error:
            failure=str(error)
        failures=(failure,)
        if world is not None:
            from pops._native_collectives import allgather_value
            failures=allgather_value(world,failure)
        after=[np.asarray(runtime.state_global(name)).copy() for name in ("radiation","matter")]
        if k < 0:
            assert all(failures)
            assert runtime.time() == 0. and runtime.macro_step() == 0
        else:
            assert not any(failures),failures
            assert runtime.time() == pytest.approx(.4) and runtime.macro_step() == 1
        def check(before=before,after=after,k=k,runtime=runtime):
            saved=tmp_path/("H05_%s_%s.npz" % (reverse,k))
            np.savez_compressed(saved,initial=np.stack(before).reshape(2,4,4),
                final=np.stack(after).reshape(2,4,4),k=k,time=runtime.time())
            with np.load(saved) as data:
                old,new=data["initial"],data["final"]
                if k < 0:
                    np.testing.assert_array_equal(new,old)
                else:
                    expected=example.backward_euler(old,k,.4,1)
                    np.testing.assert_allclose(new,expected,rtol=0,atol=2e-11)
                    np.testing.assert_allclose(new.sum(axis=0),old.sum(axis=0),rtol=0,atol=2e-11)
                    residual=np.stack((new[0]-old[0]+.4*k*(new[0]-new[1]),
                                       new[1]-old[1]-.4*k*(new[0]-new[1])))
                    assert np.max(np.abs(residual)) <= 2e-11
                    assert np.min(new) >= 0
        _root_check(world,check)
