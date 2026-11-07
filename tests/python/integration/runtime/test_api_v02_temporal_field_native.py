"""True composite solves on partial AMR with authored point reconstruction."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_temporal_field import build_temporal_field
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200


def _image(runtime, world):
    return dict(accepted=collective_call(world, lambda: bytes(runtime._executor.program_accepted_state())),
        time=runtime.time(), step=runtime.macro_step(), patches=tuple(runtime.patch_boxes()),
        states=[collective_call(world, lambda level=level: runtime.block_level_state_global("method",level))
                for level in range(runtime.n_levels())],
        observed=[collective_call(world, lambda level=level: runtime.history_global("observed",level,1))
                  for level in range(runtime.n_levels())])


def _same(a,b,world):
    with collective_check(world):
        assert a.keys()==b.keys()
        for key in ("accepted","time","step","patches"): assert a[key]==b[key],key
        for key in ("states","observed"):
            for x,y in zip(a[key],b[key],strict=True):
                if x is None: assert y is None
                else: np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize("constant_control", (False,True))
def test_subcycled_composite_temporal_equation_nonuniform_history_and_restart(
        tmp_path,isolated_native_cache,native_cxx,kokkos_root,record_property,constant_control):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("temporal_field_"+str(constant_control),builder=build_temporal_field,
                                constant_control=constant_control)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.16,max_steps=4,console=False))
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/"temporal-field")))
    restarted=_bind(spec,artifact,world)
    collective_call(world,lambda:restarted.restart(checkpoint))
    _same(_image(runtime,world),_image(restarted,world),world)
    for active in (runtime,restarted):
        collective_call(world,lambda active=active:pops.run(active,t_end=.18,max_steps=1,console=False))
        rows=profile_gather_value(world,collective_call(world,lambda active=active:spec.invariants(active)))
        with collective_check(world):
            # block_level_state_global gathers the level into every rank, so every rank
            # evaluates the same metrics from the same image.
            owners=[row for row in rows if row.get("metrics_owner",True)]
            assert len(owners)==len(rows) and all(row==owners[0] for row in owners),rows
            assert owners[0]["passed"] is (not constant_control),owners
            assert len(owners[0]["level_counts"])==2 and all(owners[0]["level_counts"])
            if constant_control: assert owners[0]["field_error"]>1.e-5
        record_property("metrics",str(rows))
    _same(_image(runtime,world),_image(restarted,world),world)
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks",1 if world is None else int(world.size))


def test_parent_rejection_revokes_temporal_sources_and_restores_field_histories(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("temporal_field_rollback",builder=build_temporal_field,reject_parent=True)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.18,max_steps=5,console=False))
    before=_image(runtime,world)
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.20,max_steps=1,console=False))
    with collective_check(world):
        assert failures and all(failures),failures
        assert all("reject" in failure[1].lower() for failure in failures),failures
    _same(before,_image(runtime,world),world)
