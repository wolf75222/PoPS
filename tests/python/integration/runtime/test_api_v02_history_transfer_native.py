"""Actual PoPS AMR history matrix, nonuniform BDF consumer, restart and rollback.

All ranks enter native collectives. The same module is a serial/MPI reception;
no local Python evaluator stands in for the generated temporal transfer.
"""
import struct
import numpy as np
import pytest
import pops
from examples.api_v02.method_history_transfer import build_history_transfer
from tests.python.support.api_v02_native import _compile, _bind
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.public_native_profile import profile_gather_value
from tests.python.support.integral_state_receipts import collective_directory

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT=900


def _sample_rows(raw):
    assert raw[:8]==b"POPSHID2"
    size,=struct.unpack_from("<Q",raw,8)
    offset=16+size
    level,count=struct.unpack_from("<qQ",raw,offset)
    offset+=16
    rows=[]
    for kind,begin,interval,ordinal,*digest in struct.iter_unpack("<8Q",raw[offset:]):
        def number(bits):
            return struct.unpack("<d",struct.pack("<Q",bits))[0]
        rows.append(dict(kind=kind,time=number(begin),interval=number(interval),
                         ordinal=ordinal,digest=tuple(digest)))
    assert len(rows)==count
    return level,rows


def _image(runtime,world):
    names=runtime.history_names()
    rings={}
    metadata={}
    states={}
    for level in range(runtime.n_levels()):
        states[level]=collective_call(world,lambda level=level:
            runtime.block_level_state_global("method",level))
    for name in names:
        for level in runtime.history_levels(name):
            for slot in range(runtime.history_depth(name)):
                rings[name,level,slot]=collective_call(world,lambda name=name,level=level,slot=slot:
                    runtime.history_global(name,level,slot))
            native=runtime._executor._s
            raw=collective_call(world,lambda name=name,level=level,native=native:
                bytes(native.history_sample_identity(name,level)))
            metadata[name,level]=(raw,tuple(runtime.history_slot_dt(name,level,slot)
                for slot in range(runtime.history_depth(name))))
    accepted=collective_call(world,lambda:bytes(runtime._executor.program_accepted_state()))
    return dict(states=states,rings=rings,metadata=metadata,time=runtime.time(),
                macro=runtime.macro_step(),patches=tuple(runtime.patch_boxes()),accepted=accepted)


def _same(before,after,world):
    with collective_check(world):
        assert before.keys()==after.keys()
        for key in ("states","rings"):
            assert before[key].keys()==after[key].keys()
            for index,value in before[key].items():
                if value is None:
                    assert after[key][index] is None
                else:
                    np.testing.assert_array_equal(value,after[key][index])
        for key in before.keys()-{"states","rings"}:
            assert before[key]==after[key],key


def _check_exact_flux_transfer(image):
    from fractions import Fraction
    from tests.python.support.history_flux_receipts import accepted_state_history_flux
    from tests.python.support.history_transfer_oracles import assert_projected_quadratic_flux
    flux_rings,payload=accepted_state_history_flux(image["accepted"])
    assert payload and flux_rings
    parent=[row for row in flux_rings if row["name"].endswith("retained_rhs")
            and all(slot and all(basis["level"]==0 for basis in slot) for slot in row["slots"])]
    fine=[row for row in flux_rings if row["name"].endswith("retained_rhs")
          and all(slot and all(basis["level"]==1 for basis in slot) for slot in row["slots"])]
    assert len(parent)==len(fine)==1
    _,sources=_sample_rows(image["metadata"]["retained_rhs",0][0])
    _,targets=_sample_rows(image["metadata"]["retained_rhs",1][0])
    interval=Fraction(image["metadata"]["retained_rhs",0][1][1])/2
    exact_targets=tuple(Fraction(image["time"])-lag*interval for lag in (3,1,2))
    assert [row["time"] for row in targets]==list(map(float,exact_targets))
    assert all(row["interval"]==float(interval) for row in targets)
    assert_projected_quadratic_flux(parent[0]["slots"],fine[0]["slots"],
        [row["time"] for row in sources],exact_targets)
    assert all(len(slot)==3 for slot in fine[0]["slots"][:2]), \
        "real transferred expressions must retain all three distinct source evaluations"
    assert any(basis["faces"] for row in flux_rings for slot in row["slots"] for basis in slot)


def _check_transferred_ring(image,world,*,linear_control):
    with collective_check(world):
        fine=[key for key in image["metadata"] if key[1]==1]
        assert len(fine)==1
        name,level=fine[0]
        raw,intervals=image["metadata"][name,level]
        actual_level,samples=_sample_rows(raw)
        assert actual_level==level and len(samples)==3
        assert all(row["kind"]==3 and any(row["digest"]) for row in samples)
        assert samples[1]["time"]>samples[2]["time"]>samples[0]["time"]
        assert abs(intervals[1]-2*intervals[2])<1.e-13
        assert abs(samples[1]["time"]-.12)<1.e-13
        assert abs(samples[2]["time"]-.10)<1.e-13
        for slot,row in enumerate(samples):
            values=image["rings"][name,level,slot]
            if values is not None:
                values=np.asarray(values).reshape(3,32,32)
                np.testing.assert_allclose(values[1],row["time"],rtol=0,atol=2.e-13)
                error=float(np.max(abs(values[0]-row["time"]**2)))
                if linear_control and slot in (0,2):
                    assert error>1.e-5
                else:
                    assert error<2.e-13


@pytest.mark.parametrize("linear_control",(False,True))
def test_native_temporal_matrix_is_consumed_by_all_lag_bdf_and_survives_restart(
        linear_control,tmp_path,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T08_history_transfer",builder=build_history_transfer,
                                linear_control=linear_control)
    runtime=_bind(spec,artifact,world)
    initial_boxes=tuple(runtime.patch_boxes())
    collective_call(world,lambda:pops.run(runtime,t_end=.16,max_steps=4,console=False))
    transferred=_image(runtime,world)
    with collective_check(world):
        assert transferred["macro"]==4 and abs(transferred["time"]-.16)<1.e-13
        assert transferred["patches"]!=initial_boxes,"the test must create actual new fine coverage"
    _check_transferred_ring(transferred,world,linear_control=linear_control)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/"transferred_history")))
    restarted=_bind(spec,artifact,world)
    collective_call(world,lambda:restarted.restart(checkpoint))
    _same(transferred,_image(restarted,world),world)
    for active in (runtime,restarted):
        collective_call(world,lambda active=active:pops.run(active,t_end=.18,max_steps=1,console=False))
        rows=profile_gather_value(world,collective_call(world,lambda active=active:spec.invariants(active)))
        with collective_check(world):
            # block_level_state_global gathers the level into every rank, so every rank
            # evaluates the same metrics from the same image.
            owners=[row for row in rows if row.get("metrics_owner",True)]
            assert len(owners)==len(rows) and all(row==owners[0] for row in owners),rows
            assert owners[0]["passed"] is (not linear_control),owners
            if linear_control:
                assert owners[0]["quadratic_error"]>1.e-5
        record_property("native_metrics",str(rows))
    _same(_image(runtime,world),_image(restarted,world),world)
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks",1 if world is None else int(world.size))


def test_parent_rejection_restores_reconstructed_samples_and_all_native_history_state(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world=_compile("T08_history_rollback",builder=build_history_transfer,reject_parent=True)
    runtime=_bind(spec,artifact,world)
    collective_call(world,lambda:pops.run(runtime,t_end=.16,max_steps=4,console=False))
    before=_image(runtime,world)
    _check_transferred_ring(before,world,linear_control=False)
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=.18,max_steps=1,console=False))
    with collective_check(world):
        assert failures and all(failures),failures
        assert all("rejected" in failure[1].lower() for failure in failures),failures
    _same(before,_image(runtime,world),world)


def test_real_flux_history_matrix_preserves_distinct_sources_reflux_and_restart(
        tmp_path,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    from examples.api_v02.method_history_transfer import build_flux_history_transfer
    spec,artifact,world=_compile("T08_flux_history",builder=build_flux_history_transfer)
    runtime=_bind(spec,artifact,world)
    initial_boxes=tuple(runtime.patch_boxes())
    collective_call(world,lambda:pops.run(runtime,t_end=.16,max_steps=4,console=False))
    image=_image(runtime,world)
    mass=collective_call(world,lambda:runtime.integral("method",component=0))
    with collective_check(world):
        assert image["patches"]!=initial_boxes
        fine_cells=sum((upper[0]-lower[0]+1)*(upper[1]-lower[1]+1)
            for level,lower,upper in image["patches"] if level==1)
        assert 0<fine_cells<64**2,"the post-transfer continuation must retain actual coarse-fine faces"
        assert abs(mass-1.)<3.e-11
        _,samples=_sample_rows(image["metadata"]["retained_rhs",1][0])
        np.testing.assert_allclose([row["time"] for row in samples],[.10,.14,.12],rtol=0,atol=1.e-13)
        assert all(row["kind"]==3 and any(row["digest"]) for row in samples)
        normalized=[]
        for slot,row in enumerate(samples):
            raw=image["rings"]["retained_rhs",1,slot]
            if raw is not None:
                field=np.asarray(raw).reshape(4,64,64)
                mask=field[1]>.5
                assert mask.sum()==fine_cells
                normalized.append(field[0][mask]/row["time"]**2)
        if normalized:
            assert np.max(abs(normalized[0]))>1.
            for other in normalized[1:]:
                np.testing.assert_allclose(other,normalized[0],rtol=0,atol=3.e-11)
        _check_exact_flux_transfer(image)
    directory=collective_directory(world,tmp_path)
    checkpoint=collective_call(world,lambda:runtime.checkpoint(str(directory/"actual_flux_history")))
    restarted=_bind(spec,artifact,world)
    collective_call(world,lambda:restarted.restart(checkpoint))
    _same(image,_image(restarted,world),world)
    for active in (runtime,restarted):
        collective_call(world,lambda active=active:pops.run(active,t_end=.32,max_steps=4,console=False))
        second=_image(active,world)
        mass=collective_call(world,lambda active=active:active.integral("method",component=0))
        with collective_check(world):
            assert abs(mass-1.)<3.e-11
            assert second["patches"]!=image["patches"],"a second new-coverage matrix must actually execute"
            _,samples=_sample_rows(second["metadata"]["retained_rhs",1][0])
            np.testing.assert_allclose([row["time"] for row in samples],[.26,.30,.28],rtol=0,atol=1.e-13)
            normalized=[]
            for slot,row in enumerate(samples):
                assert row["kind"]==3 and any(row["digest"])
                raw=second["rings"]["retained_rhs",1,slot]
                if raw is not None:
                    field=np.asarray(raw).reshape(4,64,64)
                    mask=field[1]>.5
                    normalized.append(field[0][mask]/row["time"]**2)
            if normalized:
                for other in normalized[1:]:
                    np.testing.assert_allclose(other,normalized[0],rtol=0,atol=3.e-11)
            _check_exact_flux_transfer(second)
        collective_call(world,lambda active=active:pops.run(active,t_end=.36,max_steps=1,console=False))
        mass=collective_call(world,lambda active=active:active.integral("method",component=0))
        report=collective_call(world,active.program_report)
        with collective_check(world):
            assert abs(mass-1.)<3.e-11
            assert report.flux_ledger,"the matrix must feed the real post-remap reflux ledger"
    _same(_image(runtime,world),_image(restarted,world),world)
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks",1 if world is None else int(world.size))
