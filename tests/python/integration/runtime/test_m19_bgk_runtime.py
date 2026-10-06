"""Prospective installed isolated BGK witness; Source collection is not Native proof."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from pops._generated_release_contract import NATIVE_ABI_VERSION
from tests.python.support.m19_bgk_case import build,DT,NU,NAMES
from tests.python.support.m19_bgk_oracle import initial,moments,step
from tests.python.support.collective_checks import collective_attempt,collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance,pin,save_json
from tests.python.support.evidence_json import ENCODING, evidence_dumps
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once


def retain_mapping_sources(artifact,providers,directory):
    """Join every compiled transfer to its authenticated original source package."""
    from pops.external import load,SourceComponentPackage
    packages={}
    for path in sorted(providers.rglob('physical-map.pops.json')):
        package=load(path)
        if type(package) is not SourceComponentPackage:raise TypeError('source transfer package required')
        package.verify();packages[package.identity.token]=package
    entries=[]
    for index,component in enumerate(artifact.component_artifacts):
        component.verify()
        if component.source_package is None or component.fixed_signature:
            raise ValueError('BGK transfers require actual source-compiled component authority')
        package=packages[component.source_package.token]
        root=directory/('map-%d'%index);root.mkdir()
        binary=root/('component'+component.suffix);binary.write_bytes(component.binary)
        payloads={}
        for payload in package.payloads:
            target=root/payload.path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload.content);payloads[payload.path]=pin(target)
        entries.append(dict(compiled=json.loads(evidence_dumps(component.to_data())),
            compiled_encoding=ENCODING,binary=pin(binary),payloads=payloads,
            source_manifest=save_json(root/'source-package.json',package.to_data())))
    if len(entries)!=6 or {c.source_package.token for c in artifact.component_artifacts}!=set(packages):
        raise ValueError('complete six-transfer source/binary closure required')
    return entries


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize('nx,nv,reverse',((32,32,False),(64,64,True)))
def test_installed_isolated_bgk_two_stages(nx,nv,reverse,tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else int(world.rank);ranks=1 if world is None else int(world.size)
    directory=collective_directory(world,tmp_path/'isolated-bgk')
    collective_call(world,lambda:directory.mkdir(exist_ok=True))
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert int(native.__native_dimension__)==2
        assert native.module_capabilities('production')['abi_version']==NATIVE_ABI_VERSION
    providers=directory/'providers'
    collective_call(world,lambda:providers.mkdir(exist_ok=True))
    if world is None:resolved=collective_call(world,lambda:build(providers,nx=nx,nv=nv,reverse=reverse))
    else:
        from tests.python.integration.runtime.test_m19_product_support_runtime import _load_published_provider
        resolved=collective_call(world,lambda:build(providers,nx=nx,nv=nv,reverse=reverse) if rank==0 else None)
        resolved=collective_call(world,lambda:resolved if rank==0 else build(
            providers,nx=nx,nv=nv,reverse=reverse,provider_factory=_load_published_provider))
    artifact=pops.compile(resolved) if world is None else compile_resolved_plan_once(
        world,resolved,route='isolated self BGK %d %d %s'%(nx,nv,reverse),compile_artifact=pops.compile)
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    map_directory=directory/('map-provenance-rank%d'%rank)
    collective_call(world,lambda:map_directory.mkdir())
    mappings=collective_call(world,lambda:retain_mapping_sources(artifact,providers,map_directory))
    collective_call(world,lambda:save_json(directory/('prepared-rank%d.json'%rank),dict(
        schema='sol61.isolated-self-bgk@1',rank=rank,ranks=ranks,cells=[nv,nx],
        array_axes=['component','position','velocity'],dt=[DT.numerator,DT.denominator],
        nu=[NU.numerator,NU.denominator],steps=2,temporal_method='SSPRK2',
        moment_powers=[0,1,2],all_moments_at_each_stage=True,velocity_domain=[-8,8],
        velocity_transport='absent in isolated relaxation',position_boundary='periodic',
        equilibrium='continuous self Maxwellian, no discrete fit/renormalization',
        actual_model_and_program_C25_before_bind=True,actual_transfer_sources_before_bind=mappings,
        artifact=artifact.artifact_identity.token,native=pin(native.__file__),
        native_capabilities=dict(native.module_capabilities('production')),root_received=False)))
    f=initial(nx,nv);physical=moments(f)
    seed={NAMES[0]:f.T[None,:,:].copy(),NAMES[7]:np.stack((np.full((nx,nv),7.),np.full((nx,nv),-13.)))}
    for index,value in enumerate(physical):
        seed[NAMES[4+index]]=value[None,None,:].copy()
        seed[NAMES[1+index]]=np.broadcast_to(value[:,None],(nx,nv))[None,:,:].copy()
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_state=seed,
        resources={'execution_context':artifact_execution_context(artifact)}))
    def capture(label):
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        cursors=collective_call(world,lambda:runtime.consumer_cursors.to_data())
        owners={name:collective_call(world,lambda name=name:runtime.local_boxes(name)) for name in NAMES}
        files={};receipt=directory/('%s-rank%d.json'%(label,rank))
        def write(complete):return save_json(receipt,dict(schema='sol61.bgk-state@1',phase=label,
            rank=rank,ranks=ranks,clock=clock,consumer_cursors=cursors,local_boxes=owners,
            files=files,capture_complete=complete,storage_route='public composite CP9 checkpoint'))
        collective_call(world,lambda:write(False))
        for name in NAMES:
            array=collective_call(world,lambda name=name:runtime.state_global(name))
            path=directory/('%s-rank%d-%s.npy'%(label,rank,name))
            collective_call(world,lambda:np.save(path,array,allow_pickle=False))
            files[name]=collective_call(world,lambda:pin(path))
            collective_call(world,lambda:write(False))
        checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/(label+'-checkpoint')))
        collective_call(world,lambda:save_json(directory/('%s-cp-rank%d.json'%(label,rank)),pin(checkpoint)))
        collective_call(world,lambda:write(True))
        return files
    capture('initial');expected=f.copy()
    for index in (1,2):
        original=None
        def advance():
            nonlocal original
            try:return pops.run(runtime,t_end=index*float(DT),max_steps=1,console=False)
            except Exception as error:original=error;raise
        _,failures=collective_attempt(world,advance)
        collective_call(world,lambda:save_json(directory/('attempt%d-rank%d.json'%(index,rank)),dict(failures=failures)))
        if any(failures):
            try:capture('failed%d'%index)
            except Exception as capture_error:
                if original is not None:
                    original.add_note('capture failure: '+repr(capture_error));raise original
                raise
            if original is not None:raise original
            raise RuntimeError('peer BGK step failed: '+repr(failures))
        actual=capture('accepted%d'%index);expected,stages=step(expected)
        with collective_check(world):
            result=np.load(actual[NAMES[0]]['path'],allow_pickle=False)
            assert np.isfinite(result).all() and np.min(result)>0
            np.testing.assert_allclose(result,expected.T[None,:,:],rtol=0,atol=2e-11)
            for power,value in enumerate(stages['moments']):
                np.testing.assert_allclose(np.load(actual[NAMES[4+power]]['path'],allow_pickle=False),
                    value[None,None,:],rtol=0,atol=2e-11)
                np.testing.assert_allclose(np.load(actual[NAMES[1+power]]['path'],allow_pickle=False),
                    np.broadcast_to(value[:,None],(nx,nv))[None,:,:],rtol=0,atol=2e-11)
            for before,after in zip(moments(f),moments(result[0].T),strict=True):
                np.testing.assert_allclose(after,before,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(np.load(actual[NAMES[7]]['path'],allow_pickle=False),seed[NAMES[7]])
            assert runtime.time()==index*float(DT) and runtime.macro_step()==index
    record_property('bgk_receipt',str(directory/('prepared-rank%d.json'%rank)))
