"""Installed-package cubature Raw path; no full Fan–Li/AMR qualification."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from tests.python.support.atomic_cubature_path_case import make_case
from tests.python.support.atomic_cubature_fv_oracle import DT,NX,NY,initial_averages,forward_euler,authenticate_carrier_values
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
SCHEMA='pops.atomic-cubature-raw-native-fixture@1'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(world,runtime):
    values=collective_call(world,lambda:np.array(runtime.block_level_state_global('population',0),copy=True))
    carriers=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
    clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),runtime.n_levels()))
    return values,carriers,clock


@pytest.mark.parametrize('nonconservative',[False,True],ids=['conservative-atoms','raw-density-product'])
def test_installed_atomic_cubature_raw_path(nonconservative,tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'installed package required'
    case,layout=collective_call(world,lambda:make_case(nonconservative=nonconservative,amr=True,fixed_dt=DT))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='atomic-raw-'+str(nonconservative),compile_artifact=pops.compile))
    subject=resolved.initial_condition_plan.bindings[0].subject
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={subject:initial_averages()},
        resources={'execution_context':artifact_execution_context(artifact)}))
    directory=collective_directory(world,tmp_path/'atomic-raw')
    images=[capture(world,runtime)]; reports=[]
    for step in (1,2):
        reports.append(collective_call(world,lambda:pops.run(runtime,t_end=step*DT,max_steps=1,console=False)))
        images.append(capture(world,runtime))
    def persist():
        if world is None or world.rank==0:
            for phase,image in zip(('initial','accepted','continuous'),images,strict=True):
                np.save(directory/(phase+'.npy'),image[0],allow_pickle=False)
                (directory/(phase+'.carriers')).write_bytes(image[1])
            # No regenerated fallback is permitted for provenance.
            if type(artifact._generated_cpp) is not str: raise ValueError('retained Program C++ required')
            (directory/'program.cpp').write_text(artifact._generated_cpp)
            artifact.dump_ir(directory/'program.ir.json')
            (directory/'compiled-manifest.json').write_text(json.dumps(artifact.manifest().to_dict(),sort_keys=True,allow_nan=False)+'\n')
            receipt={'schema':SCHEMA,'nonconservative':nonconservative,'shape':[6,NY,NX],'dt':DT,
                'ranks':1 if world is None else int(world.size),'clocks':[i[2] for i in images],
                'package':str(Path(pops.__file__).resolve()),'native':{'path':str(Path(native.__file__).resolve()),'sha256':digest(native.__file__)},
                'compiled':{'path':str(Path(artifact.so_path).resolve()),'sha256':digest(artifact.so_path),
                            'abi_key':artifact.abi_key,'problem_hash':artifact.problem_hash,'cache_key':artifact.cache_key},
                'files':{p.name:digest(p) for p in sorted(directory.iterdir()) if p.is_file()},
                'root_scientific_approval':False,'full_m17_qualification':False}
            (directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,allow_nan=False)+'\n')
    collective_call(world,persist)
    record_property('atomic_cubature_receipt',str(directory/'receipt.json'))
    with collective_check(world):
        expected=initial_averages()
        for step,(image,blob,clock) in enumerate(images):
            saved=np.load(directory/(('initial','accepted','continuous')[step]+'.npy'),allow_pickle=False)
            assert saved.dtype==image.dtype and saved.shape==image.shape and saved.tobytes()==image.tobytes()
            authenticate_carrier_values(blob,saved)
            np.testing.assert_allclose(saved.reshape(expected.shape),expected,rtol=2e-12,atol=2e-13)
            assert blob and clock==(step*DT,step,1)
            expected=forward_euler(expected,nonconservative=nonconservative)
        assert all(report.accepted_steps==1 for report in reports)
