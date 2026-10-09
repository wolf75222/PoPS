"""Public same-physics diffusion/Field witness with independent saved-array oracle."""
import hashlib,json,os,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
import pops
import pytest
from pops.fields import CompositeHierarchySolve
from pops.solvers.elliptic import GeometricMG
from pops.solvers.tolerances import AbsoluteFloor,Relative
from tests.python.support.public_diffusion_field_case import (
    author_case,one_level_amr_layout,CELLS,KAPPA,DT,BASE,AMPLITUDE,RHS_OFFSET,SOLVER_RTOL,SOLVER_ATOL)
from tests.python.support.diffusion_field_oracle import check_saved
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

@pytest.mark.parametrize('method,composition,variant,kappa',(
    ('euler','weighted_rates','baseline',float(KAPPA)),
    ('ssprk2','weighted_rates','baseline',float(KAPPA)),
    ('ssprk2','weighted_predictor','baseline',float(KAPPA)),
    ('euler','weighted_rates','renamed',float(KAPPA)),
    ('euler','weighted_rates','coefficient_perturbation',.125)),
    ids=('euler','ssprk2_rates','ssprk2_predictor','euler_renamed','euler_kappa0125'))
def test_public_diffusion_predictor_is_field_rhs(method,composition,variant,kappa,tmp_path,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    native=select_native_dimension(2)
    main=Path(__file__).resolve().parents[4]
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=main,text=True).strip()
    expected_native=sha(native.__file__)
    if os.environ.get('POPS_DIFFUSION_FIELD_SOURCE_SHA'):
        assert source==os.environ['POPS_DIFFUSION_FIELD_SOURCE_SHA']
    if os.environ.get('POPS_DIFFUSION_FIELD_NATIVE_SHA'):
        assert expected_native==os.environ['POPS_DIFFUSION_FIELD_NATIVE_SHA']
    package=Path(pops.__file__).resolve().parent
    site_roots={Path(sysconfig.get_path(kind)).resolve() for kind in ['purelib','platlib']}
    assert any(package.is_relative_to(root) for root in site_roots)
    assert not package.is_relative_to(main), 'Source/prototype package cannot qualify this installed witness'
    from scripts.verify_installed_native import verify_installed_native
    origin=verify_installed_native(expect_dimension=2,expect_mpi=bool(native.__has_mpi__),
        expect_parallel_hdf5=bool(native.__has_parallel_hdf5__))
    # Authenticate the loaded public package, not merely a same-header native binary.
    from scripts.check_packaging_manifest import read_manifest,PYTHON_SOURCE_SUFFIXES
    names=subprocess.check_output(['git','ls-files','--','python/pops'],cwd=main,text=True).splitlines()
    names=[n for n in names if Path(n).suffix in PYTHON_SOURCE_SUFFIXES]
    names+=['include/'+str(p) for p in read_manifest(main).installed_headers]
    names+=['include/pops_headers.manifest']
    for name in names:
        installed=package/(name.removeprefix('python/pops/') if name.startswith('python/pops/') else name)
        assert sha(installed)==sha(main/name), name
    from pops.codegen.abi import module_header_signature
    sdk=module_header_signature()
    world=native.mpi_world() if native.__has_mpi__ else None
    rank,ranks=(int(world.rank),int(world.size)) if world is not None else (0,1)
    shared_directory=collective_directory(world,tmp_path/('diffusion-field-'+method))
    directory=shared_directory/('rank%d'%rank)
    collective_call(world,lambda:directory.mkdir(exist_ok=False))
    labels={'model':'renamed_heat_model','rate':'renamed_balance','method':'renamed_integrator',
        'D':'renamed_first_slope','Y':'renamed_predictor','D1':'renamed_second_slope','accepted':'renamed_endpoint'} if variant=='renamed' else None
    authored=collective_call(world,lambda:author_case(method,rk2_composition=composition,labels=labels,kappa=kappa,layout_factory=one_level_amr_layout,
        field_solver=GeometricMG(tolerance=Relative(SOLVER_RTOL,floor=AbsoluteFloor(SOLVER_ATOL)),max_cycles=100),
        hierarchy_policy=CompositeHierarchySolve()))
    inputs=dict(method=method,rk2_composition=composition,variant=variant,labels=labels,cells=list(CELLS),kappa=kappa,dt=float(DT),mean=float(BASE),amplitude=float(AMPLITUDE),
        rhs_offset=float(RHS_OFFSET),field_stage_fraction=1,solver_rtol=SOLVER_RTOL,solver_atol=SOLVER_ATOL,
        Source=source,Native=expected_native,SDK=sdk,actual_context_scope=dict(layout='AMR',levels=1,rank=rank,ranks=ranks),
        equations=dict(state='ddt(U)=div(CoeffGradient(U[0],kappa))',field='-laplacian(phi)+phi=(3+tau)*Y'),
        source_inputs={str(Path(__file__).resolve()):sha(__file__),str(Path(author_case.__code__.co_filename).resolve()):sha(author_case.__code__.co_filename)})
    # A fresh, literal inputs record precedes compile/run; never derived from actual results.
    collective_call(world,lambda:(directory/'inputs.json').write_text(json.dumps(inputs,sort_keys=True,indent=2)+'\n'))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(authored.case),layout=authored.layout,
        compile_options={'model_source_policy':'require'}))
    artifact=(compile_resolved_plan_once(world,resolved,route='public diffusive RHS predictor Field',compile_artifact=pops.compile)
        if world is not None else pops.compile(resolved))
    subject=artifact.plan.initial_condition_plan.bindings[0].subject
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={subject:authored.initial},
        resources={'execution_context':artifact_execution_context(artifact)}))
    initial=collective_call(world,lambda:np.asarray(runtime.block_level_state_global('material',0)).reshape(authored.initial.shape).copy())
    report=collective_call(world,lambda:pops.run(runtime,t_end=float(DT),max_steps=1,console=False))
    actual=collective_call(world,lambda:np.asarray(runtime.block_level_state_global('material',0)).reshape(authored.initial.shape).copy())
    predictor=collective_call(world,lambda:np.asarray(runtime.history_global('predictor_Y',0,0)).reshape(authored.initial.shape).copy())
    phi=collective_call(world,lambda:np.asarray(runtime.history_global('phi_stage',0,0)).reshape(authored.initial.shape).copy())
    with collective_check(world):
        assert report.accepted_steps==1 and runtime.macro_step()==1 and runtime.time()==float(DT)
    for name,array in [('initial',initial),('predictor',predictor),('accepted',actual),('phi_stage',phi)]:
        collective_call(world,lambda name=name,array=array:np.save(directory/(name+'.npy'),array,allow_pickle=False))
    checkpoint=collective_call(world,lambda:runtime.checkpoint(shared_directory/'accepted'))
    # Retain actual original compiler outputs as text and pins, never regenerate a model.
    for i,block in enumerate(artifact.blocks):
        collective_call(world,lambda i=i,block=block:block.model.dump_cpp(directory/('model%d.cpp'%i)))
    for i,row in enumerate(artifact.layout_programs):
        collective_call(world,lambda i=i,row=row:row.program.dump_cpp(directory/('program%d.cpp'%i)))
    evidence=dict(Source=source,Native=expected_native,SDK=sdk,package_origin=str(package),native_origin=origin,context=native.runtime_environment_report(),
        artifact=artifact.artifact_identity.token,checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),
        compiled_manifest=artifact.manifest().to_dict(),models=[b.model.source_provenance(require_complete=True) for b in artifact.blocks])
    collective_call(world,lambda:(directory/'actual-receipt.json').write_text(json.dumps(evidence,sort_keys=True,indent=2,default=str)+'\n'))
    with collective_check(world):
        oracle=check_saved(directory)
        assert oracle['status']=='PASS'
    record_property('diffusion_field_inputs',str(directory/'inputs.json'))
